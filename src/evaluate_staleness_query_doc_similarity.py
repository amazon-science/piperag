"""
This scripts evaluate the retrieval quality of the staleness query, by comparing the stale query results with the non-stale query results.

Example usage:
    python evaluate_staleness_query_doc_similarity.py --test-dataset-spec $WORKSPACE/data/datasets/val_wikipedia/val_wikipedia_chunk9_1K/val_db_c4_to_999.json --max-len 1024 --nprobe 64
    python evaluate_staleness_query_doc_similarity.py --test-dataset-spec $WORKSPACE/data/datasets/val_realnews/val_realnews_chunk31_1K/val_db_c4_to_999.json --max-len 1024 --nprobe 64
    python evaluate_staleness_query_doc_similarity.py --test-dataset-spec $WORKSPACE/data/datasets/val_c4/val_c4_chunk1023_1K/val_db_c4_to_999.json --max-len 1024 --nprobe 64
"""
import argparse
import faiss
import json
import torch
import time
import numpy as np

from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer
import gc

from dataset_retro import RetroDataset, ChunkedSequenceDataset, RetroTrainingExample, ShardedChunkedSequenceDataset, \
    ChunkNeighbourDataset, ShardedChunkNeighbourDataset, \
    RetroDatasetRetrieveRealTime, RetroDatasetRetrieveRealTimeStale, \
    RetroDatasetRetrieveRealTimeStaleUsePerfModel, RetroDatasetRetrieveRealTimeRetrieveOnce


def get_query_embeddings(input_dataset, retrieval_interval, model_st, chunk_tokenizer, staleness=False, stale_steps=None):
    """
    Return a list of list. 
        first dimension: sequence index
        second dimension: query chunk index
    """        

    chunk_size = input_dataset.chunk_size

    query_chunk_embeddings = [] # a list of input_dataset.num_sequences elements, each is a numpy array of shape (num_query_chunks_of_seq, dim)
    for seq_index in range(input_dataset.num_sequences):
        # print("Sequence index: ", seq_index)
        input_chunk_indices = input_dataset.get_chunk_indices_of_sequence(seq_index)
        
        # input_ids, if the sequence length <= max_num_chunks, keep all, otherwise trunk to max_num_chunks
        input_ids = [input_dataset.get_chunk_tokens(chunk_index) for chunk_index in input_chunk_indices]
        input_ids = np.concatenate(input_ids, axis=0)
        num_query_chunks = int(np.ceil((input_ids.shape[0] - chunk_size) / retrieval_interval)) + 1
        query_window_tokens = []
        if staleness: # a special case: keep the same interval, only do staleness shift
            assert stale_steps <= retrieval_interval
            query_window_tokens.append(chunk_tokenizer.decode(input_ids[: chunk_size], skip_special_tokens=True))
            for i in range(1, num_query_chunks):
                query_window_tokens.append(chunk_tokenizer.decode(input_ids[i * retrieval_interval - stale_steps: i * retrieval_interval - stale_steps + chunk_size], skip_special_tokens=True))
            query_chunk_embeddings.append(model_st.encode(query_window_tokens, convert_to_numpy=True, output_value="sentence_embedding", normalize_embeddings=True))
        else: # in the normal case
            for i in range(num_query_chunks):
                query_window_tokens.append(chunk_tokenizer.decode(input_ids[i * retrieval_interval: i * retrieval_interval + chunk_size], skip_special_tokens=True))
            query_chunk_embeddings.append(model_st.encode(query_window_tokens, convert_to_numpy=True, output_value="sentence_embedding", normalize_embeddings=True))

    return query_chunk_embeddings


def evaluate_staleness_query_doc_similarity(
    spec_file: Path, 
    continuation_chunks=1,
    pad_token_idx=0,
    max_len=None,
    retrieval_interval=64,
    nprobe=None,
) -> RetroDatasetRetrieveRealTime:

    spec = json.load(spec_file.open())
    base_dir = spec_file.parent

    # input dataset
    input_dataset = ShardedChunkedSequenceDataset([
        ChunkedSequenceDataset(
            chunks=base_dir / shard["chunks"],
            seq2chunk=base_dir / shard["seq2chunk"],
            chunk2seq=base_dir / shard["chunk2seq"]
        )
        for shard in spec["shards"]
    ])
    chunk_size = input_dataset.chunk_size
    num_sequences = input_dataset.num_sequences

    # retrieval dataset
    index_spec = json.load((base_dir / spec["neighbours"]["index_spec"]).open())
    index_base_dir = base_dir / Path(spec["neighbours"]["index_spec"]).parent
    retrieval_dataset = ShardedChunkedSequenceDataset([
        ChunkedSequenceDataset(
            chunks=index_base_dir / shard["chunks"],
            seq2chunk=index_base_dir / shard["seq2chunk"],
            chunk2seq=index_base_dir / shard["chunk2seq"]
        )
        for shard in index_spec
    ])

    print("Load index...")
    start = time.time()
    index = faiss.read_index(str(base_dir / spec["neighbours"]["faiss_index"]))
    assert index.is_trained, "The index must be trained"
    index.parallel_mode = 1 # intra-query parallelism
    print("Load index takes {} seconds".format(time.time() - start))

    if nprobe is not None:
        index.nprobe = nprobe
        print("Find nprobe in input args, override the one in spec")
        print("Set nprobe as {}".format(index.nprobe))
    elif "nprobe" in spec["neighbours"]: 
        index.nprobe = int(spec["neighbours"]["nprobe"])
        print("No nprobe in input args, find it in spec")
        print("Set nprobe as {}".format(index.nprobe))
    else:
        print("No nprobe in input args or spec, use default nprobe {}".format(index.nprobe))

    # # encode all the query chunk embeddings here
    print("Loading Sentence BERT...")
    model_st = SentenceTransformer("all-MiniLM-L6-v2")
    # model_similarity_eval = SentenceTransformer("paraphrase-mpnet-base-v2")
    # model_similarity_eval = SentenceTransformer("paraphrase-MiniLM-L12-v2")
    model_similarity_eval = SentenceTransformer("msmarco-bert-base-dot-v5") # this one is the best
    # model_similarity_eval = SentenceTransformer("sentence-t5-large")
    if torch.cuda.is_available():
        print("Moving model to GPU...")
        device = "cuda"
        # device = f"cuda:{args.gpu_id}" if args.gpu_id is not None else "cuda"
        model_st.to(device)

    queries_dict = {} # key = stale_steps, value = query_chunk_embeddings
    retrieved_chunk_indices_dict = {} # key = stale_steps, value = retrieved_chunk_indices
    retrieved_doc_indices_dict = {} # key = stale_steps, value = retrieved_doc_indices
    retrieved_tokens_dict = {} # key = stale_steps, value = retrieved_tokens
    # get the search results of the stale and non-stale query
    for stale_steps in [0, 1, 2, 4, 8, 16, 32, 64]:

        staleness = False if stale_steps == 0 else True
        print("Generating sentence embeddings...")
        chunk_tokenizer = AutoTokenizer.from_pretrained("t5-base", use_fast=True, model_max_length=10000)
        query_chunk_embeddings = get_query_embeddings(input_dataset, retrieval_interval, model_st, chunk_tokenizer, staleness=staleness, stale_steps=stale_steps)
        queries_dict[stale_steps] = query_chunk_embeddings

        total_chunk_num = 0
        total_unique_doc_num_per_seq = 0
        retrieved_chunk_indices_dict[stale_steps] = {}
        retrieved_doc_indices_dict[stale_steps] = {}
        retrieved_tokens_dict[stale_steps] = {}
        for seq_id in range(num_sequences):
            # here, set num neighbours as 1, to get the average number of different documents per chunk
            query_embeddings = queries_dict[stale_steps][seq_id]
            _, retrieved_chunk_indices = index.search(query_embeddings, 1)
            # convert to 1d list
            retrieved_chunk_indices = retrieved_chunk_indices.flatten().tolist()

            # Get retrieved document IDs
            retrieved_doc_indices = []
            for retrieved_chunk_index in retrieved_chunk_indices:
                doc_id = retrieval_dataset.get_sequence_index_of_chunk(retrieved_chunk_index)
                retrieved_doc_indices.append(doc_id)
            chunk_num = len(query_embeddings)
            unique_doc_num = len(set(retrieved_doc_indices))
            total_chunk_num += chunk_num
            total_unique_doc_num_per_seq += unique_doc_num
            retrieved_chunk_indices_dict[stale_steps][seq_id] = retrieved_chunk_indices
            retrieved_doc_indices_dict[stale_steps][seq_id] = retrieved_doc_indices
    
            # Get the retrieved neighbour tokens
            retrieved_tokens_per_query = []
            for i, neighbour_chunk_idx in enumerate(retrieved_chunk_indices):
                retrieved_tokens = retrieval_dataset.get_chunk_tokens(
                    neighbour_chunk_idx, # the continuous start from old 
                    include_continuation_chunks=continuation_chunks + 1 # retrieve an extra chunk for trimming
                )
                if stale_steps > 0 and i > 0: # remove the context of the stale query
                    retrieved_tokens = retrieved_tokens[chunk_size:]
                else: # no stale, or stale but in the first chunk
                    retrieved_tokens = retrieved_tokens[:-chunk_size]
                retrieved_tokens_per_query.append(chunk_tokenizer.decode(retrieved_tokens, skip_special_tokens=True))
            retrieved_tokens_dict[stale_steps][seq_id] = retrieved_tokens_per_query
          
        print("stale_steps: {}, total_chunk_num: {}, total_unique_doc_num_per_seq: {}, average unique doc num per chunk: {}".format(stale_steps, total_chunk_num, total_unique_doc_num_per_seq, total_unique_doc_num_per_seq / total_chunk_num))
        min_doc_id = min([min(retrieved_doc_indices_dict[stale_steps][seq_id]) for seq_id in range(num_sequences)])
        max_doc_id = max([max(retrieved_doc_indices_dict[stale_steps][seq_id]) for seq_id in range(num_sequences)])
        print("doc ids (sample): {}, min doc id: {}, max doc id: {}".format(retrieved_doc_indices_dict[stale_steps][0][:10], min_doc_id, max_doc_id))
        min_chunk_id = min([min(retrieved_chunk_indices_dict[stale_steps][seq_id]) for seq_id in range(num_sequences)])
        max_chunk_id = max([max(retrieved_chunk_indices_dict[stale_steps][seq_id]) for seq_id in range(num_sequences)])
        print("chunk ids (sample): {}, min chunk id: {}, max chunk id: {}".format(retrieved_chunk_indices[:10], min_chunk_id, max_chunk_id))

        # compare non-stale query results with stale query results, especially the percentage of document id overlap
        chunk_count = 0
        overlap_count = 0
        for seq_id in range(num_sequences):
            retrieved_doc_indices_stale = retrieved_doc_indices_dict[stale_steps][seq_id]
            retrieved_doc_indices_non_stale = retrieved_doc_indices_dict[0][seq_id]
            chunk_num = len(retrieved_doc_indices_non_stale)
            for i in range(chunk_num):
                if retrieved_doc_indices_non_stale[i] == retrieved_doc_indices_stale[i]:
                    overlap_count += 1
            chunk_count += chunk_num
        print("stale_steps: {}, doc overlap percentage: {:.2f}%".format(stale_steps, overlap_count / chunk_count * 100))

        # compare the retrieved token chunk similarity using sentence bert
        similarity_list = []
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        for seq_id in range(num_sequences):
            retrieved_tokens_stale = retrieved_tokens_dict[stale_steps][seq_id]
            retrieved_tokens_non_stale = retrieved_tokens_dict[0][seq_id]
            chunk_num = len(retrieved_tokens_non_stale)
            
            embeddings_stale = model_similarity_eval.encode(retrieved_tokens_stale, convert_to_tensor=True)
            embeddings_non_stale = model_similarity_eval.encode(retrieved_tokens_non_stale, convert_to_tensor=True)

            #Compute cosine-similarities
            cosine_scores = cos(embeddings_stale, embeddings_non_stale).cpu().tolist()
            similarity_list += cosine_scores
        print("stale_steps: {}, average cosine similarity: {:.4f}".format(stale_steps, np.mean(similarity_list)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data args
    parser.add_argument("--test-dataset-spec", required=True, type=Path)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--nprobe", type=int, default=64)

    args = parser.parse_args()
    
    evaluate_staleness_query_doc_similarity(
        spec_file=args.test_dataset_spec,
        continuation_chunks=1,
        pad_token_idx=0,
        max_len=args.max_len,
        retrieval_interval=64,
        nprobe=args.nprobe,
    )
