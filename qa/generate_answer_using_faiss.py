"""
This is the PyTorch inference program, by retrieving from the Faiss server.

"""
from argparse import ArgumentError
import json
import readline
import torch
import time
import os
import sys

sys.path.append('../src')
from pathlib import Path
from dataset_retro import ChunkedSequenceDataset, ShardedChunkedSequenceDataset
from modeling_retro import RetroConfig
from sentence_transformers import SentenceTransformer
from retrieval import RetrieverWithCache, IndexServiceRetriever, IndexServiceClient, DummyRetriever
from faiss_retrieval import FaissRetriever
from train_retro import RetroModelLMHeadLightning, RetroModelLMHeadLightningInference
from data.tokenize_and_chunk import get_tokenizer
# from generate_retro_original import DemoRetriever


"""
Example Usage:

python generate_answer_using_faiss.py \
    --retro-config $WORKSPACE/data/model/retro.json \
    --checkpoint $WORKSPACE/data/model/model.ckpt \
    --num-neighbours 2 \
    --num-continuation-chunks 1 
"""

def main_autoregressive(args):
    
    config = RetroConfig(**json.load(args.retro_config.open()))
    tokenizer = get_tokenizer()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device:", device)
    
    # get the EOS token
    eos_token_id = tokenizer.eos_token_id
    print("EOS token id:", eos_token_id)
    pad_token_id = tokenizer.pad_token_id
    print("PAD token id:", pad_token_id)
        
    if args.use_stale:
        staleness_offset = 64
    else:
        staleness_offset = 0

    if args.no_retriever:
        retriever = DummyRetriever(
            num_neighbours=args.num_neighbours, 
            neighbour_len=config.chunk_size * (1 + args.num_continuation_chunks)
        )
    else:
        retriever = FaissRetriever(
            index_dir=args.index_dir,
            spec_file=args.spec_file,
            nprobe=args.nprobe,
            omp_threads=args.omp_threads,
            num_neighbours=args.num_neighbours, 
            num_continuation_chunks=args.num_continuation_chunks,
            staleness_offset=staleness_offset,
        )

    if args.use_original_model:
        print("Using original model: RetroModelLMHeadLightning")
        model = RetroModelLMHeadLightning.load_from_checkpoint(str(args.checkpoint), config=config, retriever=retriever, device=device).eval()
    else:
        raise ValueError("Not implemented yet")
        # print("Using inference model: RetroModelLMHeadLightningInference")
        # model = RetroModelLMHeadLightningInference.load_from_checkpoint(str(args.checkpoint), config=config, retriever=retriever, device=device).eval()    
    # model to device
    model.to(device)
    print("Model loaded.")
    
    all_generated_answers = []

    # load json NQ-open.efficientqa.test.1.1.jsonl
    f = open('NQ-open.efficientqa.test.1.1.jsonl') 
    for lid, line in enumerate(f):
        line_json = json.loads(line)	

        out_jsonl = {}
        out_jsonl["question"] = line_json["question"]
        out_jsonl["answer_and_def_correct_predictions"] = line_json["answer_and_def_correct_predictions"]

        prompt = line_json["question"]
        input_ids = tokenizer([prompt], add_special_tokens=False, return_tensors="pt")["input_ids"]
        input_ids = input_ids.to(device)

        # Prepend "Question: " before the prompt
        prefix_ids = tokenizer(["Question: "], add_special_tokens=False, return_tensors="pt")["input_ids"]
        prefix_ids = prefix_ids.to(device)

        # Append "\nAnswer:" after the prompt
        suffix_ids = tokenizer(["\n Answer: "], add_special_tokens=False, return_tensors="pt")["input_ids"]
        suffix_ids = suffix_ids.to(device)

        # Append empty token in the middle to fill 64 tokens
        len_to_fill = 64 - len(input_ids[0]) - len(prefix_ids[0]) - len(suffix_ids[0])
        input_ids = torch.cat([torch.zeros((1, len_to_fill), dtype=torch.int64, device=device), prefix_ids, input_ids, suffix_ids], dim=1)
        assert len(input_ids[0]) == 64


        print("Input sequence:")
        print(tokenizer.decode(input_ids[0]))

        if args.use_stale:
            kwargs = {"stale": True}
        else:
            kwargs = {}
        
        start = time.time()
        out = model.generate(
            inputs=input_ids, 
            do_sample=False,
            # do_sample=True,
            num_beams=1,
            top_k=5,
            top_p=1,
            temperature=1,
            min_length=args.min_len,
            max_length=args.max_len,
            length_penalty=1,
            early_stopping=False,
            num_beam_groups=1,
            num_return_sequences=1,
            repetition_penalty=1,
            no_repeat_ngram_size=3,
            # no_repeat_ngram_size=0, # can lead to infinite sequence generation
            encoder_no_repeat_ngram_size=0,
            diversity_penalty=0.0,
            remove_invalid_values=False,
            pad_token_id=0, 
            eos_token_id=1,
            # enable k-v cache
            output_scores=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict_in_generate=True,
            use_cache=True,
            **kwargs
        )
        end = time.time()
        print("Time elapsed: {} ms".format((end - start) * 1000))
        # print(out)

        print("Done. Final output:")
        # for i, seq in enumerate(out.sequences):
        #     print(f"Sequence {i}:")
        #     print(tokenizer.decode(seq))
        #     print("\n")
        print(tokenizer.decode(out.sequences[0]))
        question_with_answer = tokenizer.decode(out.sequences[0])
        # delete everything before "Answer:"
        answer = question_with_answer.split("Answer:")[1].strip()
        out_jsonl["generated_answer"] = answer
        all_generated_answers.append(out_jsonl)

        # append the latest output to the output jsonl file 
        if lid % 10 == 0 or lid == args.target_total_lines - 1:
            with open(args.out_file, "w") as f:
                for line in all_generated_answers:
                    f.write(json.dumps(line) + "\n")

        if lid == args.target_total_lines - 1:
            break
            

if __name__ == "__main__":

    import argparse 
    parser = argparse.ArgumentParser()

    # Retro
    parser.add_argument("--retro-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num-neighbours", type=int, default=2)
    parser.add_argument("--num-continuation-chunks", type=int, default=1)
    parser.add_argument("--use-original-model", type=int, default=1, help="Use the original model or the PipeRAG inference model")
    parser.add_argument("--min_len", type=int, default=1)
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--use-stale", type=int, default=0)
    parser.add_argument("--no-retriever", type=int, default=0)
    parser.add_argument("--out-file", type=Path, default="./generated_answers/generated_answers.jsonl")
    parser.add_argument("--target-total-lines", type=int, default=100)


    # performance model
    parser.add_argument('--use_perf_model', action='store_true', help="whether to use performance model")
    parser.add_argument('--generation_model_path', type=Path)#, default=Path('$WORKSPACE/inference/performance/p4d.24xlarge_performance_generation_len_1024_k_2.pickle'))
    parser.add_argument('--retrieval_model_path', type=Path)#, default=Path('$WORKSPACE/inference/performance/p4d.24xlarge_performance_search_c4_chunk_0_to_999_IVF16384,PQ64.pickle'))
    parser.add_argument('--sbert_model_path', type=Path)#, default=Path('$WORKSPACE/inference/performance/p4d.24xlarge_performance_SBERT.pickle'))
    parser.add_argument('--extra_overhead_ms', type=int, default=10, help="set a default extra latency on the retrieval side")
    parser.add_argument('--search_latency_budget_discount', type=float, default=1.0, help="if < 1.0, e.g., 0.9, limit the latency budget of search to 90%")
    parser.add_argument('--min_nprobe', type=int, default=None)
    parser.add_argument('--max_nprobe', type=int, default=None)

    # Faiss args
    parser.add_argument("--index_dir", type=Path, default="$WORKSPACE/data/datasets/indexes_wikipedia/wikipedia_chunk_0_to_8/IVF1024,PQ32_populated.index")
    # parser.add_argument("--index_dir", type=Path, default="$WORKSPACE/data/datasets/indexes_c4/c4_chunk_0_to_1/IVF1024,PQ64_populated.index")
    parser.add_argument("--spec_file", type=Path, default="$WORKSPACE/data/datasets/indexes_wikipedia/wikipedia_chunk_0_to_8/index.spec.json")
    # parser.add_argument("--spec_file", type=Path, default="$WORKSPACE/data/datasets/indexes_c4/c4_chunk_0_to_1/index.spec.json")
    parser.add_argument("--nprobe", type=int, default=32)
    parser.add_argument("--omp_threads", type=int, default=None)
    
    args = parser.parse_args()

    # replace $WORKSPACE with the actual path
    args.index_dir = Path(str(args.index_dir).replace("$WORKSPACE", os.environ["WORKSPACE"]))
    args.spec_file = Path(str(args.spec_file).replace("$WORKSPACE", os.environ["WORKSPACE"]))
    if args.use_perf_model:
        args.generation_model_path = Path(str(args.generation_model_path).replace("$WORKSPACE", os.environ["WORKSPACE"]))
        args.retrieval_model_path = Path(str(args.retrieval_model_path).replace("$WORKSPACE", os.environ["WORKSPACE"]))
        args.sbert_model_path = Path(str(args.sbert_model_path).replace("$WORKSPACE", os.environ["WORKSPACE"]))

    # main(args)
    main_autoregressive(args)
