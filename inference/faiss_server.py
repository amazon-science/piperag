"""
Example Usage:

Dummy server:
    python faiss_server.py --host 127.0.0.1 --port 50051 --mode dummy

Faiss server (using fixed nprobe):
    python faiss_server.py --host 127.0.0.1 --port 50051 --mode faiss \
        --nprobe 32 --omp_threads 1 \
        --index_dir $WORKSPACE/data/datasets/indexes_c4/c4_chunk_0_to_999/IVF16384,PQ64_populated.index \
        --spec_file $WORKSPACE/data/datasets/indexes_c4/c4_chunk_0_to_999/index.spec.json 

Faiss server (using performance model):
    python faiss_server.py  --host 127.0.0.1 --port 50051 --mode faiss \
        --use_perf_model \
        --generation_model_path $WORKSPACE/inference/performance/p4d.24xlarge_performance_generation_len_1024_k_2.pickle \
        --retrieval_model_path $WORKSPACE/inference/performance/p4d.24xlarge_performance_search_c4_chunk_0_to_999_IVF16384,PQ64.pickle \
        --sbert_model_path $WORKSPACE/inference/performance/p4d.24xlarge_performance_SBERT.pickle \
        --extra_overhead_ms 10 \
        --nprobe 32 --omp_threads 1 \
        --index_dir $WORKSPACE/data/datasets/indexes_c4/c4_chunk_0_to_999/IVF16384,PQ64_populated.index \
        --spec_file $WORKSPACE/data/datasets/indexes_c4/c4_chunk_0_to_999/index.spec.json 
"""

import asyncio
import logging
import json
import time
import grpc
import faiss
import numpy as np
import sys


import retrieval_pb2
import retrieval_pb2_grpc

from typing import Optional
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from pathlib import Path

sys.path.append('../src')
from dataset_retro import ChunkedSequenceDataset, ShardedChunkedSequenceDataset
from performance_model import PerformanceModel

async def serve(retrievalServiceObj, host, port) -> None:
    server = grpc.aio.server()
    retrieval_pb2_grpc.add_RetrievalServiceServicer_to_server(retrievalServiceObj, server)
    listen_addr = f"{host}:{port}"
    server.add_insecure_port(listen_addr)
    print("Starting server on %s", listen_addr)
    await server.start()
    await server.wait_for_termination()

class DummyRetrievalService(retrieval_pb2_grpc.RetrievalServiceServicer):
    """
    A dummy retriever will reply some dummy messages
    """

    async def Retrieve(
        self,
        request: retrieval_pb2.RetrievalRequest,
        context: grpc.aio.ServicerContext,
    ) -> retrieval_pb2.RetrievalReply:

        chunk_size = 64
        retrieved_tokens = []
        # for i, query in enumerate(request.query):
        #     print(f"Query {i}: {query}")
            # retrieved_tokens.append("Hello, {}!, num_continuation_chunks: {}, num_neighbours: {}, staleness_offset: {}  seq_len: {}, interval: {}, ".format(
            #     query, request.num_continuation_chunks, request.num_neighbours, request.staleness_offset, request.seq_len, request.interval))
        retrieved_tokens = list(np.ones(len(request.query) * 1 * request.num_neighbours * (1 + request.num_continuation_chunks) * chunk_size).astype(np.int32))
        print("batch size: {}\tnum_continuation_chunks: {}\tnum_neighbours: {}\tstaleness_offset: {}\tseq_len: {}\tinterval: {}".format(
            len(request.query), request.num_continuation_chunks, request.num_neighbours, request.staleness_offset, request.seq_len, request.interval))
        print("Length of retrieved tokens: ", len(retrieved_tokens))
        time.sleep(0.1)
        return retrieval_pb2.RetrievalReply(retrieved_tokens=retrieved_tokens)

class FaissRetrievalService(retrieval_pb2_grpc.RetrievalServiceServicer):
    """
    Faiss retrieve service
    """

    def __init__(self, 
            index_dir :Path = Path("$WORKSPACE/data/datasets/indexes_c4/c4_chunk_0_to_999/IVF65536,PQ64_populated.index"), 
            spec_file: Path = Path("$WORKSPACE/data/datasets/indexes_c4/c4_chunk_0_to_999/index.spec.json"),
            nprobe : Optional[int] = 32,
            omp_threads : Optional[int] = None,
            chunk_size : Optional[int] = 64,
            pad_token_idx : Optional[int] = 0,
            ):
        
        # Intialize Faiss
        self.index_dir = index_dir
        self.nprobe = nprobe
        self.chunk_size = chunk_size
        self.pad_token_idx = pad_token_idx
        
        print("Loading Faiss index...")
        self.index = faiss.read_index(str(self.index_dir))
        self.dim = self.index.d
        self.ps = faiss.ParameterSpace()
        self.ps.initialize(self.index)
        self.set_nprobe(nprobe)
        
        assert self.dim == self.index.d
        if omp_threads is not None:
            print("WARNING: setting omp thread number to", omp_threads, 
                  ", please make sure only one Faiss object exists in the current process, "
                  "otherwise it can affect the performance of other Faiss objects.")
            faiss.omp_set_num_threads(omp_threads)
        
        # Load the retrieval spec
        print("Loading retrieval spec...")
        index_spec = json.load(spec_file.open())
        index_base_dir = index_dir.parent
        self.retrieval_dataset = ShardedChunkedSequenceDataset([
            ChunkedSequenceDataset(
                chunks=index_base_dir / shard["chunks"],
                seq2chunk=index_base_dir / shard["seq2chunk"],
                chunk2seq=index_base_dir / shard["chunk2seq"]
            )
            for shard in index_spec
        ])

        print("Loading Sentence BERT...")
        self.model_st = SentenceTransformer("all-MiniLM-L6-v2")
        self.chunk_tokenizer = AutoTokenizer.from_pretrained("t5-base", use_fast=True, model_max_length=10000)

        print("Warming up Faiss index and Sentence BERT model...")
        self.index.search(np.random.random((1, self.dim)).astype(np.float32), 1)
        self.model_st.encode(["Hello world!"], convert_to_numpy=True, output_value="sentence_embedding", normalize_embeddings=True)

        print("Faiss retrieval service initialized.")

        self.perf_model = None

    def load_perf_model(
        self,
        generation_model_path : Path = Path('$WORKSPACE/inference/performance/performance_generation_len_1024_k_2.pickle'),
        retrieval_model_path : Path = Path('$WORKSPACE/inference/performance/performance_search_c4_chunk_0_to_999_IVF16384,PQ64.pickle'),
        sbert_model_path : Path = Path('$WORKSPACE/inference/performance/performance_SBERT.pickle'),
        extra_overhead_ms = 10, # set a default extra latency on the retrieval side
        search_latency_budget_discount = 1.0, # if < 1.0, e.g., 0.9, limit the latency budget of search to 90%
        min_nprobe = None,
        max_nprobe = None):
        
        self.perf_model = PerformanceModel(
            generation_model_path=generation_model_path, retrieval_model_path=retrieval_model_path, sbert_model_path=sbert_model_path,
            extra_overhead_ms=extra_overhead_ms, search_latency_budget_discount=search_latency_budget_discount,
            min_nprobe=min_nprobe, max_nprobe=max_nprobe)


    def embed(self, query_texts):
        """
        Input: a list of query texts
        Output: a np array of query embeddings
        """
        query_embeddings = self.model_st.encode(query_texts, convert_to_numpy=True, output_value="sentence_embedding", normalize_embeddings=True)
        return query_embeddings

    async def Retrieve(
        self,
        request: retrieval_pb2.RetrievalRequest,
        context: grpc.aio.ServicerContext,
    ) -> retrieval_pb2.RetrievalReply:

        query_texts = request.query
        num_continuation_chunks = request.num_continuation_chunks
        num_neighbours = request.num_neighbours
        staleness_offset = request.staleness_offset
        seq_len = request.seq_len
        interval = request.interval
        use_perf_model = request.use_perf_model

        if use_perf_model:
            assert self.perf_model is not None, "Please load the performance model first"
            nprobe = self.perf_model.predict(seq_len, interval)
            print("Predicted nprobe: ", nprobe)
            self.set_nprobe(nprobe)

        batch_size = len(query_texts)
        neighbour_size = self.chunk_size * (1 + num_continuation_chunks)

        # turn queries from strings into embeddings
        query_embeddings = self.embed(query_texts)

        # Faiss search
        _, retrieved_chunk_indices = self.index.search(query_embeddings, num_neighbours)
        print("retrieved_chunk_indices", retrieved_chunk_indices)

        # turn result IDs into a list of strings (in the format of a list of integers), length = batch_size * num_neighbours, 
        #   each string length = chunk_size * (1 + num_continuation_chunks)
        neighbour_ids = []
        for i in range(batch_size):

            neighbour_ids_per_query = []
            for neighbour_chunk_idx in retrieved_chunk_indices[i]:
                neighbour_tokens = self.retrieval_dataset.get_chunk_tokens(
                    neighbour_chunk_idx, # the continuous start from old 
                    include_continuation_chunks=num_continuation_chunks + 1 # retrieve an extra chunk for trimming
                )
                print("neighbour_tokens", neighbour_tokens)
                # Pad to extend another chunk length
                if neighbour_tokens is not None:
                    retrieved_tokens_part = np.pad(neighbour_tokens, (0, neighbour_size + self.chunk_size - len(neighbour_tokens)), constant_values=self.pad_token_idx)
                else:
                    retrieved_tokens_part = np.ones(neighbour_size + self.chunk_size) * self.pad_token_idx

                # remove the stale context
                if self.chunk_size != staleness_offset:
                    retrieved_tokens_part = retrieved_tokens_part[staleness_offset: -(self.chunk_size - staleness_offset)] 
                else:
                    retrieved_tokens_part = retrieved_tokens_part[self.chunk_size:]
                neighbour_ids_per_query.append(retrieved_tokens_part)
                
            neighbour_ids += neighbour_ids_per_query

        # For ONNX inference format (see modeling_retro_inference.py), neighbors shape -> [batch, num chunks == 1, num neighbours, neighbour len
        #    but we use flat shape here to make it pass to the protobuf
        neighbour_ids = np.array(neighbour_ids, dtype=np.int32).reshape((-1,))
        assert neighbour_ids.shape[0] == batch_size * 1 * num_neighbours * neighbour_size
        retrieved_tokens = list(neighbour_ids)

        return retrieval_pb2.RetrievalReply(retrieved_tokens=retrieved_tokens)

    def set_nprobe(self, nprobe : int):
        self.nprobe = nprobe
        self.index.nprobe = nprobe

    # grpc service
    def SetNprobe(
            self, 
            request: retrieval_pb2.SetNprobeRequest,
            context : grpc.ServicerContext
        ) -> retrieval_pb2.SetNprobeReply:
        print("Set default nprobe: ", request.nprobe)
        self.set_nprobe(request.nprobe)
        return retrieval_pb2.SetNprobeReply(reply="Set nprobe to {} succeed".format(request.nprobe))

    
if __name__ == "__main__":
   
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9091)
    parser.add_argument("--mode", type=str, default="dummy", choices=["dummy", "faiss"])

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
    parser.add_argument("--index_dir", type=Path, default="$WORKSPACE/data/datasets/indexes_c4/c4_chunk_0_to_1/IVF1024,PQ64_populated.index")
    parser.add_argument("--spec_file", type=Path, default="$WORKSPACE/data/datasets/indexes_c4/c4_chunk_0_to_1/index.spec.json")
    parser.add_argument("--nprobe", type=int, default=32)
    parser.add_argument("--omp_threads", type=int, default=None)
    

    args = parser.parse_args()
 
    host = args.host
    port = args.port
    index_dir = args.index_dir
    spec_file = args.spec_file
    nprobe = args.nprobe
    omp_threads = args.omp_threads
    

    if args.mode == "dummy":
        retrievalServiceObj = DummyRetrievalService()
    elif args.mode == "faiss":
        retrievalServiceObj = FaissRetrievalService(index_dir=index_dir, spec_file=spec_file, nprobe=nprobe, omp_threads=omp_threads)
    
    if args.use_perf_model:
        retrievalServiceObj.load_perf_model(
            generation_model_path=args.generation_model_path,
            retrieval_model_path=args.retrieval_model_path,
            sbert_model_path=args.sbert_model_path,
            extra_overhead_ms=args.extra_overhead_ms,
            search_latency_budget_discount=args.search_latency_budget_discount,
            min_nprobe=args.min_nprobe,
            max_nprobe=args.max_nprobe
        )

    asyncio.run(serve(retrievalServiceObj, host, port))
