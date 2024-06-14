"""
Example Usage:
python test_faiss_perf.py \
    --query-embedding-file ../data/datasets/wikipedia-en/09_wikipedia-en.embeddings.npy \
    --index-file ../data/datasets/indexes_wikipedia/wikipedia_chunk_0_to_8/IVF1024,PQ32_populated.index \
    --num-neighbours 100 \
    --nprobe 32 \
    --use-gpu 1 \
    --num-queries 16384 \
    --max-batch-size 1024
"""

import faiss
import json
import numpy as np
import time
from pathlib import Path

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--query-embedding-file", required=True, type=str)
parser.add_argument("--index-file", required=True, type=str)
parser.add_argument("--num-neighbours", type=int, default=100)
parser.add_argument("--nprobe", type=int, default=32)
parser.add_argument("--use-gpu", type=int, default=1)
parser.add_argument("--num-queries", type=int, default=1000)
parser.add_argument("--max-batch-size", type=int, default=1024)
args = parser.parse_args()

query_embedding_file = args.query_embedding_file
index_file = args.index_file
num_neighbours = args.num_neighbours
nprobe = args.nprobe
use_gpu = args.use_gpu
num_queries = args.num_queries
max_batch_size = args.max_batch_size
batch_sizes = [2 ** i for i in range(0, int(np.log2(max_batch_size)) + 1) if 2 ** i <= max_batch_size]

print("Load index...")
index = faiss.read_index(index_file)
assert index.is_trained, "The index must be trained"

# get number of vectors in the index
ntotal = index.ntotal
d = index.d
m = index.pq.M
nlist = index.nlist # get the product quantizer subvector number
imbalance_factor = index.invlists.imbalance_factor() # get imbalance factor
print("Number of vectors in the index: {}".format(ntotal))
print("Dimensionality of the index: {}".format(d))
print("Number of subvectors in the index: {}".format(m))
print("Number of clusters in the index: {}".format(nlist))
print("Imbalance factor of the index: {}".format(imbalance_factor))

if use_gpu:
    res = faiss.StandardGpuResources()  # use a single GPU
    index = faiss.index_cpu_to_gpu(res, 0, index)

# set number of probes -> scan all
index.nprobe = nprobe

# only load the first num_queries queries
print("Load query embeddings...")
query_embeddings = np.load(query_embedding_file).astype("float32")[0:num_queries]

print("Running search...")

for batch_size in reversed(batch_sizes):
    print("===== Batch size: {} =====".format(batch_size))
    start = time.time()
    for start_qid in range(0, num_queries, batch_size):
        end_qid = min(start_qid + batch_size, num_queries)
        _, retrieved_chunk_indices = index.search(query_embeddings[start_qid:end_qid], num_neighbours)
    end = time.time()
    print("Search took {:.2f} seconds".format(end - start))
    print("QPS: {:.2f}".format(num_queries / (end - start)))

    # calculate effective bandwidth (consider both PQ codes and inverted lists)
    bytes_per_query_scan = (nprobe / nlist * ntotal) * m + nlist * d * 4
    total_bytes_scanned = bytes_per_query_scan * num_queries
    effective_bandwidth = total_bytes_scanned / (end - start) / 1e9
    print("Effective bandwidth: {:.2f} GB/s".format(effective_bandwidth))