"""
Profile the performance of CPU Sbert

Example usage:
    python test_sbert_performance.py --nq 100 --chunk_size 64 --perf_file ./performance/p4d.24xlarge_performance_SBERT.pickle

Saved pickle object format:
    a dictionary with keys: "average_latency_ms", "latency_std_ms", "P95_latency_ms"
        "average_latency_ms": average latency in ms
        "latency_std_ms": standard deviation in ms
        "P95_latency_ms": calculated P95 latency in ms
"""

import argparse 
import numpy as np
import time
# import torch
import pickle

from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer


parser = argparse.ArgumentParser()
parser.add_argument("--nq", type=int, default=100, help="Number of queries")
parser.add_argument("--chunk_size", type=int, default=64)
# parser.add_argument("--num_threads", type=int, default=32)
parser.add_argument("--perf_file", type=Path, default="./performance/performance_SBERT.pickle")

args = parser.parse_args()

print("Loading model...")
model_st = SentenceTransformer("all-MiniLM-L6-v2")
tokenizer = AutoTokenizer.from_pretrained("t5-base", use_fast=True, model_max_length=10000)
# generate random int between 0 and 1000, shape = (nq, chunk_size)
input_ids = np.random.randint(0, 1000, size=(args.nq, args.chunk_size))
query_texts_list = []
for i in range(args.nq):
    query_texts_list.append(tokenizer.decode(input_ids[i]))
print("Finshed loading model and queries")
# torch.set_num_threads(args.num_threads)

# warm up
query_embeddings = model_st.encode(query_texts_list[-1], convert_to_numpy=True, output_value="sentence_embedding", normalize_embeddings=True)

# profile
time_list = []
for query_texts in query_texts_list:
    start = time.time()
    query_embeddings = model_st.encode(query_texts, convert_to_numpy=True, output_value="sentence_embedding", normalize_embeddings=True)
    end = time.time()
    time_list.append(end - start)

average_latency_ms = np.mean(time_list) * 1000
latency_std_ms = np.std(time_list) * 1000
# 2 standard deviation: 95%: https://www.learner.org/wp-content/uploads/2019/03/AgainstAllOdds_StudentGuide_Unit08-Normal-Calculations.pdf 
# P95_latency_ms = average_latency_ms + 2 * latency_std_ms
# Real P95 latency
P95_latency_ms = np.percentile(time_list, 95) * 1000
print(f"SBERT average latency: {average_latency_ms} ms, standard deviation: {latency_std_ms} ms, real P95: {P95_latency_ms} ms, calculated P95: {average_latency_ms + 2 * latency_std_ms} ms")

performance_SBERT = {"average_latency_ms": average_latency_ms, "latency_std_ms": latency_std_ms, "P95_latency_ms": P95_latency_ms}
with open(args.perf_file, 'wb') as handle:
    pickle.dump(performance_SBERT, handle, protocol=4)
