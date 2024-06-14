"""
The performance model, given the sequence length and retrieval interval, predict the maximum nprobe that can be used,
    using the profiling results of the generation model, the retriever, and the SBERT model.

Example usage:
    python performance_model.py \
        --generation_model_path $WORKSPACE/inference/performance/p4d.24xlarge_performance_generation_len_1024_k_2.pickle \
        --retrieval_model_path $WORKSPACE/inference/performance/p4d.24xlarge_performance_search_c4_chunk_0_to_999_IVF16384,PQ64.pickle \
        --sbert_model_path $WORKSPACE/inference/performance/p4d.24xlarge_performance_SBERT.pickle \
        --extra_overhead_ms 10 --max_nprobe 128
"""


import numpy as np
import pickle 
import time
import os

from pathlib import Path

class PerformanceModel(object):
    
    def __init__(self, 
        generation_model_path : Path = Path('$WORKSPACE/inference/performance/performance_generation_len_1024_k_2.pickle'),
        retrieval_model_path : Path = Path('$WORKSPACE/inference/performance/performance_search_c4_chunk_0_to_999_IVF16384,PQ64.pickle'),
        sbert_model_path : Path = Path('$WORKSPACE/inference/performance/performance_SBERT.pickle'),
        extra_overhead_ms = 10, # set a default extra latency on the retrieval side
        search_latency_budget_discount = 1.0, # if < 1.0, e.g., 0.9, limit the latency budget of search to 90%
        min_nprobe = None,
        max_nprobe = None
        ):

        """
        generation_model: a dictionary with keys: "average_decoder_latency_ms", "average_encoder_latency_ms"
                "average_decoder_latency_ms": an array of average decoder latency per step in ms, length = max_gen_len
                "average_encoder_latency_ms": an scalar of average encoder latency 
        """
        generation_model_path = str(generation_model_path).replace("$WORKSPACE", os.environ["WORKSPACE"])
        retrieval_model_path = str(retrieval_model_path).replace("$WORKSPACE", os.environ["WORKSPACE"])
        sbert_model_path = str(sbert_model_path).replace("$WORKSPACE", os.environ["WORKSPACE"])
        
        with open(generation_model_path, 'rb') as f:
            self.generation_model = pickle.load(f)
        
        """
        retrieval_model: a dictionary with keys: "average_latency_ms", "latency_std_ms", "P95_latency_ms", 
            "predictor_latency_ms", "predictor_nprobe_using_latency_ms"
            "average_latency_ms": a dictionary with keys: nprobe, values: average latency in ms
            "latency_std_ms": a dictionary with keys: nprobe, values: standard deviation in ms
            "P95_latency_ms": a dictionary with keys: nprobe, values: calculated P95 latency in ms
            "predictor_latency_ms": a linear regression model that predicts average latency given nprobe
            "predictor_nprobe_using_latency_ms": a linear regression model that predicts nprobe given a latency constraint, based on average retrieval latency
        """
        with open(retrieval_model_path, 'rb') as f:
            self.retrieval_model = pickle.load(f)

        """
        sbert_model: a dictionary with keys: "average_latency_ms", "latency_std_ms", "P95_latency_ms"
                "average_latency_ms": average latency in ms
                "latency_std_ms": standard deviation in ms
                "P95_latency_ms": calculated P95 latency in ms
        """
        with open(sbert_model_path, 'rb') as f:
            self.sbert_model = pickle.load(f)

        self.extra_overhead_ms = extra_overhead_ms
        self.search_latency_budget_discount = search_latency_budget_discount
        self.max_nprobe = int(max_nprobe) if max_nprobe is not None else None
        self.min_nprobe = int(min_nprobe) if min_nprobe is not None else None
        
        

    def predict(self, seq_len, retrieval_interval):
        """
        Given the current GPU states, return the maximum nprobe that can be used,
            such that the search latency is within the next generation window's latency budget.
        """

        generation_latency = 0
        for i in range(seq_len, seq_len + retrieval_interval):
            if i < len(self.generation_model["average_decoder_latency_ms"]):
                generation_latency += self.generation_model["average_decoder_latency_ms"][i]
            else:
                generation_latency += self.generation_model["average_decoder_latency_ms"][-1]
        # print("generation_latency: ", generation_latency)

        sbert_latency = self.sbert_model["average_latency_ms"]

        search_latency_budget = generation_latency - self.extra_overhead_ms - sbert_latency
        search_latency_budget = search_latency_budget * self.search_latency_budget_discount
        # print("search_latency_budget: ", search_latency_budget)
        nprobe = self.retrieval_model["predictor_nprobe_using_latency_ms"].predict(np.array(search_latency_budget).reshape(-1, 1))[0][0]

        nprobe = int(nprobe)
        if nprobe < 1:
            nprobe = 1 
        if self.max_nprobe is not None and nprobe > self.max_nprobe:
            nprobe = self.max_nprobe
        if self.min_nprobe is not None and nprobe < self.min_nprobe:
            nprobe = self.min_nprobe

        return nprobe

    def predict_nprobe_latency(self, nprobe):
        """
        Return predicted latency in ms
        """
        return self.retrieval_model["predictor_latency_ms"].predict(np.array(nprobe).reshape(-1, 1))[0][0]

if __name__ == '__main__':

    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument('--generation_model_path', type=Path, default=Path('$WORKSPACE/inference/performance/p4d.24xlarge_performance_generation_len_1024_k_2.pickle'))
    parser.add_argument('--retrieval_model_path', type=Path, default=Path('$WORKSPACE/inference/performance/p4d.24xlarge_performance_search_c4_chunk_0_to_999_IVF16384,PQ64.pickle'))
    parser.add_argument('--sbert_model_path', type=Path, default=Path('$WORKSPACE/inference/performance/p4d.24xlarge_performance_SBERT.pickle'))
    parser.add_argument('--extra_overhead_ms', type=int, default=10)
    parser.add_argument('--search_latency_budget_discount', type=float, default=1.0)
    parser.add_argument('--min_nprobe', type=int, default=None)
    parser.add_argument('--max_nprobe', type=int, default=None)
    args = parser.parse_args()

    performance_model = PerformanceModel(
        generation_model_path=args.generation_model_path,
        retrieval_model_path=args.retrieval_model_path,
        sbert_model_path=args.sbert_model_path,
        extra_overhead_ms=args.extra_overhead_ms,
        search_latency_budget_discount=args.search_latency_budget_discount,
        min_nprobe=args.min_nprobe,
        max_nprobe=args.max_nprobe
    )

    seq_len_list = np.arange(1, 1024 - 100, 100)
    retrieval_interval_list = [1, 2, 4, 8, 16, 32, 64]

    for seq_len in seq_len_list:
        for retrieval_interval in retrieval_interval_list:
            start = time.time()
            pred_nprobe = performance_model.predict(seq_len, retrieval_interval)
            end = time.time()
            print(f"seq_len: {seq_len}, retrieval_interval: {retrieval_interval}, nprobe: {pred_nprobe}, time (ms): {(end - start) * 1000}")
