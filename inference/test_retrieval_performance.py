"""
Given a Faiss index and a set of nprobe, evaluate the retrieval latency

Example usage:
    
    python test_retrieval_performance.py \
    --index_dir $WORKSPACE/data/datasets/indexes_c4/c4_chunk_0_to_999/IVF16384,PQ64_populated.index \
    --query_file $WORKSPACE/data/datasets/c4-en/c4-train.00000-of-01024.embeddings.npy \
    --nq 1000 --max_nprobe 128 \
    --save_perf --perf_file ./performance/p4d.24xlarge_performance_search_c4_chunk_0_to_999_IVF16384,PQ64.pickle

Saved pickle object format:
    a dictionary with keys: "average_latency_ms", "latency_std_ms", "P95_latency_ms", 
        "predictor_latency_ms", "predictor_nprobe_using_latency_ms"
        "average_latency_ms": a dictionary with keys: nprobe, values: average latency in ms
        "latency_std_ms": a dictionary with keys: nprobe, values: standard deviation in ms
        "P95_latency_ms": a dictionary with keys: nprobe, values: calculated P95 latency in ms
        "predictor_latency_ms": a linear regression model that predicts average latency given nprobe
        "predictor_nprobe_using_latency_ms": a linear regression model that predicts nprobe given a latency constraint, based on average retrieval latency
"""

import argparse 
import numpy as np
import faiss
import time
import pickle

from pathlib import Path
from sklearn import datasets, linear_model, neural_network


parser = argparse.ArgumentParser()

parser.add_argument("--index_dir", type=Path, default="$WORKSPACE/data/datasets/indexes_c4/c4_chunk_0_to_1/IVF1024,PQ64_populated.index")
parser.add_argument("--query_file", type=Path, default="$WORKSPACE/data/datasets/c4-en/c4-train.00000-of-01024.embeddings.npy")
parser.add_argument("--nq", type=int, default=1000, help="Number of queries")
parser.add_argument("--max_nprobe", type=int, default=32)
parser.add_argument("--save_perf", action="store_true")
parser.add_argument("--perf_file", type=Path, default="./performance/performance_search.pickle")

args = parser.parse_args()

print("Loading index and queries...")
index = faiss.read_index(str(args.index_dir))
queries = np.load(args.query_file).astype("float32")
print("Finshed loading index and queries")

# start from 1, x 2 every time

average_latency_ms_dict = {}
latency_std_ms_dict = {}
P95_latency_ms_dict = {}

nprobe = 1
nprobe_list = []

# warm up 
index.search(queries[-1].reshape(1, -1), 1)

# try_alternative_parallel_mode = False
pmodes = [0]
# if try_alternative_parallel_mode:
#     pmodes = [0, 1, 2, 3]

X_nprobe = []
Y_latency_ms = []
while nprobe <= args.max_nprobe:

    for pmode in pmodes:
        print("nprobe: ", nprobe, "parallel mode: ", pmode)
        index.parallel_mode = pmode
        index.nprobe = nprobe
        nprobe_list.append(nprobe)

        # always use batch size of 1
        time_list = []
        for i in range(args.nq):
            query = queries[i].reshape(1, -1)
            start = time.time()
            index.search(query, 1)
            end = time.time()
            time_list.append(end - start)
        average_latency_ms = np.mean(time_list) * 1000
        latency_std_ms = np.std(time_list) * 1000
        P95_latency_ms = np.percentile(time_list, 95) * 1000
        # 2 standard deviation: 95%: https://www.learner.org/wp-content/uploads/2019/03/AgainstAllOdds_StudentGuide_Unit08-Normal-Calculations.pdf 
        calculated_P95_latency_ms = average_latency_ms + 2 * latency_std_ms
        print(f"nprobe: {nprobe}, average latency: {average_latency_ms} ms, standard deviation: {latency_std_ms} ms, real P95: {P95_latency_ms}, calculated P95: {calculated_P95_latency_ms} ms")

        if pmode == 0:
            average_latency_ms_dict[nprobe] = average_latency_ms
            latency_std_ms_dict[nprobe] = latency_std_ms
            P95_latency_ms_dict[nprobe] = P95_latency_ms
            X_nprobe += [nprobe] * len(time_list)
            Y_latency_ms += list(np.array(time_list) * 1000)

    nprobe *= 2


if args.save_perf:
    
    X_nprobe = np.array(X_nprobe).reshape(-1, 1)
    Y_latency_ms = np.array(Y_latency_ms).reshape(-1, 1)

    predictor_latency_ms = linear_model.LinearRegression()
    # predictor_latency_ms = neural_network.MLPRegressor(hidden_layer_sizes=(4,))
    predictor_latency_ms.fit(X_nprobe, Y_latency_ms)

    predictor_nprobe_using_latency_ms = linear_model.LinearRegression()
    # predictor_nprobe_using_latency_ms = neural_network.MLPRegressor(hidden_layer_sizes=(4,))
    predictor_nprobe_using_latency_ms.fit(Y_latency_ms, X_nprobe)

    print("\n=== Linear regression model ===")
    for nprobe in nprobe_list:
        # show predicted average latency and P95 latency, and compare them to the actual values
        predicted_avg_latency = predictor_latency_ms.predict(np.array(nprobe).reshape(-1, 1))[0][0]
        actual_avg_latency = average_latency_ms_dict[nprobe]
        delta_avg_latency = 100 * (predicted_avg_latency - actual_avg_latency) / actual_avg_latency

        print(f"nprobe: {nprobe}, predicted avg latency: {predicted_avg_latency} ms, actual avg latency: {actual_avg_latency} ms, delta (%): {delta_avg_latency}")

        # show the predicted nprobe based on average latency and P95 latency
        predicted_nprobe_given_avg_latency = predictor_nprobe_using_latency_ms.predict(np.array(actual_avg_latency).reshape(-1, 1))[0][0]
        delta_nprobe_given_avg_latency = 100 * (predicted_nprobe_given_avg_latency - nprobe) / nprobe
        print(f"nprobe: {nprobe}, predicted nprobe given avg latency: {int(predicted_nprobe_given_avg_latency)}, delta (%) : {delta_nprobe_given_avg_latency}")

    performance_search = {"average_latency_ms": average_latency_ms_dict, "latency_std_ms": latency_std_ms_dict, "P95_latency_ms": P95_latency_ms_dict, 
        "predictor_latency_ms": predictor_latency_ms, "predictor_nprobe_using_latency_ms": predictor_nprobe_using_latency_ms}
    with open(args.perf_file, 'wb') as handle:
        pickle.dump(performance_search, handle, protocol=4)
