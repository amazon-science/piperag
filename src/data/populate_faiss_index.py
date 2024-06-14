import faiss
import json
import numpy as np
import time
from queue import Queue
from pathlib import Path
from typing import List
from threading import Thread
from time import perf_counter
from contextlib import contextmanager


@contextmanager
def catchtime() -> float:
    start = perf_counter()
    yield lambda: perf_counter() - start


def load_embeddings(shard_paths: List[Path], queue: Queue):
    for path in shard_paths:
        queue.put(np.load(path).astype("float32"))


def save_index(index, args):
    print("Saving index...")
    if args.use_gpus is True:
        index = faiss.index_gpu_to_cpu(index)
    
    imbalance_factor = index.invlists.imbalance_factor()
    print("Imbalance factor: {}".format(imbalance_factor))
    if imbalance_factor > 1.2:
        print("CRITICAL WARNING: imbalance factor is high, may hurt pipeline stability & performance model!")

    faiss.write_index(index, str(args.output_index))

def main(args):

    # Load empty (but trained) index
    print("Loading index...")
    index = faiss.read_index(str(args.trained_index))
    assert index.is_trained, "The index must be trained"

    if args.use_gpus:
        co = faiss.GpuMultipleClonerOptions()
        co.shard = args.shard_index
        if args.start_vec_id > 0:
            assert args.shard_index is True, "add-with-ids can only be used with sharding"
        co.useFloat16 = True
        index = faiss.index_cpu_to_all_gpus(index, co)
    
    print("Adding embedding shards...")
    spec = json.load(args.spec.open("r"))
    base_dir = args.spec.parent
    shard_paths = [base_dir / shard["embeddings"] for shard in spec]

    # Start background thread to load embedding shards from disk
    embeddings_queue = Queue(maxsize=3)
    worker = Thread(target=load_embeddings, args=(shard_paths, embeddings_queue))
    worker.start()

    start = time.time()
    num_embs_added = 0
    for i in range(len(shard_paths)):
        print("Processing shard {}, shard name: {}".format(i, shard_paths[i]))
        embs = embeddings_queue.get()
        index.add_with_ids(embs, np.arange(num_embs_added + args.start_vec_id, num_embs_added + embs.shape[0] + args.start_vec_id))
        
        if args.num_queries > 0:
            # Evaluate after adding each shard
            with catchtime() as t:
                _, I = index.search(embs[:args.num_queries,:], 1)
            qps = args.num_queries / t()
            recall_at_1 = np.mean(I[:,0] == (np.arange(args.num_queries) + num_embs_added))
            print("Recall@1: {recall_at_1} \tQueries / s: {int(qps)}")

        num_embs_added += embs.shape[0]
        assert num_embs_added == index.ntotal, "Index size does not match number of vectors added"
        print(f"finished shard {i}:\tadded {num_embs_added} vectors\t, total ids (added by the start-vec-id) {num_embs_added + args.start_vec_id}")
        now = time.time()
        print("Time elapsed: {} hours + {:.2f} s".format((now - start) // 3600, (now - start) % 3600))
        print("Average time per shard (s): {:.2f}".format((now - start) / (i + 1)))

        if i % 100 == 0:
            print("Saving index... shard ID: {}".format(i))
            save_index(index, args)
    
    save_index(index, args)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--trained-index", required=True, type=Path)
    parser.add_argument("--output-index", required=True, type=Path)
    parser.add_argument("--use-gpus", action="store_true")
    parser.add_argument("--shard-index", action="store_true")
    parser.add_argument("--num-queries", default=0, type=int)
    parser.add_argument("--start-vec-id", default=0, type=int)
    args = parser.parse_args()

    main(args)