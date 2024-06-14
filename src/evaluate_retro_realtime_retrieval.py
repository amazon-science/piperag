"""
Evaluate one configuration, and report perplexity.

Example usage (without performance model):
    python evaluate_retro_realtime_retrieval.py  --test-dataset-spec /fsx/retro_tobias/data/datasets/val_wikipedia/val_wikipedia_chunk9_1K/val_db_c4_to_999.json \
        --num-neighbours 2 --max-len 1024 --num-continuation-chunks 1 --staleness 1 --remove_stale_context 1 --no-retrieval 0 --retrieval-interval 64 \
        --checkpoint /fsx/retro_tobias/data/model/model.ckpt --retro-config /fsx/retro_tobias/data/model/retro.json \
        --batch-size 64 --gpus-per-node 1 --num-nodes 1 --num-workers 0 --nprobe 1

Example usage (with performance model):
    python evaluate_retro_realtime_retrieval.py  --test-dataset-spec /fsx/retro_tobias/data/datasets/val_wikipedia/val_wikipedia_chunk9_1K/val_db_c4_to_999.json \
        --num-neighbours 2 --max-len 1024 --num-continuation-chunks 1 --staleness 1 --remove_stale_context 1 --no-retrieval 0 --retrieval-interval 64 \
        --checkpoint /fsx/retro_tobias/data/model/model.ckpt --retro-config /fsx/retro_tobias/data/model/retro.json \
        --use_perf_model --generation_model_path /fsx/retro_tobias/inference/performance/p4d.24xlarge_performance_generation_len_1024_k_2.pickle \
        --retrieval_model_path /fsx/retro_tobias/inference/performance/p4d.24xlarge_performance_search_c4_chunk_0_to_999_IVF16384,PQ64.pickle \
        --sbert_model_path /fsx/retro_tobias/inference/performance/p4d.24xlarge_performance_SBERT.pickle \
        --extra_overhead_ms 10 --search_latency_budget_discount 1.0 --batch-size 64 --gpus-per-node 1 --num-nodes 1 --num-workers 0 --nprobe 1

"""

import argparse
import json
import faiss # Faiss import must before pytorch lightning
import pytorch_lightning as pl

from pathlib import Path
from functools import partial
from torch.utils.data import DataLoader
from modeling_retro import RetroConfig
from train_retro import RetroModelLMHeadLightning, get_retro_dataset_from_spec, get_realtime_retrieval_retro_dataset_from_spec, retro_collate_fn


def main(args):

    config = RetroConfig(**json.load(args.retro_config.open()))
    
    test_dss = [get_realtime_retrieval_retro_dataset_from_spec(
        spec_file=args.test_dataset_spec[i],
        num_neighbours=args.num_neighbours,
        continuation_chunks=args.num_continuation_chunks,
        pad_token_idx=config.pad_token_idx,
        max_len=args.max_len,
        use_gpus=args.use_gpus, 
        staleness=args.staleness,
        stale_steps=args.stale_steps,
        remove_stale_context=args.remove_stale_context,
        no_retrieval=args.no_retrieval,
        one_retrieval=args.one_retrieval,
        retrieval_interval=args.retrieval_interval,
        nprobe=args.nprobe,
        use_perf_model=args.use_perf_model,
        generation_model_path=args.generation_model_path,   
        retrieval_model_path=args.retrieval_model_path,
        sbert_model_path=args.sbert_model_path,
        extra_overhead_ms=args.extra_overhead_ms,
        search_latency_budget_discount=args.search_latency_budget_discount,
        min_nprobe=args.min_nprobe,
        max_nprobe=args.max_nprobe,
    ) for i in range(len(args.test_dataset_spec))]

    collate_fn = partial(retro_collate_fn, pad_token_idx=config.pad_token_idx)

    test_dls = [DataLoader(
        test_ds, 
        batch_size=args.batch_size, 
        collate_fn=collate_fn,
        num_workers=args.num_workers
    ) for test_ds in test_dss]

    model = RetroModelLMHeadLightning.load_from_checkpoint(str(args.checkpoint), config=config, strict=False)
    model.set_retrieval_interval(args.retrieval_interval)

    trainer = pl.Trainer(
        strategy="ddp_find_unused_parameters_false" if args.gpus_per_node is not None else None,
        gpus=args.gpus_per_node, 
        num_nodes=args.num_nodes,
        logger=None,
    )

    trainer.test(model, dataloaders=test_dls) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data args
    parser.add_argument("--test-dataset-spec", nargs='+', required=True, type=Path)
    parser.add_argument("--num-neighbours", type=int)
    parser.add_argument("--num-continuation-chunks", type=int, default=1)
    parser.add_argument("--max-len", type=int)
    parser.add_argument("--use-gpus")
    parser.add_argument("--staleness", type=int, default=0)
    parser.add_argument("--stale_steps", type=int, default=None)
    parser.add_argument("--remove_stale_context", type=int, default=0, help="Whether to remove stale context from the query, such that" + 
        " num-continuation-chunks is practically num-continuation-chunks - 1, makes sure the staleness computation demans is the same.")
    parser.add_argument("--no-retrieval", type=int, default=0, help="1 = disable retrieval, 0 = enable retrieval")
    parser.add_argument("--one-retrieval", type=int, default=0, help="1 = only retrieve once at the beginning, 0 = normal periodic retrieval")
    parser.add_argument("--retrieval-interval", type=int, default=64)
    parser.add_argument("--nprobe", default=None)

    # Model args
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--retro-config", required=True, type=Path)

    # performance model
    parser.add_argument('--use_perf_model', action='store_true', help="whether to use performance model")
    parser.add_argument('--generation_model_path', type=Path)#, default=Path('$WORKSPACE/inference/performance/p4d.24xlarge_performance_generation_len_1024_k_2.pickle'))
    parser.add_argument('--retrieval_model_path', type=Path)#, default=Path('$WORKSPACE/inference/performance/p4d.24xlarge_performance_search_c4_chunk_0_to_999_IVF16384,PQ64.pickle'))
    parser.add_argument('--sbert_model_path', type=Path)#, default=Path('$WORKSPACE/inference/performance/p4d.24xlarge_performance_SBERT.pickle'))
    parser.add_argument('--extra_overhead_ms', type=int, default=10, help="set a default extra latency on the retrieval side")
    parser.add_argument('--search_latency_budget_discount', type=float, default=1.0, help="if < 1.0, e.g., 0.9, limit the latency budget of search to 90%")
    parser.add_argument('--min_nprobe', type=int, default=None)
    parser.add_argument('--max_nprobe', type=int, default=None)

    # Training args
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--gpus-per-node", type=int)
    parser.add_argument("--num-nodes", type=int)
    parser.add_argument("--num-workers", type=int, default=0) # once pytorch lightning spawns multiple workers, each worker uses only one thread; 0 = use main process to load data

    args = parser.parse_args()
    if args.nprobe is not None:
        args.nprobe = int(args.nprobe)
    main(args)

