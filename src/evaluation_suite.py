"""
Deprecated script! Used for some initial experiments. Use "evaluate_perplexity_all.py" instead, which imports some of the functions here. 

Example Usage: 
    # Staleness experiments, Eval on RealNews
    python evaluation_suite.py --config-set-name realnews_eval_chunk31_db_chunk0_0 > ../logs/log_realnews_eval_chunk31_db_chunk0_0.txt 
    python evaluation_suite.py --config-set-name realnews_eval_chunk31_db_chunk0_9 > ../logs/log_realnews_eval_chunk31_db_chunk0_9.txt 
    python evaluation_suite.py --config-set-name realnews_eval_chunk31_db_chunk0_29 > ../logs/log_realnews_eval_chunk31_db_chunk0_29.txt 
    python evaluation_suite.py --config-set-name realnews_eval_mix_c4_to_99_realnews_to_29_wiki_to_8.json > ../logs/log_realnews_eval_mix_c4_to_99_realnews_to_29_wiki_to_8.txt

    # Staleness experiments, Eval on wikipedia
    python evaluation_suite.py --config-set-name wikipedia_eval_chunk9_db_chunk0_0 > ../logs/log_wikipedia_eval_chunk9_db_chunk0_0.txt
    python evaluation_suite.py --config-set-name wikipedia_eval_chunk9_db_chunk0_8 > ../logs/log_wikipedia_eval_chunk9_db_chunk0_8.txt
    python evaluation_suite.py --config-set-name wikipedia_eval_mix_c4_to_99_realnews_to_29_wiki_to_8 > ../logs/log_wikipedia_eval_mix_c4_to_99_realnews_to_29_wiki_to_8.txt
    python evaluation_suite.py --config-set-name wikipedia_eval_c4_to_999 > ../logs/log_wikipedia_eval_c4_to_999.txt
    
    # Staleness experiments, val set overlap with DB
    python evaluation_suite.py --config-set-name realnews_eval_chunk0_db_chunk0_0 > ../logs/log_realnews_eval_chunk0_db_chunk0_0.txt 
    python evaluation_suite.py --config-set-name wikipedia_eval_chunk0_db_chunk0_0 > ../logs/log_wikipedia_eval_chunk0_db_chunk0_0.txt


    # nprobe experiments
    python evaluation_suite.py --config-set-name realnews_eval_chunk31_db_chunk0_29 --mode nprobe --min-nprobe 1 --max-nprobe 16  > ../logs/log_realnews_eval_chunk31_db_chunk0_29_nprobe_1_16.txt
    python evaluation_suite.py --config-set-name wikipedia_eval_c4_to_999 --mode nprobe --min-nprobe 1 --max-nprobe 16 > ../logs/log_wikipedia_eval_c4_to_999_nprobe_1_16.txt
    python evaluation_suite.py --config-set-name wikipedia_eval_c4_to_499 --mode nprobe --min-nprobe 1 --max-nprobe 16 > ../logs/log_wikipedia_eval_c4_to_499_nprobe_1_16.txt

    # interval experiments
    python evaluation_suite.py --config-set-name realnews_eval_chunk31_db_chunk0_29 --mode interval --min-interval 8 --max-interval 64 --batch-size 16 > ../logs/log_realnews_eval_chunk31_db_chunk0_29_interval_8_64.txt
    python evaluation_suite.py --config-set-name wikipedia_eval_chunk9_db_chunk0_8 --mode interval --min-interval 8 --max-interval 64 --batch-size 16 > ../logs/log_wikipedia_eval_chunk9_db_chunk0_8_interval_8_64.txt
"""


import argparse
import os
import subprocess
import time


class TestConfig:

    def __init__(
        self,
        exp_name="test",
        test_dataset_spec=None,
        num_continuation_chunks=1,
        staleness=0,
        stale_steps=None,
        remove_stale_context=0,
        max_len=512,
        num_neighbours=2,
        no_retrieval=0,
        one_retrieval=0,
        retrieval_interval=64,
        nprobe=None,
        # model info
        checkpoint=None,
        retro_config=None,
        # worker info
        batch_size=None,
        gpus_per_node=None,
        num_nodes=None,
        num_workers=None,

        # performance model
        use_perf_model=False,
        generation_model_path=None,
        retrieval_model_path=None,
        sbert_model_path=None,
        extra_overhead_ms=None,
        search_latency_budget_discount=None,
        min_nprobe=None,
        max_nprobe=None,
    ):

        self.test_dataset_spec = test_dataset_spec
        self.num_neighbours = num_neighbours
        self.num_continuation_chunks = num_continuation_chunks
        self.staleness = staleness
        self.stale_steps = stale_steps
        self.remove_stale_context = remove_stale_context if staleness else 0
        
        self.max_len = max_len
        self.no_retrieval = no_retrieval
        self.one_retrieval = one_retrieval
        self.retrieval_interval = retrieval_interval
        self.nprobe = nprobe
        self.exp_name = exp_name

        self.checkpoint = checkpoint
        self.retro_config = retro_config

        self.batch_size = batch_size
        self.gpus_per_node = gpus_per_node
        self.num_nodes = num_nodes
        self.num_workers = num_workers

        # performance model
        self.use_perf_model = use_perf_model
        self.generation_model_path = generation_model_path
        self.retrieval_model_path = retrieval_model_path
        self.sbert_model_path = sbert_model_path
        self.extra_overhead_ms = extra_overhead_ms
        self.search_latency_budget_discount = search_latency_budget_discount
        self.min_nprobe = min_nprobe
        self.max_nprobe = max_nprobe


    def print_info(self):
        # print all members in one line
        print(self.exp_name)
        for k, v in self.__dict__.items():
            print(k, ":", v)


    def eval(self):

        out_log = "/tmp/out" + str(os.getpid()) + ".log"

        cmd = "python evaluate_retro_realtime_retrieval.py " 
        cmd += " --test-dataset-spec {}".format(self.test_dataset_spec)  
        cmd += " --num-neighbours {}".format(self.num_neighbours)  
        cmd += " --max-len {}".format(self.max_len) 
        cmd += " --num-continuation-chunks {}".format(self.num_continuation_chunks)
        cmd += " --staleness {}".format(self.staleness)
        if self.staleness:
            cmd +=  " --remove_stale_context {}".format(self.remove_stale_context)
            if self.stale_steps is not None:
                cmd += " --stale_steps {}".format(self.stale_steps)
        cmd += " --no-retrieval {}".format(self.no_retrieval)
        cmd += " --one-retrieval {}".format(self.one_retrieval)
        cmd += " --retrieval-interval {}".format(self.retrieval_interval)

        # model info
        cmd += " --checkpoint {}".format(self.checkpoint)
        cmd += " --retro-config {}".format(self.retro_config) 

        # performance model
        if self.use_perf_model:
            cmd += " --use_perf_model"
            cmd += " --generation_model_path {}".format(self.generation_model_path.replace('-', '_'))
            cmd += " --retrieval_model_path {}".format(self.retrieval_model_path.replace('-', '_'))
            cmd += " --sbert_model_path {}".format(self.sbert_model_path.replace('-', '_'))
            cmd += " --extra_overhead_ms {}".format(self.extra_overhead_ms)
            cmd += " --search_latency_budget_discount {}".format(self.search_latency_budget_discount)
            if self.min_nprobe is not None:
                cmd += " --min_nprobe {}".format(self.min_nprobe)
            if self.max_nprobe is not None:
                cmd += " --max_nprobe {}".format(self.max_nprobe)

        # worker info
        cmd += " --batch-size {}".format(self.batch_size)
        cmd += " --gpus-per-node {}".format(self.gpus_per_node)
        cmd += " --num-nodes {}".format(self.num_nodes)
        cmd += " --num-workers {}".format(self.num_workers)

        # if set nprobe, otherwise rely on the default nprobe in the config file
        if self.nprobe is not None:
            cmd += " --nprobe {}".format(self.nprobe)

        cmd += " > {}".format(out_log) 

        print("\n======================")
        self.print_info()
        print("Executing: ", cmd, flush=True)
        start = time.time()
        proc = subprocess.run(cmd, shell=True)
        end = time.time()
        print("Finshed, time for execution: {:.2f} seconds".format(end - start))

        # get the row containing 'perplexity' in the log file
        test_loss = None
        perplexity = None
        with open(out_log, "r") as f:
            lines = f.readlines()
            for line in lines:
                if 'test_loss' in line:
                    test_loss = float(line.split(" ")[-1])
                if "perplexity" in line:
                    perplexity = float(line.split(" ")[-1])
        
        print("test_loss: {:.2f}".format(test_loss))
        print("perplexity: {:.2f}".format(perplexity))
        print("======================\n", flush=True)

        return perplexity

def staleness_exp_config_set(test_dataset_spec, args):
    """
    Return a standard set of configs for evaluation
    """
    config_set = [
        
        TestConfig(
            exp_name="no retrieval",
            test_dataset_spec=test_dataset_spec,
            num_continuation_chunks=1,
            staleness=0,
            remove_stale_context=0,
            max_len=512,
            num_neighbours=2,
            no_retrieval=1,
            retrieval_interval=args.retrieval_interval,
            nprobe=args.nprobe,
            # model info
            checkpoint=args.checkpoint,
            retro_config=args.retro_config,
            # worker info
            batch_size=args.batch_size,
            gpus_per_node=args.gpus_per_node,
            num_nodes=args.num_nodes,
            num_workers=args.num_workers,
        ),

        TestConfig(
            exp_name="retrieval no stale",
            test_dataset_spec=test_dataset_spec,
            num_continuation_chunks=1,
            staleness=0,
            remove_stale_context=0,
            max_len=512,
            num_neighbours=2,
            no_retrieval=0,
            retrieval_interval=args.retrieval_interval,
            nprobe=args.nprobe,
            # model info
            checkpoint=args.checkpoint,
            retro_config=args.retro_config,
            # worker info
            batch_size=args.batch_size,
            gpus_per_node=args.gpus_per_node,
            num_nodes=args.num_nodes,
            num_workers=args.num_workers,
        ),
        
        # TestConfig(
        #     exp_name="retrieval with stale, continuation = 1",
        #     test_dataset_spec=test_dataset_spec,
        #     num_continuation_chunks=1,
        #     staleness=1,
        #     remove_stale_context=0,
        #     max_len=512,
        #     num_neighbours=2,
        #     no_retrieval=0,
        #     retrieval_interval=args.retrieval_interval,
            # nprobe=args.nprobe,
            # model info
            # checkpoint=args.checkpoint,
            # retro_config=args.retro_config,
            # # worker info
            # batch_size=args.batch_size,
            # gpus_per_node=args.gpus_per_node,
            # num_nodes=args.num_nodes,
            # num_workers=args.num_workers,
        # ),

        # TestConfig(
        #     exp_name="retrieval with stale, continuation = 2",
        #     test_dataset_spec=test_dataset_spec,
        #     num_continuation_chunks=2,
        #     staleness=1,
        #     remove_stale_context=0,
        #     max_len=512,
        #     num_neighbours=2,
        #     no_retrieval=0,
        #     retrieval_interval=args.retrieval_interval,
            # nprobe=args.nprobe,
            # # model info
            # checkpoint=args.checkpoint,
            # retro_config=args.retro_config,
            # # worker info
            # batch_size=args.batch_size,
            # gpus_per_node=args.gpus_per_node,
            # num_nodes=args.num_nodes,
            # num_workers=args.num_workers,
        # ),

        TestConfig(
            exp_name="retrieval with stale, continuation = 1, shift to exclude the stale context",
            test_dataset_spec=test_dataset_spec,
            num_continuation_chunks=1,
            staleness=1,
            remove_stale_context=1,
            max_len=512,
            num_neighbours=2,
            no_retrieval=0,
            retrieval_interval=args.retrieval_interval,
            nprobe=args.nprobe,
            # model info
            checkpoint=args.checkpoint,
            retro_config=args.retro_config,
            # worker info
            batch_size=args.batch_size,
            gpus_per_node=args.gpus_per_node,
            num_nodes=args.num_nodes,
            num_workers=args.num_workers,
        ),
    ]

    return config_set

def nprobe_exp_config_set(test_dataset_spec, args):
    """
    Return a standard set of configs for evaluation
    """
    config_set = []

    nprobe = args.min_nprobe
    while nprobe <= args.max_nprobe:
        
        config_set.append(TestConfig(
            exp_name="retrieval no stale, nprobe = {}".format(nprobe),
            test_dataset_spec=test_dataset_spec,
            num_continuation_chunks=1,
            staleness=0,
            remove_stale_context=0,
            max_len=512,
            num_neighbours=2,
            no_retrieval=0,
            retrieval_interval=args.retrieval_interval,
            nprobe=nprobe,
            # model info
            checkpoint=args.checkpoint,
            retro_config=args.retro_config,
            # worker info
            batch_size=args.batch_size,
            gpus_per_node=args.gpus_per_node,
            num_nodes=args.num_nodes,
            num_workers=args.num_workers,
        ))

        nprobe *= 2

    return config_set

def interval_exp_config_set(test_dataset_spec, args):
    """
    Return a standard set of configs for evaluation
    """
    config_set = []

    # baseline RETRO
    config_set.append(TestConfig(
        exp_name="retrieval no stale, interval = {}".format(args.retrieval_interval),
        test_dataset_spec=test_dataset_spec,
        num_continuation_chunks=1,
        staleness=0,
        remove_stale_context=0,
        max_len=512,
        num_neighbours=2,
        no_retrieval=0,
        retrieval_interval=args.retrieval_interval,
        nprobe=args.nprobe,
        # model info
        checkpoint=args.checkpoint,
        retro_config=args.retro_config,
        # worker info
        batch_size=args.batch_size,
        gpus_per_node=args.gpus_per_node,
        num_nodes=args.num_nodes,
        num_workers=args.num_workers,
    ))

    interval = args.max_interval
    while interval >= args.min_interval:
        
        config_set.append(TestConfig(
            exp_name="retrieval with stale, continuation = 1, shift to exclude the stale context, interval = {}".format(interval),
            test_dataset_spec=test_dataset_spec,
            num_continuation_chunks=1,
            staleness=1,
            remove_stale_context=1,
            max_len=512,
            num_neighbours=2,
            no_retrieval=0,
            retrieval_interval=interval,
            nprobe=args.nprobe,
            # model info
            checkpoint=args.checkpoint,
            retro_config=args.retro_config,
            # worker info
            batch_size=args.batch_size,
            gpus_per_node=args.gpus_per_node,
            num_nodes=args.num_nodes,
            num_workers=args.num_workers,
        ))

        interval = int(interval / 2)

    return config_set

if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument("--config-set-name", type=str, default="realnews_eval_chunk0_db_chunk0")
    parser.add_argument("--checkpoint", type=str, default="$WORKSPACE/data/model/model.ckpt")
    parser.add_argument("--retro-config", type=str, default="$WORKSPACE/data/model/retro.json")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--retrieval-interval", type=int, default=64)
    parser.add_argument("--gpus-per-node", type=int, default=1)
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0) # once pytorch lightning spawns multiple workers, each worker uses only one thread; 0 = use main process to load data
    parser.add_argument("--nprobe", default=None)

    parser.add_argument("--mode", default="staleness", choices=["staleness", "nprobe", "interval"])

    # parameters for nprobe (set min max), here, the arg's nprobe will be negated
    parser.add_argument("--min-nprobe", type=int, default=1)
    parser.add_argument("--max-nprobe", type=int, default=64)

    # parameters for interval (set min max), here, the arg's interval will be negated
    parser.add_argument("--min-interval", type=int, default=8)
    parser.add_argument("--max-interval", type=int, default=64)

    args = parser.parse_args()

    if args.nprobe is not None:
        args.nprobe = int(args.nprobe)

    config_set_name = args.config_set_name

    if args.gpus_per_node > 1:
        print("WARNING: the index will be forked to each GPU process, make sure don't overflow the memeory")

    print("=== Running config set: {} ===".format(config_set_name))

    if config_set_name == 'realnews_eval_chunk0_db_chunk0_0':
        # RealNews, eval chunk 0, db chunk 0
        test_dataset_spec = "$WORKSPACE/data/datasets/val_realnews/val_realnews_chunk0_1K/val_db_chunk_0_0.spec.json"
    elif config_set_name == 'realnews_eval_chunk31_db_chunk0_0':
        # RealNews, eval chunk 31, db chunk 0~0
        test_dataset_spec = "$WORKSPACE/data/datasets/val_realnews/val_realnews_chunk31_1K/val_db_chunk_0_0.spec.json"
    elif config_set_name == 'realnews_eval_chunk31_db_chunk0_9':
        # RealNews, eval chunk 31, db chunk 0~9
        test_dataset_spec = "$WORKSPACE/data/datasets/val_realnews/val_realnews_chunk31_1K/val_db_chunk_0_9.spec.json"
    elif config_set_name == 'realnews_eval_chunk31_db_chunk0_29':
        # RealNews, eval chunk 31, db chunk 0~29
        test_dataset_spec = "$WORKSPACE/data/datasets/val_realnews/val_realnews_chunk31_1K/val_db_chunk_0_29.spec.json"
    elif config_set_name == 'realnews_eval_mix_c4_to_99_realnews_to_29_wiki_to_8.json':
        # RealNews, eval = mixed db
        test_dataset_spec = '$WORKSPACE/data/datasets/val_realnews/val_realnews_chunk31_1K/val_db_mix_c4_to_99_realnews_to_29_wiki_to_8.json'

    elif config_set_name == 'wikipedia_eval_chunk0_db_chunk0_0':
        # Wikipedia-en, eval chunk 0, db chunk 0~0
        test_dataset_spec = "$WORKSPACE/data/datasets/val_wikipedia/val_wikipedia_chunk0_1K/val_db_chunk_0_0.spec.json"
    elif config_set_name == 'wikipedia_eval_chunk9_db_chunk0_0':
        # Wikipedia-en, eval chunk 9, db chunk 0~0
        test_dataset_spec = "$WORKSPACE/data/datasets/val_wikipedia/val_wikipedia_chunk9_1K/val_db_chunk_0_0.spec.json"
    elif config_set_name == 'wikipedia_eval_chunk9_db_chunk0_8':
        # Wikipedia-en, eval chunk 9, db chunk 0~9
        test_dataset_spec = "$WORKSPACE/data/datasets/val_wikipedia/val_wikipedia_chunk9_1K/val_db_chunk_0_8.spec.json"
    elif config_set_name == 'wikipedia_eval_mix_c4_to_99_realnews_to_29_wiki_to_8':
        # Wikipedia-en, eval = mixed db
        test_dataset_spec = '$WORKSPACE/data/datasets/val_wikipedia/val_wikipedia_chunk9_1K/val_db_mix_c4_to_99_realnews_to_29_wiki_to_8.json'

    elif config_set_name == 'wikipedia_eval_c4_to_999':
        test_dataset_spec = '$WORKSPACE/data/datasets/val_wikipedia/val_wikipedia_chunk9_1K/val_db_c4_to_999.json'
    elif config_set_name == 'wikipedia_eval_c4_to_499':
        test_dataset_spec = '$WORKSPACE/data/datasets/val_wikipedia/val_wikipedia_chunk9_1K/val_db_c4_to_499.json'
        
    else:
        raise ValueError("Unknown config set: {}".format(config_set_name))

    if args.mode == 'staleness':
        config_set = staleness_exp_config_set(test_dataset_spec, args)
    elif args.mode == 'nprobe':
        config_set = nprobe_exp_config_set(test_dataset_spec, args)
    elif args.mode == 'interval':
        config_set = interval_exp_config_set(test_dataset_spec, args)

    for config in config_set:
        config.eval()
    