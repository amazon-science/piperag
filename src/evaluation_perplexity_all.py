"""
This is the final evaluation script for the perplexity of models configurations. 

Store all the perplexity data in the following dataframe format:
    Schema (inputs): eval_set, dbname, index_type, seq_len, nprobe, retrieval_interval, staleness, num_continuation_chunks, num_neighbours

        eval_set: str, name of the evaluation set
        dbname: str, name of the database, e.g., c4_to_999
        index_type: str, name of the index type, e.g., IVF16384,PQ64
        seq_len: int, maximum sequence length
        nprobe: str, fixed int nprobe is converted str, use dynamic -> "performance_model_using_average" or "performance_model_using_P95"
        retrieval_interval: int, retrieval interval
        staleness: bool, False: original RETRO, True: our solution with staleness
        num_continuation_chunks: int, number of continuation chunks
        num_neighbours: int, number of neighbours

    Schema (outputs): perplexity

Example usage:
    # original RETRO
    python evaluation_perplexity_all.py --test_dataset_spec $WORKSPACE/data/datasets/val_wikipedia/val_wikipedia_chunk9_1K/val_db_c4_to_999.json --mode original
    python evaluation_perplexity_all.py --test_dataset_spec $WORKSPACE/data/datasets/val_realnews/val_realnews_chunk31_1K/val_db_c4_to_999.json --mode original
    python evaluation_perplexity_all.py --test_dataset_spec $WORKSPACE/data/datasets/val_c4/val_c4_chunk1023_1K/val_db_c4_to_999.json --mode original
    
    # our solution with staleness
    python evaluation_perplexity_all.py --test_dataset_spec $WORKSPACE/data/datasets/val_wikipedia/val_wikipedia_chunk9_1K/val_db_c4_to_999.json --mode stale
    python evaluation_perplexity_all.py --test_dataset_spec $WORKSPACE/data/datasets/val_realnews/val_realnews_chunk31_1K/val_db_c4_to_999.json --mode stale
    python evaluation_perplexity_all.py --test_dataset_spec $WORKSPACE/data/datasets/val_c4/val_c4_chunk1023_1K/val_db_c4_to_999.json --mode stale

    # our solution with staleness + dynamic nprobe
    python evaluation_perplexity_all.py --test_dataset_spec $WORKSPACE/data/datasets/val_wikipedia/val_wikipedia_chunk9_1K/val_db_c4_to_999.json --mode stale_dynamic_nprobe \
        --perf_model_server_inference p4d.24xlarge --perf_model_server_retrieval m5.metal.2.5GHz \
        --generation_model_path $WORKSPACE/inference/performance/p4d.24xlarge_performance_generation_len_1024_k_2.pickle \
        --retrieval_model_path $WORKSPACE/inference/performance/m5.metal.2.5GHz_performance_search_c4_chunk_0_to_999_IVF16384,PQ64.pickle \
        --sbert_model_path $WORKSPACE/inference/performance/m5.metal.2.5GHz_performance_SBERT.pickle 

    # our solution with staleness, but without additional retrievals (set less stale steps)
    python evaluation_perplexity_all.py --test_dataset_spec $WORKSPACE/data/datasets/val_wikipedia/val_wikipedia_chunk9_1K/val_db_c4_to_999.json --mode stale_without_additional_retrievals
    python evaluation_perplexity_all.py --test_dataset_spec $WORKSPACE/data/datasets/val_realnews/val_realnews_chunk31_1K/val_db_c4_to_999.json --mode stale_without_additional_retrievals
    python evaluation_perplexity_all.py --test_dataset_spec $WORKSPACE/data/datasets/val_c4/val_c4_chunk1023_1K/val_db_c4_to_999.json --mode stale_without_additional_retrievals

    # baseline: no retrieval (still need to load a small db to do dummy retrieval)
    python evaluation_perplexity_all.py --test_dataset_spec $WORKSPACE/data/datasets/val_wikipedia/val_wikipedia_chunk9_1K/val_db_c4_to_999.json --mode no_retrieval
    python evaluation_perplexity_all.py --test_dataset_spec $WORKSPACE/data/datasets/val_realnews/val_realnews_chunk31_1K/val_db_c4_to_999.json --mode no_retrieval
    python evaluation_perplexity_all.py --test_dataset_spec $WORKSPACE/data/datasets/val_c4/val_c4_chunk1023_1K/val_db_c4_to_999.json --mode no_retrieval

    # baseline: only retrieve once (and retrieve the entire doc instead of just the following chunk)
    python evaluation_perplexity_all.py --test_dataset_spec $WORKSPACE/data/datasets/val_wikipedia/val_wikipedia_chunk9_1K/val_db_c4_to_999.json --mode one_retrieval
    python evaluation_perplexity_all.py --test_dataset_spec $WORKSPACE/data/datasets/val_realnews/val_realnews_chunk31_1K/val_db_c4_to_999.json --mode one_retrieval
    python evaluation_perplexity_all.py --test_dataset_spec $WORKSPACE/data/datasets/val_c4/val_c4_chunk1023_1K/val_db_c4_to_999.json --mode one_retrieval
"""

import argparse
import os
import subprocess
import pandas as pd
import pickle
import json

from evaluation_suite import TestConfig
from pathlib import Path

parser = argparse.ArgumentParser()

parser.add_argument("--test_dataset_spec", type=str, default='$WORKSPACE/data/datasets/val_wikipedia/val_wikipedia_chunk9_1K/val_db_c4_to_999.json')
parser.add_argument("--out_df_path", type=str, default="$WORKSPACE/plots/generation_perplexity_df.pickle")
parser.add_argument("--overwrite_entries", action="store_true", help="Whether to overwrite the existing dataframe entries if the data exists")
parser.add_argument("--mode", type=str, default="original", choices=["original", "stale", "stale_without_additional_retrievals", "stale_dynamic_nprobe", "no_retrieval", "one_retrieval"], help="Whether to run original RETRO or our solution with staleness")


# performance model
parser.add_argument('--perf_model_server_inference', type=str, default=None, help="None for fixed nprobe, or str, e.g., 'p4d.24xlarge' for dynamic nprobe")
parser.add_argument('--perf_model_server_retrieval', type=str, default=None, help="None for fixed nprobe, or str, e.g., 'p4d.24xlarge' for dynamic nprobe")
parser.add_argument('--generation_model_path', type=Path)#, default=Path('$WORKSPACE/inference/performance/p4d.24xlarge_performance_generation_len_1024_k_2.pickle'))
parser.add_argument('--retrieval_model_path', type=Path)#, default=Path('$WORKSPACE/inference/performance/p4d.24xlarge_performance_search_c4_chunk_0_to_999_IVF16384,PQ64.pickle'))
parser.add_argument('--sbert_model_path', type=Path)#, default=Path('$WORKSPACE/inference/performance/p4d.24xlarge_performance_SBERT.pickle'))
parser.add_argument('--extra_overhead_ms', type=int, default=0, help="set a default extra latency on the retrieval side")
parser.add_argument('--search_latency_budget_discount', type=float, default=1.0, help="if < 1.0, e.g., 0.9, limit the latency budget of search to 90%")
parser.add_argument('--min_nprobe', type=int, default=None)
parser.add_argument('--max_nprobe', type=int, default=None)

# worker info
# parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--gpus-per-node", type=int, default=1)
parser.add_argument("--num-nodes", type=int, default=1)
parser.add_argument("--num-workers", type=int, default=0)

args = parser.parse_args()

test_dataset_spec = os.path.expandvars(args.test_dataset_spec)
out_df_path = os.path.expandvars(args.out_df_path)

# performance model
perf_model_server_inference = args.perf_model_server_inference
perf_model_server_retrieval = args.perf_model_server_retrieval
generation_model_path = os.path.expandvars(str(args.generation_model_path))
retrieval_model_path = os.path.expandvars(str(args.retrieval_model_path))
sbert_model_path = os.path.expandvars(str(args.sbert_model_path))
extra_overhead_ms = args.extra_overhead_ms
search_latency_budget_discount = args.search_latency_budget_discount
min_nprobe = args.min_nprobe
max_nprobe = args.max_nprobe

# batch_size = args.batch_size
gpus_per_node = args.gpus_per_node
num_nodes = args.num_nodes
num_workers = 0 # once pytorch lightning spawns multiple workers, each worker uses only one thread; 0 = use main process to load data

with open(test_dataset_spec, "r") as f:
    test_dataset_spec_json = json.load(f)
eval_set = test_dataset_spec_json["eval_set"]
dbname = test_dataset_spec_json["dbname"]
index_type = test_dataset_spec_json["index_type"]

# Constants across all experiments
checkpoint = os.path.expandvars("$WORKSPACE/data/model/model.ckpt")
retro_config = os.path.expandvars("$WORKSPACE/data/model/retro.json")
num_continuation_chunks = 1
num_neighbours = 2
max_len = 1024

# define the schema of the dataframe, in flatten format
schema_inputs = {
    "eval_set": str,
    "dbname": str,
    "index_type": str,
    "seq_len": int,
    "nprobe": str,  # or str
    "retrieval_interval": int,
    "staleness": bool,
    "stale_steps": int, # by default, equal to retrieval_interval if staleness == True; if keep the retrieval frequency, the staleness can be arbitrary here
    "num_continuation_chunks": int,
    "num_neighbours": int,
    "perf_model_server_inference": str, # None for fixed nprobe, or str, e.g., "p4d.24xlarge" for dynamic nprobe
    "perf_model_server_retrieval": str, # None for fixed nprobe, or str, e.g., "p4d.24xlarge" for dynamic nprobe
}

schema_outputs = {
    "perplexity": float,
}

# merge inputs and outputs to a single schema
schema = {**schema_inputs, **schema_outputs} 

def get_batch_size(retrieval_interval):
    # based on NVIDIA A100 40GB
    if retrieval_interval >= 64:
        return 64
    elif retrieval_interval >= 32:
        return 32
    elif retrieval_interval >= 16:
        return 16
    elif retrieval_interval >= 8:
        return 8
    else:
        return 1

def read_data_from_pickle(pickle_path):
    with open(pickle_path, "rb") as f:
        return pickle.load(f)

def write_data_to_pickle(data, pickle_path):
    with open(pickle_path, "wb") as f:
        pickle.dump(data, f, protocol=4)

def check_entry_exists(pickle_path, entry_dict, content_keys={"perplexity"}):
    """ 
    Check if the entry exists in the dataframe
        entry_dict: a dict with a subset of schema as the dataframe, and contains values, e.g.,
            {"eval_set": "val_wikipedia", "dbname": "c4_to_999", "index_type": "IVF16384,PQ64", "seq_len": 1024, "nprobe": "1", "retrieval_interval": 64, "staleness": False, "num_continuation_chunks": 1, "num_neighbours": 2}
    """
    assert set(entry_dict.keys()).issubset(set(schema.keys())), "entry_dict must be a subset of schema"
    assert set(schema_inputs.keys()).issubset(set(entry_dict.keys())), "entry_dict must contain all inputs in schema"

    results_df = read_data_from_pickle(pickle_path)
    # check if the entry exists
    for _, row in results_df.iterrows():
        all_match = True
        for key in entry_dict.keys():
            if row[key] != entry_dict[key]:
                all_match = False
        if all_match:
            # check whether if all content keys are not None
            if all([row[key] is not None for key in content_keys]):
                return True
            else: # entry exists but not all content keys are not None
                return True
    return False # entry does not exist

def delete_entry(pickle_path, entry_dict, content_keys={"perplexity"}):
    """ 
    Delete the entry in the dataframe
        entry_dict: a dict with a subset of schema as the dataframe, and contains values, e.g.,
            {"eval_set": "val_wikipedia", "dbname": "c4_to_999", "index_type": "IVF16384,PQ64", "seq_len": 1024, "nprobe": "1", "retrieval_interval": 64, "staleness": False, "num_continuation_chunks": 1, "num_neighbours": 2}
    """
    assert set(entry_dict.keys()).issubset(set(schema.keys())), "entry_dict must be a subset of schema"
    assert set(schema_inputs.keys()).issubset(set(entry_dict.keys())), "entry_dict must contain all inputs in schema"

    results_df = read_data_from_pickle(pickle_path)
    written_results_df = pd.DataFrame(columns=results_df.columns)
    drop_count = 0
    # check if the entry exists
    for idx, row in results_df.iterrows():
        all_match = True
        for key in entry_dict.keys():
            if row[key] != entry_dict[key]:
                all_match = False
        if all_match:
            # check whether if all content keys are not None
            if all([row[key] is not None for key in content_keys]):
                # written_results_df = written_results_df.drop(idx, inplace=True)
                drop_count += 1
            else:
                all_match = False
        if not all_match:
            written_results_df = pd.concat([written_results_df, pd.DataFrame([row])], ignore_index=True)
    if drop_count > 0:
        print(written_results_df)
        print("Deleted {} entries".format(drop_count))
        write_data_to_pickle(written_results_df, pickle_path)
    assert not check_entry_exists(pickle_path, entry_dict, content_keys=content_keys), "Entry still exists after deletion"


def automic_write(out_df_path, new_row):
    if os.path.exists(out_df_path):
        results_df = read_data_from_pickle(out_df_path)
    else:
        # create a dataframe with the above schema, and write
        results_df = pd.DataFrame(columns=schema.keys())
    # add a random row to the dataframe
    results_df = pd.concat([results_df, new_row], ignore_index=False)
    pd.DataFrame.sort_values(results_df, by=["eval_set", "dbname", "index_type", "seq_len", "nprobe", "retrieval_interval", "staleness", "num_continuation_chunks", "num_neighbours"], inplace=True)
    # print(results_df)
    write_data_to_pickle(results_df, out_df_path)

def run_original_RETRO():

    nprobe_list = [1, 2, 4, 8, 16, 32, 64]
    # We only use 64 as the retrieval interval for original RETRO, but would be interesting to see how it affects the performance in other cases
    retrieval_interval_list = [64, 32, 16, 8] 
    # retrieval_interval_list = [64] 

    config_set = []
    for retrieval_interval in retrieval_interval_list:
        for nprobe in nprobe_list:
            batch_size = get_batch_size(retrieval_interval)
            config_set.append(TestConfig(
                    exp_name="retrieval no stale, nprobe: {}, retrieval_interval: {}".format(nprobe, retrieval_interval),
                    test_dataset_spec=test_dataset_spec,
                    num_continuation_chunks=num_continuation_chunks,
                    staleness=0,
                    stale_steps=None,
                    remove_stale_context=0,
                    max_len=max_len,
                    num_neighbours=num_neighbours,
                    no_retrieval=0,
                    retrieval_interval=retrieval_interval,
                    nprobe=nprobe,
                    # model info
                    checkpoint=checkpoint,
                    retro_config=retro_config,
                    # worker info
                    batch_size=batch_size,
                    gpus_per_node=gpus_per_node,
                    num_nodes=num_nodes,
                    num_workers=num_workers,
                ))

    for config in config_set:

        schema_inputs_values = {
            "eval_set": eval_set,
            "dbname": dbname,
            "index_type": index_type,
            "seq_len": max_len,
            "nprobe": str(config.nprobe),
            "retrieval_interval": config.retrieval_interval,
            "staleness": False,
            "stale_steps": 0,
            "num_continuation_chunks": num_continuation_chunks,
            "num_neighbours": num_neighbours,
            "perf_model_server_inference": None,
            "perf_model_server_retrieval": None,
            }

        if os.path.exists(out_df_path) and check_entry_exists(out_df_path, schema_inputs_values, content_keys={"perplexity"}):
            print("Entry exists, skip...")
            print("Entry config: ", schema_inputs_values)
        else:
            delete_entry(out_df_path, schema_inputs_values, content_keys={"perplexity"})
            print("Entry does not exist or overwrite, run...")
            perplexity = config.eval()

            schema_outputs_values = {
                "perplexity": perplexity,
            }

            # merge the two dicts as the new row
            new_row = pd.DataFrame([{**schema_inputs_values, **schema_outputs_values}])

            automic_write(out_df_path, new_row)


def run_PipeRAG():

    nprobe_list = [1, 2, 4, 8, 16, 32, 64]
    retrieval_interval_list = [64, 32, 16, 8] 

    config_set = []
    for retrieval_interval in retrieval_interval_list:
        for nprobe in nprobe_list:
            batch_size = get_batch_size(retrieval_interval)
            config_set.append(TestConfig(
                    exp_name="retrieval with stale, nprobe: {}, retrieval_interval: {}".format(nprobe, retrieval_interval),
                    test_dataset_spec=test_dataset_spec,
                    num_continuation_chunks=num_continuation_chunks,
                    staleness=1,
                    stale_steps=retrieval_interval,
                    remove_stale_context=1,
                    max_len=max_len,
                    num_neighbours=num_neighbours,
                    no_retrieval=0,
                    retrieval_interval=retrieval_interval,
                    nprobe=nprobe,
                    # model info
                    checkpoint=checkpoint,
                    retro_config=retro_config,
                    # worker info
                    batch_size=batch_size,
                    gpus_per_node=gpus_per_node,
                    num_nodes=num_nodes,
                    num_workers=num_workers,
                ))

    for config in config_set:

        schema_inputs_values = {
            "eval_set": eval_set,
            "dbname": dbname,
            "index_type": index_type,
            "seq_len": max_len,
            "nprobe": str(config.nprobe),
            "retrieval_interval": config.retrieval_interval,
            "staleness": True,
            "stale_steps": config.retrieval_interval,
            "num_continuation_chunks": num_continuation_chunks,
            "num_neighbours": num_neighbours,
            "perf_model_server_inference": None,
            "perf_model_server_retrieval": None,
            }

        if os.path.exists(out_df_path) and check_entry_exists(out_df_path, schema_inputs_values, content_keys={"perplexity"}) \
            and not args.overwrite_entries:
            print("Entry exists, skip...")
            print("Entry config: ", schema_inputs_values)
        else:
            delete_entry(out_df_path, schema_inputs_values, content_keys={"perplexity"})
            print("Entry does not exist or overwrite, run...")
            perplexity = config.eval()

            schema_outputs_values = {
                "perplexity": perplexity,
            }

            # merge the two dicts as the new row
            new_row = pd.DataFrame([{**schema_inputs_values, **schema_outputs_values}])

            automic_write(out_df_path, new_row)


def run_PipeRAG_with_dynamic_nprobe():

    assert perf_model_server_inference is not None and perf_model_server_retrieval is not None, "perf_model_server_inference and perf_model_server_retrieval must be specified"
    retrieval_interval_list = [64, 32, 16, 8] 

    config_set = []
    for retrieval_interval in retrieval_interval_list:
        batch_size = get_batch_size(retrieval_interval)
        config_set.append(TestConfig(
                exp_name="retrieval with stale, nprobe: dynamic, retrieval_interval: {}".format(retrieval_interval),
                test_dataset_spec=test_dataset_spec,
                num_continuation_chunks=num_continuation_chunks,
                staleness=1,
                stale_steps=retrieval_interval,
                remove_stale_context=1,
                max_len=max_len,
                num_neighbours=num_neighbours,
                no_retrieval=0,
                retrieval_interval=retrieval_interval,
                nprobe=1,
                # model info
                checkpoint=checkpoint,
                retro_config=retro_config,
                # worker info
                batch_size=batch_size,
                gpus_per_node=gpus_per_node,
                num_nodes=num_nodes,
                num_workers=num_workers,

                # performance model
                use_perf_model = True,
                generation_model_path = generation_model_path,
                retrieval_model_path = retrieval_model_path,
                sbert_model_path = sbert_model_path,
                extra_overhead_ms = extra_overhead_ms,
                search_latency_budget_discount = search_latency_budget_discount,
                min_nprobe = min_nprobe,
                max_nprobe = max_nprobe,
            ))

    for config in config_set:

        schema_inputs_values = {
            "eval_set": eval_set,
            "dbname": dbname,
            "index_type": index_type,
            "seq_len": max_len,
            "nprobe": "dynamic",
            "retrieval_interval": config.retrieval_interval,
            "staleness": True,
            "stale_steps": config.retrieval_interval,
            "num_continuation_chunks": num_continuation_chunks,
            "num_neighbours": num_neighbours,
            "perf_model_server_inference": perf_model_server_inference,
            "perf_model_server_retrieval": perf_model_server_retrieval,
            }

        if os.path.exists(out_df_path) and check_entry_exists(out_df_path, schema_inputs_values, content_keys={"perplexity"}) \
            and not args.overwrite_entries:
            print("Entry exists, skip...")
            print("Entry config: ", schema_inputs_values)
        else:
            delete_entry(out_df_path, schema_inputs_values, content_keys={"perplexity"})
            print("Entry does not exist or overwrite, run...")
            perplexity = config.eval()

            schema_outputs_values = {
                "perplexity": perplexity,
            }

            # merge the two dicts as the new row
            new_row = pd.DataFrame([{**schema_inputs_values, **schema_outputs_values}])

            automic_write(out_df_path, new_row)


def run_PipeRAG_without_additional_retrievals():

    nprobe_list = [1, 2, 4, 8, 16, 32, 64]
    retrieval_interval = 64
    stale_steps_list = [32, 16, 8] 
    # stale_steps_list = [64, 32, 16, 8] 

    config_set = []
    for stale_steps in stale_steps_list:
        for nprobe in nprobe_list:
            batch_size = get_batch_size(retrieval_interval)
            config_set.append(TestConfig(
                    exp_name="retrieval with stale (stale_steps != retrieval_interval), nprobe: {}, retrieval_interval: {}, stale_steps: {}".format(
                        nprobe, retrieval_interval, stale_steps),
                    test_dataset_spec=test_dataset_spec,
                    num_continuation_chunks=num_continuation_chunks,
                    staleness=1,
                    stale_steps=stale_steps,
                    remove_stale_context=1,
                    max_len=max_len,
                    num_neighbours=num_neighbours,
                    no_retrieval=0,
                    retrieval_interval=retrieval_interval,
                    nprobe=nprobe,
                    # model info
                    checkpoint=checkpoint,
                    retro_config=retro_config,
                    # worker info
                    batch_size=batch_size,
                    gpus_per_node=gpus_per_node,
                    num_nodes=num_nodes,
                    num_workers=num_workers,
                ))

    for config in config_set:

        schema_inputs_values = {
            "eval_set": eval_set,
            "dbname": dbname,
            "index_type": index_type,
            "seq_len": max_len,
            "nprobe": str(config.nprobe),
            "retrieval_interval": config.retrieval_interval,
            "staleness": True,
            "stale_steps": config.stale_steps,
            "num_continuation_chunks": num_continuation_chunks,
            "num_neighbours": num_neighbours,
            "perf_model_server_inference": None,
            "perf_model_server_retrieval": None,
            }

        if os.path.exists(out_df_path) and check_entry_exists(out_df_path, schema_inputs_values, content_keys={"perplexity"}) \
            and not args.overwrite_entries:
            print("Entry exists, skip...")
            print("Entry config: ", schema_inputs_values)
        else:
            delete_entry(out_df_path, schema_inputs_values, content_keys={"perplexity"})
            print("Entry does not exist or overwrite, run...")
            perplexity = config.eval()

            schema_outputs_values = {
                "perplexity": perplexity,
            }

            # merge the two dicts as the new row
            new_row = pd.DataFrame([{**schema_inputs_values, **schema_outputs_values}])

            automic_write(out_df_path, new_row)


def run_no_retrieval():

    nprobe = 1
    retrieval_interval = 64
    batch_size = get_batch_size(retrieval_interval)
    config = TestConfig(
            exp_name="retrieval no stale, nprobe: {}, retrieval_interval: {}".format(nprobe, retrieval_interval),
            test_dataset_spec=test_dataset_spec,
            num_continuation_chunks=num_continuation_chunks,
            staleness=0,
            stale_steps=None,
            remove_stale_context=0,
            max_len=max_len,
            num_neighbours=num_neighbours,
            no_retrieval=1, # no retrieval
            retrieval_interval=retrieval_interval,
            nprobe=nprobe,
            # model info
            checkpoint=checkpoint,
            retro_config=retro_config,
            # worker info
            batch_size=batch_size,
            gpus_per_node=gpus_per_node,
            num_nodes=num_nodes,
            num_workers=num_workers,
        )

    # only 'eval_set' and 'seq_len' are not None
    schema_inputs_values = {
        "eval_set": eval_set,
        "dbname": None,
        "index_type": None,
        "seq_len": max_len,
        "nprobe": None,
        "retrieval_interval": None,
        "staleness": None,
        "stale_steps": 0,
        "num_continuation_chunks": None,
        "num_neighbours": None,
        "perf_model_server_inference": None,
        "perf_model_server_retrieval": None,
        }

    if os.path.exists(out_df_path) and check_entry_exists(out_df_path, schema_inputs_values, content_keys={"perplexity"}):
        print("Entry exists, skip...")
        print("Entry config: ", schema_inputs_values)
    else:
        delete_entry(out_df_path, schema_inputs_values, content_keys={"perplexity"})
        print("Entry does not exist or overwrite, run...")
        perplexity = config.eval()

        schema_outputs_values = {
            "perplexity": perplexity,
        }

        # merge the two dicts as the new row
        new_row = pd.DataFrame([{**schema_inputs_values, **schema_outputs_values}])

        automic_write(out_df_path, new_row)

def run_RETRO_with_only_one_retrieval():

    nprobe_list = [1, 2, 4, 8, 16, 32, 64]
    # We only use 64 as the retrieval interval for original RETRO, but would be interesting to see how it affects the performance in other cases
    retrieval_interval = 64

    one_retrieval_num_continuation_chunks = 1 # retrieve the entire document once
    # one_retrieval_num_continuation_chunks = 16 - 1 # retrieve the entire document once

    config_set = []
    for nprobe in nprobe_list:
        batch_size = get_batch_size(retrieval_interval)
        config_set.append(TestConfig(
                exp_name="retrieval only once, no stale, nprobe: {}".format(nprobe),
                test_dataset_spec=test_dataset_spec,
                num_continuation_chunks=one_retrieval_num_continuation_chunks,
                staleness=0,
                stale_steps=None,
                remove_stale_context=0,
                max_len=max_len,
                num_neighbours=num_neighbours,
                no_retrieval=0,
                one_retrieval=1,
                retrieval_interval=retrieval_interval, # keep this as retrieval interval to launch the program
                nprobe=nprobe,
                # model info
                checkpoint=checkpoint,
                retro_config=retro_config,
                # worker info
                batch_size=batch_size,
                gpus_per_node=gpus_per_node,
                num_nodes=num_nodes,
                num_workers=num_workers,
            ))

    for config in config_set:

        # for retrieve only once, set the retrieval interval to be equal to sequence length
        schema_inputs_values = {
            "eval_set": eval_set,
            "dbname": dbname,
            "index_type": index_type,
            "seq_len": max_len,
            "nprobe": str(config.nprobe),
            "retrieval_interval": max_len,
            "staleness": False,
            "stale_steps": 0,
            "num_continuation_chunks": one_retrieval_num_continuation_chunks,
            "num_neighbours": num_neighbours,
            "perf_model_server_inference": None,
            "perf_model_server_retrieval": None,
            }

        if os.path.exists(out_df_path) and check_entry_exists(out_df_path, schema_inputs_values, content_keys={"perplexity"}):
            print("Entry exists, skip...")
            print("Entry config: ", schema_inputs_values)
        else:
            delete_entry(out_df_path, schema_inputs_values, content_keys={"perplexity"})
            print("Entry does not exist or overwrite, run...")
            perplexity = config.eval()

            schema_outputs_values = {
                "perplexity": perplexity,
            }

            # merge the two dicts as the new row
            new_row = pd.DataFrame([{**schema_inputs_values, **schema_outputs_values}])

            automic_write(out_df_path, new_row)


if __name__ == "__main__":

    if args.mode == "original":
        run_original_RETRO()
    elif args.mode == "stale":
        run_PipeRAG()
    elif args.mode == "stale_without_additional_retrievals":
        run_PipeRAG_without_additional_retrievals()
    elif args.mode == "stale_dynamic_nprobe":
        run_PipeRAG_with_dynamic_nprobe()
    elif args.mode == "no_retrieval":
        run_no_retrieval()
    elif args.mode == "one_retrieval":
        run_RETRO_with_only_one_retrieval()