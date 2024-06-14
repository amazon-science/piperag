"""
Evaluate the end-to-end RAG performance, assuming the retrieval database is already started.

Example Usage:

    # Fixed nprobe
    python evaluate_rag_performance.py --host 127.0.0.1 --port 50051 \
        --out_df_path $WORKSPACE/plots/generation_performance_df.pickle \
        --num_runs 5 \
        --dbname c4_chunk_0_to_999 --index_type IVF16384,PQ64 \
        --server_inference p4d.24xlarge --server_retrieval p4d.24xlarge --mode fixed_nprobe
    
    # Dynamic nprobe
    python evaluate_rag_performance.py --host 127.0.0.1 --port 50051 \
        --out_df_path $WORKSPACE/plots/generation_performance_df.pickle \
        --num_runs 5 \
        --dbname c4_chunk_0_to_999 --index_type IVF16384,PQ64 \
        --server_inference p4d.24xlarge --server_retrieval p4d.24xlarge --mode dynamic_nprobe

    # RETRO + no retrieval
    python evaluate_rag_performance.py --host 127.0.0.1 --port 50051 \
        --out_df_path $WORKSPACE/plots/generation_performance_df.pickle \
        --num_runs 5 \
        --server_inference p4d.24xlarge --server_retrieval p4d.24xlarge --mode RETRO_no_retrieval

    # RETRO + one retrieval
    python evaluate_rag_performance.py --host 127.0.0.1 --port 50051 \
        --out_df_path $WORKSPACE/plots/generation_performance_df.pickle \
        --num_runs 5 \
        --dbname c4_chunk_0_to_999 --index_type IVF16384,PQ64 \
        --server_inference p4d.24xlarge --server_retrieval p4d.24xlarge --mode RETRO_one_retrieval
"""

import asyncio
import gc
import os 
import numpy as np
import pandas as pd
import pickle

from pathlib import Path
from inference_client import InferenceClient


batch_size = 1
chunk_size = 64
num_continuation_chunks = 1
num_neighbours = 2
max_len = 1024
prompt = "Start to" # need to at least have two tokens for ONNX to initialize the input states

# define the schema of the dataframe, in flatten format
schema_inputs = {
    "server_inference": str,
    "server_retrieval": str,
    "dbname": str,
    "index_type": str,
    "seq_len": int,
    "nprobe": str,  # or str
    "retrieval_interval": int,
    "staleness": bool,
    "stale_steps": int, # by default, equal to retrieval_interval if staleness == True; if keep the retrieval frequency, the staleness can be arbitrary here
    "num_continuation_chunks": int,
    "num_neighbours": int,
}

schema_outputs = {
    "sequence_latency_total_ms": list,  # a 1-d np array storing total sequence generation time, shape: (nruns, )
    "sequence_latency_per_token_ms": list,  # a 2-d array of sequence generation latency per token, (nruns, seq_len)
}

# merge inputs and outputs to a single schema
schema = {**schema_inputs, **schema_outputs} 

def read_data_from_pickle(pickle_path):
    with open(pickle_path, "rb") as f:
        return pickle.load(f)

def write_data_to_pickle(data, pickle_path):
    with open(pickle_path, "wb") as f:
        pickle.dump(data, f, protocol=4)

def check_entry_exists(pickle_path, entry_dict, content_keys={"sequence_latency_total_ms", "sequence_latency_per_token_ms"}):
    """ 
    Check if the entry exists in the dataframe
        entry_dict: a dict with a subset of schema as the dataframe, and contains values, e.g.,
            {"val_wikipedia", "dbname": "c4_to_999", "index_type": "IVF16384,PQ64", "seq_len": 1024, "nprobe": "1", "retrieval_interval": 64, "staleness": False, "num_continuation_chunks": 1, "num_neighbours": 2}
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

def delete_entry(pickle_path, entry_dict, content_keys={"sequence_latency_total_ms", "sequence_latency_per_token_ms"}):
    """ 
    Delete the entry in the dataframe
        entry_dict: a dict with a subset of schema as the dataframe, and contains values, e.g.,
            {"val_wikipedia", "dbname": "c4_to_999", "index_type": "IVF16384,PQ64", "seq_len": 1024, "nprobe": "1", "retrieval_interval": 64, "staleness": False, "num_continuation_chunks": 1, "num_neighbours": 2}
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
    pd.DataFrame.sort_values(results_df, by=["dbname", "index_type", "seq_len", "nprobe", "retrieval_interval", "staleness", "num_continuation_chunks", "num_neighbours"], inplace=True)
    # print(results_df)
    write_data_to_pickle(results_df, out_df_path)

def run_fixed_nprobe(args):

    nprobe_list = [1, 2, 4, 8, 16, 32, 64]
    # We only use 64 as the retrieval interval for original RETRO, but would be interesting to see how it affects the performance in other cases
    retrieval_interval_list = [64, 32, 16, 8] 
    # retrieval_interval_list = [64] 

    dbname = args.dbname
    index_type = args.index_type

    config_set = []
    for retrieval_interval in retrieval_interval_list:
        for nprobe in nprobe_list:
            for staleness in [False, True]:
            
                schema_inputs_values = {
                    "server_inference": args.server_inference,
                    "server_retrieval": args.server_retrieval,
                    "dbname": dbname,
                    "index_type": index_type,
                    "seq_len": max_len,
                    "nprobe": str(nprobe),
                    "retrieval_interval": retrieval_interval,
                    "staleness": staleness,
                    "num_continuation_chunks": num_continuation_chunks,
                    "num_neighbours": num_neighbours}
                if staleness:
                    schema_inputs_values["stale_steps"] = retrieval_interval
                else:
                    schema_inputs_values["stale_steps"] = 0

                if os.path.exists(args.out_df_path) and not args.overwrite_entries and \
                    check_entry_exists(args.out_df_path, schema_inputs_values, content_keys={"sequence_latency_total_ms", "sequence_latency_per_token_ms"}):
                    print("Entry exists, skip...")
                    print("Entry config: ", schema_inputs_values)
                else:
                    delete_entry(args.out_df_path, schema_inputs_values, content_keys={"sequence_latency_total_ms", "sequence_latency_per_token_ms"})
                    print("Entry does not exist or overwrite, run...")
                    print("Entry config: ", schema_inputs_values)

                    staleness_offset = retrieval_interval # currently, staleness_offset == interval, but in the future can be decoupled to be different
                    retriever = InferenceClient(host=args.host, port=args.port,
                        checkpoint=args.checkpoint, retro_config=args.retro_config,
                        encoder_dir=args.encoder_dir, decoder_dir=args.decoder_dir,
                        # parameterrs below are not from args
                        chunk_size=chunk_size, prompt=prompt, batch_size=batch_size, max_gen_len=max_len,
                        num_neighbours=num_neighbours, num_continuation_chunks=num_continuation_chunks,
                        staleness_offset=staleness_offset, interval=retrieval_interval
                        )
                    # reset server default nprobe
                    asyncio.run(retriever.set_server_nprobe(nprobe))

                    sequence_latency_total_ms = [] 
                    sequence_latency_per_token_ms = []
                    for _ in range(args.num_runs):
                        time_per_step_sec = asyncio.run(retriever.generation(staleness=staleness, use_perf_model=False))
                        time_per_step_ms = np.array(time_per_step_sec) * 1000
                        sequence_latency_total_ms.append(np.sum(time_per_step_ms))
                        sequence_latency_per_token_ms.append(time_per_step_ms)
                    sequence_latency_total_ms = list(np.array(sequence_latency_total_ms))
                    sequence_latency_per_token_ms = list(np.array(sequence_latency_per_token_ms))
                    print("sequence_latency_total_ms: ", sequence_latency_total_ms)
                    print("average per token latency (ms): ", np.mean(sequence_latency_per_token_ms))

                    schema_outputs_values = {
                        "sequence_latency_total_ms": sequence_latency_total_ms,
                        "sequence_latency_per_token_ms": sequence_latency_per_token_ms,
                    }

                    # merge the two dicts as the new row
                    new_row = pd.DataFrame([{**schema_inputs_values, **schema_outputs_values}])

                    automic_write(args.out_df_path, new_row)

                    del retriever
                    gc.collect()


def run_dynamic_nprobe(args):

    retrieval_interval_list = [64, 32, 16, 8] 
    staleness = True # staleness is always True for dynamic nprobe
    nprobe = 'dynamic'

    dbname = args.dbname
    index_type = args.index_type

    config_set = []
    for retrieval_interval in retrieval_interval_list:
        schema_inputs_values = {
            "server_inference": args.server_inference,
            "server_retrieval": args.server_retrieval,
            "dbname": dbname,
            "index_type": index_type,
            "seq_len": max_len,
            "nprobe": str(nprobe),
            "retrieval_interval": retrieval_interval,
            "staleness": staleness,
            "stale_steps": retrieval_interval,
            "num_continuation_chunks": num_continuation_chunks,
            "num_neighbours": num_neighbours}

        if os.path.exists(args.out_df_path) and not args.overwrite_entries and \
            check_entry_exists(args.out_df_path, schema_inputs_values, content_keys={"sequence_latency_total_ms", "sequence_latency_per_token_ms"}):
            print("Entry exists, skip...")
            print("Entry config: ", schema_inputs_values)
        else:
            delete_entry(args.out_df_path, schema_inputs_values, content_keys={"sequence_latency_total_ms", "sequence_latency_per_token_ms"})
            print("Entry does not exist or overwrite, run...")
            print("Entry config: ", schema_inputs_values)

            staleness_offset = retrieval_interval # currently, staleness_offset == interval, but in the future can be decoupled to be different
            retriever = InferenceClient(host=args.host, port=args.port,
                checkpoint=args.checkpoint, retro_config=args.retro_config,
                encoder_dir=args.encoder_dir, decoder_dir=args.decoder_dir,
                # parameterrs below are not from args
                chunk_size=chunk_size, prompt=prompt, batch_size=batch_size, max_gen_len=max_len,
                num_neighbours=num_neighbours, num_continuation_chunks=num_continuation_chunks,
                staleness_offset=staleness_offset, interval=retrieval_interval
                )

            sequence_latency_total_ms = [] 
            sequence_latency_per_token_ms = []
            for _ in range(args.num_runs):
                time_per_step_sec = asyncio.run(retriever.generation(staleness=staleness, use_perf_model=True))
                time_per_step_ms = np.array(time_per_step_sec) * 1000
                sequence_latency_total_ms.append(np.sum(time_per_step_ms))
                sequence_latency_per_token_ms.append(time_per_step_ms)
            sequence_latency_total_ms = list(np.array(sequence_latency_total_ms))
            sequence_latency_per_token_ms = list(np.array(sequence_latency_per_token_ms))
            print("sequence_latency_total_ms: ", sequence_latency_total_ms)
            print("average per token latency (ms): ", np.mean(sequence_latency_per_token_ms))

            schema_outputs_values = {
                "sequence_latency_total_ms": sequence_latency_total_ms,
                "sequence_latency_per_token_ms": sequence_latency_per_token_ms,
            }

            # merge the two dicts as the new row
            new_row = pd.DataFrame([{**schema_inputs_values, **schema_outputs_values}])

            automic_write(args.out_df_path, new_row)

            del retriever
            gc.collect()


def run_RETRO_no_retrieval(args):

    # baselines: RETRO + no retrieval 
    config_set = []
            
    schema_inputs_values = {
        "server_inference": args.server_inference,
        "server_retrieval": args.server_retrieval,
        "dbname": None,
        "index_type": None,
        "seq_len": max_len,
        "nprobe": None,
        "retrieval_interval": None,
        "staleness": None,
        "stale_steps": 0,
        "num_continuation_chunks": None,
        "num_neighbours": None}

    if os.path.exists(args.out_df_path) and not args.overwrite_entries and \
        check_entry_exists(args.out_df_path, schema_inputs_values, content_keys={"sequence_latency_total_ms", "sequence_latency_per_token_ms"}):
        print("Entry exists, skip...")
        print("Entry config: ", schema_inputs_values)
    else:
        delete_entry(args.out_df_path, schema_inputs_values, content_keys={"sequence_latency_total_ms", "sequence_latency_per_token_ms"})
        print("Entry does not exist or overwrite, run...")
        print("Entry config: ", schema_inputs_values)

        no_retrieval_staleness_offset = 0
        no_retrieval_num_neighbours = 1
        no_retrieval_num_continuation_chunks = 0
        no_retrieval_retrieval_interval = None
        retriever = InferenceClient(host=args.host, port=args.port,
            checkpoint=args.checkpoint, retro_config=args.retro_config,
            encoder_dir=args.encoder_dir, decoder_dir=args.decoder_dir,
            # parameterrs below are not from args
            chunk_size=chunk_size, prompt=prompt, batch_size=batch_size, max_gen_len=max_len,
            num_neighbours=no_retrieval_num_neighbours, num_continuation_chunks=no_retrieval_num_continuation_chunks,
            staleness_offset=no_retrieval_staleness_offset, interval=no_retrieval_retrieval_interval
            )

        sequence_latency_total_ms = [] 
        sequence_latency_per_token_ms = []
        for _ in range(args.num_runs):
            decoder_time_per_step_sec, encoder_time_per_step_sec, total_time_per_step_sec = retriever.generation_without_retrieval()
            time_per_step_ms = np.array(total_time_per_step_sec) * 1000 
            sequence_latency_total_ms.append(np.sum(time_per_step_ms))
            sequence_latency_per_token_ms.append(time_per_step_ms)
        sequence_latency_total_ms = list(np.array(sequence_latency_total_ms))
        sequence_latency_per_token_ms = list(np.array(sequence_latency_per_token_ms))
        print("sequence_latency_total_ms: ", sequence_latency_total_ms)
        print("average per token latency (ms): ", np.mean(sequence_latency_per_token_ms))

        schema_outputs_values = {
            "sequence_latency_total_ms": sequence_latency_total_ms,
            "sequence_latency_per_token_ms": sequence_latency_per_token_ms,
        }

        # merge the two dicts as the new row
        new_row = pd.DataFrame([{**schema_inputs_values, **schema_outputs_values}])

        automic_write(args.out_df_path, new_row)

        del retriever
        gc.collect()

def run_RETRO_one_retrieval(args):

    # baselines: RETRO + only one retrieval at the beginning
    nprobe_list = [1, 2, 4, 8, 16, 32, 64]
    
    dbname = args.dbname
    index_type = args.index_type

    retrieval_interval = 64
    one_retrieval_num_continuation_chunks = 1
    staleness = False

    config_set = []
    for nprobe in nprobe_list:
        
        # for retrieve only once, set the retrieval interval to be equal to sequence length
        schema_inputs_values = {
            "server_inference": args.server_inference,
            "server_retrieval": args.server_retrieval,
            "dbname": dbname,
            "index_type": index_type,
            "seq_len": max_len,
            "nprobe": str(nprobe),
            "retrieval_interval": max_len,
            "staleness": staleness,
            "stale_steps": 0,
            "num_continuation_chunks": num_continuation_chunks,
            "num_neighbours": num_neighbours}

        if os.path.exists(args.out_df_path) and not args.overwrite_entries and \
            check_entry_exists(args.out_df_path, schema_inputs_values, content_keys={"sequence_latency_total_ms", "sequence_latency_per_token_ms"}):
            print("Entry exists, skip...")
            print("Entry config: ", schema_inputs_values)
        else:
            delete_entry(args.out_df_path, schema_inputs_values, content_keys={"sequence_latency_total_ms", "sequence_latency_per_token_ms"})
            print("Entry does not exist or overwrite, run...")
            print("Entry config: ", schema_inputs_values)

            staleness_offset = 0 
            retriever = InferenceClient(host=args.host, port=args.port,
                checkpoint=args.checkpoint, retro_config=args.retro_config,
                encoder_dir=args.encoder_dir, decoder_dir=args.decoder_dir,
                # parameterrs below are not from args
                chunk_size=chunk_size, prompt=prompt, batch_size=batch_size, max_gen_len=max_len,
                num_neighbours=num_neighbours, num_continuation_chunks=num_continuation_chunks,
                staleness_offset=staleness_offset, interval=retrieval_interval
                )
            # reset server default nprobe
            asyncio.run(retriever.set_server_nprobe(nprobe))

            sequence_latency_total_ms = [] 
            sequence_latency_per_token_ms = []
            for _ in range(args.num_runs):
                time_per_step_sec = asyncio.run(retriever.generation(staleness=staleness, use_perf_model=False, one_retrieval=True))
                time_per_step_ms = np.array(time_per_step_sec) * 1000
                sequence_latency_total_ms.append(np.sum(time_per_step_ms))
                sequence_latency_per_token_ms.append(time_per_step_ms)
            sequence_latency_total_ms = list(np.array(sequence_latency_total_ms))
            sequence_latency_per_token_ms = list(np.array(sequence_latency_per_token_ms))
            print("sequence_latency_total_ms: ", sequence_latency_total_ms)
            print("average per token latency (ms): ", np.mean(sequence_latency_per_token_ms))

            schema_outputs_values = {
                "sequence_latency_total_ms": sequence_latency_total_ms,
                "sequence_latency_per_token_ms": sequence_latency_per_token_ms,
            }

            # merge the two dicts as the new row
            new_row = pd.DataFrame([{**schema_inputs_values, **schema_outputs_values}])

            automic_write(args.out_df_path, new_row)

            del retriever
            gc.collect()


if __name__ == '__main__':
    
    import argparse 
    parser = argparse.ArgumentParser()

    parser.add_argument('--host', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=50051)
    parser.add_argument('--checkpoint', type=Path, default=Path('../data/model/model.ckpt'))
    parser.add_argument('--retro_config', type=Path, default=Path('../data/model/retro.json'))
    parser.add_argument('--encoder_dir', type=Path, default=Path('../src/onnx_retro_encoder/retro_encoder.onnx'))
    parser.add_argument('--decoder_dir', type=Path, default=Path('../src/onnx_retro_decoder/retro_decoder.onnx'))

    parser.add_argument("--mode", type=str, choices=["fixed_nprobe", "dynamic_nprobe", "RETRO_no_retrieval", "RETRO_one_retrieval"])
    parser.add_argument("--dbname", type=str)#, default="c4_chunk_0_to_999")
    parser.add_argument("--index_type", type=str)#, default="IVF16384,PQ64")
    parser.add_argument("--server_inference", type=str) #, default="p4d.24xlarge")
    parser.add_argument("--server_retrieval", type=str) #, default="p4d.24xlarge")
    parser.add_argument("--out_df_path", type=str, default="$WORKSPACE/plots/generation_performance_df.pickle")
    parser.add_argument("--overwrite_entries", action="store_true", help="Whether to overwrite the existing dataframe entries if the data exists")
    parser.add_argument('--num_runs', type=int, default=5, help="number of runs")

    args = parser.parse_args()
    if args.mode == "dynamic_nprobe":
        run_dynamic_nprobe(args)
    elif args.mode == "fixed_nprobe":
        run_fixed_nprobe(args)
    elif args.mode == "RETRO_no_retrieval":
        run_RETRO_no_retrieval(args)
    elif args.mode == "RETRO_one_retrieval":
        run_RETRO_one_retrieval(args)