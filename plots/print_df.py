"""
Example usage:
    python print_df.py --df_path generation_perplexity_df.pickle

If want to select rows:
    python print_df.py --df_path generation_perplexity_df.pickle \
        --eval_set wikipedia_chunk9_1K \
        --dbname c4_chunk_0_to_999 \
        --index_type IVF16384,PQ64 \
        --seq_len 1024 \
        --nprobe 1 \
        --retrieval_interval 64 \
        --staleness 0 \
        --num_continuation_chunks 1 \
        --num_neighbours 2 \
        --stale_steps 64
"""

import argparse
import pandas as pd
import pickle
import os

"""
schema_inputs = {
    "eval_set": str,
    "dbname": str,
    "index_type": str,
    "seq_len": int,
    "nprobe": str,  # or str
    "retrieval_interval": int,
    "staleness": bool,
    "num_continuation_chunks": int,
    "num_neighbours": int,
}
"""


def read_data_from_pickle(pickle_path):
    with open(pickle_path, "rb") as f:
        return pickle.load(f)

def write_data_to_pickle(data, pickle_path):
    with open(pickle_path, "wb") as f:
        pickle.dump(data, f, protocol=4)

def select_rows(df, key_values):
    """
    Select rows from a dataframe that match the given key-value pairs
    """
    # create an dataframe with the same schema
    selected_rows = pd.DataFrame(columns=df.columns)
    # iterate through the dataframe
    for _, row in df.iterrows():
        all_match = True
        for key in key_values.keys():
            if row[key] != key_values[key]:
                all_match = False
        if all_match:
            selected_rows = pd.concat([selected_rows, pd.DataFrame([row])], ignore_index=True)
    return selected_rows

def select_rows_aligned_stale_and_interval(df):
    """
    For normal RETRO (staleness = False), "stale_steps" == 0;
    For PipeRAGE (staleness = True), "stale_steps" == "retrieval_interval" 
    """
    # create an dataframe with the same schema
    selected_rows = pd.DataFrame(columns=df.columns)
    # iterate through the dataframe
    for _, row in df.iterrows():
        if row["dbname"] == None: # no retrieval
            continue
        if row["staleness"] == False:
            if row["stale_steps"] == 0:
                selected_rows = pd.concat([selected_rows, pd.DataFrame([row])], ignore_index=True)
        else: # Stale
            if row["stale_steps"] == row["retrieval_interval"]:
                selected_rows = pd.concat([selected_rows, pd.DataFrame([row])], ignore_index=True)

    return selected_rows

def find_duplicate_entries(df, df_path=None, delete_entries=True):
    """
    Duplicate values in all but the perplexity column
    """
    count = 0
    schema = df.columns
    # iterate through the dataframe
    for _, row in df.iterrows():
        # turn the row into key values
        key_values = {}
        for key in schema:
            if key != "perplexity":
                key_values[key] = row[key]
        # find the duplicate entries
        duplicate_entries = select_rows(df, key_values)
        if duplicate_entries.index.size > 1:
            print("Duplicate entries found: {}".format(key_values))
            print(duplicate_entries)
            count += 1
            if delete_entries:
                delete_entry(df_path, key_values)

    print("Total {} duplicate entries found".format(count))

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

# def add_server_columns(df):
#     perf_model_server_inference = []
#     perf_model_server_retrieval = []
#     for _, row in df.iterrows():
#         if row["nprobe"] == "dynamic":
#             perf_model_server_inference.append("p4d.24xlarge")
#             perf_model_server_retrieval.append("p4d.24xlarge")
#         else: # None
#             perf_model_server_inference.append(None)
#             perf_model_server_retrieval.append(None)
#     df["perf_model_server_inference"] = perf_model_server_inference
#     df["perf_model_server_retrieval"] = perf_model_server_retrieval
#     return df

# def add_stale_steps_columns(df):
#     stale_steps = []
#     for _, row in df.iterrows():
#         if row["staleness"] == True:
#             stale_steps.append(row["retrieval_interval"])
#         else: # None
#             stale_steps.append(0)
#     df["stale_steps"] = stale_steps
#     return df

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--df_path", type=str, default="$WORKSPACE/plots/generation_perplexity_df.pickle")

    # if sepcify the following arguments, trigger the selection
    parser.add_argument("--eval_set", type=str) #, default="wikipedia_chunk9_1K")
    parser.add_argument("--dbname", type=str) #, default="c4_chunk_0_to_999")
    parser.add_argument("--index_type", type=str) #, default="IVF16384,PQ64")
    parser.add_argument("--seq_len", type=int) #, default=1024)
    parser.add_argument("--nprobe", type=str) #, default="1")
    parser.add_argument("--retrieval_interval", type=int) #, default=64)
    parser.add_argument("--staleness", type=int) #, default=0)
    parser.add_argument("--num_continuation_chunks", type=int) #, default=1)
    parser.add_argument("--num_neighbours", type=int) #, default=2)
    parser.add_argument("--stale_steps", type=int) #, default=64)

    args = parser.parse_args()
    df_path = os.path.expandvars(args.df_path)
    df = pd.DataFrame(read_data_from_pickle(df_path))

    key_values = {}
    if args.eval_set is not None:
        key_values["eval_set"] = args.eval_set
    if args.dbname is not None:
        key_values["dbname"] = args.dbname
    if args.index_type is not None:
        key_values["index_type"] = args.index_type
    if args.seq_len is not None:
        key_values["seq_len"] = args.seq_len
    if args.nprobe is not None:
        key_values["nprobe"] = args.nprobe
    if args.retrieval_interval is not None:
        key_values["retrieval_interval"] = args.retrieval_interval
    if args.staleness is not None:
        if args.staleness == 1:
            key_values["staleness"] = True
        elif args.staleness == 0:
            key_values["staleness"] = False
    if args.num_continuation_chunks is not None:
        key_values["num_continuation_chunks"] = args.num_continuation_chunks
    if args.num_neighbours is not None:
        key_values["num_neighbours"] = args.num_neighbours
    if args.stale_steps is not None:
        key_values["stale_steps"] = args.stale_steps

    if len(key_values) > 0:
        df = select_rows(df, key_values)
    print(df)

    # find_duplicate_entries(df, df_path=df_path, delete_entries=False)
    # find_duplicate_entries(df, df_path=df_path, delete_entries=True)
