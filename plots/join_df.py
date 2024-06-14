"""
Example usage:
    python join_df.py --df_ppl_path $WORKSPACE/plots/generation_perplexity_df.pickle \
        --df_perf_path $WORKSPACE/plots/generation_performance_df.pickle \
        --out_df_path $WORKSPACE/plots/generation_join_perplexity_and_performance_df.pickle
"""

import argparse
import pandas as pd
import pickle
import os

from print_df import read_data_from_pickle, write_data_to_pickle, select_rows

"""
schema_common_inputs = {
    "dbname": str,
    "index_type": str,
    "seq_len": int,
    "nprobe": str,  # or str
    "retrieval_interval": int,
    "staleness": bool,
    "stale_steps", int,
    "num_continuation_chunks": int,
    "num_neighbours": int,
}
"""

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--df_ppl_path", type=str, default="$WORKSPACE/plots/generation_perplexity_df.pickle")
    parser.add_argument("--df_perf_path", type=str, default="$WORKSPACE/plots/generation_performance_df.pickle")
    parser.add_argument("--out_df_path", type=str, default="$WORKSPACE/plots/generation_join_perplexity_and_performance_df.pickle")

    args = parser.parse_args()
    df_ppl_path = os.path.expandvars(args.df_ppl_path)
    df_perf_path = os.path.expandvars(args.df_perf_path)
    out_df_path = os.path.expandvars(args.out_df_path)

    df_ppl = pd.DataFrame(read_data_from_pickle(df_ppl_path))
    df_perf = pd.DataFrame(read_data_from_pickle(df_perf_path))
    print("Perplexity dataframe sample:")
    print(df_ppl.iloc[0])
    print("\n\nPerformance dataframe sample:")
    print(df_perf.iloc[0])

    # join the two dataframes
    df = pd.merge(df_ppl, df_perf, 
        left_on=["dbname", "index_type", "seq_len", "nprobe", "retrieval_interval", "staleness", "stale_steps", "num_continuation_chunks", "num_neighbours"],
        right_on=["dbname", "index_type", "seq_len", "nprobe", "retrieval_interval", "staleness", "stale_steps", "num_continuation_chunks", "num_neighbours"],)
    # df = df_ppl.join(df_perf, on=["dbname", "index_type", "seq_len", "nprobe", "retrieval_interval", "staleness", "num_continuation_chunks", "num_neighbours"], how='right')

    # for those dynamic nprobe, make sure the inference and retrieval server are the same
    df_dynamic_nprobe = df.loc[(df['server_inference'] == df['perf_model_server_inference']) & (df['server_retrieval'] == df['perf_model_server_retrieval'])]
    df_fixed_nprobe = df.loc[df['perf_model_server_inference'].isna() & df['perf_model_server_retrieval'].isna()]
    df = pd.concat([df_dynamic_nprobe, df_fixed_nprobe])

    print(df)
    write_data_to_pickle(df, out_df_path)