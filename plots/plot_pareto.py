import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pickle
import pandas as pd

from matplotlib.ticker import FuncFormatter
from print_df import read_data_from_pickle, select_rows

plt.style.use('seaborn-v0_8-deep')

print("WARNING: please join the performance df with perplexity df if you have updated either!")

num_continuation_chunks = 1
num_neighbours = 2
max_len = 1024

max_speedup = 1.0
max_ppl_diff = 0.0

# get pareto frontier between perplexity and sequence_latency_total_ms
def get_pareto(df):
    """
    return a list of (perplexity, performance) tuple
    """
    # sort by perplexity
    df = df.sort_values(by=["perplexity"])
    # get the pareto frontier
    pareto_frontier = []
    pareto_frontier_full_row = pd.DataFrame(columns=df.columns)
    
    min_sequence_latency_total_ms = np.inf
    for _, row in df.iterrows():
        if np.median(row["sequence_latency_total_ms"]) <= min_sequence_latency_total_ms:
            pareto_frontier.append((row["perplexity"], np.median(row["sequence_latency_total_ms"])))
            pareto_frontier_full_row = pd.concat([pareto_frontier_full_row, row.to_frame().transpose()])
        min_sequence_latency_total_ms = np.min([min_sequence_latency_total_ms, np.median(row["sequence_latency_total_ms"])])
    return pareto_frontier, pareto_frontier_full_row

def plot(setting_kv, df_RETRO, df_stale):


    setting_baseline = setting_kv.copy()
    setting_baseline["dbname"] = None
    setting_baseline["index_type"] = None
    setting_baseline["num_continuation_chunks"] = None
    setting_baseline["num_neighbours"] = None
    df_baseline = select_rows(df, setting_baseline)
    assert len(df_baseline.values) == 1
    baseline_ppl = df_baseline["perplexity"].values[0]
    baseline_latency = np.median(df_baseline["sequence_latency_total_ms"].values[0]) / 1000.0
    print("Baseline", df_baseline)

    pareto_frontier_RETRO, _ = get_pareto(df_RETRO)
    pareto_frontier_stale, pareto_frontier_full_row = get_pareto(df_stale)
    print("The pareto frontier of PipeRAG:")
    print(pareto_frontier_full_row[["retrieval_interval", "nprobe"]])

    # x = perplexity, y = sequence_latency_total_ms
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 3))

    label_font = 18
    markersize = 8
    tick_font = 16
    legen_font = 16

    # plot RETRO
    data_x_RETRO = [ppl for ppl, _ in pareto_frontier_RETRO]
    data_y_RETRO = [latency  for _, latency in pareto_frontier_RETRO]
    max_latency_RETRO = np.max(data_y_RETRO)
    max_perplexity_RETRO = np.max(data_x_RETRO)

    # plot stale
    # drop all tuples where the latency is larger than the max latency of RETRO * 1.2
    pareto_frontier_stale = [t for t in pareto_frontier_stale if t[1] <= max_latency_RETRO * 1.0]
    data_x_stale = [ppl for ppl, _ in pareto_frontier_stale]
    data_y_stale = [latency for _, latency in pareto_frontier_stale]

    # drop all tuples where the perplexity is larger than the maxmimum perplexity of RETRO
    pareto_frontier_stale = [t for t in pareto_frontier_stale if t[0] <= max_perplexity_RETRO]
    data_x_stale = [ppl for ppl, _ in pareto_frontier_stale]
    data_y_stale = [latency for _, latency in pareto_frontier_stale]

    # convert to seconds
    data_y_RETRO = np.array(data_y_RETRO) / 1000.0
    data_y_stale = np.array(data_y_stale) / 1000.0

    plot_RETRO = ax.plot(data_x_RETRO, data_y_RETRO, marker='X', markersize=markersize)
    plot_stale = ax.plot(data_x_stale, data_y_stale, marker='o', markersize=markersize)

    ax.legend([plot_RETRO[0], plot_stale[0]], ["RETRO", "PipeRAG"], loc="upper right", ncol=1, fontsize=legen_font, frameon=True)
    ax.tick_params(top=False, bottom=True, left=True, right=False, labelleft=True, labelsize=tick_font)
    ax.get_xaxis().set_visible(True)
    ax.set_xlabel('Perplexity', fontsize=label_font)
    ax.set_ylabel('Latency (s)', fontsize=label_font)

    # print baseline latency as a horizontal line
    # ax.axhline(y=baseline_latency, color='#101010', linestyle='--', label="baseline")
    min_ppl = np.min([np.min(data_x_RETRO), np.min(data_x_stale)])
    max_ppl = np.max([np.max(data_x_RETRO), np.max(data_x_stale)])
    ax.text(min_ppl, baseline_latency - 1, f"No retrieval: latency={baseline_latency:.2f} s, perplexity={baseline_ppl:.2f}", fontsize=tick_font, color='black', horizontalalignment='left', verticalalignment='top')
    
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)

    # set y limit
    # min_ppl = np.min([np.min(plot[0].get_ydata()) for plot in plots])
    min_latency = np.min([np.min(data_y_RETRO), np.min(data_y_stale)])
    max_latency = np.max([np.max(data_y_RETRO), np.max(data_y_stale)])
    ax.set_ylim([baseline_latency - 5, max_latency + 1])

    eval_set = setting_kv["eval_set"]
    if eval_set == "wikipedia_chunk9_1K":
        eval_set = "Wikipedia"
    elif eval_set == "realnews_chunk31_1K":
        eval_set = "RealNews"
    elif eval_set == "c4_chunk1023_1K":
        eval_set = "C4"
    ax.text(max_ppl, min_latency + 0.35 * (max_latency - min_latency),  f'Eval set: {eval_set}', fontsize=tick_font, ha='right', va='bottom')


    fig_name = 'ppl_pareto_eval_{}'.format(setting_kv["eval_set"])
    plt.savefig('./out_img/pareto/{}.png'.format(fig_name), transparent=False, dpi=200, bbox_inches="tight")
    # plt.show()

    # for each point in data_RETRO, find the point in data_stale which has less pereplexity, and compute the latency speedup
    data_RETRO = list(zip(data_x_RETRO, data_y_RETRO))
    data_stale = list(zip(data_x_stale, data_y_stale))
    speedup_list = []
    for ppl_RETRO, latency_RETRO in data_RETRO:
        # find the point in data_stale which has less pereplexity
        latency_stale = [latency for ppl, latency in data_stale if ppl < ppl_RETRO]
        assert len(latency_stale) > 0
        speedup = latency_RETRO / np.min(latency_stale)
        speedup_list.append(speedup)
        print("speedup: {:.2f} x".format(speedup))
        # update max_speedup
        global max_speedup
        max_speedup = np.max([max_speedup, speedup])

    # for each point in data_RETRO, find the point in data_stale which has lower latency, and with the minimum perplexity, show perplexity difference
    data_RETRO = list(zip(data_x_RETRO, data_y_RETRO))
    data_stale = list(zip(data_x_stale, data_y_stale))
    speedup_list = []
    for ppl_RETRO, latency_RETRO in data_RETRO:
        # find the point in data_stale which has lower latency, and with the minimum perplexity
        latency_stale = [latency for ppl, latency in data_stale if latency < latency_RETRO]
        assert len(latency_stale) > 0
        ppl_stale = [ppl for ppl, latency in data_stale if latency < latency_RETRO]
        assert len(ppl_stale) > 0
        ppl_stale = np.min(ppl_stale)
        print("ppl difference: {:.2f}\tRETRO: ppl={:.2f}\tPipeRAG: ppl={:.2f}".format(ppl_RETRO - ppl_stale, ppl_RETRO, ppl_stale))
        # update max_ppl_diff
        global max_ppl_diff
        max_ppl_diff = np.max([max_ppl_diff, ppl_RETRO - ppl_stale])
        


if __name__ == "__main__":

    df_path = "generation_join_perplexity_and_performance_df.pickle"
    df = pd.DataFrame(read_data_from_pickle(df_path))


    setting_kv_list = [
        {"eval_set": "wikipedia_chunk9_1K", "dbname": "c4_chunk_0_to_999", "index_type": "IVF16384,PQ64", "seq_len": 1024, "num_continuation_chunks": 1, "num_neighbours": 2, "server_inference": "p4d.24xlarge", "server_retrieval": "m5.metal.2.5GHz"},
        {"eval_set": "realnews_chunk31_1K", "dbname": "c4_chunk_0_to_999", "index_type": "IVF16384,PQ64", "seq_len": 1024, "num_continuation_chunks": 1, "num_neighbours": 2, "server_inference": "p4d.24xlarge", "server_retrieval": "m5.metal.2.5GHz"},
        {"eval_set": "c4_chunk1023_1K", "dbname": "c4_chunk_0_to_999", "index_type": "IVF16384,PQ64", "seq_len": 1024, "num_continuation_chunks": 1, "num_neighbours": 2, "server_inference": "p4d.24xlarge", "server_retrieval": "m5.metal.2.5GHz"},
    ]

    for setting_kv in setting_kv_list:
        # RETRO = no stale + fixed interval
        setting_kv_RETRO = {"staleness": False, "retrieval_interval": 64, "stale_steps": 0}
        setting_kv_RETRO = {**setting_kv_RETRO, **setting_kv}
        df_RETRO = select_rows(df, setting_kv_RETRO)

        # ours = stale + variable interval
        setting_kv_stale = {"staleness": True}
        setting_kv_stale = {**setting_kv_stale, **setting_kv}
        df_stale = select_rows(df, setting_kv_stale)

        plot(setting_kv, df_RETRO, df_stale)

    print("Across datasets, the maximum speedup is {:.2f}x".format(max_speedup))
    print("Across datasets, the maximum perplexity difference is {:.2f}".format(max_ppl_diff))