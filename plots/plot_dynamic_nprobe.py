import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pickle
import pandas as pd

from matplotlib.ticker import FuncFormatter
from print_df import read_data_from_pickle, select_rows

# plt.style.use('seaborn-v0_8-pastel')
plt.style.use('seaborn-v0_8-deep')
# plt.style.use('seaborn-v0_8-whitegrid')

print("WARNING: please join the performance df with perplexity df if you have updated either!")

num_continuation_chunks = 1
num_neighbours = 2
max_len = 1024

df_path = "generation_join_perplexity_and_performance_df.pickle"

def plot(df, setting_kv):

    setting_baseline = setting_kv.copy()
    setting_baseline["dbname"] = None
    setting_baseline["index_type"] = None
    setting_baseline["seq_len"] = max_len
    setting_baseline["nprobe"] = None
    setting_baseline["retrieval_interval"] = None
    setting_baseline["staleness"] = None
    setting_baseline["stale_steps"] = 0
    setting_baseline["num_continuation_chunks"] = None
    setting_baseline["num_neighbours"] = None
    base_ppl = select_rows(df, setting_baseline)["perplexity"].values[0]
    baseline_latency = np.median(select_rows(df, setting_baseline)["sequence_latency_total_ms"].values[0]) / 1000.0

    # RETRO
    setting_kv_RETRO = setting_kv.copy()
    setting_kv_RETRO["staleness"] = False
    setting_kv_RETRO["retrieval_interval"] = 64
    df_RETRO = select_rows(df, setting_kv_RETRO)

    df = select_rows(df, setting_kv)

    nprobe_list = [1, 2, 4, 8, 16, 32, 64]
    retrieval_interval_list = [64, 32, 16, 8] 

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    label_font = 18
    markersize = 8
    tick_font = 16
    legend_font = 16

    data_x_fixed_nprobe = []
    data_y_fixed_nprobe = []
    perplexity_latency_fixed_nprobe = [] # each element is a tuple (perplexity, latency)

    data_x_dynamic_nprobe = []
    data_y_dynamic_nprobe = []
    pereplexity_latency_dynamic_nprobe = [] # each element is a tuple (perplexity, latency)

    # x: perplexity; y: latency
    for pid, retrieval_interval in enumerate(retrieval_interval_list):
        for nprobe in nprobe_list:
            out = select_rows(df, {"retrieval_interval": retrieval_interval, "stale_steps": retrieval_interval, "nprobe": str(nprobe)})
            assert out.index.size == 1
            # get the perplexity column
            y = np.median(out["sequence_latency_total_ms"].values[0]) / 1000.0
            if y < 20:
                data_x_fixed_nprobe.append(out["perplexity"].values[0])
                data_y_fixed_nprobe.append(y)
                perplexity_latency_fixed_nprobe.append((out["perplexity"].values[0], y))

        # get dynamic nprobe plot
        nprobe = 'dynamic'
        out = select_rows(df, {"retrieval_interval": retrieval_interval, "stale_steps": retrieval_interval, "nprobe": str(nprobe)})
        assert out.index.size == 1
        # get the perplexity column
        data_x_dynamic_nprobe.append(out["perplexity"].values[0])
        data_y_dynamic_nprobe.append(np.median(out["sequence_latency_total_ms"].values[0]) / 1000.0)
        pereplexity_latency_dynamic_nprobe.append((out["perplexity"].values[0], np.median(out["sequence_latency_total_ms"].values[0]) / 1000.0))

    # get the tuple with minimum perplexity of dynamic nprobe
    min_ppl_dynamic_nprobe_ppl = np.inf
    min_ppl_dynamic_nprobe_ppl_latency = np.inf
    for ppl, latency in pereplexity_latency_dynamic_nprobe:
        if ppl < min_ppl_dynamic_nprobe_ppl:
            min_ppl_dynamic_nprobe_ppl = ppl
            min_ppl_dynamic_nprobe_ppl_latency = latency
    
    # from the fixed nprobe data, get the tuple with less perplexity than the minimum perplexity of dynamic nprobe, get the one with minimum latency
    min_ppl_fixed_nprobe = np.inf
    min_ppl_fixed_nprobe_latency = np.inf
    for ppl, latency in perplexity_latency_fixed_nprobe:
        if ppl < min_ppl_dynamic_nprobe_ppl and latency < min_ppl_fixed_nprobe_latency:
            min_ppl_fixed_nprobe = ppl
            min_ppl_fixed_nprobe_latency = latency
    
    print("dataset: {}".format(setting_kv["eval_set"]))
    print("Baseline perplexity: {:.2f}, latency: {:.2f}".format(base_ppl, baseline_latency))
    print("Dynamic nprobe perplexity: {:.2f}, latency: {:.2f}, latency_diff: {:.2f}, perplexity_diff: {:.2f}".format(min_ppl_dynamic_nprobe_ppl, min_ppl_dynamic_nprobe_ppl_latency, (min_ppl_dynamic_nprobe_ppl_latency - baseline_latency), (min_ppl_dynamic_nprobe_ppl - base_ppl)))
    print("Fixed nprobe perplexity: {:.2f}, latency: {:.2f}, latency_diff: {:.2f}, perplexity_diff: {:.2f}".format(min_ppl_fixed_nprobe, min_ppl_fixed_nprobe_latency, (min_ppl_fixed_nprobe_latency - baseline_latency), (min_ppl_fixed_nprobe - base_ppl)))
    
    # find RETRO points whose perplexity > dynamic nprobe, and latency > dynamic nprobe
    for _, row in df_RETRO.iterrows():
        latency = np.median(row["sequence_latency_total_ms"])
        if row["perplexity"] > min_ppl_dynamic_nprobe_ppl and latency > min_ppl_dynamic_nprobe_ppl_latency:
            print("RETRO: perplexity:{:.2f}, latency:{:.2f}, latency_diff: {:.2f}, perplexity_diff: {:.2f}".format(
                row["perplexity"], np.median(latency) / 1000.0, (np.median(latency) / 1000.0 - baseline_latency), (row["perplexity"] - base_ppl)))

    # draw scatter plots for fixed and dynamic nprobe, separately
    plots = []
    legend_labels = []
    area = 100
    plots.append(ax.scatter(data_x_fixed_nprobe, data_y_fixed_nprobe, s=area, alpha=0.7))
    legend_labels.append("PipeRAG (fixed nprobe)")
    plots.append(ax.scatter(data_x_dynamic_nprobe, data_y_dynamic_nprobe, s=area, alpha=0.7))
    legend_labels.append("PipeRAG (performance model)")

    # set plot configs
    
    ax.legend([plot for plot in plots], legend_labels, loc="upper right", ncol=1, fontsize=legend_font)
    ax.tick_params(top=False, bottom=True, left=True, right=False, labelleft=True, labelsize=tick_font)
    ax.get_xaxis().set_visible(True)
    ax.set_xlabel('Perplexity', fontsize=label_font)
    ax.set_ylabel('Latency (s)', fontsize=label_font)

    max_ppl = np.max(data_x_fixed_nprobe)
    min_ppl = np.min(data_x_fixed_nprobe)
    max_latency = np.max(data_y_fixed_nprobe)
    min_latency = np.min(data_y_fixed_nprobe)

    eval_set = setting_kv["eval_set"]
    if eval_set == "wikipedia_chunk9_1K":
        eval_set = "Wikipedia"
    elif eval_set == "realnews_chunk31_1K":
        eval_set = "RealNews"
    elif eval_set == "c4_chunk1023_1K":
        eval_set = "C4"
    ax.text(max_ppl - 0.1, min_latency + 0.6 * (max_latency - min_latency), f'Eval set: {eval_set}', fontsize=legend_font, ha='right', va='bottom')

    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)

    # plt.rcParams.update({'figure.autolayout': True})

    # # draw line of base ppl, and annotate it with texts
    # ax.axhline(y=base_ppl, color='r', linestyle='--', label="base ppl")
    # ax.text(0.5, base_ppl + 0.2, f"No retrieval, PPL={base_ppl}", fontsize=14, color='r')

    # # set y limit
    # min_ppl = np.min([np.min(plot[0].get_ydata()) for plot in plots])
    # ax.set_ylim([min_ppl - 0.5, base_ppl + 0.8])

    fig_name = 'ppl_dynamic_nprobe_eval_{}_db_{}'.format(setting_kv["eval_set"], setting_kv["dbname"])
    plt.savefig('./out_img/dynamic_nprobe/{}.png'.format(fig_name), transparent=False, dpi=200, bbox_inches="tight")
    # plt.show()


if __name__ == "__main__":

    # x axis: increase nprobe; different curves = different retrievals; 2 plots = with/without staleness
    setting_kv_list = [\
        {"eval_set": "wikipedia_chunk9_1K", "dbname": "c4_chunk_0_to_999", "index_type": "IVF16384,PQ64", "staleness": True, "seq_len": 1024,  "num_continuation_chunks": 1, "num_neighbours": 2, "server_inference": "p4d.24xlarge", "server_retrieval": "m5.metal.2.5GHz"},
        {"eval_set": "realnews_chunk31_1K", "dbname": "c4_chunk_0_to_999", "index_type": "IVF16384,PQ64", "staleness": True, "seq_len": 1024,  "num_continuation_chunks": 1, "num_neighbours": 2, "server_inference": "p4d.24xlarge", "server_retrieval": "m5.metal.2.5GHz"},
        {"eval_set": "c4_chunk1023_1K", "dbname": "c4_chunk_0_to_999", "index_type": "IVF16384,PQ64", "staleness": True, "seq_len": 1024,  "num_continuation_chunks": 1, "num_neighbours": 2, "server_inference": "p4d.24xlarge", "server_retrieval": "m5.metal.2.5GHz"},
    ]
    df = pd.DataFrame(read_data_from_pickle(df_path))
    for setting_kv in setting_kv_list:
        plot(df, setting_kv)


