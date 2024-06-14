import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pickle
import pandas as pd

from matplotlib.ticker import FuncFormatter
from print_df import read_data_from_pickle, select_rows, select_rows_aligned_stale_and_interval

print("WARNING: please join the performance df with perplexity df if you have updated either!")

num_continuation_chunks = 1
num_neighbours = 2
max_len = 1024

df_path = "generation_perplexity_df.pickle"

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

    df = select_rows(df, setting_kv)

    nprobe_list = [1, 2, 4, 8, 16, 32, 64]
    retrieval_interval_list = [64, 32, 16, 8] 
    data_x = [str(nprobe) for nprobe in nprobe_list]

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    label_font = 16
    markersize = 8
    tick_font = 14

    plots = []
    legend_labels = []
    for retrieval_interval in retrieval_interval_list:
        data_y = []
        for nprobe in nprobe_list:
            out = select_rows(df, {"retrieval_interval": retrieval_interval, "nprobe": str(nprobe)})
            out = select_rows_aligned_stale_and_interval(out)
            assert out.index.size == 1
            # get the perplexity column
            data_y.append(out["perplexity"].values[0])
        plot = ax.plot(data_x, data_y, marker='o', markersize=markersize)
        plots.append(plot)
        legend_labels.append("retrieval_interval={}".format(retrieval_interval))
    ax.legend([plot[0] for plot in plots], legend_labels, loc=(0, 1.05), ncol=2, fontsize=label_font)
    ax.tick_params(top=False, bottom=True, left=True, right=False, labelleft=True, labelsize=tick_font)
    ax.get_xaxis().set_visible(True)
    ax.set_xlabel('nprobe (search space)', fontsize=label_font)
    ax.set_ylabel('Perplexity', fontsize=label_font)

    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)

    # plt.rcParams.update({'figure.autolayout': True})

    # draw line of base ppl, and annotate it with texts
    ax.axhline(y=base_ppl, color='r', linestyle='--', label="base ppl")
    ax.text(0.5, base_ppl + 0.2, f"No retrieval, PPL={base_ppl}", fontsize=14, color='r')

    # set y limit
    min_ppl = np.min([np.min(plot[0].get_ydata()) for plot in plots])
    ax.set_ylim([min_ppl - 0.5, base_ppl + 0.8])

    fig_name = 'ppl_nprobe_eval_{}_db_{}_staleness_{}'.format(setting_kv["eval_set"], setting_kv["dbname"], setting_kv["staleness"])
    plt.savefig('./out_img/ppl_nprobe/{}.png'.format(fig_name), transparent=False, dpi=200, bbox_inches="tight")
    # plt.show()


if __name__ == "__main__":

    # x axis: increase nprobe; different curves = different retrievals; 2 plots = with/without staleness
    setting_kv_list = [\
        {"eval_set": "wikipedia_chunk9_1K", "dbname": "c4_chunk_0_to_999", "index_type": "IVF16384,PQ64", "staleness": False, "seq_len": 1024,  "num_continuation_chunks": 1, "num_neighbours": 2},
        {"eval_set": "wikipedia_chunk9_1K", "dbname": "c4_chunk_0_to_999", "index_type": "IVF16384,PQ64", "staleness": True, "seq_len": 1024,  "num_continuation_chunks": 1, "num_neighbours": 2},
        {"eval_set": "realnews_chunk31_1K", "dbname": "c4_chunk_0_to_999", "index_type": "IVF16384,PQ64", "staleness": False, "seq_len": 1024,  "num_continuation_chunks": 1, "num_neighbours": 2},
        {"eval_set": "realnews_chunk31_1K", "dbname": "c4_chunk_0_to_999", "index_type": "IVF16384,PQ64", "staleness": True, "seq_len": 1024,  "num_continuation_chunks": 1, "num_neighbours": 2},
        {"eval_set": "c4_chunk1023_1K", "dbname": "c4_chunk_0_to_999", "index_type": "IVF16384,PQ64", "staleness": False, "seq_len": 1024,  "num_continuation_chunks": 1, "num_neighbours": 2},
        {"eval_set": "c4_chunk1023_1K", "dbname": "c4_chunk_0_to_999", "index_type": "IVF16384,PQ64", "staleness": True, "seq_len": 1024,  "num_continuation_chunks": 1, "num_neighbours": 2},
    ]
    df = pd.DataFrame(read_data_from_pickle(df_path))
    for setting_kv in setting_kv_list:
        plot(df, setting_kv)


