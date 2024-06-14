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

    df = select_rows(df, setting_kv)

    nprobe_list = [1, 2, 4, 8, 16, 32, 64]
    retrieval_interval_list = [64, 32, 16, 8] 
    data_x = [str(retrieval_interval) for retrieval_interval in retrieval_interval_list]

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    label_font = 16
    markersize = 8
    tick_font = 14

    plots = []
    legend_labels = []
    for nprobe in nprobe_list:
        data_y = []
        for retrieval_interval in retrieval_interval_list:
            out = select_rows(df, {"retrieval_interval": retrieval_interval, "nprobe": str(nprobe)})
            out = select_rows_aligned_stale_and_interval(out)
            assert out.index.size == 1
            # get the perplexity column
            data_y.append(out["perplexity"].values[0])
        plot = ax.plot(data_x, data_y, marker='o', markersize=markersize)
        plots.append(plot)
        legend_labels.append("nprobe={}".format(nprobe))
    ax.legend([plot[0] for plot in plots], legend_labels, loc=(0, 1.05), ncol=3, fontsize=label_font)
    ax.tick_params(top=False, bottom=True, left=True, right=False, labelleft=True, labelsize=tick_font)
    ax.get_xaxis().set_visible(True)
    ax.set_xlabel('Retrieval interval', fontsize=label_font)
    ax.set_ylabel('Perplexity', fontsize=label_font)

    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)

    # plt.rcParams.update({'figure.autolayout': True})

    fig_name = 'ppl_retrieval_interval_eval_{}_db_{}_staleness_{}'.format(setting_kv["eval_set"], setting_kv["dbname"], setting_kv["staleness"])
    plt.savefig('./out_img/ppl_interval/{}.png'.format(fig_name), transparent=False, dpi=200, bbox_inches="tight")
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


