import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pickle
import pandas as pd

from matplotlib.ticker import FuncFormatter
from print_df import read_data_from_pickle, select_rows, select_rows_aligned_stale_and_interval

# print(plt.style.available)
# plt.style.use('seaborn-v0_8')
# plt.style.use('seaborn-v0_8-pastel')
plt.style.use('seaborn-v0_8-deep')
# plt.style.use('seaborn-v0_8-whitegrid')

print("WARNING: please join the performance df with perplexity df if you have updated either!")

num_continuation_chunks = 1
num_neighbours = 2
max_len = 1024

df_path = "generation_perplexity_df.pickle"

def plot(df, setting_kv):

    setting_kv_RETRO = setting_kv.copy()
    setting_kv_RETRO["staleness"] = False
    setting_kv_RETRO["retrieval_interval"] = 64
    df_RETRO = select_rows(df, setting_kv_RETRO)

    setting_kv_PipeRAG = setting_kv.copy()
    setting_kv_PipeRAG["staleness"] = True
    df_PipeRAG = select_rows(df, setting_kv_PipeRAG)

    nprobe_list = [1, 2, 4, 8, 16, 32, 64]
    retrieval_interval_list = [64, 32, 16, 8] 
    data_x = [str(nprobe) for nprobe in nprobe_list]

    fig, ax = plt.subplots(1, 1, figsize=(8, 3))

    label_font = 18
    markersize = 8
    tick_font = 16
    legen_font = 14

    plots = []
    legend_labels = []

    # PipeRAG
    for retrieval_interval in retrieval_interval_list:
        data_y = []
        for nprobe in nprobe_list:
            out = select_rows(df_PipeRAG, {"retrieval_interval": retrieval_interval, "nprobe": str(nprobe)})
            out = select_rows_aligned_stale_and_interval(out)
            assert out.index.size == 1
            # get the perplexity column
            data_y.append(out["perplexity"].values[0])
        plot = ax.plot(data_x, data_y, marker='o', markersize=markersize)
        plots.append(plot)
        legend_labels.append("interval={}".format(retrieval_interval))
    
    # RETRO
    data_y = []
    for nprobe in nprobe_list:
        out = select_rows(df_RETRO, {"retrieval_interval": 64, "nprobe": str(nprobe)})
        out = select_rows_aligned_stale_and_interval(out)
        assert out.index.size == 1
        # get the perplexity column
        data_y.append(out["perplexity"].values[0])
    plot = ax.plot(data_x, data_y, marker='X', markersize=markersize, color='#CCCCCC')
    plots.append(plot)
    legend_labels.append("RETRO".format(retrieval_interval))

    ax.legend([plot[0] for plot in plots], legend_labels, loc=(-0.15, 1.05), ncol=3, fontsize=label_font)
    ax.tick_params(top=False, bottom=True, left=True, right=False, labelleft=True, labelsize=tick_font)
    ax.get_xaxis().set_visible(True)
    ax.set_xlabel('nprobe (search space)', fontsize=label_font)
    ax.set_ylabel('Perplexity', fontsize=label_font)


    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # plt.rcParams.update({'figure.autolayout': True})

    # set y limit
    min_ppl = np.min([np.min(plot[0].get_ydata()) for plot in plots])
    max_ppl = np.max([np.max(plot[0].get_ydata()) for plot in plots])
    # ax.set_ylim([min_ppl - 0.5, max_ppl + 0.5])

    eval_set = setting_kv["eval_set"]
    if eval_set == "wikipedia_chunk9_1K":
        eval_set = "Wikipedia"
    elif eval_set == "realnews_chunk31_1K":
        eval_set = "RealNews"
    elif eval_set == "c4_chunk1023_1K":
        eval_set = "C4"
    ax.text(6, max_ppl - 0.1, f'Eval set: {eval_set}', fontsize=tick_font, ha='right', va='top')

    fig_name = 'ppl_eval_{}_db_{}'.format(setting_kv["eval_set"], setting_kv["dbname"])
    plt.savefig('./out_img/ppl_nprobe_interval_paper/{}.png'.format(fig_name), transparent=False, dpi=200, bbox_inches="tight")
    # plt.show()


if __name__ == "__main__":

    # x axis: increase nprobe; different curves = different retrievals; 2 plots = with/without staleness
    setting_kv_list = [\
        {"eval_set": "wikipedia_chunk9_1K", "dbname": "c4_chunk_0_to_999", "index_type": "IVF16384,PQ64", "seq_len": 1024,  "num_continuation_chunks": 1, "num_neighbours": 2},
        {"eval_set": "realnews_chunk31_1K", "dbname": "c4_chunk_0_to_999", "index_type": "IVF16384,PQ64", "seq_len": 1024,  "num_continuation_chunks": 1, "num_neighbours": 2},
        {"eval_set": "c4_chunk1023_1K", "dbname": "c4_chunk_0_to_999", "index_type": "IVF16384,PQ64", "seq_len": 1024,  "num_continuation_chunks": 1, "num_neighbours": 2},
    ]
    df = pd.DataFrame(read_data_from_pickle(df_path))
    for setting_kv in setting_kv_list:
        plot(df, setting_kv)


