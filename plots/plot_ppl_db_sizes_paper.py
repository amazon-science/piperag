import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pickle
import pandas as pd

from matplotlib.ticker import FuncFormatter
from print_df import read_data_from_pickle, select_rows, select_rows_aligned_stale_and_interval

# print(plt.style.available)
plt.style.use('seaborn-v0_8-pastel')
# plt.style.use('seaborn-v0_8-whitegrid')

print("WARNING: please join the performance df with perplexity df if you have updated either!")

num_continuation_chunks = 1
num_neighbours = 2
max_len = 1024

df_path = "generation_perplexity_df.pickle"

def plot_db_size(df, setting_kv, dbnames=['c4_chunk_0_to_99', 'c4_chunk_0_to_349', 'c4_chunk_0_to_999']):
    """
    different curves = different interval; fix nprobe
    """
    
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
    
    setting_RETRO_one_retrieval = setting_kv.copy()
    setting_RETRO_one_retrieval["retrieval_interval"] = 1024
    setting_RETRO_one_retrieval["nprobe"] = str(64)
    setting_RETRO_one_retrieval["staleness"] = False
    df_RETRO_one_retrieval = select_rows(df, setting_RETRO_one_retrieval)

    setting_RETRO = setting_kv.copy()
    setting_RETRO["retrieval_interval"] = 64
    setting_RETRO["nprobe"] = str(64)
    setting_RETRO["staleness"] = False
    df_RETRO = select_rows(df, setting_RETRO)

    setting_PipeRAG = setting_kv.copy()
    setting_PipeRAG["retrieval_interval"] = 8
    setting_PipeRAG["nprobe"] = str(64)
    setting_PipeRAG["staleness"] = True
    df_PipeRAG = select_rows(df, setting_PipeRAG)

    data_x = ["Small DB (10%)", "Medium DB (35%)", "Large DB (100%)"]

    fig, ax = plt.subplots(1, 1, figsize=(8, 3))

    label_font = 18
    markersize = 8
    tick_font = 16
    legen_font = 14

    legend_labels = ["RAG w/ one retrieval", "RETRO", "PipeRAG"]
    
    data_y_RETRO_one_retrieval = []
    data_y_RETRO = []
    data_y_PipeRAG = []
    for dbname in dbnames:
        out_RETRO_one_retrieval = select_rows(df_RETRO_one_retrieval, {"dbname": dbname})
        out_RETRO_one_retrieval = select_rows_aligned_stale_and_interval(out_RETRO_one_retrieval)
        assert out_RETRO_one_retrieval.index.size == 1
        data_y_RETRO_one_retrieval.append(out_RETRO_one_retrieval["perplexity"].values[0])

        out_RETRO = select_rows(df_RETRO, {"dbname": dbname})
        out_RETRO = select_rows_aligned_stale_and_interval(out_RETRO)
        assert out_RETRO.index.size == 1
        data_y_RETRO.append(out_RETRO["perplexity"].values[0])

        out_PipeRAG = select_rows(df_PipeRAG, {"dbname": dbname})
        out_PipeRAG = select_rows_aligned_stale_and_interval(out_PipeRAG)
        assert out_PipeRAG.index.size == 1
        data_y_PipeRAG.append(out_PipeRAG["perplexity"].values[0])

    x = np.arange(len(data_x))  # the label locations
    width = 0.2  # the width of the bars

    rects1  = ax.bar(x - width, data_y_RETRO_one_retrieval, width)#, label='Men')
    rects2 = ax.bar(x , data_y_RETRO, width)#, label='Women')
    rects3 = ax.bar(x + width, data_y_PipeRAG, width)#, label='Women')
    rects = [rects1, rects2, rects3]

    ax.legend([plot[0] for plot in rects], legend_labels, loc=(0., 1.05), ncol=3, fontsize=legen_font)
    ax.tick_params(top=False, bottom=True, left=True, right=False, labelleft=True, labelsize=tick_font)
    ax.get_xaxis().set_visible(True)
    ax.set_ylabel('Perplexity', fontsize=label_font)
    ax.set_xticks((0, 1, 2))
    ax.set_xticklabels(data_x, fontsize=tick_font)

    # mark the y value of each bar
    for rect in rects:
        for plot in rect:
            height = plot.get_height()
            ax.annotate('{:.2f}'.format(height),
                        xy=(plot.get_x() + plot.get_width() / 2, height - 0.2),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        fontsize=tick_font,
                        ha='center', va='top', rotation=90)
    
    eval_set = setting_kv["eval_set"]
    if eval_set == "wikipedia_chunk9_1K":
        eval_set = "Wikipedia"
    elif eval_set == "realnews_chunk31_1K":
        eval_set = "RealNews"
    elif eval_set == "c4_chunk1023_1K":
        eval_set = "C4"
    ax.text(2.0 + width, base_ppl + 0.2, f'Eval set: {eval_set}', fontsize=tick_font, ha='right', va='bottom')

    # draw line of base ppl, and annotate it with texts
    ax.axhline(y=base_ppl, color='r', linestyle='--', label="base ppl")
    # right align the text
    ax.text(0. - width, base_ppl + 0.2, "No retrieval: PPL={:.2f}".format(base_ppl), fontsize=tick_font, color='r', ha='left', va='bottom')

    # set y limit
    min_ppl = np.min([np.min(plot[-1].get_height()) for plot in rects])
    max_ppl = base_ppl
    delta = max_ppl - min_ppl
    ax.set_ylim([min_ppl - 0.6 * delta, base_ppl + 0.3 * delta])

    fig_name = 'paper_ppl_db_size_{}'.format(setting_kv["eval_set"])
    plt.savefig('./out_img/ppl_db_size/{}.png'.format(fig_name), transparent=False, dpi=200, bbox_inches="tight")
    # plt.show()



if __name__ == "__main__":

    # x axis: increase nprobe; different curves = different retrievals; 2 plots = with/without staleness
    setting_kv_list = [\
        {"eval_set": "wikipedia_chunk9_1K", "index_type": "IVF16384,PQ64", "staleness": False, "num_continuation_chunks": 1, "num_neighbours": 2},
        {"eval_set": "realnews_chunk31_1K", "index_type": "IVF16384,PQ64", "staleness": False, "num_continuation_chunks": 1, "num_neighbours": 2},
        {"eval_set": "c4_chunk1023_1K", "index_type": "IVF16384,PQ64", "staleness": True, "num_continuation_chunks": 1, "num_neighbours": 2},
    ]
    df = pd.DataFrame(read_data_from_pickle(df_path))
    for setting_kv in setting_kv_list:
        plot_db_size(df, setting_kv)


