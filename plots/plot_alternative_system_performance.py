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

import sys
sys.path.append('../inference')
from performance_model import PerformanceModel

print("WARNING: please join the performance df with perplexity df if you have updated either!")

num_continuation_chunks = 1
num_neighbours = 2
max_len = 1024

df_path = "generation_join_perplexity_and_performance_df.pickle"
# df_path = "generation_perplexity_df.pickle"
perf_model = PerformanceModel(
    generation_model_path="$WORKSPACE/inference/performance/p4d.24xlarge_performance_generation_len_1024_k_2.pickle",
    retrieval_model_path="$WORKSPACE/inference/performance/m5.metal.2.5GHz_performance_search_c4_chunk_0_to_999_IVF16384,PQ64.pickle",
    sbert_model_path="$WORKSPACE/inference/performance/m5.metal.2.5GHz_performance_SBERT.pickle",
    extra_overhead_ms=10,
    search_latency_budget_discount=1.0,
    min_nprobe=None,
    max_nprobe=None
)

performance_diff_percentage_all = []

def predict_performance(df, perf_model, staleness, speedup_compute=1.0, speedup_retrieval=1.0):
    """
    Return perplexity-performance pairs for each row in df
    """
    perplexity_performance_list = []
    estimated_performance = None
    # iterate over all rows
    for index, row in df.iterrows():
        num_retrieval = int(max_len / row["retrieval_interval"])
        if row["nprobe"] == "dynamic":
            continue
        else:
            nprobe = int(row["nprobe"])
        if staleness:
            estimated_performance = 0
            # calculate the latency of the retrieval, if retrieval latency < generation latency, then add nothing
            retrieval_latency = (perf_model.predict_nprobe_latency(nprobe) + perf_model.sbert_model["average_latency_ms"]) / speedup_retrieval
            for i in range(0, max_len, row["retrieval_interval"]):
                generation_latency = np.sum(perf_model.generation_model["average_decoder_latency_ms"][i:i+row["retrieval_interval"]]) / speedup_compute
                if i == 0: # first step always invoke
                    estimated_performance += retrieval_latency + generation_latency
                else:
                    if retrieval_latency >= generation_latency:
                        estimated_performance += retrieval_latency
                    else:
                        estimated_performance += generation_latency
        else:
            estimated_performance = (num_retrieval * perf_model.predict_nprobe_latency(nprobe)) / speedup_retrieval + \
                (num_retrieval * perf_model.sbert_model["average_latency_ms"]) / speedup_compute + \
                np.sum(perf_model.generation_model["average_decoder_latency_ms"][:max_len]) / speedup_compute
        perplexity_performance_list.append((row["perplexity"], estimated_performance))

    # sort by perplexity
    perplexity_performance_list = sorted(perplexity_performance_list, key=lambda x: x[0])
    
    return perplexity_performance_list

def verify_performance_model(df, perf_model, staleness):
    """
    Return perplexity-performance pairs for each row in df
    """
    speedup_compute = 1.0 
    speedup_retrieval=1.0
    max_diff_perc = 0.0
    min_diff_perc = 0.0
    diff_percentage_list = []
    
    estimated_performance = None
    # iterate over all rows
    for index, row in df.iterrows():
        num_retrieval = int(max_len / row["retrieval_interval"])
        if row["nprobe"] == "dynamic":
            continue
        else:
            nprobe = int(row["nprobe"])
        if staleness:
            estimated_performance = 0
            # calculate the latency of the retrieval, if retrieval latency < generation latency, then add nothing
            retrieval_latency = (perf_model.predict_nprobe_latency(nprobe) + perf_model.sbert_model["average_latency_ms"]) / speedup_retrieval
            for i in range(0, max_len, row["retrieval_interval"]):
                generation_latency = np.sum(perf_model.generation_model["average_decoder_latency_ms"][i:i+row["retrieval_interval"]]) / speedup_compute
                if i == 0: # first step always invoke
                    estimated_performance += retrieval_latency + generation_latency
                else:
                    if retrieval_latency >= generation_latency:
                        estimated_performance += retrieval_latency
                    else:
                        estimated_performance += generation_latency
        else:
            estimated_performance = (num_retrieval * perf_model.predict_nprobe_latency(nprobe)) / speedup_retrieval + \
                (num_retrieval * perf_model.sbert_model["average_latency_ms"]) / speedup_compute + \
                np.sum(perf_model.generation_model["average_decoder_latency_ms"][:max_len]) / speedup_compute
        real_performance = np.median(row["sequence_latency_total_ms"]) 
        diff_percentage = (estimated_performance - real_performance) / real_performance * 100.0
        diff_percentage_list.append(diff_percentage)
        if diff_percentage > max_diff_perc:
            max_diff_perc = diff_percentage
        if diff_percentage < min_diff_perc:
            min_diff_perc = diff_percentage
        # if abs(diff_percentage) > 10:
        #     print("Warning: large difference between estimated and real performance: {:.2f}%".format(diff_percentage))
        #     print("nprobe: {}, retrieval_interval: {}, staleness: {}".format(nprobe, row["retrieval_interval"], staleness))
        #     print("estimated performance: {:.2f}, real performance: {:.2f}".format(estimated_performance, real_performance))

    diff_percentage_list = np.abs(diff_percentage_list)
    print("max_diff_perc: {:.2f}, min_diff_perc: {:.2f}".format(max_diff_perc, min_diff_perc))
    print("average diff percentage: {:.2f}, median diff percentage: {:.2f}".format(np.mean(diff_percentage_list), np.median(diff_percentage_list)))
    
    return diff_percentage_list

def get_pareto(perplexity_performance_list):
    """
    Return pareto frontier
    """
    pareto_frontier = []
    min_performance = np.inf
    for perplexity, performance in perplexity_performance_list:
        if performance < min_performance:
            pareto_frontier.append((perplexity, performance))
            min_performance = performance
    return pareto_frontier

def plot(df, setting_kv):

    setting_kv_RETRO = setting_kv.copy()
    setting_kv_RETRO["staleness"] = False
    setting_kv_RETRO["retrieval_interval"] = 64

    df_RETRO = select_rows(df, setting_kv_RETRO)

    setting_stale = setting_kv.copy()
    setting_stale["staleness"] = True

    df_stale = select_rows(df, setting_stale)

    # X: perplexity; Y: latency
    nprobe_list = [1, 2, 4, 8, 16, 32, 64]
    retrieval_interval_list = [64, 32, 16, 8] 

    fig, ax_lists = plt.subplots(2, 2, figsize=(10, 8))

    label_font = 18
    markersize = 8
    tick_font = 16
    legend_font = 16

    # verify_performance_model(df_stale, perf_model, staleness=True)
    # verify_performance_model(df_RETRO, perf_model, staleness=False)


    for icol, speedup in enumerate([4.0, 16.0]):

        for irow, (speedup_compute, speedup_retrieval) in enumerate([(speedup, 1.0), (1.0, speedup)]):

            plots = []
            legend_labels = []

            # RETRO
            perplexity_performance_list_RETRO = predict_performance(df_RETRO, perf_model, staleness=False, speedup_compute=speedup_compute, speedup_retrieval=speedup_retrieval)
            pareto_frontier_RETRO = get_pareto(perplexity_performance_list_RETRO)
            data_x_RETRO = [ppl for ppl, _ in pareto_frontier_RETRO]
            data_y_RETRO = [latency for _, latency in pareto_frontier_RETRO]
            max_latency_RETRO = np.max(data_y_RETRO)

            # stale
            perplexity_performance_list_stale = predict_performance(df_stale, perf_model, staleness=True, speedup_compute=speedup_compute, speedup_retrieval=speedup_retrieval)
            pareto_frontier_stale = get_pareto(perplexity_performance_list_stale)
            # drop all tuples where the latency is larger than the max latency of RETRO * 1.2
            pareto_frontier_stale = [t for t in pareto_frontier_stale if t[1] <= max_latency_RETRO * 1.0]
            data_x_stale = [ppl for ppl, _ in pareto_frontier_stale]
            data_y_stale = [latency  for _, latency in pareto_frontier_stale]
            
            data_y_RETRO = np.array(data_y_RETRO) / 1000.0
            data_y_stale = np.array(data_y_stale) / 1000.0
            plot = ax_lists[irow][icol].plot(data_x_RETRO, data_y_RETRO, marker='X', markersize=markersize)
            plots.append(plot)
            legend_labels.append("RETRO")
            plot = ax_lists[irow][icol].plot(data_x_stale, data_y_stale, marker='o', markersize=markersize)
            plots.append(plot)
            legend_labels.append("PipeRAG")
            
            ax_lists[irow][icol].legend([plot[0] for plot in plots], legend_labels, loc="upper right", ncol=1, fontsize=legend_font)

            # text at top right
            if speedup_compute > 1:
                ax_lists[irow][icol].text(0.98, 0.5, "Inference with\n{}x speedup".format(int(speedup_compute)), horizontalalignment='right', verticalalignment='top', transform=ax_lists[irow][icol].transAxes, fontsize=legend_font)
            else:
                ax_lists[irow][icol].text(0.98, 0.5, "Retrieval with\n{}x speedup".format(int(speedup_retrieval)), horizontalalignment='right', verticalalignment='top', transform=ax_lists[irow][icol].transAxes, fontsize=legend_font)

            max_latency = np.max(data_y_RETRO)
            min_latency = np.min(data_y_RETRO)
            max_ppl = np.max(data_x_RETRO + data_x_stale)
            min_ppl = np.min(data_x_RETRO + data_x_stale)

            eval_set = setting_kv["eval_set"]
            if eval_set == "wikipedia_chunk9_1K":
                eval_set = "Wikipedia"
            elif eval_set == "realnews_chunk31_1K":
                eval_set = "RealNews"
            elif eval_set == "c4_chunk1023_1K":
                eval_set = "C4"
            # ax_lists[irow][icol].text(0.95, 0.7, f'Eval set: {eval_set}', fontsize=legend_font, ha='right', va='bottom')
            ax_lists[irow][icol].text(max_ppl, min_latency + 0.6 * (max_latency - min_latency), f'Eval set: {eval_set}', fontsize=legend_font, ha='right', va='bottom')

            # for each point in data_RETRO, find the point in data_stale which has less pereplexity, and compute the latency speedup
            data_RETRO = list(zip(data_x_RETRO, data_y_RETRO))
            data_stale = list(zip(data_x_stale, data_y_stale))
            speedup_list = []
            ppl_reduction_list = []
            for ppl_RETRO, latency_RETRO in data_RETRO:
                # find the point in data_stale which has less pereplexity
                latency_stale = [latency for ppl, latency in data_stale if ppl < ppl_RETRO]
                assert len(latency_stale) > 0
                speedup = latency_RETRO / np.min(latency_stale)
                speedup_list.append(speedup)
            # for each point in data_RETRO, find the point in data_stale which has lower latency, and with the minimum perplexity, show perplexity difference
            for ppl_RETRO, latency_RETRO in data_RETRO:
                # find the point in data_stale which has lower latency
                latency_stale = [latency for ppl, latency in data_stale if latency < latency_RETRO]
                assert len(latency_stale) > 0
                ppl_stale = [ppl for ppl, latency in data_stale if latency < latency_RETRO]
                assert len(ppl_stale) > 0
                ppl_stale = np.min(ppl_stale)
                ppl_diff = ppl_RETRO - ppl_stale
                ppl_reduction_list.append(ppl_diff)
            speedup_list = np.array(speedup_list)
            print("Speedup compute: {}\t Speedup retrieval: {}".format(speedup_compute, speedup_retrieval))
            print("maximum speedup: {:.2f}".format(np.max(speedup_list)))
            print("maximum ppl reduction: {:.2f}".format(np.max(ppl_reduction_list)))

    # plot settings
    for irow in range(2):
        for icol in range(2):
            ax_lists[irow][icol].tick_params(top=False, bottom=True, left=True, right=False, labelleft=True, labelsize=tick_font)
            ax_lists[irow][icol].get_xaxis().set_visible(True)
            ax_lists[irow][icol].set_xlabel('Perplexity', fontsize=label_font)
            ax_lists[irow][icol].set_ylabel('Latency (s)', fontsize=label_font)
            ax_lists[irow][icol].yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
            # ax_lists[irow][icol].set_ylim([0, 25])
            # ax_lists[irow][icol].set_xlim([18.5, 20.5])



    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)

    # plt.rcParams.update({'figure.autolayout': True})

    fig.tight_layout()
    fig_name = 'ppl_alternative_system_performance_eval_{}_db_{}'.format(setting_kv["eval_set"], setting_kv["dbname"])
    plt.savefig('./out_img/alternative_system_performance/{}.png'.format(fig_name), transparent=False, dpi=200, bbox_inches="tight")
    # plt.show()


if __name__ == "__main__":

    # x axis: increase nprobe; different curves = different retrievals; 2 plots = with/without staleness
    setting_kv_list = [\
        {"eval_set": "wikipedia_chunk9_1K", "dbname": "c4_chunk_0_to_999", "index_type": "IVF16384,PQ64", "seq_len": 1024,  "num_continuation_chunks": 1, "num_neighbours": 2, "server_retrieval": "m5.metal.2.5GHz"},
        {"eval_set": "realnews_chunk31_1K", "dbname": "c4_chunk_0_to_999", "index_type": "IVF16384,PQ64", "seq_len": 1024,  "num_continuation_chunks": 1, "num_neighbours": 2, "server_retrieval": "m5.metal.2.5GHz"},
        {"eval_set": "c4_chunk1023_1K", "dbname": "c4_chunk_0_to_999", "index_type": "IVF16384,PQ64", "seq_len": 1024,  "num_continuation_chunks": 1, "num_neighbours": 2, "server_retrieval": "m5.metal.2.5GHz"},
    ]
    df = pd.DataFrame(read_data_from_pickle(df_path))
    for setting_kv in setting_kv_list:
        plot(df, setting_kv)

    for setting_kv in [setting_kv_list[0]]: # all three datasets share the same performance
        setting_stale = setting_kv.copy()
        setting_stale["staleness"] = True
        df_stale = select_rows(df, setting_stale)

        setting_kv_RETRO = setting_kv.copy()
        setting_kv_RETRO["staleness"] = False
        setting_kv_RETRO["retrieval_interval"] = 64
        df_RETRO = select_rows(df, setting_kv_RETRO)

        print("Stale performance verification:")
        diff_percentage_list_stale = verify_performance_model(df_stale, perf_model, staleness=True)
        print("Retro performance verification:")
        diff_percentage_list_RETRO = verify_performance_model(df_RETRO, perf_model, staleness=False)
        # concatenate the two lists
        diff_percentage_list = list(diff_percentage_list_stale) + list(diff_percentage_list_RETRO)
        print("Overall (stale + non-stale) performance verification:")
        print("Average diff percentage: {:.2f}".format(np.mean(diff_percentage_list)))
        print("Median diff percentage: {:.2f}".format(np.median(diff_percentage_list)))
        