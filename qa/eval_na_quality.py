"""
Given the results of the NQ-open-domain, compute the scores

Example usage:
	python eval_na_quality.py --fname generated_answers/generated_answers.jsonl
"""

import json

from compute_f1_score import compute_f1_score

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--fname", type=str, default="generated_answers/generated_answers.jsonl", help="Only compatible with mode 'use_huggingface_greedy_search'")

args = parser.parse_args()

fname_input = args.fname

f1_list = []
recall_list = []
precision_list = []

with open(fname_input) as f:
	for lid, line in enumerate(f):
		line_json = json.loads(line)
		answer_list = line_json["answer_and_def_correct_predictions"]
		generated_answer = line_json["generated_answer"]

		max_f1 = 0
		max_recall = 0
		max_precision = 0

		for answer in answer_list:
			print("Answer:", answer)
			f1, precision, recall = compute_f1_score(answer, generated_answer)
			if f1 > max_f1:
				max_f1 = f1
			if recall > max_recall:
				max_recall = recall
			if precision > max_precision:
				max_precision = precision
			
		f1_list.append(max_f1)
		recall_list.append(max_recall)
		precision_list.append(max_precision)

print("Average F1:", sum(f1_list) / len(f1_list))
print("Average Recall:", sum(recall_list) / len(recall_list))
print("Average Precision:", sum(precision_list) / len(precision_list))