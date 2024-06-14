import json

fname = "v1.0-simplified_simplified-nq-train.jsonl"
target_total_lines = 100
target_max_seq_len = 1200
target_min_seq_len = 800
answer_len_list = []

my_json_output = []
valid_lines_cnt = 0

# iterate all lines in the file
with open(fname) as f:

	for lid, line in enumerate(f):
		line_json = json.loads(line)

		# annotations
		# [{'yes_no_answer': 'NONE', 'long_answer': {'start_token': 1952, 'candidate_index': 54, 'end_token': 2019}, 'short_answers': [{'start_token': 1960, 'end_token': 1969}], 'annotation_id': 593165450220027640}]
		line_json_long_answer = line_json["annotations"][0]["long_answer"]
		start_token = line_json_long_answer["start_token"]
		end_token = line_json_long_answer["end_token"]
		if start_token == end_token: # no long answer
			continue
		elif end_token - start_token <= target_max_seq_len and end_token - start_token >= target_min_seq_len:

			# get from long answer's start_token to end_token from the document_text
			answer = " ".join(line_json["document_text"].split(" ")[start_token: end_token])
			print("\nQuestion:", line_json["question_text"])
			print("Answer:", answer)
			my_json_output.append({"question": line_json["question_text"], "answer": answer})

			answer_len_list.append(end_token - start_token)	
			valid_lines_cnt += 1

		if valid_lines_cnt >= target_total_lines:
			break

print("Average answer length:", sum(answer_len_list) / len(answer_len_list))
print("Max answer length:", max(answer_len_list))
print("Min answer length:", min(answer_len_list))

# write the output to a new jsonl file
with open("filtered_nq.jsonl", "w") as f:
	for line in my_json_output:
		f.write(json.dumps(line) + "\n")
