import json

fname = "v1.0-simplified_simplified-nq-train.jsonl"

# load the first line of the file, print the schema
with open(fname) as f:
	line = f.readline()
	line_json = json.loads(line)
	print(line_json)

	print("\n", "Keys in the JSON object:", line_json.keys(), "\n")
	for key in line_json.keys():
		print("\n")
		print(key)
		print(line_json[key])
		print("\n")

	# print the long answer, e.g., 
	# annotations
	# [{'yes_no_answer': 'NONE', 'long_answer': {'start_token': 1952, 'candidate_index': 54, 'end_token': 2019}, 'short_answers': [{'start_token': 1960, 'end_token': 1969}], 'annotation_id': 593165450220027640}]
	line_json_long_answer = line_json["annotations"][0]["long_answer"]
	# get from long answer's start_token to end_token from the document_text
	print("\n", "Long Answer:", line_json_long_answer, "\n")
	start_token = line_json_long_answer["start_token"]
	end_token = line_json_long_answer["end_token"]
	# get tokens from doc: https://github.com/google-research-datasets/natural-questions/blob/master/text_utils.py#L44
	answer = " ".join(line_json["document_text"].split(" ")[start_token: end_token])
	print("Question:", line_json["question_text"])
	print("Answer:", answer)