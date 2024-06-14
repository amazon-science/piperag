import time
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using {} device".format(device))

# Original demo uses GPT2Model, which does not work, need to use GPT2LMHeadModel instead
modelname = 'gpt2-medium' # medium: 380M
# modelname = 'gpt2' # 137M

tokenizer = GPT2Tokenizer.from_pretrained(modelname)
model = GPT2LMHeadModel.from_pretrained(modelname)
model.to(device)

text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
encoded_input.to(device)
print(encoded_input)
print(encoded_input)
output = model(**encoded_input)
print(output)
print(key for key in output.__dict__.keys())
print(output.__dict__.keys())
input_ids = encoded_input["input_ids"]

# model.half()

max_len = 512
start = time.time()
res = model.generate(input_ids,
            do_sample=False,
            num_beams=1,
            top_k=5,
            top_p=1,
            temperature=1,
            min_length=max_len,
            max_length=max_len,
            length_penalty=1,
            early_stopping=False,
            num_beam_groups=1,
            num_return_sequences=1,
            repetition_penalty=1,
            no_repeat_ngram_size=3,
            encoder_no_repeat_ngram_size=0,
            diversity_penalty=0.0,
            remove_invalid_values=False,
            pad_token_id=0, 
            eos_token_id=1,
            output_scores=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict_in_generate=False,
            use_cache=False,)
end = time.time()
print(res)
print("Time taken: {:.1f} ms".format(1000 * (end - start)))
print("Time per token: {:.1f} ms".format(1000 * (end - start) / len(res[0])))
