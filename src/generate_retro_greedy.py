"""
Example Usage (the sys profiling part is optional): 
nsys profile -t cuda,nvtx \
             --capture-range=cudaProfilerApi \
             --capture-range-end=none \
             --backtrace none \
             -s none \
             --show-output=true \
             --force-overwrite=true \
             --export=sqlite,text \
             -o ./traces/sample \
python generate_retro_greedy.py \
    --retro-config $WORKSPACE/data/model/retro.json \
    --checkpoint $WORKSPACE/data/model/model.ckpt \
    --prompt "A retrieval-enhanced language model is" \
    --num-neighbours 1 \
    --num-continuation-chunks 1 \
    --incremental_token_num 10 \
    --use-cache 1 \
    --use-our-greedy 1 \
    --use-huggingface-greedy-search 0 \
    --use-huggingface-generate 0 \
    --use-float16 1 \
    --use-original-model 0 
"""


from argparse import ArgumentError
import json
import readline
import torch
import time
from pathlib import Path
from dataset_retro import ChunkedSequenceDataset, ShardedChunkedSequenceDataset
from modeling_retro import RetroConfig
from sentence_transformers import SentenceTransformer
from retrieval import RetrieverWithCache, IndexServiceRetriever, IndexServiceClient, DummyRetriever
from train_retro import RetroModelLMHeadLightning, RetroModelLMHeadLightningInference
from data.tokenize_and_chunk import get_tokenizer
from transformers import LogitsProcessorList

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--retro-config", type=Path, required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--prompt")
parser.add_argument("--num-neighbours", type=int, default=2)
parser.add_argument("--num-continuation-chunks", type=int, default=1)
parser.add_argument("--incremental_token_num", type=int, default=1)
parser.add_argument("--use-our-greedy", type=int, default=0)
parser.add_argument("--use-huggingface-greedy-search", type=int, default=0)
parser.add_argument("--use-huggingface-generate", type=int, default=0)
parser.add_argument("--use-cache", type=int, default=1)
parser.add_argument("--use-float16", type=int, default=1)
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--use-original-model", type=int, default=0, help="Use the original model or the inference model")
parser.add_argument("--retrieval-interval", type=int, default=64, help="Only compatible with mode 'use_huggingface_greedy_search'")

args = parser.parse_args()
incremental_token_num = args.incremental_token_num

use_our_greedy=args.use_our_greedy
use_huggingface_greedy_search=args.use_huggingface_greedy_search
use_huggingface_generate=args.use_huggingface_generate
use_cache=True if args.use_cache else False

use_original_model = True if args.use_original_model else False
retrieval_interval = args.retrieval_interval
if retrieval_interval != 64:
    assert use_huggingface_greedy_search, "ERROR: retrieval_interval is only compatible with mode 'use_huggingface_greedy_search'"


if args.use_float16:
    float_type = torch.float16
else:
    float_type = torch.float32

# assert one of the three option is one, others are zeros

@torch.no_grad() # no grad improves latency from e.g, 33 -> 27 ms
def generate_greedy(model, input_ids, logits_processor, retrieval_interval=64, use_original_model=False):
    """
> $WORKSPACE/src/generate_retro_greedy.py(62)main_autoregressive()
-> iter_count = 0
(Pdb) model_inputs = model.prepare_inputs_for_generation(input_ids)
(Pdb) model(model_inputs)
*** TypeError: forward() missing 1 required positional argument: 'neighbour_ids'
(Pdb) model_inputs
{'input_ids': tensor([[   71, 24515,   138,    18,    35,   107,   663,    26,  1612,   825,
            19]]), 'neighbour_ids': tensor([[[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]])}

(Pdb) outputs = model(**model_inputs)
(Pdb) outputs
RetroModelLMHeadOutput(hidden_states=tensor([[[ 0.0291, -0.4603,  0.9976,  ..., -0.0571, -0.7826, -0.9045],
         [ 0.5305,  1.9860, -0.2031,  ..., -1.7691, -0.3677, -1.1979],
         [-0.0259,  0.7580,  0.1563,  ..., -0.6771,  0.1087, -1.5425],
         ...,
         [ 0.4000,  0.1890, -0.1693,  ...,  0.4072,  0.4206, -0.6877],
         [ 0.0307,  0.9190,  1.4587,  ...,  1.6908, -0.9797, -1.1908],
         [-0.6635, -1.7058, -0.4785,  ...,  0.1777, -0.0978, -1.0301]]],
       grad_fn=<MulBackward0>), neighbour_hidden_states=tensor([[[[[ 0.5755,  1.2693, -0.2792,  ...,  0.9419,  0.6808, -0.4154],
           [ 0.5635,  1.2481, -0.2839,  ...,  0.9411,  0.6717, -0.4056],
           [ 0.5564,  1.2634, -0.2687,  ...,  0.9445,  0.6819, -0.4243],
           ...,
           [ 0.5734,  1.2615, -0.2805,  ...,  0.9429,  0.6774, -0.4141],
           [ 0.5734,  1.2615, -0.2805,  ...,  0.9429,  0.6774, -0.4141],
           [ 0.5734,  1.2615, -0.2805,  ...,  0.9429,  0.6774, -0.4141]]]]],
       grad_fn=<AddBackward0>), logits=tensor([[[-8.6393, -0.2064,  3.0579,  ..., -8.6675, -8.6072, -8.6384],
         [-7.7779,  2.6336,  4.2337,  ..., -7.7934, -7.8173, -7.8286],
         [-9.2258,  2.5481,  2.8686,  ..., -9.1716, -9.1315, -9.0912],
         ...,
         [-9.2560,  4.6650,  1.9768,  ..., -9.3683, -9.4503, -9.3982],
         [-9.7617,  4.0080,  3.0843,  ..., -9.7873, -9.8571, -9.7301],
         [-9.5021,  1.7600,  1.2725,  ..., -9.5084, -9.4993, -9.4128]]],
       grad_fn=<UnsafeViewBackward0>), loss=None)

Logits can be used for future word prediction, but what is the 11 here? 

(Pdb) outputs.logits.shape, 11 is the current length
torch.Size([1, 11, 32128])

next_token_logits = outputs.logits[:, -1, :]

from transformers import LogitsProcessorList
logits_processor =  LogitsProcessorList()
next_tokens_scores = logits_processor(input_ids, next_token_logits)

(Pdb) next_tokens_scores
tensor([[-9.5021,  1.7600,  1.2725,  ..., -9.5084, -9.4993, -9.4128]],
       grad_fn=<SliceBackward0>)
(Pdb) next_tokens_scores.shape
torch.Size([1, 32128])

(Pdb) next_tokens = torch.argmax(next_tokens_scores, dim=-1)
(Pdb) next_tokens
tensor([3])

(Pdb) input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
(Pdb) input_ids
tensor([[   71, 24515,   138,    18,    35,   107,   663,    26,  1612,   825,
            19,     3]])

model_kwargs = model._update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder)
    """

    model_inputs = model.prepare_inputs_for_generation(input_ids, retrieval_interval=retrieval_interval)
    outputs = model(**model_inputs)
    if use_original_model:
        next_token_logits = outputs.logits[:, -1, :]
    else:
        next_token_logits = outputs[:, -1, :]

    next_tokens_scores = logits_processor(input_ids, next_token_logits)

    next_tokens = torch.argmax(next_tokens_scores, dim=-1)
    input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

    return input_ids


@torch.no_grad() # no grad improves latency from e.g, 33 -> 27 ms
def generate_greedy_with_decoder_cache(model, input_ids, logits_processor, use_original_model=False, past_key_value=None):
    # Only for our inference model implementation
    assert not use_original_model

    model_inputs = model.prepare_inputs_for_generation(input_ids)
    if past_key_value is not None:
        model_inputs["past_key_value"] = past_key_value
    outputs, past_key_value = model(**model_inputs)
    next_token_logits = outputs[:, -1, :]

    next_tokens_scores = logits_processor(input_ids, next_token_logits)

    next_tokens = torch.argmax(next_tokens_scores, dim=-1)
    input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

    return input_ids, past_key_value

@torch.no_grad() # no grad improves latency from e.g, 33 -> 27 ms
def generate_greedy_with_encoder_decoder_cache(
    model, encoder,
    input_ids, 
    neighbour_hidden_states=None, 
    neighbour_attention_mask=None, 
    past_key_value=None,
    past_ca_hidden_states=None,
    logits_processor=None, use_original_model=False):

    assert not use_original_model

    if neighbour_hidden_states is None or neighbour_attention_mask is None:
        # generate zero tensors
        # neighbour_ids - [batch, num chunks == 1, num neighbours, neighbour length]
        # neighbour_hidden_states - [batch, num chunks == 1, num neighbours, neighbour length, hidden size]
        # neighbour_attention_mask - [batch, num chunks == 1, num neighbours, neighbour length]
        model_inputs = model.prepare_inputs_for_generation(input_ids)
        neighbour_ids = model_inputs["neighbour_ids"][:,-1:,:,:] # last chunk
        neighbour_hidden_states = torch.zeros((neighbour_ids.shape[0], 1, neighbour_ids.shape[2], neighbour_ids.shape[3], model.config.enc_hidden_dim), dtype=float_type, device=neighbour_ids.device)
        neighbour_attention_mask = torch.zeros((neighbour_ids.shape[0], 1, neighbour_ids.shape[2], neighbour_ids.shape[3]), dtype=float_type, device=neighbour_ids.device)
        
    outputs, past_key_value, past_ca_hidden_states, past_ca_attention_mask = model(input_ids, neighbour_hidden_states, neighbour_attention_mask, past_key_value, past_ca_hidden_states)
    next_token_logits = outputs[:, -1, :]

    if logits_processor is not None: # a default logits_processor does not change the input logits
        next_tokens_scores = logits_processor(input_ids, next_token_logits)
    else:
        next_tokens_scores = next_token_logits

    next_tokens = torch.argmax(next_tokens_scores, dim=-1)
    input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

    # if across 64-token boundary, update the encoder cache
    if input_ids.shape[1] % model.config.chunk_size == 0:
        model_inputs = model.prepare_inputs_for_generation(input_ids)
        neighbour_ids = model_inputs["neighbour_ids"][:,-1:,:,:]
        # invoke encoder forward
        neighbour_hidden_states, neighbour_attention_mask = encoder(neighbour_ids, past_ca_hidden_states, past_ca_attention_mask)

    return input_ids, neighbour_hidden_states, neighbour_attention_mask, past_key_value, past_ca_hidden_states, past_ca_attention_mask


def main_autoregressive(args):

    """ Starts loading model """

    config = RetroConfig(**json.load(args.retro_config.open()))
    tokenizer = get_tokenizer()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    retriever = DummyRetriever(
        num_neighbours=args.num_neighbours, 
        neighbour_len=config.chunk_size * (1 + args.num_continuation_chunks))
        
    if use_original_model:
        print("Using original model: RetroModelLMHeadLightning")
        model = RetroModelLMHeadLightning.load_from_checkpoint(str(args.checkpoint), config=config, retriever=retriever, device=device).eval()
    else:
        print("Using inference model: RetroModelLMHeadLightningInference")
        model = RetroModelLMHeadLightningInference.load_from_checkpoint(str(args.checkpoint), config=config, retriever=retriever, device=device).eval()

    if args.use_float16:
        print("Using float16")
        model = model.half() # use fp16
        # model = torch.compile(model, mode="reduce-overhead")
        
    model.to(device) # move to GPU if available

    dec_params = 0
    enc_params = 0
    for name, param in model.named_parameters():
        print(name, param.shape, param.dtype)
        if name.startswith("base.dec"):
            dec_params += param.numel()
        elif name.startswith("base.enc"):
            enc_params += param.numel()
    print("dec params:", dec_params)
    print("enc params:", enc_params)
    # time.sleep(1000)

    encoder = model.base.encoder

    """ Ends loading model """

    """ Starts getting inputs """ 
    prompt = args.prompt

    if prompt is None:
        print("Input prompt:")
        prompt = input()

    input_ids = tokenizer([prompt], add_special_tokens=False, return_tensors="pt")["input_ids"]
    input_ids = input_ids.to(device)
    # replicate it by batch size times, on the second dimension
    # https://pytorch.org/docs/stable/generated/torch.Tensor.repeat.html
    input_ids = input_ids.repeat(args.batch_size, 1)
    print("Input ID shape:", input_ids.shape)
    iter_count = 0

    """ Ends getting inputs """

    if use_our_greedy:
        logits_processor =  LogitsProcessorList()
        neighbour_hidden_states = None 
        neighbour_attention_mask = None 
        past_key_value = None 
        past_ca_hidden_states = None 
        past_ca_attention_mask = None
    elif use_huggingface_greedy_search:
        logits_processor =  LogitsProcessorList()

    # max_len = 20
    max_len = 512
    torch.cuda.cudart().cudaProfilerStart()
    t_start_all = time.time()
    while len(input_ids[0]) < max_len:

        print("iter_count:", iter_count)
        print("input_ids len:", len(input_ids[0]))
        
        start = time.time()

        if use_our_greedy:
            print("Generation strategy: our greedy implementation")
            ### Our greedy implementation

            # input_ids = generate_greedy(model, input_ids, logits_processor, use_original_model=use_original_model)
            
            # input_ids, past_key_value = generate_greedy_with_decoder_cache(model, input_ids, logits_processor, use_original_model=False, past_key_value=past_key_value)
            # print("past_key_value len: ", len(past_key_value))
            # print("past k v shape: ", past_key_value[0].shape, past_key_value[1].shape)

            input_ids, neighbour_hidden_states, neighbour_attention_mask, past_key_value, past_ca_hidden_states, past_ca_attention_mask = generate_greedy_with_encoder_decoder_cache(
                model, encoder,
                input_ids, 
                neighbour_hidden_states=neighbour_hidden_states, 
                neighbour_attention_mask=neighbour_attention_mask, 
                past_key_value=past_key_value,
                past_ca_hidden_states=past_ca_hidden_states,
                logits_processor=logits_processor, use_original_model=False)

            print("input_ids shape: ", input_ids.shape)
            print("neighbour_hidden_states shape: ", neighbour_hidden_states.shape)
            print("neighbour_attention_mask shape: ", neighbour_attention_mask.shape)
            print("past_key_value len: ", len(past_key_value))
            print("each past_key_value shape: ", past_key_value[0].shape, past_key_value[1].shape)
            print("past_ca_hidden_states shape: ", past_ca_hidden_states.shape)
            print("past_ca_attention_mask shape: ", past_ca_attention_mask.shape)

        elif use_huggingface_greedy_search:
            print("Generation strategy: Huggingface greedy_search")
            # print("After changing the model outputs, this mode is not supported")
            # break
            ### the greedy_search API in Huggingface
            # with torch.no_grad():
            #     out = model.greedy_search(
            #         input_ids=input_ids,
            #         logits_processor=None,
            #         stopping_criteria=None,
            #         max_length=len(input_ids[0]) + incremental_token_num,
            #         pad_token_id=0, 
            #         eos_token_id=1,
            #         output_attentions=False,
            #         output_hidden_states=False,
            #         output_scores=False,
            #         return_dict_in_generate=True,
            #         synced_gpus=False,
            #         # **model_kwargs ?,
            #     )
            # input_ids = out.sequences
            input_ids = generate_greedy(model, input_ids, logits_processor, retrieval_interval=retrieval_interval, use_original_model=use_original_model)

        elif use_huggingface_generate:
            print("Generation strategy: Huggingface generate")
            # print("After changing the model outputs, this mode is not supported")
            # break
            ### original Huggingface generate wrapper
            with torch.no_grad():
                out = model.generate(
                    inputs=input_ids, 
                    do_sample=False,
                    num_beams=1,
                    top_k=5,
                    top_p=1,
                    temperature=1,
                    min_length=len(input_ids[0]) + incremental_token_num,
                    max_length=len(input_ids[0]) + incremental_token_num,
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
                    # enable k-v cache
                    output_scores=False,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict_in_generate=True,
                    use_cache=use_cache,
                )
                input_ids = out.sequences

        end = time.time()
        print("Time elapsed of this iteration: {} ms".format((end - start) * 1000))
        # print(out)

        torch.cuda.cudart().cudaProfilerStop()

        """\
        (Pdb) out
        GreedySearchDecoderOnlyOutput(sequences=tensor([[   71, 24515,   138,    18,    35,   107,   663,    26,  1612,   825,
                    19,     3]]), scores=None, attentions=None, hidden_states=None)
        """

        # print("\n-- Generation of this round complete --\n")
        print(tokenizer.decode(input_ids[0]))
        # print("\n-------------------------\n")
        iter_count += 1

    t_end_all = time.time()
    print("Total time elapsed: {} ms".format((t_end_all - t_start_all) * 1000))
    print("output length: ", len(input_ids[0]))
    print("output shape: ", input_ids.shape)


if __name__ == "__main__":

    # main(args)
    main_autoregressive(args)
