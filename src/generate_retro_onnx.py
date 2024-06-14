"""
Example Usage: 

python generate_retro_onnx.py \
    --retro-config $WORKSPACE/data/model/retro.json \
    --checkpoint $WORKSPACE/data/model/model.ckpt \
    --prompt "A retrieval-enhanced language model is" \
    --num-neighbours 1 \
    --num-continuation-chunks 1 \
    --enable-profiling 0 \
    --need-export 0 \
    --run-with-binding 1

# WARNING: enable profiling will slow down the program by a factor of 3
nsys profile -t cuda,nvtx \
             --capture-range=cudaProfilerApi \
             --capture-range-end=none \
             --backtrace none \
             -s none \
             --show-output=true \
             --force-overwrite=true \
             --export=sqlite,text \
             -o ./traces/sample \
    ... (python script)
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

import numpy as np
import onnx
import onnxruntime as ort

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--retro-config", type=Path, required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--prompt")
parser.add_argument("--num-neighbours", type=int, default=2)
parser.add_argument("--num-continuation-chunks", type=int, default=1)
parser.add_argument("--use-float16", type=int, default=0)
parser.add_argument("--enable-profiling", type=int, default=0)
parser.add_argument("--need-export", type=int, default=0, help="0 = directly load; 1 = export to ONNX and load")
parser.add_argument("--run-with-binding", type=int, default=1, help="0 = run directly; 1 = run with io bindings")

args = parser.parse_args()

encoder_dir = 'onnx_retro_encoder/retro_encoder.onnx'
decoder_dir = 'onnx_retro_decoder/retro_decoder.onnx'

args.use_float16 = 0
print("Force using float32, because (a) output data type issue when using float16 and (b) for small amount of compute fp32 is not faster than fp16")
if args.use_float16:
    float_type = np.float16
else:
    float_type = np.float32

batch_size = 1

encoder_input_names = ['neighbour_ids', 'past_ca_hidden_states', 'past_ca_attention_mask']
encoder_output_names = ['neighbour_hidden_states', 'neighbour_attention_mask']

decoder_input_names=['input_ids'] + ['neighbour_hidden_states', 'neighbour_attention_mask'] + [f'past_key_value.{i}' for i in range(24)] + ['past_ca_hidden_states']
decoder_output_names = ['outputs'] + [f'out_key_values.{i}' for i in range(24)] + ['out_ca_hidden_states', 'out_ca_attention_mask']


def main_autoregressive(args):

    """ Starts loading model """

    config = RetroConfig(**json.load(args.retro_config.open()))
    tokenizer = get_tokenizer()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    retriever = DummyRetriever(
        num_neighbours=args.num_neighbours, 
        neighbour_len=config.chunk_size * (1 + args.num_continuation_chunks))
        
    model = RetroModelLMHeadLightningInference.load_from_checkpoint(str(args.checkpoint), config=config, retriever=retriever, device=device).eval()

    if args.use_float16:
        print("Using float16")
        model = model.half() # use fp16
    model.to(device) # move to GPU if available

    """ Ends loading model """

    """ Starts getting inputs & run demo inference """ 
    prompt = args.prompt

    # Get input
    input_ids = tokenizer([prompt], add_special_tokens=False, return_tensors="pt")["input_ids"]
    input_ids = input_ids.to(device)
    input_ids = input_ids.repeat(batch_size, 1)
    print("Input ID shape:", input_ids.shape)

    # run inference given prompt - 1 tokens, to get the past k-v cache & ca hidden states
    model_inputs = model.prepare_inputs_for_generation(input_ids)
    neighbour_ids = model_inputs["neighbour_ids"][:,-1:,:,:] # last chunk
    neighbour_hidden_states = torch.zeros((neighbour_ids.shape[0], 1, neighbour_ids.shape[2], neighbour_ids.shape[3], model.config.enc_hidden_dim), dtype=torch.float32, device=neighbour_ids.device)
    neighbour_attention_mask = torch.zeros((neighbour_ids.shape[0], 1, neighbour_ids.shape[2], neighbour_ids.shape[3]), dtype=torch.float32, device=neighbour_ids.device)
    outputs, past_key_value, past_ca_hidden_states, past_ca_attention_mask = model(input_ids, neighbour_hidden_states, neighbour_attention_mask)

    encoder = model.base.encoder
    model_inputs = model.prepare_inputs_for_generation(input_ids)
    neighbour_ids = model_inputs["neighbour_ids"][:,-1:,:,:] # last chunk
    neighbour_hidden_states, neighbour_attention_mask = encoder(neighbour_ids, past_ca_hidden_states, past_ca_attention_mask)
    
    
    if args.need_export:
        """ Export Encoder """
        print("\n===== Start exporting Encoder to ONNX =====\n", flush=True)
        # neighbour_ids - [batch, num chunks == 1, num neighbours, neighbour length]
        # past_ca_hidden_states - [batch, num chunks == 1, chunk length, hidden size]
        # neighbour_hidden_states torch.Size([batch, num chunks == 1, num neighbours, neighbour length, encoder hidden size])
        encoder_dynamic_axes={'neighbour_ids': [2, 3], 
            'past_ca_hidden_states': [2], 
            'past_ca_attention_mask': [2],
            # 'neighbour_hidden_states': [2, 3], 
            # 'neighbour_attention_mask': [2, 3]
            }
        
        # https://pytorch.org/docs/2.1/onnx.html#torch.onnx.export
        torch.onnx.export(model=encoder, args=(neighbour_ids, past_ca_hidden_states, past_ca_attention_mask), f=encoder_dir, export_params=True, verbose=True,
            input_names=encoder_input_names, output_names=encoder_output_names, dynamic_axes=encoder_dynamic_axes)
        print("Encoder Model saved.")

    # run inference given prompt - 1 tokens, to get the past k-v cache & ca hidden states
    input_ids_last = input_ids[:, :-1]
    print("input_ids_last shape:", input_ids_last.shape)
    input_ids_last = input_ids_last.to(device)
    outputs, past_key_value, past_ca_hidden_states, past_ca_attention_mask = model(
        input_ids_last, neighbour_hidden_states, neighbour_attention_mask, past_key_value=None, past_ca_hidden_states=None)
    print("past_key_value len: ", len(past_key_value))
    print("past k v shape: ", past_key_value[0].shape, past_key_value[1].shape)

    if args.need_export:

        """ Export Decoder """
        print("\n===== Start exporting Decoder to ONNX =====\n", flush=True)
        # dim 0 = batch size
        decoder_dynamic_axes={'input_ids': [1], 
            'neighbour_hidden_states': [2, 3], 
            'neighbour_attention_mask': [2, 3], 
            'past_ca_hidden_states': [2], }
            # 'outputs': # [batch size, vocab size]
            # 'out_ca_hidden_states': [2], 
            # 'out_ca_attention_mask': [2]}
        for i in range(len(past_key_value)):
            decoder_dynamic_axes[f'past_key_value.{i}'] = [2] # torch.Size([1, 12, 10, 128]) -> batch size, num_heads, len, dim/num_heads
            decoder_dynamic_axes[f'out_key_value.{i}'] = [2] # torch.Size([1, 12, 10, 128]) -> batch size, num_heads, len, dim/num_heads

        # https://pytorch.org/docs/2.1/onnx.html#torch.onnx.export
        past_ca_hidden_states = past_ca_hidden_states
        torch.onnx.export(model=model, args=(input_ids, neighbour_hidden_states, neighbour_attention_mask, past_key_value, past_ca_hidden_states), f=decoder_dir, export_params=True, verbose=True,
            input_names=decoder_input_names, output_names=decoder_output_names, dynamic_axes=decoder_dynamic_axes)

        # # ONNX does not support converting models larger than 2 GB.... https://github.com/gmalivenko/onnx-opcounter/issues/4
        # if args.use_float16:
        #     print("Converting to float16")
        #     from onnxconverter_common import float16
        #     model = onnx.load(decoder_dir)
        #     model = float16.convert_float_to_float16(model)
        #     onnx.save(model, decoder_dir)
        print("Decoder Model saved.")


    iter_count = 0
    # max_len = 20
    max_len = 1024
    # torch.cuda.cudart().cudaProfilerStart()
    logits_processor =  LogitsProcessorList()


    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if args.enable_profiling:
        print("Warning: profiling will slow down the program by a factor of 3")
        so.enable_profiling=True
    ort_sess_encoder = ort.InferenceSession(encoder_dir, so, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    ort_sess_decoder = ort.InferenceSession(decoder_dir, so, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    # using I/O binding
    if args.run_with_binding: 

        @torch.no_grad() 
        def invoke_encoder_with_binding(
            ort_sess_encoder, io_binding_encoder):

            start = time.time()
            ort_sess_encoder.run_with_iobinding(io_binding_encoder)
            end = time.time()
            print("decoder kernel time: {:.2f} ms".format((end - start) * 1000))

        @torch.no_grad() # no grad improves latency from e.g, 33 -> 27 ms
        def invoke_decoder_with_binding(
                input_ids : np.ndarray,
                logits_processor, ort_sess_decoder, io_binding_decoder) -> np.ndarray:

            # outputs = model(input_ids, neighbour_ids)
            # outputs = ort_sess_decoder.run(None, model_inputs)
            
            start = time.time()
            ort_sess_decoder.run_with_iobinding(io_binding_decoder)
            # outputs = io_binding_decoder.copy_outputs_to_cpu()[0]
            outputs = io_binding_decoder.get_outputs()[0].numpy() # io_binding_decoder.get_outputs() has a length of 25
            end = time.time()
            print("decoder kernel time: {:.2f} ms".format((end - start) * 1000))

            outputs = torch.tensor(outputs)

            input_ids = torch.tensor(input_ids)
            # next_token_logits = outputs.logits[:, -1, :]
            next_token_logits = outputs[:, -1, :]

            next_tokens_scores = logits_processor(input_ids, next_token_logits)

            next_tokens = torch.argmax(next_tokens_scores, dim=-1)
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            input_ids = input_ids.detach().cpu().numpy()

            return input_ids

        io_binding_encoder = ort_sess_encoder.io_binding()
        io_binding_decoder = ort_sess_decoder.io_binding()
        # OnnxRuntime will copy the data over to the CUDA device if 'input' is consumed by nodes on the CUDA device
        
        # get some initial values
        neighbour_ids = model_inputs["neighbour_ids"][:,-1:,:,:] # last chunk
        neighbour_hidden_states, neighbour_attention_mask = encoder(neighbour_ids, past_ca_hidden_states, past_ca_attention_mask)
        neighbour_hidden_states = torch.zeros((neighbour_ids.shape[0], 1, neighbour_ids.shape[2], neighbour_ids.shape[3], model.config.enc_hidden_dim), dtype=torch.float32, device=neighbour_ids.device)
        neighbour_attention_mask = torch.zeros((neighbour_ids.shape[0], 1, neighbour_ids.shape[2], neighbour_ids.shape[3]), dtype=torch.float32, device=neighbour_ids.device)

        # Convert all inputs to numpy array
        input_ids = input_ids.cpu().numpy() if isinstance(input_ids, torch.Tensor) else input_ids
        neighbour_ids = neighbour_ids.cpu().numpy() if isinstance(neighbour_ids, torch.Tensor) else neighbour_ids
        neighbour_hidden_states = neighbour_hidden_states.cpu().numpy() if isinstance(neighbour_hidden_states, torch.Tensor) else neighbour_hidden_states
        neighbour_attention_mask = neighbour_attention_mask.cpu().numpy() if isinstance(neighbour_attention_mask, torch.Tensor) else neighbour_attention_mask
        past_ca_hidden_states = past_ca_hidden_states.cpu().numpy() if isinstance(past_ca_hidden_states, torch.Tensor) else past_ca_hidden_states
        past_ca_attention_mask = past_ca_attention_mask.cpu().numpy() if isinstance(past_ca_attention_mask, torch.Tensor) else past_ca_attention_mask
        for i in range(len(past_key_value)):
            past_key_value[i] = past_key_value[i].cpu().numpy() if isinstance(past_key_value[i], torch.Tensor) else past_key_value[i]


        # io_binding_decoder.bind_cpu_input('input_ids', input_ids)
        ortvalue_input_ids = ort.OrtValue.ortvalue_from_numpy(input_ids, device_type='cuda', device_id=0)
        io_binding_decoder.bind_input(name='input_ids', device_type='cuda', device_id=0, element_type=np.int64, shape=ortvalue_input_ids.shape(), buffer_ptr=ortvalue_input_ids.data_ptr())
        ortvalue_neighbour_hidden_states = ort.OrtValue.ortvalue_from_numpy(neighbour_hidden_states, device_type='cuda', device_id=0)
        io_binding_decoder.bind_input(name='neighbour_hidden_states', device_type='cuda', device_id=0, element_type=float_type, shape=ortvalue_neighbour_hidden_states.shape(), buffer_ptr=ortvalue_neighbour_hidden_states.data_ptr())
        ortvalue_neighbour_attention_mask = ort.OrtValue.ortvalue_from_numpy(neighbour_attention_mask, device_type='cuda', device_id=0)
        io_binding_decoder.bind_input(name='neighbour_attention_mask', device_type='cuda', device_id=0, element_type=float_type, shape=ortvalue_neighbour_attention_mask.shape(), buffer_ptr=ortvalue_neighbour_attention_mask.data_ptr())
        ortvalue_past_ca_hidden_states = ort.OrtValue.ortvalue_from_numpy(past_ca_hidden_states, device_type='cuda', device_id=0)
        io_binding_decoder.bind_input(name='past_ca_hidden_states', device_type='cuda', device_id=0, element_type=float_type, shape=ortvalue_past_ca_hidden_states.shape(), buffer_ptr=ortvalue_past_ca_hidden_states.data_ptr())
        ortvalue_past_key_value = [ort.OrtValue.ortvalue_from_numpy(past_key_value[i], device_type='cuda', device_id=0) for i in range(len(past_key_value))]
        for i in range(len(past_key_value)):
            io_binding_decoder.bind_input(name=f'past_key_value.{i}', device_type='cuda', device_id=0, element_type=float_type, shape=ortvalue_past_key_value[i].shape(), buffer_ptr=ortvalue_past_key_value[i].data_ptr())

        io_binding_decoder.bind_output('outputs', 'cuda', element_type=float_type)
        for i in range(len(past_key_value)):
            # io_binding_decoder.bind_ortvalue_output(f'out_key_values.{i}', ortvalue_past_key_value[i]) # different shape, cannot bind the same input
            io_binding_decoder.bind_output(f'out_key_values.{i}', 'cuda', element_type=float_type)
        io_binding_decoder.bind_ortvalue_output('out_ca_hidden_states', ortvalue_past_ca_hidden_states)
        ortvalue_past_ca_attention_mask = ort.OrtValue.ortvalue_from_numpy(past_ca_attention_mask, device_type='cuda', device_id=0)
        io_binding_decoder.bind_ortvalue_output('out_ca_attention_mask', ortvalue_past_ca_attention_mask)
        

        t_start_all = time.time()
        while len(input_ids[0]) < max_len:

            print("iter_count:", iter_count)
            
            start = time.time()

            input_ids = invoke_decoder_with_binding(input_ids, logits_processor, ort_sess_decoder, io_binding_decoder)

            # updates next iteration inputs
            io_binding_decoder.bind_cpu_input('input_ids', input_ids)
            ortvalue_past_key_value_updated = io_binding_decoder.get_outputs()[1:1 + 24] # io_binding_decoder.get_outputs() has a length of 25, is a list of OrtValue objects
            ortvalue_past_ca_hidden_states_updated = io_binding_decoder.get_outputs()[-2]
            ortvalue_past_ca_attention_mask_updated = io_binding_decoder.get_outputs()[-1]
            for i in range(len(past_key_value)):
                io_binding_decoder.bind_ortvalue_input(f'past_key_value.{i}', ortvalue_past_key_value_updated[i])
            io_binding_decoder.bind_ortvalue_input('past_ca_hidden_states', io_binding_decoder.get_outputs()[-2])

            io_binding_decoder.bind_output('outputs', 'cuda', element_type=float_type)
            for i in range(len(past_key_value)):
                # io_binding_decoder.bind_ortvalue_output(f'out_key_values.{i}', ortvalue_past_key_value_updated[i])  # different shape, cannot bind the same input
                io_binding_decoder.bind_output(f'out_key_values.{i}', 'cuda', element_type=float_type)
            io_binding_decoder.bind_ortvalue_output('out_ca_hidden_states', ortvalue_past_ca_hidden_states_updated)
            io_binding_decoder.bind_ortvalue_output('out_ca_attention_mask', ortvalue_past_ca_attention_mask_updated)

            end = time.time()
            print("Time elapsed of this iteration: {} ms".format((end - start) * 1000))

            # print("\n-- Generation of this round complete --\n")
            # print(input_ids)
            print(tokenizer.decode(input_ids[0]))
            # print("\n-------------------------\n")
            iter_count += 1

            if len(input_ids[0]) % config.chunk_size == 0:
                # update encoder inputs
                print(neighbour_ids)
                print(neighbour_ids.shape)
                io_binding_encoder.bind_cpu_input('neighbour_ids', neighbour_ids)
                io_binding_encoder.bind_ortvalue_input('past_ca_hidden_states', io_binding_decoder.get_outputs()[-2])
                io_binding_encoder.bind_ortvalue_input('past_ca_attention_mask', io_binding_decoder.get_outputs()[-1])
                io_binding_encoder.bind_ortvalue_output('neighbour_hidden_states', ortvalue_neighbour_hidden_states)
                io_binding_encoder.bind_ortvalue_output('neighbour_attention_mask', ortvalue_neighbour_attention_mask)
                

                invoke_encoder_with_binding(ort_sess_encoder, io_binding_encoder)

                # updates decoder inputs
                io_binding_decoder.bind_ortvalue_input('neighbour_hidden_states', io_binding_encoder.get_outputs()[0])
                io_binding_decoder.bind_ortvalue_input('neighbour_attention_mask', io_binding_encoder.get_outputs()[1])

        t_end_all = time.time()
        print("Total time elapsed: {} ms".format((t_end_all - t_start_all) * 1000))
        print("output length: ", len(input_ids[0]))
        print("output shape: ", input_ids.shape)

    else: # run without io binding

        @torch.no_grad() 
        def invoke_encoder(
            neighbour_ids : np.ndarray, 
            past_ca_hidden_states : np.ndarray, 
            past_ca_attention_mask : np.ndarray, 
            ort_sess_encoder) -> (np.ndarray, np.ndarray):

            inputs = {'neighbour_ids': neighbour_ids, 'past_ca_hidden_states': past_ca_hidden_states, 'past_ca_attention_mask': past_ca_attention_mask}
            start = time.time()
            all_outputs = ort_sess_encoder.run(encoder_output_names, inputs) # returns numpy array
            end = time.time()
            print("encoder kernel + cp cpu -> gpu; gpu -> cpu + run time: {:.2f} ms".format((end - start) * 1000))
            neighbour_hidden_states = all_outputs[0]
            neighbour_attention_mask = all_outputs[1]

            return neighbour_hidden_states, neighbour_attention_mask

        @torch.no_grad() 
        def invoke_decoder(
            input_ids : np.ndarray,
            neighbour_hidden_states : np.ndarray,
            neighbour_attention_mask : np.ndarray,
            past_key_value : list[np.ndarray],
            past_ca_hidden_states : np.ndarray,
            logits_processor, 
            ort_sess_decoder) -> (np.ndarray, list[np.ndarray], np.ndarray, np.ndarray):

            # outputs = model(input_ids, neighbour_ids)
            # outputs = ort_sess_decoder.run(None, model_inputs)


            start = time.time()
            inputs = {'input_ids': input_ids, 'neighbour_hidden_states': neighbour_hidden_states, 'neighbour_attention_mask': neighbour_attention_mask, 'past_ca_hidden_states': past_ca_hidden_states}
            for i in range(len(past_key_value)):
                    inputs[f'past_key_value.{i}'] = np.array(past_key_value[i], dtype=float_type)
            start_run = time.time()
            all_outputs = ort_sess_decoder.run(decoder_output_names, inputs) # returns numpy array
            end_run = time.time()
            print("decoder kernel + cp cpu -> gpu + run time: {:.2f} ms".format((end_run - start_run) * 1000))
            outputs = all_outputs[0]
            past_key_value = all_outputs[1: 1 + 24]
            past_ca_hidden_states = all_outputs[-2]
            past_ca_attention_mask = all_outputs[-1]
            end = time.time()
            print("decoder kernel + cp cpu -> gpu; gpu -> cpu + run time: {:.2f} ms".format((end - start) * 1000))
            outputs = torch.tensor(outputs)

            input_ids = torch.tensor(input_ids)
            # next_token_logits = outputs.logits[:, -1, :]
            next_token_logits = outputs[:, -1, :]

            next_tokens_scores = logits_processor(input_ids, next_token_logits)

            next_tokens = torch.argmax(next_tokens_scores, dim=-1)
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            input_ids = input_ids.detach().cpu().numpy()

            return input_ids, past_key_value, past_ca_hidden_states, past_ca_attention_mask

        neighbour_ids = model_inputs["neighbour_ids"][:,-1:,:,:] # last chunk
        neighbour_hidden_states, neighbour_attention_mask = encoder(neighbour_ids, past_ca_hidden_states, past_ca_attention_mask)
        neighbour_hidden_states = torch.zeros((neighbour_ids.shape[0], 1, neighbour_ids.shape[2], neighbour_ids.shape[3], model.config.enc_hidden_dim), dtype=torch.float32, device=neighbour_ids.device)
        neighbour_attention_mask = torch.zeros((neighbour_ids.shape[0], 1, neighbour_ids.shape[2], neighbour_ids.shape[3]), dtype=torch.float32, device=neighbour_ids.device)


        # Convert all inputs to numpy array
        input_ids = input_ids.cpu().numpy() if isinstance(input_ids, torch.Tensor) else input_ids
        neighbour_ids = neighbour_ids.cpu().numpy() if isinstance(neighbour_ids, torch.Tensor) else neighbour_ids
        neighbour_hidden_states = neighbour_hidden_states.cpu().numpy() if isinstance(neighbour_hidden_states, torch.Tensor) else neighbour_hidden_states
        neighbour_attention_mask = neighbour_attention_mask.cpu().numpy() if isinstance(neighbour_attention_mask, torch.Tensor) else neighbour_attention_mask
        past_ca_hidden_states = past_ca_hidden_states.cpu().numpy() if isinstance(past_ca_hidden_states, torch.Tensor) else past_ca_hidden_states
        past_ca_attention_mask = past_ca_attention_mask.cpu().numpy() if isinstance(past_ca_attention_mask, torch.Tensor) else past_ca_attention_mask
        for i in range(len(past_key_value)):
            past_key_value[i] = past_key_value[i].cpu().numpy() if isinstance(past_key_value[i], torch.Tensor) else past_key_value[i]

        t_start_all = time.time()
        while len(input_ids[0]) < max_len:

            print("iter_count:", iter_count)
            
            start = time.time()
            input_ids, past_key_value, past_ca_hidden_states, past_ca_attention_mask = invoke_decoder(
                input_ids, neighbour_hidden_states, neighbour_attention_mask, past_key_value, past_ca_hidden_states, logits_processor, ort_sess_decoder)
            end = time.time()
            print("Time elapsed of this iteration: {} ms".format((end - start) * 1000))
            print(tokenizer.decode(input_ids[0]))
            iter_count += 1

            if len(input_ids[0]) % config.chunk_size == 0:
                neighbour_hidden_states, neighbour_attention_mask = invoke_encoder(
                    neighbour_ids, past_ca_hidden_states, past_ca_attention_mask, ort_sess_encoder)

        t_end_all = time.time()
        print("Total time elapsed: {} ms".format((t_end_all - t_start_all) * 1000))
        print("output length: ", len(input_ids[0]))
        print("output shape: ", input_ids.shape)


if __name__ == "__main__":

    # main(args)
    main_autoregressive(args)
