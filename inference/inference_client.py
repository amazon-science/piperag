"""
This is the ONNX inference program, by retrieving from the Faiss server.

Example Usage:
    python inference_client.py --host 127.0.0.1 --port 50051  \
        --checkpoint $WORKSPACE/data/model/model.ckpt \
        --retro_config $WORKSPACE/data/model/retro.json \
        --encoder_dir $WORKSPACE/src/onnx_retro_encoder/retro_encoder.onnx \
        --decoder_dir $WORKSPACE/src/onnx_retro_decoder/retro_decoder.onnx \
        --batch_size 1 --chunk_size 64 --num_continuation_chunks 1 --num_neighbours 2 \
        --interval 64 --max_seq_len 1024 --staleness 1

Profile the performance:
    python inference_client.py --host 127.0.0.1 --port 50051  \
        --checkpoint $WORKSPACE/data/model/model.ckpt \
        --retro_config $WORKSPACE/data/model/retro.json \
        --encoder_dir $WORKSPACE/src/onnx_retro_encoder/retro_encoder.onnx \
        --decoder_dir $WORKSPACE/src/onnx_retro_decoder/retro_decoder.onnx \
        --batch_size 1 --chunk_size 64 --num_continuation_chunks 1 --num_neighbours 2 \
        --interval 64 --max_seq_len 1024 --staleness 1 \
        --mode profile_local_generation --num_runs 5 --perf_file ./performance/p4d.24xlarge_performance_generation_len_1024_k_2.pickle
"""

import asyncio
import numpy as np
import time
import pickle 

import grpc
import retrieval_pb2
import retrieval_pb2_grpc

from typing import Optional, List

import sys
sys.path.append('../src')
from argparse import ArgumentError
import json
import readline
import torch
from pathlib import Path
from dataset_retro import ChunkedSequenceDataset, ShardedChunkedSequenceDataset
from modeling_retro import RetroConfig
from sentence_transformers import SentenceTransformer
from retrieval import RetrieverWithCache, IndexServiceRetriever, IndexServiceClient, DummyRetriever
from train_retro import RetroModelLMHeadLightning, RetroModelLMHeadLightningInference
from data.tokenize_and_chunk import get_tokenizer
from transformers import LogitsProcessorList

import onnx
import onnxruntime as ort

class InferenceClient(object):

    def __init__(self, 
        host: Optional[str] = None, port: Optional[int] = 50051, 
        checkpoint: Optional[Path] = Path('../data/model/model.ckpt'),
        retro_config : Optional[Path] = Path('../data/model/retro.json'),
        encoder_dir: Optional[Path] = Path('../src/onnx_retro_encoder/retro_encoder.onnx'),
        decoder_dir: Optional[Path] = Path('../src/onnx_retro_decoder/retro_decoder.onnx'),
        chunk_size : Optional[int] = 64, prompt : Optional[str] = "Retrieval augmentation", 
        batch_size : Optional[int] = 1, max_gen_len : Optional[int] = 1024,
        num_neighbours : Optional[int] = 2, num_continuation_chunks : Optional[int] = 1,
        staleness_offset : Optional[int] = 0, interval : Optional[int] = 64
        ):
          
        self.host = host
        self.port = port
        self.checkpoint = checkpoint
        self.retro_config = retro_config
        self.encoder_dir = encoder_dir
        self.decoder_dir = decoder_dir

        self.chunk_size = chunk_size
        self.prompt = prompt
        self.batch_size = batch_size
        self.max_gen_len = max_gen_len
        self.num_neighbours = num_neighbours
        self.num_continuation_chunks = num_continuation_chunks
        self.staleness_offset = staleness_offset
        self.interval = interval

        self.float_type = np.float32
        self.logits_processor =  LogitsProcessorList()

        self.config = RetroConfig(**json.load(self.retro_config.open()))
        self.tokenizer = get_tokenizer()

        self.model_state_init()

    def model_state_init(self):
        """
        Load the original model as well as the ONNX model, prepare the states for the generation
        """
        retriever = DummyRetriever(
            num_neighbours=self.num_neighbours, 
            neighbour_len=self.config.chunk_size * (1 + self.num_continuation_chunks))
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = RetroModelLMHeadLightningInference.load_from_checkpoint(str(self.checkpoint), config=self.config, retriever=retriever, device=device).eval()
        model.to(device) # move to GPU if available

        """ Starts getting inputs & run demo inference """ 
        # Get input
        input_ids = self.tokenizer([self.prompt], add_special_tokens=False, return_tensors="pt")["input_ids"]
        input_ids = input_ids.to(device)
        input_ids = input_ids.repeat(self.batch_size, 1)
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
    
        # run inference given prompt - 1 tokens, to get the past k-v cache & ca hidden states
        input_ids_last = input_ids[:, :-1]
        print("input_ids_last shape:", input_ids_last.shape)
        input_ids_last = input_ids_last.to(device)
        outputs, past_key_value, past_ca_hidden_states, past_ca_attention_mask = model(
            input_ids_last, neighbour_hidden_states, neighbour_attention_mask, past_key_value=None, past_ca_hidden_states=None)
        print("past_key_value len: ", len(past_key_value))
        print("past k v shape: ", past_key_value[0].shape, past_key_value[1].shape)
        
        # get some initial values
        neighbour_ids = model_inputs["neighbour_ids"][:,-1:,:,:] # last chunk
        neighbour_hidden_states, self.init_neighbour_attention_mask = encoder(neighbour_ids, past_ca_hidden_states, past_ca_attention_mask)
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



        # Save all needed states to self.init_xxx
        self.init_input_ids = input_ids
        self.init_neighbour_ids = neighbour_ids
        self.init_neighbour_hidden_states = neighbour_hidden_states
        self.init_neighbour_attention_mask = neighbour_attention_mask
        self.init_past_ca_hidden_states = past_ca_hidden_states
        self.init_past_ca_attention_mask = past_ca_attention_mask
        self.init_past_key_value = past_key_value


        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.ort_sess_encoder = ort.InferenceSession(str(self.encoder_dir), so, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.ort_sess_decoder = ort.InferenceSession(str(self.decoder_dir), so, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    def generation_without_retrieval(self):
        """
        Pure generation, without real retrieval
        """
        # using I/O binding
        io_binding_encoder = self.ort_sess_encoder.io_binding()
        io_binding_decoder = self.ort_sess_decoder.io_binding()
        # OnnxRuntime will copy the data over to the CUDA device if 'input' is consumed by nodes on the CUDA device

        # load all init states to local variables
        input_ids = self.init_input_ids.copy()
        neighbour_ids = self.init_neighbour_ids.copy()
        neighbour_hidden_states = self.init_neighbour_hidden_states.copy()
        neighbour_attention_mask = self.init_neighbour_attention_mask.copy()
        past_ca_hidden_states = self.init_past_ca_hidden_states.copy()
        past_ca_attention_mask = self.init_past_ca_attention_mask.copy()
        past_key_value = [self.init_past_key_value[i].copy() for i in range(len(self.init_past_key_value))]

        iter_count = 0
        t_start_all = time.time()

        # io_binding_decoder.bind_cpu_input('input_ids', input_ids)
        ortvalue_input_ids = ort.OrtValue.ortvalue_from_numpy(input_ids, device_type='cuda', device_id=0)
        io_binding_decoder.bind_input(name='input_ids', device_type='cuda', device_id=0, element_type=np.int64, shape=ortvalue_input_ids.shape(), buffer_ptr=ortvalue_input_ids.data_ptr())
        ortvalue_neighbour_hidden_states = ort.OrtValue.ortvalue_from_numpy(neighbour_hidden_states, device_type='cuda', device_id=0)
        io_binding_decoder.bind_input(name='neighbour_hidden_states', device_type='cuda', device_id=0, element_type=self.float_type, shape=ortvalue_neighbour_hidden_states.shape(), buffer_ptr=ortvalue_neighbour_hidden_states.data_ptr())
        ortvalue_neighbour_attention_mask = ort.OrtValue.ortvalue_from_numpy(neighbour_attention_mask, device_type='cuda', device_id=0)
        io_binding_decoder.bind_input(name='neighbour_attention_mask', device_type='cuda', device_id=0, element_type=self.float_type, shape=ortvalue_neighbour_attention_mask.shape(), buffer_ptr=ortvalue_neighbour_attention_mask.data_ptr())
        ortvalue_past_ca_hidden_states = ort.OrtValue.ortvalue_from_numpy(past_ca_hidden_states, device_type='cuda', device_id=0)
        io_binding_decoder.bind_input(name='past_ca_hidden_states', device_type='cuda', device_id=0, element_type=self.float_type, shape=ortvalue_past_ca_hidden_states.shape(), buffer_ptr=ortvalue_past_ca_hidden_states.data_ptr())
        ortvalue_past_key_value = [ort.OrtValue.ortvalue_from_numpy(past_key_value[i], device_type='cuda', device_id=0) for i in range(len(past_key_value))]
        for i in range(len(past_key_value)):
            io_binding_decoder.bind_input(name=f'past_key_value.{i}', device_type='cuda', device_id=0, element_type=self.float_type, shape=ortvalue_past_key_value[i].shape(), buffer_ptr=ortvalue_past_key_value[i].data_ptr())

        io_binding_decoder.bind_output('outputs', 'cuda', element_type=self.float_type)
        for i in range(len(past_key_value)):
            # io_binding_decoder.bind_ortvalue_output(f'out_key_values.{i}', ortvalue_past_key_value[i]) # different shape, cannot bind the same input
            io_binding_decoder.bind_output(f'out_key_values.{i}', 'cuda', element_type=self.float_type)
        io_binding_decoder.bind_ortvalue_output('out_ca_hidden_states', ortvalue_past_ca_hidden_states)
        ortvalue_past_ca_attention_mask = ort.OrtValue.ortvalue_from_numpy(past_ca_attention_mask, device_type='cuda', device_id=0)
        io_binding_decoder.bind_ortvalue_output('out_ca_attention_mask', ortvalue_past_ca_attention_mask)

        decoder_time_per_step = []
        encoder_time_per_step = [] # only for those retrieval step
        total_time_per_step = []

        while len(input_ids[0]) < self.max_gen_len:

            print("iter_count:", iter_count)

            start = time.time()
            input_ids = self.invoke_decoder_with_binding(input_ids, self.logits_processor, self.ort_sess_decoder, io_binding_decoder)
            self.update_decoder_binding_after_decoding(input_ids, len(past_key_value), io_binding_decoder)
            end_decoder = time.time()
            decoder_time_per_step.append(end_decoder - start)

            # print("\n-- Generation of this iteration complete --")
            # print(self.tokenizer.decode(input_ids[0]))
            iter_count += 1

            if len(input_ids[0]) % self.config.chunk_size == 0:
                start_encoder = time.time()
                self.update_encoder_binding_after_decoding(io_binding_encoder, neighbour_ids, io_binding_decoder, 
                    ortvalue_neighbour_hidden_states, ortvalue_neighbour_attention_mask)
                self.invoke_encoder_with_binding(self.ort_sess_encoder, io_binding_encoder)
                self.update_decoder_binding_after_encoding(io_binding_decoder, io_binding_encoder)
                end_encoder = time.time()
                encoder_time_per_step.append(end_encoder - start_encoder)
            
            end = time.time()
            total_time_per_step.append(end - start)
            print("iteration time: {} ms".format((end - start) * 1000))

        t_end_all = time.time()
        print(self.tokenizer.decode(input_ids[0]))
        print("Total time elapsed: {} ms".format((t_end_all - t_start_all) * 1000))
        print("output length: ", len(input_ids[0]))
        print("output shape: ", input_ids.shape)

        return decoder_time_per_step, encoder_time_per_step, total_time_per_step

    def profile_local_generation(self, num_runs=5, perf_file='./performance/performance_generation.pickle'):
        """
        Run the generation for num_runs times, and profile the average latency

        Saved pickle object format:
            a dictionary with keys: "average_decoder_latency_ms", "average_encoder_latency_ms"
                "average_decoder_latency_ms": an array of average decoder latency per step in ms, length = max_gen_len
                "average_encoder_latency_ms": an scalar of average encoder latency 
        """
        decoder_time_per_run = []
        encoder_time_per_run = []
        for i in range(num_runs):
            print("Run: ", i)
            decoder_time_per_step, encoder_time_per_step, total_time_per_step = self.generation_without_retrieval()
            decoder_time_per_run.append(decoder_time_per_step)
            encoder_time_per_run.append(encoder_time_per_step)
        decoder_time_per_run = np.array(decoder_time_per_run)
        encoder_time_per_run = np.array(encoder_time_per_run)

        decoder_time_per_run = np.mean(decoder_time_per_run, axis=0)
        encoder_time_per_run = np.mean(encoder_time_per_run, axis=0)

        if decoder_time_per_run.shape[0] < self.max_gen_len:
            # pad the first tokens as the first value
            decoder_time_per_run = np.pad(decoder_time_per_run, (self.max_gen_len - decoder_time_per_run.shape[0], 0), constant_values=decoder_time_per_run[0])

        # Convert to milliseconds
        decoder_time_per_run = decoder_time_per_run * 1000  
        encoder_time_per_run = encoder_time_per_run * 1000 

        print("Decoder first ten steps (ms): ", decoder_time_per_run[:10])
        print("Decoder last ten steps (ms): ", decoder_time_per_run[-10:])
        print("Decoder min: {} ms, max: {} ms, average: {} ms".format(np.min(decoder_time_per_run), np.max(decoder_time_per_run), np.mean(decoder_time_per_run)))
        print("Encoder min: {} ms, max: {} ms, average: {} ms".format(np.min(encoder_time_per_run), np.max(encoder_time_per_run), np.mean(encoder_time_per_run)))

        performance_generation = {"average_decoder_latency_ms": decoder_time_per_run, "average_encoder_latency_ms": encoder_time_per_run}
        with open(perf_file, 'wb') as handle:
            pickle.dump(performance_generation, handle, protocol=4)

    # grpc client
    async def set_server_nprobe(self, nprobe):
        self.create_channel()
        nprobe_request = self.stub.SetNprobe(retrieval_pb2.SetNprobeRequest(nprobe=nprobe))
        nprobe_reply = await nprobe_request
        print("Server replied: " + nprobe_reply.reply)


    # grpc client
    async def generation(self, staleness=True, use_perf_model=False, one_retrieval=False):
        """
        Pure generation, without real retrieval
        """
        self.create_channel()

        # using I/O binding
        io_binding_encoder = self.ort_sess_encoder.io_binding()
        io_binding_decoder = self.ort_sess_decoder.io_binding()
        # OnnxRuntime will copy the data over to the CUDA device if 'input' is consumed by nodes on the CUDA device

        # load all init states to local variables
        input_ids = self.init_input_ids.copy()
        neighbour_ids = self.init_neighbour_ids.copy()
        neighbour_hidden_states = self.init_neighbour_hidden_states.copy()
        neighbour_attention_mask = self.init_neighbour_attention_mask.copy()
        past_ca_hidden_states = self.init_past_ca_hidden_states.copy()
        past_ca_attention_mask = self.init_past_ca_attention_mask.copy()
        past_key_value = [self.init_past_key_value[i].copy() for i in range(len(self.init_past_key_value))]

        iter_count = 0
        t_start_all = time.time()

        # io_binding_decoder.bind_cpu_input('input_ids', input_ids)
        ortvalue_input_ids = ort.OrtValue.ortvalue_from_numpy(input_ids, device_type='cuda', device_id=0)
        io_binding_decoder.bind_input(name='input_ids', device_type='cuda', device_id=0, element_type=np.int64, shape=ortvalue_input_ids.shape(), buffer_ptr=ortvalue_input_ids.data_ptr())
        ortvalue_neighbour_hidden_states = ort.OrtValue.ortvalue_from_numpy(neighbour_hidden_states, device_type='cuda', device_id=0)
        io_binding_decoder.bind_input(name='neighbour_hidden_states', device_type='cuda', device_id=0, element_type=self.float_type, shape=ortvalue_neighbour_hidden_states.shape(), buffer_ptr=ortvalue_neighbour_hidden_states.data_ptr())
        ortvalue_neighbour_attention_mask = ort.OrtValue.ortvalue_from_numpy(neighbour_attention_mask, device_type='cuda', device_id=0)
        io_binding_decoder.bind_input(name='neighbour_attention_mask', device_type='cuda', device_id=0, element_type=self.float_type, shape=ortvalue_neighbour_attention_mask.shape(), buffer_ptr=ortvalue_neighbour_attention_mask.data_ptr())
        ortvalue_past_ca_hidden_states = ort.OrtValue.ortvalue_from_numpy(past_ca_hidden_states, device_type='cuda', device_id=0)
        io_binding_decoder.bind_input(name='past_ca_hidden_states', device_type='cuda', device_id=0, element_type=self.float_type, shape=ortvalue_past_ca_hidden_states.shape(), buffer_ptr=ortvalue_past_ca_hidden_states.data_ptr())
        ortvalue_past_key_value = [ort.OrtValue.ortvalue_from_numpy(past_key_value[i], device_type='cuda', device_id=0) for i in range(len(past_key_value))]
        for i in range(len(past_key_value)):
            io_binding_decoder.bind_input(name=f'past_key_value.{i}', device_type='cuda', device_id=0, element_type=self.float_type, shape=ortvalue_past_key_value[i].shape(), buffer_ptr=ortvalue_past_key_value[i].data_ptr())

        io_binding_decoder.bind_output('outputs', 'cuda', element_type=self.float_type)
        for i in range(len(past_key_value)):
            # io_binding_decoder.bind_ortvalue_output(f'out_key_values.{i}', ortvalue_past_key_value[i]) # different shape, cannot bind the same input
            io_binding_decoder.bind_output(f'out_key_values.{i}', 'cuda', element_type=self.float_type)
        io_binding_decoder.bind_ortvalue_output('out_ca_hidden_states', ortvalue_past_ca_hidden_states)
        ortvalue_past_ca_attention_mask = ort.OrtValue.ortvalue_from_numpy(past_ca_attention_mask, device_type='cuda', device_id=0)
        io_binding_decoder.bind_ortvalue_output('out_ca_attention_mask', ortvalue_past_ca_attention_mask)

        time_per_step = []

        if staleness:
            first_retrieval = True 
            has_unconsumed_retrieval = False
            response = None
            while len(input_ids[0]) < self.max_gen_len:

                print("iter_count:", iter_count)

                start = time.time()

                loop = asyncio.get_event_loop()
                input_ids =  await loop.run_in_executor(None, self.invoke_decoder_with_binding, input_ids, self.logits_processor, self.ort_sess_decoder, io_binding_decoder)
                await loop.run_in_executor(None, self.update_decoder_binding_after_decoding, input_ids, len(past_key_value), io_binding_decoder)

                # print("\n-- Generation of this iteration complete --")
                # print(self.tokenizer.decode(input_ids[0]))
                iter_count += 1

                if len(input_ids[0]) % self.interval == 0 and len(input_ids[0]) >= self.chunk_size:
                    
                    # last chunk of input_ids
                    query = []
                    for b in range(self.batch_size):
                        query.append(self.tokenizer.decode(input_ids[b][-self.chunk_size:]))
                    seq_len = len(query[0])

                    if first_retrieval:
                        # print("First retrieval")
                        if use_perf_model:
                            nprobe_perf_model_first_iter = 32
                            nprobe_request = self.stub.SetNprobe(retrieval_pb2.SetNprobeRequest(nprobe=nprobe_perf_model_first_iter))
                            nprobe_reply = await nprobe_request
                        # first iteration, not use performance model
                        self.request = self.stub.Retrieve(retrieval_pb2.RetrievalRequest(
                            query=query, num_continuation_chunks=self.num_continuation_chunks, num_neighbours=self.num_neighbours, 
                            staleness_offset=self.staleness_offset, seq_len=seq_len, interval=self.interval, use_perf_model=False))
                        first_retrieval = False
                        response = await self.request
                        has_unconsumed_retrieval = False
                        retrieved_tokens = response.retrieved_tokens
                        neighbour_ids = np.array(retrieved_tokens).reshape((self.batch_size, 1, self.num_neighbours, (1 + self.num_continuation_chunks) * self.chunk_size))
                        await loop.run_in_executor(None, self.update_encoder_binding_after_decoding, io_binding_encoder, neighbour_ids, io_binding_decoder, 
                            ortvalue_neighbour_hidden_states, ortvalue_neighbour_attention_mask)
                        await loop.run_in_executor(None, self.invoke_encoder_with_binding, self.ort_sess_encoder, io_binding_encoder)
                        await loop.run_in_executor(None, self.update_decoder_binding_after_encoding, io_binding_decoder, io_binding_encoder)
                    else:
                        # print("Continuation retrieval")
                        if has_unconsumed_retrieval:
                            # print("Using unconsumed retrieval")
                            response = await self.request
                            retrieved_tokens = response.retrieved_tokens
                            neighbour_ids = np.array(retrieved_tokens).reshape((self.batch_size, 1, self.num_neighbours, (1 + self.num_continuation_chunks) * self.chunk_size))
                            await loop.run_in_executor(None, self.update_encoder_binding_after_decoding, io_binding_encoder, neighbour_ids, io_binding_decoder, 
                                ortvalue_neighbour_hidden_states, ortvalue_neighbour_attention_mask)
                            await loop.run_in_executor(None, self.invoke_encoder_with_binding, self.ort_sess_encoder, io_binding_encoder)
                            await loop.run_in_executor(None, self.update_decoder_binding_after_encoding, io_binding_decoder, io_binding_encoder)
                        else: # use the staled encoder states
                            pass 
                        
                        if len(input_ids[0]) < self.max_gen_len - 1:
                            # print("Sending new retrieval")
                            self.request = self.stub.Retrieve(retrieval_pb2.RetrievalRequest(
                                query=query, num_continuation_chunks=self.num_continuation_chunks, num_neighbours=self.num_neighbours, 
                                staleness_offset=self.staleness_offset, seq_len=seq_len, interval=self.interval, use_perf_model=use_perf_model))
                            has_unconsumed_retrieval = True
                    
                end = time.time()
                print("iteration time: {} ms".format((end - start) * 1000))
                time_per_step.append(end - start)
        else: # no staleness

            assert use_perf_model == False, "No staleness: cannot set use_perf_model as True"

            while len(input_ids[0]) < self.max_gen_len:

                print("iter_count:", iter_count)

                start = time.time()

                loop = asyncio.get_event_loop()
                input_ids =  await loop.run_in_executor(None, self.invoke_decoder_with_binding, input_ids, self.logits_processor, self.ort_sess_decoder, io_binding_decoder)
                await loop.run_in_executor(None, self.update_decoder_binding_after_decoding, input_ids, len(past_key_value), io_binding_decoder)

                # print("\n-- Generation of this iteration complete --")
                # print(self.tokenizer.decode(input_ids[0]))
                iter_count += 1

                if len(input_ids[0]) % self.interval == 0 and len(input_ids[0]) >= self.chunk_size:
                    
                    if not one_retrieval or (one_retrieval and len(input_ids[0]) == self.chunk_size):
                        # last chunk of input_ids
                        query = []
                        for b in range(self.batch_size):
                            query.append(self.tokenizer.decode(input_ids[b][-self.chunk_size:]))
                        seq_len = len(query[0])

                        self.request = self.stub.Retrieve(retrieval_pb2.RetrievalRequest(
                            query=query, num_continuation_chunks=self.num_continuation_chunks, num_neighbours=self.num_neighbours, 
                            staleness_offset=self.staleness_offset, seq_len=seq_len, interval=self.interval, use_perf_model=False))
                        response = await self.request
                        retrieved_tokens = response.retrieved_tokens
                        neighbour_ids = np.array(retrieved_tokens).reshape((self.batch_size, 1, self.num_neighbours, (1 + self.num_continuation_chunks) * self.chunk_size))
                        await loop.run_in_executor(None, self.update_encoder_binding_after_decoding, io_binding_encoder, neighbour_ids, io_binding_decoder, 
                            ortvalue_neighbour_hidden_states, ortvalue_neighbour_attention_mask)
                        await loop.run_in_executor(None, self.invoke_encoder_with_binding, self.ort_sess_encoder, io_binding_encoder)
                        await loop.run_in_executor(None, self.update_decoder_binding_after_encoding, io_binding_decoder, io_binding_encoder)

                end = time.time()
                print("iteration time: {} ms".format((end - start) * 1000))
                time_per_step.append(end - start)

        t_end_all = time.time()
        print(self.tokenizer.decode(input_ids[0]))
        print("Total time elapsed: {} ms".format((t_end_all - t_start_all) * 1000))
        print("output length: ", len(input_ids[0]))
        print("output shape: ", input_ids.shape)

        return time_per_step


    @torch.no_grad() 
    def invoke_encoder_with_binding(self, ort_sess_encoder, io_binding_encoder):

        # start = time.time()
        ort_sess_encoder.run_with_iobinding(io_binding_encoder)
        # end = time.time()
        # print("encoder kernel time: {:.2f} ms".format((end - start) * 1000))

    @torch.no_grad() # no grad improves latency from e.g, 33 -> 27 ms
    def invoke_decoder_with_binding(self, input_ids : np.ndarray, 
        logits_processor, ort_sess_decoder, io_binding_decoder) -> np.ndarray:

        
        # start = time.time()
        ort_sess_decoder.run_with_iobinding(io_binding_decoder)
        # outputs = io_binding_decoder.copy_outputs_to_cpu()[0]
        outputs = io_binding_decoder.get_outputs()[0].numpy() # io_binding_decoder.get_outputs() has a length of 25
        # end = time.time()
        # print("decoder kernel time: {:.2f} ms".format((end - start) * 1000))

        outputs = torch.tensor(outputs)

        input_ids = torch.tensor(input_ids)
        # next_token_logits = outputs.logits[:, -1, :]
        next_token_logits = outputs[:, -1, :]

        next_tokens_scores = logits_processor(input_ids, next_token_logits)

        next_tokens = torch.argmax(next_tokens_scores, dim=-1)
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        input_ids = input_ids.detach().cpu().numpy()

        return input_ids

    def update_decoder_binding_after_decoding(self, input_ids : np.ndarray, past_key_value_len : int, io_binding_decoder : ort.InferenceSession.io_binding) -> None:
        # updates next iteration inputs
        io_binding_decoder.bind_cpu_input('input_ids', input_ids)
        ortvalue_past_key_value_updated = io_binding_decoder.get_outputs()[1:1 + 24] # io_binding_decoder.get_outputs() has a length of 25, is a list of OrtValue objects
        ortvalue_past_ca_hidden_states_updated = io_binding_decoder.get_outputs()[-2]
        ortvalue_past_ca_attention_mask_updated = io_binding_decoder.get_outputs()[-1]
        for i in range(past_key_value_len):
            io_binding_decoder.bind_ortvalue_input(f'past_key_value.{i}', ortvalue_past_key_value_updated[i])
        io_binding_decoder.bind_ortvalue_input('past_ca_hidden_states', io_binding_decoder.get_outputs()[-2])

        io_binding_decoder.bind_output('outputs', 'cuda', element_type=self.float_type)
        for i in range(past_key_value_len):
            # io_binding_decoder.bind_ortvalue_output(f'out_key_values.{i}', ortvalue_past_key_value_updated[i])  # different shape, cannot bind the same input
            io_binding_decoder.bind_output(f'out_key_values.{i}', 'cuda', element_type=self.float_type)
        io_binding_decoder.bind_ortvalue_output('out_ca_hidden_states', ortvalue_past_ca_hidden_states_updated)
        io_binding_decoder.bind_ortvalue_output('out_ca_attention_mask', ortvalue_past_ca_attention_mask_updated)

    def update_encoder_binding_after_decoding(self, io_binding_encoder : ort.InferenceSession.io_binding, neighbour_ids : np.ndarray, io_binding_decoder : ort.InferenceSession.io_binding,
        ortvalue_neighbour_hidden_states : ort.OrtValue, ortvalue_neighbour_attention_mask : ort.OrtValue) -> None:
        # update encoder inputs, bind outputs
        io_binding_encoder.bind_cpu_input('neighbour_ids', neighbour_ids)
        io_binding_encoder.bind_ortvalue_input('past_ca_hidden_states', io_binding_decoder.get_outputs()[-2])
        io_binding_encoder.bind_ortvalue_input('past_ca_attention_mask', io_binding_decoder.get_outputs()[-1])
        io_binding_encoder.bind_ortvalue_output('neighbour_hidden_states', ortvalue_neighbour_hidden_states)
        io_binding_encoder.bind_ortvalue_output('neighbour_attention_mask', ortvalue_neighbour_attention_mask)

    def update_decoder_binding_after_encoding(self, io_binding_decoder : ort.InferenceSession.io_binding,
        io_binding_encoder : ort.InferenceSession.io_binding) -> None:   
        # updates decoder inputs
        io_binding_decoder.bind_ortvalue_input('neighbour_hidden_states', io_binding_encoder.get_outputs()[0])
        io_binding_decoder.bind_ortvalue_input('neighbour_attention_mask', io_binding_encoder.get_outputs()[1])
        
    def create_channel(self):
        self.channel = grpc.aio.insecure_channel(f"{self.host}:{self.port}") 
        self.stub = retrieval_pb2_grpc.RetrievalServiceStub(self.channel)

if __name__ == '__main__':
    
    import argparse 
    parser = argparse.ArgumentParser()

    parser.add_argument('--host', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=50051)
    parser.add_argument('--checkpoint', type=Path, default=Path('../data/model/model.ckpt'))
    parser.add_argument('--retro_config', type=Path, default=Path('../data/model/retro.json'))
    parser.add_argument('--encoder_dir', type=Path, default=Path('../src/onnx_retro_encoder/retro_encoder.onnx'))
    parser.add_argument('--decoder_dir', type=Path, default=Path('../src/onnx_retro_decoder/retro_decoder.onnx'))
    # three modes: generation, local_generation, and profile_local_generation)
    parser.add_argument('--mode', type=str, default="generation", choices=["generation", "local_generation", "profile_local_generation"], help="generation, local_generation, or profile_local_generation")

    parser.add_argument('--batch_size', type=int, default=1, help="batch size of the query")
    parser.add_argument('--staleness', type=int, default=1, help="whether to use staleness generation")
    parser.add_argument('--chunk_size', type=int, default=64, help="chunk size")
    parser.add_argument('--num_continuation_chunks', type=int, default=1, help="number of continuation chunks")
    parser.add_argument('--num_neighbours', type=int, default=1, help="number of neighbours")
    # parser.add_argument('--staleness_offset', type=int, default=0, help="staleness offset")
    parser.add_argument('--interval', type=int, default=64, help="interval")
    parser.add_argument('--max_seq_len', type=int, default=1024, help="maximum sequence length")

    # if mode == profile_local_generation
    parser.add_argument('--num_runs', type=int, default=5, help="number of runs")
    parser.add_argument('--perf_file', type=Path, default=Path('./performance/performance_generation.pickle'), help="performance file")


    args = parser.parse_args()

    staleness_offset = args.interval # currently, staleness_offset == interval, but in the future can be decoupled to be different

    retriever = InferenceClient(host=args.host, port=args.port,
        checkpoint=args.checkpoint, retro_config=args.retro_config,
        encoder_dir=args.encoder_dir, decoder_dir=args.decoder_dir,
        chunk_size=args.chunk_size, prompt="Retrieval augmentation", batch_size=args.batch_size, max_gen_len=args.max_seq_len,
        num_neighbours=args.num_neighbours, num_continuation_chunks=args.num_continuation_chunks,
        staleness_offset=staleness_offset, interval=args.interval
        )

    if args.mode == "generation":
        # Note! asyncio.run() will run everything, must wrap all the logic including channel and stub creating into a single 
        #  top-level function; cannot decouple channel creation, request send, and wait request in several components
        asyncio.run(retriever.generation(staleness=args.staleness))
    elif args.mode == "local_generation":
        # # Normal ONNX inference without retrieval
        decoder_time_per_step, encoder_time_per_step = retriever.generation_without_retrieval()
    elif args.mode == "profile_local_generation":
        retriever.profile_local_generation(args.num_runs, args.perf_file)
