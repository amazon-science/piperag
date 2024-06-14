"""
Example Usage: 
python onnx_gpt2_profiling.py

It seems that ONNX is not so compatible with nsys
nsys profile -t cuda,nvtx \
             --capture-range=cudaProfilerApi \
             --capture-range-end=none \
             --backtrace none \
             -s none \
             --show-output=true \
             --force-overwrite=true \
             --export=sqlite,text \
             -o ./traces_gpt2/sample \
python onnx_gpt2_profiling.py

Below is the error message when using nsys:
Traceback (most recent call last):
  File "/data/retro_tobias/test_funcs/onnx_gpt2_profiling.py", line 211, in <module>
    gpt2_test()
  File "/data/retro_tobias/test_funcs/onnx_gpt2_profiling.py", line 180, in gpt2_test
    ort_session = ort.InferenceSession(model_dir, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
  File "/opt/conda/envs/retro/lib/python3.9/site-packages/onnxruntime/capi/onnxruntime_inference_collection.py", line 419, in __init__
    self._create_inference_session(providers, provider_options, disabled_optimizers)
  File "/opt/conda/envs/retro/lib/python3.9/site-packages/onnxruntime/capi/onnxruntime_inference_collection.py", line 463, in _create_inference_session
    sess.initialize_session(providers, provider_options, disabled_optimizers)
onnxruntime.capi.onnxruntime_pybind11_state.RuntimeException: [ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION : Exception during initialization: /onnxruntime_src/onnxruntime/core/providers/cuda/cuda_call.cc:121 std::conditional_t<THRW, void, onnxruntime::common::Status> onnxruntime::CudaCall(ERRTYPE, const char*, const char*, ERRTYPE, const char*, const char*, int) [with ERRTYPE = cudnnStatus_t; bool THRW = true; std::conditional_t<THRW, void, onnxruntime::common::Status> = void] /onnxruntime_src/onnxruntime/core/providers/cuda/cuda_call.cc:114 std::conditional_t<THRW, void, onnxruntime::common::Status> onnxruntime::CudaCall(ERRTYPE, const char*, const char*, ERRTYPE, const char*, const char*, int) [with ERRTYPE = cudnnStatus_t; bool THRW = true; std::conditional_t<THRW, void, onnxruntime::common::Status> = void] CUDNN failure 1: CUDNN_STATUS_NOT_INITIALIZED ; GPU=0 ; hostname=ip-172-31-16-187 ; file=/onnxruntime_src/onnxruntime/core/providers/cuda/cuda_execution_provider.cc ; line=172 ; expr=cudnnCreate(&cudnn_handle_); 

"""

import torch
import onnxruntime as ort
import onnx
from onnx import numpy_helper
from transformers import GPT2Model, GPT2LMHeadModel, GPT2Tokenizer

import time
import numpy as np
import os
from onnxconverter_common import float16

# Transformers has a unified API
# for 8 transformer architectures and 30 pretrained weights.
#          Model          | Tokenizer          | Pretrained weights shortcut          | save_name
MODELS = [
    # (GPT2Model, GPT2Tokenizer, 'gpt2', 'gpt2'),
    # (GPT2LMHeadModel, GPT2Tokenizer, 'gpt2', 'gpt2-lm-head'),
    (GPT2LMHeadModel, GPT2Tokenizer, 'gpt2-medium', 'gpt2-medium-lm-head'),
]
data_dir = 'test_data_set_0'

def gpt2_test():

    input_length = 10

    for model_class, tokenizer_class, pretrained_weights, save_name in MODELS:
        # Load pretrained model/tokenizer
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        model = model_class.from_pretrained(pretrained_weights)
        model.eval()
        # Encode text
        # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
        input_ids_1 = torch.tensor(
            [[tokenizer.encode("Here " * input_length, add_special_tokens=True)]])
        print("input_ids_1.shape: {}".format(input_ids_1.shape))
        with torch.no_grad():
            output_1 = model(input_ids_1)  # Models outputs are now tuples
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        input_ids_1 = input_ids_1.to(device)

        # torch.cuda.cudart().cudaProfilerStart()

        n_run = 100
        print("Running {} times".format(n_run))
        time_ms_array = []
        with torch.no_grad():
            for i in range(n_run):
                # torch.cuda.nvtx.range_push(f"PyTorch run {i}")
                start = time.time()
                output_1 = model(input_ids_1)
                end = time.time()
                # torch.cuda.nvtx.range_pop()
                time_ms_array.append((end - start) * 1000)
                # print("Time: {} ms".format(time_ms_array[-1]))
        time_ms_array = time_ms_array[1:]
        print("Average time with PyTorch: {} ms".format(sum(time_ms_array) / len(time_ms_array)))
        pytorch_time_ms_array = time_ms_array.copy()

        model_dir, data_dir = os.path.join('test_' + save_name, 'model.onnx'), 'test_data_set_0'
        
        # move to cpu, convert to numpy
        input_ids_1 = input_ids_1.cpu().numpy()

        ort_session = ort.InferenceSession(model_dir, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

        io_binding = ort_session.io_binding()
        # OnnxRuntime will copy the data over to the CUDA device if 'input' is consumed by nodes on the CUDA device 
        io_binding.bind_cpu_input('input1', input_ids_1)
        io_binding.bind_output('output1')

        print("Running {} times".format(n_run))
        time_ms_array = []
        for i in range(n_run):
            # torch.cuda.nvtx.range_push(f"ONNX run {i}")
            start = time.time()
            # outputs = ort_session.run(
            #     None,
            #     {"actual_input_1": ortvalue},
            # )
            output_1 = ort_session.run_with_iobinding(io_binding)
            output_1 = io_binding.copy_outputs_to_cpu()[0]
            end = time.time()
            # torch.cuda.nvtx.range_pop()
            time_ms_array.append((end - start) * 1000)
            # print("Time: {} ms".format(time_ms_array[-1]))
        # remove the first run
        time_ms_array = time_ms_array[1:]
        print("Average time with ONNX: {} ms".format(sum(time_ms_array) / len(time_ms_array)))
        onnx_time_ms_array = time_ms_array.copy()

        print("ONNX speedup over PyTorch: {:.2f} x".format(sum(pytorch_time_ms_array) / sum(onnx_time_ms_array)))

        # torch.cuda.cudart().cudaProfilerStop()

        # test fp 16
        model_fp16_dir = os.path.join('test_' + save_name, "model_fp16.onnx")
        if not os.path.exists(model_fp16_dir):
            print("Converting to fp16")
            model = onnx.load(model_dir)
            model_fp16 = float16.convert_float_to_float16(model)
            onnx.save(model_fp16, model_fp16_dir)
        else:
            model_fp16 = onnx.load(model_fp16_dir)
        
        ort_session = ort.InferenceSession(model_fp16_dir, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

        io_binding = ort_session.io_binding()
        # OnnxRuntime will copy the data over to the CUDA device if 'input' is consumed by nodes on the CUDA device 
        io_binding.bind_cpu_input('input1', input_ids_1)
        io_binding.bind_output('output1')

        print("Running {} times".format(n_run))
        time_ms_array = []
        for i in range(n_run):
            # torch.cuda.nvtx.range_push(f"ONNX run {i}")
            start = time.time()
            # outputs = ort_session.run(
            #     None,
            #     {"actual_input_1": ortvalue},
            # )
            output_1 = ort_session.run_with_iobinding(io_binding)
            output_1 = io_binding.copy_outputs_to_cpu()[0]
            end = time.time()
            # torch.cuda.nvtx.range_pop()
            time_ms_array.append((end - start) * 1000)
            # print("Time: {} ms".format(time_ms_array[-1]))
        # remove the first run
        time_ms_array = time_ms_array[1:]
        print("Average time with ONNX fp16: {} ms".format(sum(time_ms_array) / len(time_ms_array)))
        onnx_fp16_time_ms_array = time_ms_array.copy()

        print("ONNX fp16 speedup over PyTorch: {:.2f} x".format(sum(pytorch_time_ms_array) / sum(onnx_fp16_time_ms_array)))
    


gpt2_test()