# https://pytorch.org/docs/2.0/onnx.html
# APIs: https://onnxruntime.ai/docs/api/python/api_summary.html#data-inputs-and-outputs

import onnx
import onnxruntime as ort

import numpy as np
import time

# Load the ONNX model
model = onnx.load("alexnet.onnx")

# Check that the model is well formed
onnx.checker.check_model(model)

# Print a human readable representation of the graph
print(onnx.helper.printable_graph(model.graph))


ort_session = ort.InferenceSession("alexnet.onnx", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
# ort_session = ort.InferenceSession("alexnet.onnx")

X = np.random.randn(10, 3, 224, 224).astype(np.float32)
# ortvalue = ort.OrtValue.ortvalue_from_numpy(X, 'cuda', 0)
ortvalue = ort.OrtValue.ortvalue_from_numpy(X)
ortvalue.device_name()  # 'cpu'
ortvalue.shape()        # shape of the numpy array X
ortvalue.data_type()    # 'tensor(float)'
ortvalue.is_tensor()    # 'True'
np.array_equal(ortvalue.numpy(), X)  # 'True'

io_binding = ort_session.io_binding()
# OnnxRuntime will copy the data over to the CUDA device if 'input' is consumed by nodes on the CUDA device
io_binding.bind_cpu_input('actual_input_1', X)
io_binding.bind_output('output1')


# dummy_input = torch.randn(10, 3, 224, 224, device="cuda")
n_run = 100
print("Running {} times".format(n_run))
time_ms_array = []
for i in range(n_run):
    start = time.time()
    # outputs = ort_session.run(
    #     None,
    #     {"actual_input_1": ortvalue},
    # )
    ort_session.run_with_iobinding(io_binding)
    Y = io_binding.copy_outputs_to_cpu()[0]
    end = time.time()
    time_ms_array.append((end - start) * 1000)
    print("Time: {} ms".format(time_ms_array[-1]))
# remove the first run
time_ms_array = time_ms_array[1:]
print("Average time: {} ms".format(sum(time_ms_array) / len(time_ms_array)))
print("Done")