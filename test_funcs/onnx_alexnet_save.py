# https://pytorch.org/docs/2.0/onnx.html

import torch
import torchvision

import time

dummy_input = torch.randn(10, 3, 224, 224, device="cuda")
model = torchvision.models.alexnet(pretrained=True).cuda()

# Providing input and output names sets the display names for values
# within the model's graph. Setting these does not change the semantics
# of the graph; it is only for readability.
#
# The inputs to the network consist of the flat list of inputs (i.e.
# the values you would pass to the forward() method) followed by the
# flat list of parameters. You can partially specify names, i.e. provide
# a list here shorter than the number of inputs to the model, and we will
# only set that subset of names, starting from the beginning.
input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
output_names = [ "output1" ]

torch.onnx.export(model, dummy_input, "alexnet.onnx", verbose=True, input_names=input_names, output_names=output_names)

n_run = 100
print("Running {} times".format(n_run))
time_ms_array = []
for i in range(n_run):
    start = time.time()
    outputs = model(dummy_input)
    end = time.time()
    time_ms_array.append((end - start) * 1000)
    print("Time: {} ms".format(time_ms_array[-1]))
# remove the first run
time_ms_array = time_ms_array[1:]
print("Average time: {} ms".format(sum(time_ms_array) / len(time_ms_array)))
print("Done")