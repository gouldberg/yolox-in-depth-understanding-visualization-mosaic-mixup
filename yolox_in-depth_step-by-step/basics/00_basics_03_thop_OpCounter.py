
import os
import torch

from torchinfo import summary

# REFERENCE:
# https://github.com/Lyken17/pytorch-OpCounter


# ------------------------------------------------------------------------------------------
# basic usage
# ------------------------------------------------------------------------------------------

from torchvision.models import resnet50

from thop import profile
from thop import clever_format


# ----------
model = resnet50()

for (module_name, module) in model.named_modules():
    # print(module_name, module)
    print(module_name)


print(summary(model))


# ----------
# (height, width)
input_size = (640, 640)
# input_size = (320, 320)

batch_size = 1

summary(
    model,
    input_size=(batch_size, 3, input_size[0], input_size[1]),
    col_names=['input_size', 'output_size', 'num_params', 'kernel_size'],
)


# ----------
input = torch.randn(1, 3, 224, 224)
print(input.shape)


# input data is in cpu, so now set model to cpu
model.to('cpu')

macs, params = profile(model, inputs=(input, ))
print(f'macs: {macs}  params: {params}')

macs, params = clever_format([macs, params], "%.3f")
print(f'macs: {macs}  params: {params}')

