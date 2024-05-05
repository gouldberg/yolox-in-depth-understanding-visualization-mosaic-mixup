
import os
import sys
import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchinfo import summary

# import math

from yolox.utils import *
# from yolox.data import COCODataset, TrainTransform
# from yolox.data.datasets import *

from yolox.models.network_blocks import *


############################################################################################
# ------------------------------------------------------------------------------------------
# get experiment configuration
# ------------------------------------------------------------------------------------------

base_dir = '/home/kswada/kw/yolox/YOLOX'

exp_file = os.path.join(base_dir, 'exps/default/yolox_s.py')

sys.path.append(os.path.dirname(exp_file))

current_exp = importlib.import_module(os.path.basename(exp_file).split(".")[0])

print(current_exp)


# ----------
exp = current_exp.Exp()

print(exp)
print(dir(exp))



# ------------------------------------------------------------------------------------------
# setup
# ------------------------------------------------------------------------------------------

is_distributed = get_world_size() > 1

rank = get_rank()

local_rank = get_local_rank()

device = "cuda:{}".format(local_rank)


# ----------
print(f'is_distributed: {is_distributed}')
print(f'rank: {rank}')
print(f'local_rank: {local_rank}')
print(f'device: {device}')



# ----------
torch.cuda.set_device(local_rank)


############################################################################################
# ------------------------------------------------------------------------------------------
# network blocks:  CSPLayer
#  - C3 in yolov5, CSP Bottleneck with 3 convolutions
# ------------------------------------------------------------------------------------------

in_channels = [256, 512, 1024]


# ----------
# C3_p4:  in: 512 --> out: 256
in_channels2 = int(2 * in_channels[1] * exp.width)
out_channels = int(in_channels[1] * exp.width)
n = round(3 * exp.depth)
shortcut = False
expansion = 0.5
depthwise = False
act = exp.act

print(f'n: {n}')


# ----------
# ch_in, ch_out, number, shortcut, groups, expansion
# 128
hidden_channels = int(out_channels * expansion)

# in: 512 --> out: 128
conv1 = BaseConv(in_channels2, hidden_channels, 1, stride=1, act=act)

# in: 512 --> out: 128
conv2 = BaseConv(in_channels2, hidden_channels, 1, stride=1, act=act)

# in: 256 --> out: 256
conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)

# 128, 128
module_list = [
    Bottleneck(
        hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act
    )
    for _ in range(n)
]

m = nn.Sequential(*module_list)


# ----------
# to device
conv1.to(device)
conv2.to(device)
conv3.to(device)
m.to(device)


# ----------
data_type = torch.float32
input = torch.rand((8, 512, 40, 40)).to(data_type)
input = input.to(device)
# input.requires_grad = False
print(input.shape)


# ---------
# (8, 512, 40, 40)
print(input.shape)

x_1 = conv1(input)
# (8, 128, 40, 40)
print(x_1.shape)

x_2 = conv2(input)
# (8, 128, 40, 40)
print(x_2.shape)

x_1 = m(x_1)
# (8, 128, 40, 40)
print(x_1.shape)

x = torch.cat((x_1, x_2), dim=1)
# (8, 256, 40, 40)
print(x.shape)

output = conv3(x)
# (8, 256, 40, 40)
print(output.shape)


############################################################################################
# ------------------------------------------------------------------------------------------
# network blocks:  Focus
#  - Focus width and height information into channel space
#    stem = Focus(3, base_channels, ksize=3, act=act)
#  - This is applied by YOLOv5 to replace the three first layers of the network.
#    It helped reducing the number of parameters, the number of FLOPS and the CUDA memory
#    while improving the speed of the forward and backward passes with minor effects on the mAP (mean Average Precision).
# ------------------------------------------------------------------------------------------

wid_mul = exp.width
dep_mul = exp.depth

base_channels = int(wid_mul * 64)
base_depth = max(round(dep_mul * 3), 1)

print(f'wid_mul: {wid_mul}')
print(f'base_channels: {base_channels}')


# ----------
in_channels = 3
out_channels = base_channels
ksize = 3

conv = BaseConv(in_channels * 4, out_channels, ksize=ksize, stride=1, act=exp.act)


# ----------
# to device
conv.to(device)


# ----------
data_type = torch.float32
input = (torch.rand((8, 3, 640, 640)) * 255).to(torch.int).to(data_type)
input = input.to(device)
# input.requires_grad = False
print(input.shape)


# ----------
x = input

# shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
patch_top_left = x[..., ::2, ::2]
patch_top_right = x[..., ::2, 1::2]
patch_bot_left = x[..., 1::2, ::2]
patch_bot_right = x[..., 1::2, 1::2]

# (8, 3, 640, 640)
print(x.shape)

# (8, 3, 320, 320)
print(patch_top_left.shape)
print(patch_top_right.shape)
print(patch_bot_left.shape)
print(patch_bot_right.shape)

x = torch.cat(
    (
        patch_top_left,
        patch_bot_left,
        patch_top_right,
        patch_bot_right,
    ),
    dim=1,
)

# (8, 12, 320, 320)
print(x.shape)

# ----------
output = conv(x)
# (8, 32, 320, 320)
print(output.shape)


############################################################################################
# ------------------------------------------------------------------------------------------
# darknet:  CSPDarknet
# ------------------------------------------------------------------------------------------

wid_mul = exp.width
dep_mul = exp.depth
depthwise = False
act = exp.act


Conv = DWConv if depthwise else BaseConv

base_channels = int(wid_mul * 64)
base_depth = max(round(dep_mul * 3), 1)

print(f'wid_mul: {wid_mul}')
print(f'base_channels: {base_channels}')

print(f'dep_mul: {dep_mul}')
print(f'base_depth: {base_depth}')


# ----------
# stem
stem = Focus(3, base_channels, ksize=3, act=act)


# dark2
dark2 = nn.Sequential(
    Conv(base_channels, base_channels * 2, 3, 2, act=act),
    CSPLayer(
        base_channels * 2,
        base_channels * 2,
        n=base_depth,
        depthwise=depthwise,
        act=act,
        ),
    )

# dark3
dark3 = nn.Sequential(
    Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
    CSPLayer(
        base_channels * 4,
        base_channels * 4,
        n=base_depth * 3,
        depthwise=depthwise,
        act=act,
    ),
)

# dark4
dark4 = nn.Sequential(
    Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
    CSPLayer(
        base_channels * 8,
        base_channels * 8,
        n=base_depth * 3,
        depthwise=depthwise,
        act=act,
    ),
)

# dark5: this has SPPBottleneck
dark5 = nn.Sequential(
    Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
    SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
    CSPLayer(
        base_channels * 16,
        base_channels * 16,
        n=base_depth,
        shortcut=False,
        depthwise=depthwise,
        act=act,
    ),
)

print(dark3)
print(dark4)
print(dark5)


# ----------
batch_size = 1

summary(
    stem,
    input_size=(batch_size, 3, exp.input_size[0], exp.input_size[1]),
    col_names=['input_size', 'output_size', 'num_params', 'kernel_size'],
)

summary(dark2)
summary(dark3)
summary(dark4)
summary(dark5)


# ----------
# to device
stem.to(device)
dark2.to(device)
dark3.to(device)
dark4.to(device)
dark5.to(device)


# ----------
data_type = torch.float32
input = (torch.rand((8, 3, 640, 640)) * 255).to(torch.int).to(data_type)
input = input.to(device)
# input.requires_grad = False
print(input.shape)


# ----------
outputs = {}

x1 = stem(input)
# (8, 32, 320, 320)
print(x1.shape)

outputs["stem"] = x1

# ----------
x2 = dark2(x1)
# (8, 64, 160, 160)
print(x2.shape)

outputs["dark2"] = x2

# ----------
x3 = dark3(x2)
# (8, 128, 80, 80)
print(x3.shape)

outputs["dark3"] = x3

# ----------
x4 = dark4(x3)
# (8, 256, 40, 40)
print(x4.shape)

outputs["dark4"] = x4


# ----------
x5 = dark5(x4)
# (8, 512, 20, 20)
print(x5.shape)

outputs["dark5"] = x5

out_features = ('dark3', 'dark4', 'dark5')
output = {k: v for k, v in outputs.items() if k in out_features}


# ------------------------------------------------------------------------------------------
# darknet:  only dark5 in CSPDarknet
#   with SPP Bottleneck (in network_blocks.py)
#   Spatial pyramid pooling layer used in YOLOv3-SPP
# ------------------------------------------------------------------------------------------

# (8, 256, 40, 40)  -->  finally to be (8, 512, 20, 20)
print(x4.shape)


# ----------
# 1.  conv
conv0 = Conv(base_channels * 8, base_channels * 16, 3, 2, act=act)


# ----------
# 2.  SPP BottleNeck
in_channels_spp = base_channels * 16
hidden_channels_spp = in_channels_spp // 2

conv1_spp = BaseConv(in_channels_spp, hidden_channels_spp, 1, stride=1, act=exp.act)

kernel_sizes=(5, 9, 13)
m_spp = nn.ModuleList(
    [
        nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
        for ks in kernel_sizes
    ]
)

conv2_channels_spp = hidden_channels_spp * (len(kernel_sizes) + 1)
out_channels_spp = base_channels * 16
conv2_spp = BaseConv(conv2_channels_spp, out_channels_spp, 1, stride=1, act=exp.act)


# ----------
# 3.  CSPLayer
in_channels_csp = base_channels * 16
out_channels_csp = base_channels * 16
n = base_depth
shortcut = False
expansion = 0.5
depthwise = False

hidden_channels_csp = int(out_channels_csp * expansion)

conv1_csp = BaseConv(in_channels_csp, hidden_channels_csp, 1, stride=1, act=exp.act)
conv2_csp = BaseConv(in_channels_csp, hidden_channels_csp, 1, stride=1, act=exp.act)
conv3_csp = BaseConv(2 * hidden_channels_csp, out_channels_csp, 1, stride=1, act=exp.act)

module_list_csp = [
    Bottleneck(
        hidden_channels_csp, hidden_channels_csp, shortcut, 1.0, depthwise, act=act
    )
    for _ in range(n)
]

m_csp = nn.Sequential(*module_list_csp)


# ----------
# to device
conv0.to(device)
conv1_spp.to(device)
m_spp.to(device)
conv2_spp.to(device)
conv1_csp.to(device)
conv2_csp.to(device)
conv3_csp.to(device)
m_csp.to(device)


# ----------
# 4. all through dark5

x4_1 = conv0(x4)
# (8, 256, 40, 40)
print(x4.shape)
# (8, 512, 20, 20)
print(x4_1.shape)


###### SPP
x4_2 = conv1_spp(x4_1)
# (8, 256, 20, 20)
print(x4_2.shape)

x4_3 = torch.cat([x4_2] + [m(x4_2) for m in m_spp], dim=1)
# (8, 1024, 20, 20)
print(x4_3.shape)

x4_4 = conv2_spp(x4_3)
# (8, 512, 20, 20)
print(x4_4.shape)

###### CSPLayer
x4_5 = conv1_csp(x4_4)
# (8, 256, 20, 20)
print(x4_5.shape)

x4_6 = conv2_csp(x4_4)
# (8, 256, 20, 20)
print(x4_6.shape)

x4_7 = m_csp(x4_5)
# (8, 256, 20, 20)
print(x4_7.shape)

x4_8 = torch.cat((x4_6, x4_7), dim=1)
# (8, 512, 20, 20)
print(x4_8.shape)

output = conv3_csp(x4_8)
# (8, 512, 20, 20)
print(output.shape)

