
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

from yolox.models.yolo_pafpn import YOLOPAFPN
from yolox.models.yolo_head import YOLOXHead
from yolox.models.yolox import YOLOX

from yolox.models.darknet import CSPDarknet

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
# YOLO-PAFPN (backbone) step by step
# ------------------------------------------------------------------------------------------

# CSPDarknet
out_features=("dark3", "dark4", "dark5")
depthwise = False

backbone = CSPDarknet(dep_mul=exp.depth, wid_mul=exp.width, out_features=out_features, act=exp.act)

batch_size = 1

summary(
    backbone,
    input_size=(batch_size, 3, exp.input_size[0], exp.input_size[1]),
    col_names=['input_size', 'output_size', 'num_params', 'kernel_size'],
)

# ----------
Conv = DWConv if depthwise else BaseConv

upsample = nn.Upsample(scale_factor=2, mode="nearest")

# ----------
in_channels = [256, 512, 1024]

lateral_conv0 = BaseConv(
    in_channels=int(in_channels[2] * exp.width),
    out_channels=int(in_channels[1] * exp.width),
    ksize=1,
    stride=1,
    groups=1,
    bias=False,
    act=exp.act
)

reduce_conv1 = BaseConv(
    in_channels=int(in_channels[1] * exp.width),
    out_channels=int(in_channels[0] * exp.width),
    ksize=1,
    stride=1,
    groups=1,
    bias=False,
    act=exp.act
)

# bottom-up
# here Conv is BaseConv (since depthwise = False)
bu_conv1 = Conv(
    in_channels=int(in_channels[1] * exp.width),
    out_channels=int(in_channels[1] * exp.width),
    ksize=3,
    stride=2,
    groups=1,
    bias=False,
    act=exp.act
)

bu_conv2 = Conv(
    in_channels=int(in_channels[0] * exp.width),
    out_channels=int(in_channels[0] * exp.width),
    ksize=3,
    stride=2,
    groups=1,
    bias=False,
    act=exp.act
)


# ----------
# in: 256 --> out: 128
C3_p3 = CSPLayer(
    in_channels=int(2 * in_channels[0] * exp.width),
    out_channels=int(in_channels[0] * exp.width),
    n=round(3 * exp.depth),
    shortcut=False,
    expansion=0.5,
    depthwise=depthwise,
    act=exp.act,
)

# in: 512 --> out: 256
C3_p4 = CSPLayer(
    in_channels=int(2 * in_channels[1] * exp.width),
    out_channels=int(in_channels[1] * exp.width),
    n=round(3 * exp.depth),
    shortcut=False,
    expansion=0.5,
    depthwise=depthwise,
    act=exp.act,
)

# in: 256 --> out: 256
C3_n3 = CSPLayer(
    in_channels=int(2 * in_channels[0] * exp.width),
    out_channels=int(in_channels[1] * exp.width),
    n=round(3 * exp.depth),
    shortcut=False,
    expansion=0.5,
    depthwise=depthwise,
    act=exp.act,
)

# in: 512 --> out: 512
C3_n4 = CSPLayer(
    in_channels=int(2 * in_channels[1] * exp.width),
    out_channels=int(in_channels[2] * exp.width),
    n=round(3 * exp.depth),
    shortcut=False,
    expansion=0.5,
    depthwise=depthwise,
    act=exp.act,
)

# ----------
# to device
backbone.to(device)
upsample.to(device)
reduce_conv1.to(device)
bu_conv1.to(device)
bu_conv2.to(device)
C3_p3.to(device)
C3_p4.to(device)
C3_n3.to(device)
C3_n4.to(device)


# ----------
data_type = torch.float32
input = (torch.rand((8, 3, 640, 640)) * 255).to(torch.int).to(data_type)
input = input.to(device)
# input.requires_grad = False
print(input.shape)

# ----------
out_features = backbone(input)

in_features = out_features
# features = [out_features[f] for f in in_features]
features = [out_features[i] for i, f in enumerate(in_features)]

# dark3, dark4, dark5
[x2, x1, x0] = features


# ----------
# 1. start from x0
# --> lateral + upsample + add x1 + C3_p4 + reduce  -->  fpn_out1
fpn_out0 = lateral_conv0(x0)
# (8, 512, 20, 20)
print(x0.shape)
# (8, 256, 20, 20)
print(fpn_out0.shape)

f_out0 = upsample(fpn_out0)
# (8, 256, 40, 40)
print(f_out0.shape)

f_out0 = torch.cat([f_out0, x1], 1)
# (8, 512, 40, 40)
print(f_out0.shape)

f_out0 = C3_p4(f_out0)
# (8, 256, 40, 40)
print(f_out0.shape)

fpn_out1 = reduce_conv1(f_out0)
# (8, 128, 40, 40)
print(fpn_out1.shape)


# ----------
# 2. upsample + add x2 + C3_p3  -->  pan_out2
f_out1 = upsample(fpn_out1)
# (8, 128, 80, 80)
print(f_out1.shape)

f_out1 = torch.cat([f_out1, x2], 1)
# (8, 256, 80, 80)
print(f_out1.shape)

pan_out2 = C3_p3(f_out1)
# (8, 128, 80, 80)
print(pan_out2.shape)


# ----------
# 3. bottom up + add fpn_out1 + C3_n3  -->  pan_out1
p_out1 = bu_conv2(pan_out2)
# (8, 128, 40, 40)
print(p_out1.shape)

p_out1 = torch.cat([p_out1, fpn_out1], 1)
# (8, 256, 40, 40)
print(p_out1.shape)

pan_out1 = C3_n3(p_out1)
# (8, 256, 40, 40)
print(pan_out1.shape)


# ----------
# 4. bottom up + add fpn_out0 + C3_n4  -->  pan_out0
p_out0 = bu_conv1(pan_out1)
# (8, 256, 20, 20)
print(p_out0.shape)

p_out0 = torch.cat([p_out0, fpn_out0], 1)
# (8, 512, 20, 20)
print(p_out0.shape)

pan_out0 = C3_n4(p_out0)
# (8, 512, 20, 20)
print(p_out0.shape)


# ----------
outputs = (pan_out2, pan_out1, pan_out0)

