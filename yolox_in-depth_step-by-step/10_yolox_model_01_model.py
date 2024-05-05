
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

# from yolox.models.darknet import CSPDarknet

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
# backbone:  YOLO-PAFPN
# ------------------------------------------------------------------------------------------

# YOLOv3 model. Darknet 53 is the default backbone of this model.

in_channels = [256, 512, 1024]
in_features=("dark3", "dark4", "dark5")

backbone = YOLOPAFPN(
    depth=exp.depth,
    width=exp.width,
    in_features=in_features,
    in_channels=in_channels,
    act=exp.act
    )


# ----------
for (module_name, module) in backbone.named_modules():
    # print(module_name, module)
    print(module_name)


# ------------------------------------------------------------------------------------------
# head:  YOLOXHead
# ------------------------------------------------------------------------------------------

## strides determines feature map H,W and its varieties
## if set 3 strides, 3 variation of feature map

strides = [8, 16, 32]
# strides = [32]

# depthwise True reduce num of parameters
depthwise = False
# depthwise = True

head = YOLOXHead(
    num_classes=exp.num_classes,
    width=exp.width,
    strides=strides,
    in_channels=in_channels,
    act=exp.act,
    depthwise=depthwise
    )


# ----------
for (module_name, module) in head.named_modules():
    # print(module_name, module)
    print(module_name)


# ------------------------------------------------------------------------------------------
# model
# ------------------------------------------------------------------------------------------

model = YOLOX(backbone, head)


for (module_name, module) in model.named_modules():
    # print(module_name, module)
    print(module_name)


# ----------
# (height, width)
print(exp.input_size)

batch_size = 1

summary(
    model,
    input_size=(batch_size, 3, exp.input_size[0], exp.input_size[1]),
    col_names=['input_size', 'output_size', 'num_params', 'kernel_size'],
)

