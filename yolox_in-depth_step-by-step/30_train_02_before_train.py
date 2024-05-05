
import os
from copy import deepcopy

import torch
import torch.nn as nn

from yolox.utils import *

from yolox.models.yolo_pafpn import YOLOPAFPN
from yolox.models.yolo_head import YOLOXHead
from yolox.models.yolox import YOLOX

# THOP: PyTorch-OpCounter
# https://github.com/Lyken17/pytorch-OpCounter
from thop import profile


from torchinfo import summary



############################################################################################
# ------------------------------------------------------------------------------------------
# used methods
# ------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------
# yolox.exp : yolo_base.py
# Exp.get_model()
# ------------------------------------------------------------------------------------------

def get_model(self):
    from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead

    def init_yolo(M):
        for m in M.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03

    if getattr(self, "model", None) is None:
        in_channels = [256, 512, 1024]
        backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, act=self.act)
        head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, act=self.act)
        self.model = YOLOX(backbone, head)

    self.model.apply(init_yolo)
    self.model.head.initialize_biases(1e-2)
    self.model.train()
    return self.model


# ------------------------------------------------------------------------------------------
# yolox.utils : model_utils.py
# get_model_info()
# ------------------------------------------------------------------------------------------

def get_model_info(model: nn.Module, tsize: Sequence[int]) -> str:
    from thop import profile

    stride = 64
    img = torch.zeros((1, 3, stride, stride), device=next(model.parameters()).device)
    flops, params = profile(deepcopy(model), inputs=(img,), verbose=False)
    params /= 1e6
    flops /= 1e9
    flops *= tsize[0] * tsize[1] / stride / stride * 2  # Gflops
    info = "Params: {:.2f}M, Gflops: {:.2f}".format(params, flops)
    return info


# ------------------------------------------------------------------------------------------
# yolox.core : trainer.py
# resume_train()
# ------------------------------------------------------------------------------------------

def resume_train(self, model):
    if self.args.resume:
        logger.info("resume training")
        if self.args.ckpt is None:
            ckpt_file = os.path.join(self.file_name, "latest" + "_ckpt.pth")
        else:
            ckpt_file = self.args.ckpt

        ckpt = torch.load(ckpt_file, map_location=self.device)
        # resume the model/optimizer state dict
        model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.best_ap = ckpt.pop("best_ap", 0)
        # resume the training states variables
        start_epoch = (
            self.args.start_epoch - 1
            if self.args.start_epoch is not None
            else ckpt["start_epoch"]
        )
        self.start_epoch = start_epoch
        logger.info(
            "loaded checkpoint '{}' (epoch {})".format(
                self.args.resume, self.start_epoch
            )
        )  # noqa
    else:
        if self.args.ckpt is not None:
            logger.info("loading checkpoint for fine tuning")
            ckpt_file = self.args.ckpt
            ckpt = torch.load(ckpt_file, map_location=self.device)["model"]
            model = load_ckpt(model, ckpt)
        self.start_epoch = 0

    return model


############################################################################################
# ------------------------------------------------------------------------------------------
# yolox.core : trainer.py
# in before_train()
# ------------------------------------------------------------------------------------------

# ----------
# THIS IS THE PART OF ORIGINAL SCRIPT
# ----------
# torch.cuda.set_device(self.local_rank)
# model = self.exp.get_model()
# logger.info(
#     "Model Summary: {}".format(get_model_info(model, self.exp.test_size))
# )
# model.to(self.device)
#
# value of epoch will be set in `resume_train`
# model = self.resume_train(model)
# ----------

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


##################
# model = exp.get_model()
# ---------->
# factor of model depth
depth = 1.00

# factor of model width
width = 1.00

# activation name. For example, if using "relu", then "silu" will be replaced to "relu".
act = "silu"

num_classes = 80
in_channels = [256, 512, 1024]

backbone = YOLOPAFPN(depth, width, in_channels=in_channels, act=act)

head = YOLOXHead(num_classes, width, in_channels=in_channels, act=act)

model = YOLOX(backbone, head)

def init_yolo(M):
    for m in M.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eps = 1e-3
            m.momentum = 0.03

model.apply(init_yolo)

model.head.initialize_biases(1e-2)

model.train()
##################


##################
# get_model_info(model, self.exp.test_size)
# ---------->

print(next(model.parameters()))
print(next(model.parameters()).shape)


# ----------
stride = 64

img = torch.zeros((1, 3, stride, stride), device=next(model.parameters()).device)

print(img.shape)

flops, params = profile(deepcopy(model), inputs=(img,), verbose=True)
# flops, params = profile(deepcopy(model), inputs=(img,), verbose=False)

print(f'flops : {flops}   params : {params}')


# output image size during evaluation/test
tsize = (640, 640)

params /= 1e6
flops /= 1e9
flops *= tsize[0] * tsize[1] / stride / stride * 2  # Gflops


info = "Params: {:.2f}M, Gflops: {:.2f}".format(params, flops)

print(img.shape)
print(info)
##################


model.to(device)


##################
# model = self.resume_train(model)
# ---------->

file_name = '/home/kswada/kw/yolox/YOLOX/YOLOX_outputs/yolox_s/'

ckpt_file = os.path.join(file_name, "latest" + "_ckpt.pth")

ckpt = torch.load(ckpt_file, map_location=device)

for k, v in ckpt.items():
    print(k)

# ----------
# resume the model/optimizer state dict

model.load_state_dict(ckpt["model"])

# optimizer.load_state_dict(ckpt["optimizer"])
