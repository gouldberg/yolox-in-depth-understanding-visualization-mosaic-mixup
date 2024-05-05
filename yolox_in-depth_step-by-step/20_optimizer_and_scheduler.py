
import os

import torch
import torch.nn as nn


# ------------------------------------------------------------------------------------------
# yolox.exp yolox_base.py
# get_optimizer()
# ------------------------------------------------------------------------------------------

def get_optimizer(self, batch_size):
    if "optimizer" not in self.__dict__:
        if self.warmup_epochs > 0:
            lr = self.warmup_lr
        else:
            lr = self.basic_lr_per_img * batch_size

        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups

        for k, v in self.model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)  # biases
            if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                pg0.append(v.weight)  # no decay
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)  # apply decay

        optimizer = torch.optim.SGD(
            pg0, lr=lr, momentum=self.momentum, nesterov=True
        )
        optimizer.add_param_group(
            {"params": pg1, "weight_decay": self.weight_decay}
        )  # add pg1 with weight_decay
        optimizer.add_param_group({"params": pg2})
        self.optimizer = optimizer

    return self.optimizer


# ------------------------------------------------------------------------------------------
# check:
# get_optimizer()
# ------------------------------------------------------------------------------------------

from yolox.models.yolo_pafpn import YOLOPAFPN
from yolox.models.yolo_head import YOLOXHead
from yolox.models.yolox import YOLOX


# factor of model depth
depth = 1.00

# factor of model width
width = 1.00

# activation name. For example, if using "relu", then "silu" will be replaced to "relu".
act = "silu"

in_channels = [256, 512, 1024]
num_classes = 80

backbone = YOLOPAFPN(depth, width, in_channels=in_channels, act=act)

head = YOLOXHead(num_classes, width, in_channels=in_channels, act=act)

model = YOLOX(backbone, head)


# ----------
# optimizer parameter groups
pg0, pg1, pg2 = [], [], []

for k, v in model.named_modules():
    if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
        pg2.append(v.bias)  # biases
    if isinstance(v, nn.BatchNorm2d) or "bn" in k:
        pg0.append(v.weight)  # no decay
    elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
        pg1.append(v.weight)  # apply decay


# ----------
# nn.BatchNorm2d
print(len(pg0))
print(pg0[0])

# weight
print(len(pg1))
print(pg1[0].shape)

# bias
print(len(pg2))
print(pg2[0].shape)


# ----------
# learning rate for one image. During training, lr will multiply batchsize.
basic_lr_per_img = 0.01 / 64.0

batch_size = 32
lr = basic_lr_per_img * batch_size

print(f'learing rate: {lr}')

# momentum of optimizer
momentum = 0.9


# ----------
optimizer = torch.optim.SGD(pg0, lr=lr, momentum=momentum, nesterov=True)

# only Parameter Group 0
print(optimizer)


# ----------
# weight decay of optimizer
weight_decay = 5e-4

# add pg1 with weight_decay
optimizer.add_param_group(
    {"params": pg1, "weight_decay": weight_decay}
)

# add pg2
optimizer.add_param_group({"params": pg2})


# ----------
print(optimizer)

for i in range(3):
    pg = optimizer.param_groups[i]
    print({j: pg[j] for j in list(pg.keys())[1:]})


# ------------------------------------------------------------------------------------------
# yolox.exp yolox_base.py
# get_lr_scheduler()
# ------------------------------------------------------------------------------------------

import math
from functools import partial

def get_lr_scheduler(self, lr, iters_per_epoch):
    from yolox.utils import LRScheduler

    scheduler = LRScheduler(
        self.scheduler,
        lr,
        iters_per_epoch,
        self.max_epoch,
        warmup_epochs=self.warmup_epochs,
        warmup_lr_start=self.warmup_lr,
        no_aug_epochs=self.no_aug_epochs,
        min_lr_ratio=self.min_lr_ratio,
    )
    return scheduler


def cos_lr(lr, total_iters, iters):
    """Cosine learning rate"""
    lr *= 0.5 * (1.0 + math.cos(math.pi * iters / total_iters))
    return lr


# ----------
# check functools/partial

lr = 0.005
total_iters = 300

lr_func = partial(cos_lr, lr, total_iters)

for iters in range(0, 300, 10):
    print(f'iters: {iters} - {cos_lr(lr=lr, total_iters=total_iters, iters=iters)}')
    print(f'iters: {iters} - {lr_func(iters)}')


# ----------
scheduler = "yoloxwarmcos"
