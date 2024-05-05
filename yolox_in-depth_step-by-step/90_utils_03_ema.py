
import os

import torch
import torch.nn as nn

import math
from copy import deepcopy


# ------------------------------------------------------------------------------------------
# yolox.utils  ema.py
# is_parallel
# ------------------------------------------------------------------------------------------

def is_parallel(model):
    """check if model is in parallel mode."""
    parallel_type = (
        nn.parallel.DataParallel,
        nn.parallel.DistributedDataParallel,
    )
    return isinstance(model, parallel_type)



base_dir = '/home/kswada/kw/yolox/YOLOX'

# ckpt_file = os.path.join(base_dir, 'YOLOX_outputs/yolox_s/latest_ckpt.pth')
ckpt_file = os.path.join(base_dir, 'YOLOX_outputs/yolox_s/best_ckpt.pth')

device = 'cuda'
ckpt = torch.load(ckpt_file, map_location=device)


# # ----------
# for k, v in ckpt['model'].items():
#     print(k)


# parallel_type = (
#     nn.parallel.DataParallel,
#     nn.parallel.DistributedDataParallel,
# )

# isinstance(ckpt['model'], parallel_type)


# ------------------------------------------------------------------------------------------
# yolox.utils  ema.py
# ModelEMA
# ------------------------------------------------------------------------------------------

class ModelEMA:
    """
    Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, updates=0):
        """
        Args:
            model (nn.Module): model to apply EMA.
            decay (float): ema decay rate.
            updates (int): counter of EMA updates.
        """
        # Create EMA(FP32)
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()
        self.updates = updates
        # decay exponential ramp (to help early epochs)
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = (
                model.module.state_dict() if is_parallel(model) else model.state_dict()
            )  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1.0 - d) * msd[k].detach()


# ----------
ema = deepcopy(model.module if is_parallel(model) else model).eval()

