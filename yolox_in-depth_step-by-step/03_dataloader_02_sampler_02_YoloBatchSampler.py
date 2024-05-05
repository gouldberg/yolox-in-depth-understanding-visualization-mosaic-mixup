
import os
import sys
import importlib
import random

import cv2
import PIL.Image

from yolox.utils import *
from yolox.data import COCODataset, TrainTransform
from yolox.data.datasets import *

from yolox.data import InfiniteSampler, YoloBatchSampler


############################################################################################
# ------------------------------------------------------------------------------------------
# COCODataset
# yolox.data.datasets : coco.py  COCODataset
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


# ----------
# COCODataset
max_labels = 50

dataset = COCODataset(
    # data_dir=exp.data_dir,
    json_file=exp.train_ann,
    img_size=exp.input_size,
    preproc=TrainTransform(
        max_labels=max_labels,
        flip_prob=exp.flip_prob,
        hsv_prob=exp.hsv_prob
        ),
        cache=False,
        cache_type='ram',
    )


# ------------------------------------------------------------------------------------------
# MosaicDetection
# yolox.exp : yolox_base.py  get_data_loader() MosaicDetection
#  --> yolox.data.datasets : mosaicdetection.py  class MosaicDetection(Dataset)
# ------------------------------------------------------------------------------------------

no_aug = False

input_size = exp.input_size

dataset_md = MosaicDetection(
    dataset=dataset,
    mosaic=not no_aug,
    img_size=input_size,
    preproc=TrainTransform(
        max_labels=max_labels,
        flip_prob=exp.flip_prob,
        hsv_prob=exp.hsv_prob),
    degrees=exp.degrees,
    translate=exp.translate,
    mosaic_scale=exp.mosaic_scale,
    mixup_scale=exp.mixup_scale,
    shear=exp.shear,
    enable_mixup=exp.enable_mixup,
    mosaic_prob=exp.mosaic_prob,
    mixup_prob=exp.mixup_prob,
)


# ------------------------------------------------------------------------------------------
# InifiniteSampler + YoloBatchSampler
# yolox.exp : yolox_base.py  get_data_loader()
# ------------------------------------------------------------------------------------------

sampler = InfiniteSampler(size=len(dataset_md), shuffle=True, seed=0, rank=0, world_size=1)

print(next(iter(sampler)))


# ----------
batch_size = 8

batch_sampler = YoloBatchSampler(
    sampler=sampler,
    batch_size=batch_size,
    drop_last=False,
    mosaic=not no_aug,
)

# list of (mosaic(bool), index)
print(next(iter(batch_sampler)))

