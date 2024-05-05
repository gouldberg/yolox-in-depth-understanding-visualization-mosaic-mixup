
import os
import sys
import importlib

import random
import numpy as np

import cv2
import PIL.Image

from yolox.utils import *
from yolox.data import COCODataset, TrainTransform
# from yolox.data.datasets import *

from yolox.data import *


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
# MosaicDetection dataset
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
# InfinteSampler + YoloBatchSampler
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
    # mosaic=False,
)

# list of (mosaic(bool), index)
print(next(iter(batch_sampler)))


############################################################################################
# ------------------------------------------------------------------------------------------
# dataset + dataloader
# yolox.exp : yolox_base.py  get_data_loader()
# ------------------------------------------------------------------------------------------

# set worker to 4 for shorter dataloader init time
# If your training process cost many memory, reduce this value.
data_num_workers = 4

dataloader_kwargs = {"num_workers": data_num_workers, "pin_memory": True}

dataloader_kwargs["batch_sampler"] = batch_sampler

# Make sure each process has different random seed, especially for 'fork' method.
# Check https://github.com/pytorch/pytorch/issues/63311 for more details.
dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed


# ----------
train_loader = DataLoader(dataset_md, **dataloader_kwargs)


# ----------
print(dir(train_loader))


######################
# ----------
img, target, img_info, img_id = next(iter(train_loader))


# ----------
# (batch_size, 3, input_size[0], input_size[1])
print(img.shape)
# RGB values...
print(np.array(img[0]).transpose(1,2,0).astype('uint8'))

# (batch_size, max_labels, 5)
print(target.shape)

# height (annotation)
print(img_info[0])

# width (annotation)
print(img_info[1])

# image id
print(img_id)


# ----------
for idx in range(batch_size):
    # ----------
    tmp_img = np.array(img[idx]).transpose(1,2,0).astype('uint8').copy()
    tmp_target = np.array(target[idx])
    # ----------
    # print(tmp_img.shape)
    # print(tmp_target.shape)
    # ----------
    for i in range(len(tmp_target)):
        tmp_label = tmp_target[i]
        tlx = max(int(tmp_label[1] - tmp_label[3] / 2), 0)
        tly = max(int(tmp_label[2] - tmp_label[4] / 2), 0)
        brx = max(int(tmp_label[1] + tmp_label[3] / 2), 0)
        bry = max(int(tmp_label[2] + tmp_label[4] / 2), 0)
        if tlx > 0 or tly > 0 or brx > 0 or bry > 0:
            cv2.rectangle(
                img=tmp_img,
                pt1=(tlx, tly),
                pt2=(brx, bry),
                color=(255, 0, 0),
                thickness=2
            )
    # ----------
    PIL.Image.fromarray(cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB)).show()

