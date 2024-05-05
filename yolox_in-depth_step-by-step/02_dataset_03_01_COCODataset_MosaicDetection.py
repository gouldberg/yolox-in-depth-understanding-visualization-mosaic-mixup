
import os
import sys
import importlib
import random

import cv2
import PIL.Image

from yolox.utils import *
from yolox.data import COCODataset, TrainTransform
from yolox.data.datasets import MosaicDetection


############################################################################################
# ------------------------------------------------------------------------------------------
# COCODataset + torchDataLoader
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

# loading annotations into memory, creating index
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

# ----------
# 118287
print(len(dataset.annotations))

# annotation
#  - bounxing box is converted from coco (xmin, ymin, width, height)
#     --> (xmin, ymin, xmax, ymax), resized
print(dataset.annotations[0])


# ------------------------------------------------------------------------------------------
# COCODataset + torchDataLoader + 
# yolox.exp : yolox_base.py  get_data_loader() MosaicDetection
#  --> yolox.data.datasets : mosaicdetection.py  class MosaicDetection(Dataset)
# ------------------------------------------------------------------------------------------

### NOTE THAT mosaic + mixup augmentation requires YoloBatchSampler !!!!

no_aug = False

input_size = exp.input_size


# now set transform probs are all 1.0
dataset_md = MosaicDetection(
    dataset=dataset,
    mosaic=True,
    img_size=input_size,
    preproc=TrainTransform(
        max_labels=max_labels,
        flip_prob=1.0,
        hsv_prob=1.0),
    degrees=exp.degrees,
    translate=exp.translate,
    mosaic_scale=exp.mosaic_scale,
    mixup_scale=exp.mixup_scale,
    shear=exp.shear,
    enable_mixup=True,
    mosaic_prob=1.0,
    mixup_prob=1.0,
)

# dataset_md = MosaicDetection(
#     dataset=dataset,
#     mosaic=not no_aug,
#     img_size=input_size,
#     preproc=TrainTransform(
#         max_labels=max_labels,
#         flip_prob=exp.flip_prob,
#         hsv_prob=exp.hsv_prob),
#     degrees=exp.degrees,
#     translate=exp.translate,
#     mosaic_scale=exp.mosaic_scale,
#     mixup_scale=exp.mixup_scale,
#     shear=exp.shear,
#     enable_mixup=exp.enable_mixup,
#     mosaic_prob=exp.mosaic_prob,
#     mixup_prob=exp.mixup_prob,
# )

# -->
# note that this dataset_md (from MosaicDetection) object has no attribute 'pull_item'

print(dir(dataset_md))


# ----------
# mosaic_getitem(idx)
print(len(dataset_md))

idx = random.randint(0, len(dataset_md) - 1)

# original COCODataset
img, labels, img_info0, img_id0 = dataset.pull_item(idx)

# MosaicDetection dataset
mix_img, padded_labels, img_info1, img_id1 = dataset.__getitem__(idx)
# mix_img, padded_labels, img_info1, img_id1 = dataset.mosaic_getitem(idx)


# ----------
print(img.shape)
print(mix_img.shape)


# ----------
# (num of labels, 5)
print(labels.shape)
# (max_labels, 5)
print(padded_labels.shape)


# ----------
print(labels)

# bounding boxes are transformed (flip + padded)
print(padded_labels)


# ----------
# height, width
print(img_id0)
print(img_id1)

print(img_info0)
print(img_info1)


# ----------
tmp_img0 = img.copy()
tmp_img1 = mix_img.transpose(1,2,0).astype('uint8').copy()
print(tmp_img1.shape)

# bounding box coordinate is (xmin, ymin, xmax, ymax)
for i in range(len(labels)):
    tmp_label = labels[i]
    cv2.rectangle(
        img=tmp_img0,
        pt1=(int(tmp_label[0]), int(tmp_label[1])),
        pt2=(int(tmp_label[2]), int(tmp_label[3])),
        color=(255, 0, 0),
        thickness=3
    )


# bounding box coordinate is (cx, cy, w, h)
for i in range(len(padded_labels)):
    tmp_label = padded_labels[i]
    tlx = int(tmp_label[1] - tmp_label[3] / 2)
    tly = int(tmp_label[2] - tmp_label[4] / 2)
    brx = int(tmp_label[1] + tmp_label[3] / 2)
    bry = int(tmp_label[2] + tmp_label[4] / 2)
    if tlx > 0 and tly > 0 and brx > 0 and bry > 0:
        cv2.rectangle(
            img=tmp_img1,
            pt1=(tlx, tly),
            pt2=(brx, bry),
            color=(255, 0, 0),
            thickness=3
        )

PIL.Image.fromarray(cv2.cvtColor(tmp_img0, cv2.COLOR_BGR2RGB)).show()

# NO MOSAIC AUGMENTED
PIL.Image.fromarray(cv2.cvtColor(tmp_img1, cv2.COLOR_BGR2RGB)).show()

