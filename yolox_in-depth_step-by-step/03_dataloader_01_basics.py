
import os
import sys
import importlib

import random
import numpy as np

import cv2
import PIL.Image

from torch.utils.data.dataloader import DataLoader as torchDataLoader


from yolox.utils import *
from yolox.data import COCODataset, TrainTransform


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
# torchDataLoader

batch_size = 8

data_loader = torchDataLoader(
    dataset=dataset,
    batch_size=batch_size,
)


# ----------
img, target, img_info, img_id = next(iter(data_loader))


# ----------
# (batch_size, 3, input_size[0], input_size[1])
print(img.shape)

# (batch_size, max_labels, 5)
print(target.shape)

# height (annotation)
print(img_info[0])

# width (annotation)
print(img_info[1])

# image id
print(img_id)


############################################################################################
# ------------------------------------------------------------------------------------------
# exp.get_data_loader
# yolox.exp : yolox_base.py  Exp(BaseExp).get_data_loader()
# ------------------------------------------------------------------------------------------

# ----------
# exp.get_data_loader()
batch_size = 8

is_distributed = get_world_size() > 1

# ----------
# only resized
# no_aug = True

# hsv, flip, resized, padded
no_aug = False
# ----------

train_loader = exp.get_data_loader(
    batch_size=batch_size,
    is_distributed=is_distributed,
    no_aug=no_aug,
    cache_img=None,
)

print(dir(train_loader))


# ----------
img, target, img_info, img_id = next(iter(data_loader))


# ----------
# (batch_size, 3, input_size[0], input_size[1])
print(img.shape)

# (batch_size, max_labels, 5)
print(target.shape)

# bounding box coordinate is (cx, cy, w, h)
print(target[0][0])

# height
print(img_info[0])

# width
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

