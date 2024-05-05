
import os
import sys
import importlib

import numpy as np
import random
import math

from yolox.utils import *
from yolox.data import COCODataset, TrainTransform

import cv2
import PIL.Image
import shutil


############################################################################################
# ------------------------------------------------------------------------------------------
# yolox.data.datasets : mosaicdetection.py
# get_mosaic_coordinate
# ------------------------------------------------------------------------------------------

def get_mosaic_coordinate(mosaic_image, mosaic_index, xc, yc, w, h, input_h, input_w):
    # TODO update doc
    # index0 to top left part of image
    if mosaic_index == 0:
        x1, y1, x2, y2 = max(xc - w, 0), max(yc - h, 0), xc, yc
        small_coord = w - (x2 - x1), h - (y2 - y1), w, h
    # index1 to top right part of image
    elif mosaic_index == 1:
        x1, y1, x2, y2 = xc, max(yc - h, 0), min(xc + w, input_w * 2), yc
        small_coord = 0, h - (y2 - y1), min(w, x2 - x1), h
    # index2 to bottom left part of image
    elif mosaic_index == 2:
        x1, y1, x2, y2 = max(xc - w, 0), yc, xc, min(input_h * 2, yc + h)
        small_coord = w - (x2 - x1), 0, w, min(y2 - y1, h)
    # index2 to bottom right part of image
    elif mosaic_index == 3:
        x1, y1, x2, y2 = xc, yc, min(xc + w, input_w * 2), min(input_h * 2, yc + h)  # noqa
        small_coord = 0, 0, min(w, x2 - x1), min(y2 - y1, h)
    return (x1, y1, x2, y2), small_coord


# ------------------------------------------------------------------------------------------
# yolox.data : data_augment.py
# ------------------------------------------------------------------------------------------

def get_aug_params(value, center=0):
    if isinstance(value, float):
        return random.uniform(center - value, center + value)
    elif len(value) == 2:
        return random.uniform(value[0], value[1])
    else:
        raise ValueError(
            "Affine params should be either a sequence containing two values\
             or single float values. Got {}".format(value)
        )

def get_affine_matrix(
    target_size,
    degrees=10,
    translate=0.1,
    scales=0.1,
    shear=10,
):
    twidth, theight = target_size

    # Rotation and Scale
    angle = get_aug_params(degrees)
    scale = get_aug_params(scales, center=1.0)

    if scale <= 0.0:
        raise ValueError("Argument scale should be positive")

    R = cv2.getRotationMatrix2D(angle=angle, center=(0, 0), scale=scale)

    M = np.ones([2, 3])
    # Shear
    shear_x = math.tan(get_aug_params(shear) * math.pi / 180)
    shear_y = math.tan(get_aug_params(shear) * math.pi / 180)

    M[0] = R[0] + shear_y * R[1]
    M[1] = R[1] + shear_x * R[0]

    # Translation
    translation_x = get_aug_params(translate) * twidth  # x translation (pixels)
    translation_y = get_aug_params(translate) * theight  # y translation (pixels)
    M[0, 2] = translation_x
    M[1, 2] = translation_y
    
    return M, scale


def apply_affine_to_bboxes(targets, target_size, M, scale):
    num_gts = len(targets)
    # warp corner points
    twidth, theight = target_size
    corner_points = np.ones((4 * num_gts, 3))
    corner_points[:, :2] = targets[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
        4 * num_gts, 2
    )  # x1y1, x2y2, x1y2, x2y1
    corner_points = corner_points @ M.T  # apply affine transform
    corner_points = corner_points.reshape(num_gts, 8)
    # create new boxes
    corner_xs = corner_points[:, 0::2]
    corner_ys = corner_points[:, 1::2]
    new_bboxes = (
        np.concatenate(
            (corner_xs.min(1), corner_ys.min(1), corner_xs.max(1), corner_ys.max(1))
        )
        .reshape(4, num_gts)
        .T
    )
    # clip boxes
    new_bboxes[:, 0::2] = new_bboxes[:, 0::2].clip(0, twidth)
    new_bboxes[:, 1::2] = new_bboxes[:, 1::2].clip(0, theight)
    targets[:, :4] = new_bboxes
    return targets


def random_affine(
    img,
    targets=(),
    target_size=(640, 640),
    degrees=10,
    translate=0.1,
    scales=0.1,
    shear=10,
):
    ###### M, scale = get_affine_matrix(target_size, degrees, translate, scales, shear)
    # ---------->
    twidth, theight = target_size
    # ----------
    # Rotation and Scale
    angle = get_aug_params(degrees)
    scale = get_aug_params(scales, center=1.0)
    # ----------
    if scale <= 0.0:
        raise ValueError("Argument scale should be positive")
    # ----------
    R = cv2.getRotationMatrix2D(angle=angle, center=(0, 0), scale=scale)
    # ----------
    M = np.ones([2, 3])
    # Shear
    shear_x = math.tan(get_aug_params(shear) * math.pi / 180)
    shear_y = math.tan(get_aug_params(shear) * math.pi / 180)
    # ----------
    M[0] = R[0] + shear_y * R[1]
    M[1] = R[1] + shear_x * R[0]
    # ----------
    # Translation
    translation_x = get_aug_params(translate) * twidth  # x translation (pixels)
    translation_y = get_aug_params(translate) * theight  # y translation (pixels)
    M[0, 2] = translation_x
    M[1, 2] = translation_y
    # ----------
    img = cv2.warpAffine(img, M, dsize=target_size, borderValue=(114, 114, 114))
    # Transform label coordinates
    if len(targets) > 0:
        targets = apply_affine_to_bboxes(targets, target_size, M, scale)
    return img, targets


# ----------
target_size = (640, 640)
degrees = 0.0
translate = 0.0
scales = (1.0, 1.0)
shear = 0.0

# M, scale = get_affine_matrix(
#     target_size=target_size,
#     degrees=degrees,
#     translate=translate,
#     scales=scales,
#     shear=shear
#     )


############################################################################################
# ------------------------------------------------------------------------------------------
# yolox.data.datasets : mosaicdetection.py
# class MosaicDetection(DataSets).mixup()
# ------------------------------------------------------------------------------------------

def mixup(self, origin_img, origin_labels, input_dim):
    jit_factor = random.uniform(*self.mixup_scale)
    FLIP = random.uniform(0, 1) > 0.5
    cp_labels = []
    while len(cp_labels) == 0:
        cp_index = random.randint(0, self.__len__() - 1)
        cp_labels = self._dataset.load_anno(cp_index)
    img, cp_labels, _, _ = self._dataset.pull_item(cp_index)

    if len(img.shape) == 3:
        cp_img = np.ones((input_dim[0], input_dim[1], 3), dtype=np.uint8) * 114
    else:
        cp_img = np.ones(input_dim, dtype=np.uint8) * 114

    cp_scale_ratio = min(input_dim[0] / img.shape[0], input_dim[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * cp_scale_ratio), int(img.shape[0] * cp_scale_ratio)),
        interpolation=cv2.INTER_LINEAR,
    )

    cp_img[
        : int(img.shape[0] * cp_scale_ratio), : int(img.shape[1] * cp_scale_ratio)
    ] = resized_img

    cp_img = cv2.resize(
        cp_img,
        (int(cp_img.shape[1] * jit_factor), int(cp_img.shape[0] * jit_factor)),
    )
    cp_scale_ratio *= jit_factor

    if FLIP:
        cp_img = cp_img[:, ::-1, :]

    origin_h, origin_w = cp_img.shape[:2]
    target_h, target_w = origin_img.shape[:2]
    padded_img = np.zeros(
        (max(origin_h, target_h), max(origin_w, target_w), 3), dtype=np.uint8
    )
    padded_img[:origin_h, :origin_w] = cp_img

    x_offset, y_offset = 0, 0
    if padded_img.shape[0] > target_h:
        y_offset = random.randint(0, padded_img.shape[0] - target_h - 1)
    if padded_img.shape[1] > target_w:
        x_offset = random.randint(0, padded_img.shape[1] - target_w - 1)
    padded_cropped_img = padded_img[
        y_offset: y_offset + target_h, x_offset: x_offset + target_w
    ]

    cp_bboxes_origin_np = adjust_box_anns(
        cp_labels[:, :4].copy(), cp_scale_ratio, 0, 0, origin_w, origin_h
    )
    if FLIP:
        cp_bboxes_origin_np[:, 0::2] = (
            origin_w - cp_bboxes_origin_np[:, 0::2][:, ::-1]
        )
    cp_bboxes_transformed_np = cp_bboxes_origin_np.copy()
    cp_bboxes_transformed_np[:, 0::2] = np.clip(
        cp_bboxes_transformed_np[:, 0::2] - x_offset, 0, target_w
    )
    cp_bboxes_transformed_np[:, 1::2] = np.clip(
        cp_bboxes_transformed_np[:, 1::2] - y_offset, 0, target_h
    )

    cls_labels = cp_labels[:, 4:5].copy()
    box_labels = cp_bboxes_transformed_np
    labels = np.hstack((box_labels, cls_labels))
    origin_labels = np.vstack((origin_labels, labels))
    origin_img = origin_img.astype(np.float32)
    origin_img = 0.5 * origin_img + 0.5 * padded_cropped_img.astype(np.float32)

    return origin_img.astype(np.uint8), origin_labels


############################################################################################
# ------------------------------------------------------------------------------------------
# get and set experiment
# ------------------------------------------------------------------------------------------

base_dir = '/home/kswada/kw/yolox/YOLOX'

# experiment base module
exp_base_module = 'exps.default.yolox_s'
current_exp = importlib.import_module(exp_base_module)
exp = current_exp.Exp()


# ----------
exp.depth = 0.33
exp.width = 0.50
# ----------
exp.input_size = (640, 640)
# exp.input_size = (480, 480)
# exp.input_size = (800, 800)
exp.multiscale_range = 5
# ----------
############################
# MIXUP
# apply mixup aug or not  -->  but mosaic is required
exp.enable_mixup = True
# exp.enable_mixup = False
exp.mixup_prob = 1.0
exp.mixup_scale = (0.5, 1.5)
# exp.mixup_scale = (0.8, 1.2)
############################
# MOSAIC, random affine  (yolox.data.datasets : mosaicdetection.py)
exp.enable_mosaic = True
# exp.enable_mosaic = False
exp.mosaic_prob = 1.0
# -----
# random affine:
# -----
exp.mosaic_scale = (0.1, 2)
# exp.mosaic_scale = (0.8, 1.2)
# -----
# rotation angle range, for example, if set to 2, the true range is (-2, 2)
exp.degrees = 10.0
# exp.degrees = 0.0
# -----
# translate range, for example, if set to 0.1, the true range is (-0.1, 0.1)
exp.translate = 0.1
# exp.translate = 0.00
# -----
# shear angle range, for example, if set to 2, the true range is (-2, 2)
exp.shear = 2.0
# exp.shear = 0.0

# ----------
exp.seed = 42

# ----------
max_labels = 50


# ----------

# save path
save_dir_vis_mosaicmixup = os.path.join(base_dir, 'YOLOX_outputs/vis_mosaicmixup_640640_default')
# save_dir_vis_mosaicmixup = os.path.join(base_dir, 'YOLOX_outputs/vis_mosaicmixup_640640_revise')

num_images = 10
num_randomaffine = 5


# ----------
print(exp)
print(dir(exp))


############################################################################################
# ------------------------------------------------------------------------------------------
# dataset and preprocess
# ------------------------------------------------------------------------------------------

# COCODataset
target_size = (exp.input_size[0], exp.input_size[1])

dataset = COCODataset(
    # data_dir=exp.data_dir,
    json_file=exp.train_ann,
    img_size=target_size,
    preproc=TrainTransform(
        max_labels=max_labels,
        flip_prob=exp.flip_prob,
        hsv_prob=exp.hsv_prob
        ),
        cache=False,
        cache_type='ram',
    )


# preprocess
preproc=TrainTransform(
    max_labels=max_labels,
    flip_prob=exp.flip_prob,
    hsv_prob=exp.hsv_prob)


############################################################################################
# ------------------------------------------------------------------------------------------
# yolox.data.datasets : mosaicdetection.py
# class MosaicDetection(DataSets)
# when call:
#   @Dataset.mosaic_getitem
#   def __getitem__(self, idx):
# ------------------------------------------------------------------------------------------

if os.path.exists(save_dir_vis_mosaicmixup):
    shutil.rmtree(save_dir_vis_mosaicmixup)

os.makedirs(save_dir_vis_mosaicmixup, exist_ok=True)


# ----------
input_dim = (exp.input_size[0], exp.input_size[1], 3)
input_h, input_w = input_dim[0], input_dim[1]

for i_ in range(num_images):
    # ------------------------------------------------------------------------------------------
    # mosaic center is randomly calculated
    # ------------------------------------------------------------------------------------------
    # mosaic center is randomly calculated:
    yc = int(random.uniform(0.5 * input_h, 1.5 * input_h))
    xc = int(random.uniform(0.5 * input_w, 1.5 * input_w))
    print(f'xc : {xc}   yc : {yc}')
    # ------------------------------------------------------------------------------------------
    # 3 additional images are randomly selected
    # ------------------------------------------------------------------------------------------
    idx = random.randint(0, len(dataset) - 1)
    indices = [idx] + [random.randint(0, len(dataset) - 1) for _ in range(3)]
    # print(f'selected image: {idx}')
    # print(f'including additional images: {indices}')
    # ------------------------------------------------------------------------------------------
    # combine 3 images to generate 1 mosaic image
    # ------------------------------------------------------------------------------------------
    mosaic_labels = []
    for i_mosaic, index in enumerate(indices):
        # i_mosaic = 0
        # index = indices[i_mosaic]
        # ----------
        # original image and label
        img_orig, _labels, _, img_id = dataset.pull_item(index)
        tmp_img = img_orig.copy()
        for i in range(_labels.shape[0]):
            tmp_label = _labels[i]
            tmp_img = cv2.rectangle(
                img=tmp_img,
                pt1=(int(tmp_label[0]), int(tmp_label[1])),
                pt2=(int(tmp_label[2]), int(tmp_label[3])),
                color=(255, 0, 0),
                thickness=2
            )
        save_name = str(i_).zfill(3) + "_" + str(i_mosaic) + "_" + str(index).zfill(6) + "_0_original.png"
        cv2.imwrite(os.path.join(save_dir_vis_mosaicmixup, save_name), img_orig)
        del img_id, tmp_img, tmp_label
        # ----------
        # original h, w
        h0, w0 = img_orig.shape[:2]
        scale = min(1. * input_h / h0, 1. * input_w / w0)
        # ----------
        img_resized = cv2.resize(
            img_orig, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR
        )
        # ----------
        # generate output mosaic image
        (h, w, c) = img_resized.shape[:3]
        # print((h, w, c))
        # ----------
        # width and height is two-times !!
        if i_mosaic == 0:
            mosaic_img = np.full((input_h * 2, input_w * 2, c), 114, dtype=np.uint8)
        # ----------
        # suffix l means large image, while s means small image in mosaic aug.
        (l_x1, l_y1, l_x2, l_y2), (s_x1, s_y1, s_x2, s_y2) = get_mosaic_coordinate(
            mosaic_img, i_mosaic, xc, yc, w, h, input_h, input_w
        )
        # ----------
        mosaic_img[l_y1:l_y2, l_x1:l_x2] = img_resized[s_y1:s_y2, s_x1:s_x2]
        # print((l_x1, l_y1, l_x2, l_y2))
        # # border coordinate is cropped
        # print((s_x1, s_y1, s_x2, s_y2))
        # PIL.Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).show()
        # PIL.Image.fromarray(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)).show()
        # PIL.Image.fromarray(cv2.cvtColor(img_resized[s_y1:s_y2, s_x1:s_x2], cv2.COLOR_BGR2RGB)).show()
        # PIL.Image.fromarray(cv2.cvtColor(mosaic_img, cv2.COLOR_BGR2RGB)).show()
        # ----------
        padw, padh = l_x1 - s_x1, l_y1 - s_y1
        labels = _labels.copy()
        # Normalized xywh to pixel xyxy format
        if _labels.size > 0:
            labels[:, 0] = scale * _labels[:, 0] + padw
            labels[:, 1] = scale * _labels[:, 1] + padh
            labels[:, 2] = scale * _labels[:, 2] + padw
            labels[:, 3] = scale * _labels[:, 3] + padh
        # ----------
        mosaic_labels.append(labels)
        # ----------
        # print(f'image id : {index}')
        # print(f'len labels : {len(_labels)}')
        # print(f'scale : {scale}')
        # print(f'original labels : {_labels}')
        # print(f'changed labels : {labels}')
    # # ----------
    # print(len(mosaic_labels))
    # print(mosaic_labels[0].shape)
    # print(mosaic_labels[1].shape)
    # print(mosaic_labels[2].shape)
    # print(mosaic_labels[3].shape)
    # # ----------
    # print(mosaic_labels)
    # print(mosaic_labels[:, 1])
    # print(np.clip(mosaic_labels[:, 1], 0, 2 * input_h))
    # ----------
    if len(mosaic_labels):
        mosaic_labels = np.concatenate(mosaic_labels, 0)
        np.clip(mosaic_labels[:, 0], 0, 2 * input_w, out=mosaic_labels[:, 0])
        np.clip(mosaic_labels[:, 1], 0, 2 * input_h, out=mosaic_labels[:, 1])
        np.clip(mosaic_labels[:, 2], 0, 2 * input_w, out=mosaic_labels[:, 2])
        np.clip(mosaic_labels[:, 3], 0, 2 * input_h, out=mosaic_labels[:, 3])
    # print(len(mosaic_labels))
    # print(f'mosaic labels: {mosaic_labels.shape}')
    # ----------
    # show bbox in mosaic_img
    tmp_img = mosaic_img.copy()
    for i in range(mosaic_labels.shape[0]):
        tmp_label = mosaic_labels[i]
        tmp_img = cv2.rectangle(
            img=tmp_img,
            pt1=(int(tmp_label[0]), int(tmp_label[1])),
            pt2=(int(tmp_label[2]), int(tmp_label[3])),
            color=(255, 0, 0),
            thickness=2
        )
    # PIL.Image.fromarray(cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB)).show()
    # PIL.Image.fromarray(cv2.cvtColor(mosaic_img, cv2.COLOR_BGR2RGB)).show()
    save_name = str(i_).zfill(3) + "_" + str(indices[0]).zfill(6) + "_1_mosaic.png"
    cv2.imwrite(os.path.join(save_dir_vis_mosaicmixup, save_name), tmp_img)
    del tmp_img, tmp_label
    # ------------------------------------------------------------------------------------------
    # random affine is applied
    # ------------------------------------------------------------------------------------------
    for j_ in range(num_randomaffine):
        mosaic_img2, mosaic_labels2 = random_affine(
            mosaic_img.copy(),
            mosaic_labels.copy(),
            target_size=(input_w, input_h),
            degrees=exp.degrees,
            translate=exp.translate,
            scales=exp.mosaic_scale,
            shear=exp.shear,
        )
        # print(mosaic_img.shape)
        # print(mosaic_labels.shape)
        # show bbox in mosaic_img
        tmp_img = mosaic_img2.copy()
        for i in range(mosaic_labels2.shape[0]):
            tmp_label = mosaic_labels2[i]
            tmp_img = cv2.rectangle(
                img=tmp_img,
                pt1=(int(tmp_label[0]), int(tmp_label[1])),
                pt2=(int(tmp_label[2]), int(tmp_label[3])),
                color=(255, 0, 0),
                thickness=2
            )
        save_name = str(i_).zfill(3) + "_" + str(indices[0]).zfill(6) + "_2_rdmaffine_" + str(j_).zfill(2) + ".png"
        cv2.imwrite(os.path.join(save_dir_vis_mosaicmixup, save_name), tmp_img)
        del tmp_img, tmp_label
        # ------------------------------------------------------------------------------------------
        # yolox.data.datasets : mosaicdetection.py
        # class MosaicDetection(DataSets)
        # CopyPaste: https://arxiv.org/abs/2012.07177
        # ------------------------------------------------------------------------------------------
        origin_img = mosaic_img2.copy()
        origin_labels = mosaic_labels2
        jit_factor = random.uniform(*exp.mixup_scale)
        FLIP = random.uniform(0, 1) > 0.5
        # print(f'jit_factor : {jit_factor}')
        # print(f'FLIP : {FLIP}')
        # ----------
        # get image and labels to be mix-up
        cp_labels = []
        while len(cp_labels) == 0:
            cp_index = random.randint(0, len(dataset) - 1)
            cp_labels = dataset.load_anno(cp_index)
        img_cp, cp_labels, _, _ = dataset.pull_item(cp_index)
        tmp_img = img_cp.copy()
        for i in range(cp_labels.shape[0]):
            tmp_label = cp_labels[i]
            tmp_img = cv2.rectangle(
                img=tmp_img,
                pt1=(int(tmp_label[0]), int(tmp_label[1])),
                pt2=(int(tmp_label[2]), int(tmp_label[3])),
                color=(255, 0, 0),
                thickness=2
            )
        save_name = str(i_).zfill(3) + "_" + str(indices[0]).zfill(6) + "_2_rdmaffine_" + str(j_).zfill(2) + "_0_mixupcopy.png"
        cv2.imwrite(os.path.join(save_dir_vis_mosaicmixup, save_name), tmp_img)
        del tmp_img, tmp_label
        # ----------
        # resize image for mix-up
        if len(img_cp.shape) == 3:
            cp_img = np.ones((input_dim[0], input_dim[1], 3), dtype=np.uint8) * 114
        else:
            cp_img = np.ones(input_dim, dtype=np.uint8) * 114
        cp_scale_ratio = min(input_dim[0] / img_cp.shape[0], input_dim[1] / img_cp.shape[1])
        resized_img = cv2.resize(
            img_cp,
            (int(img_cp.shape[1] * cp_scale_ratio), int(img_cp.shape[0] * cp_scale_ratio)),
            interpolation=cv2.INTER_LINEAR,
        )
        cp_img[
            : int(img_cp.shape[0] * cp_scale_ratio), : int(img_cp.shape[1] * cp_scale_ratio)
        ] = resized_img
        # ----------
        # resize image for mix-up by jitter factor
        cp_img = cv2.resize(
            cp_img,
            (int(cp_img.shape[1] * jit_factor), int(cp_img.shape[0] * jit_factor)),
        )
        cp_scale_ratio *= jit_factor
        # and flip
        if FLIP:
            cp_img = cp_img[:, ::-1, :]
        # print(cp_img.shape)
        # PIL.Image.fromarray(cv2.cvtColor(cp_img, cv2.COLOR_BGR2RGB)).show()
        # ----------
        origin_h, origin_w = cp_img.shape[:2]
        target_h, target_w = origin_img.shape[:2]
        padded_img = np.zeros(
            (max(origin_h, target_h), max(origin_w, target_w), 3), dtype=np.uint8
        )
        padded_img[:origin_h, :origin_w] = cp_img
        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h - 1)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w - 1)
        padded_cropped_img = padded_img[
            y_offset: y_offset + target_h, x_offset: x_offset + target_w
        ]
        # print(padded_img.shape)
        # (640, 640, 3)
        # print(padded_cropped_img.shape)
        # PIL.Image.fromarray(cv2.cvtColor(padded_cropped_img, cv2.COLOR_BGR2RGB)).show()
        # ----------
        cp_bboxes_origin_np = adjust_box_anns(
            cp_labels[:, :4].copy(), cp_scale_ratio, 0, 0, origin_w, origin_h
        )
        if FLIP:
            cp_bboxes_origin_np[:, 0::2] = (
                origin_w - cp_bboxes_origin_np[:, 0::2][:, ::-1]
            )
        cp_bboxes_transformed_np = cp_bboxes_origin_np.copy()
        cp_bboxes_transformed_np[:, 0::2] = np.clip(
            cp_bboxes_transformed_np[:, 0::2] - x_offset, 0, target_w
        )
        cp_bboxes_transformed_np[:, 1::2] = np.clip(
            cp_bboxes_transformed_np[:, 1::2] - y_offset, 0, target_h
        )
        # print(cp_labels[0])
        # print(cp_bboxes_origin_np[0])
        # print(cp_bboxes_transformed_np[0])
        # tmp_label = cp_bboxes_transformed_np[2]
        # tmp_img0 = cv2.rectangle(
        #     img=padded_cropped_img,
        #     pt1=(int(tmp_label[0]), int(tmp_label[1])),
        #     pt2=(int(tmp_label[2]), int(tmp_label[3])),
        #     color=(0,0,0)
        # )
        # PIL.Image.fromarray(cv2.cvtColor(tmp_img0, cv2.COLOR_BGR2RGB)).show()
        # ----------
        cls_labels = cp_labels[:, 4:5].copy()
        box_labels = cp_bboxes_transformed_np
        labels = np.hstack((box_labels, cls_labels))
        origin_labels = np.vstack((origin_labels, labels))
        origin_img = origin_img.astype(np.float32)
        # blending with 0.5, 0.5
        origin_img = 0.5 * origin_img + 0.5 * padded_cropped_img.astype(np.float32)
        # ----------
        mosaic_mixup_img = origin_img.astype(np.uint8)
        mosaic_mixup_labels = origin_labels
        # PIL.Image.fromarray(cv2.cvtColor(mosaic_img, cv2.COLOR_BGR2RGB)).show()
        # PIL.Image.fromarray(cv2.cvtColor(mosaic_mixup_img, cv2.COLOR_BGR2RGB)).show()
        # ----------
        # labels are increased by added mix-up images
        # print(mosaic_labels.shape)
        # print(mosaic_mixup_labels.shape)
        # print(mosaic_labels[0])
        # print(mosaic_mixup_labels[0])
        tmp_img = mosaic_mixup_img.copy()
        for i in range(mosaic_mixup_labels.shape[0]):
            tmp_label = mosaic_mixup_labels[i]
            tmp_img = cv2.rectangle(
                img=tmp_img,
                pt1=(int(tmp_label[0]), int(tmp_label[1])),
                pt2=(int(tmp_label[2]), int(tmp_label[3])),
                color=(255, 0, 0),
                thickness=2
            )
        save_name = str(i_).zfill(3) + "_" + str(indices[0]).zfill(6) + "_2_rdmaffine_" + str(j_).zfill(2) + "_1_mixup" + ".png"
        cv2.imwrite(os.path.join(save_dir_vis_mosaicmixup, save_name), tmp_img)
        del tmp_img, tmp_label
        ############################################################################################
        # ------------------------------------------------------------------------------------------
        # preprocess in training
        # ------------------------------------------------------------------------------------------
        mix_img, padded_labels = preproc(mosaic_mixup_img, mosaic_mixup_labels, input_dim)
        tmp_img = mix_img.copy()
        for i in range(padded_labels.shape[0]):
            tmp_label = padded_labels[i]
            tmp_img = cv2.rectangle(
                img=tmp_img,
                pt1=(int(tmp_label[0]), int(tmp_label[1])),
                pt2=(int(tmp_label[2]), int(tmp_label[3])),
                color=(255, 0, 0),
                thickness=2
            )
        save_name = str(i_).zfill(3) + "_" + str(indices[0]).zfill(6) + "_2_rdmaffine_" + str(j_).zfill(2) + "_2_preproc" + ".png"
        cv2.imwrite(os.path.join(save_dir_vis_mosaicmixup, save_name), tmp_img)
        del tmp_img, tmp_label



# img_info = (mix_img.shape[1], mix_img.shape[0])

# # ----------
# # img_info and img_id are not used for training.
# # They are also hard to be specified on a mosaic image.

# return mix_img, padded_labels, img_info, img_id





############################################################################################
# ------------------------------------------------------------------------------------------
# check:
# get_mosaic_coordinate
# ------------------------------------------------------------------------------------------

def get_mosaic_coordinate(mosaic_image, mosaic_index, xc, yc, w, h, input_h, input_w):
    # TODO update doc
    # index0 to top left part of image
    if mosaic_index == 0:
        x1, y1, x2, y2 = max(xc - w, 0), max(yc - h, 0), xc, yc
        small_coord = w - (x2 - x1), h - (y2 - y1), w, h
    # index1 to top right part of image
    elif mosaic_index == 1:
        x1, y1, x2, y2 = xc, max(yc - h, 0), min(xc + w, input_w * 2), yc
        small_coord = 0, h - (y2 - y1), min(w, x2 - x1), h
    # index2 to bottom left part of image
    elif mosaic_index == 2:
        x1, y1, x2, y2 = max(xc - w, 0), yc, xc, min(input_h * 2, yc + h)
        small_coord = w - (x2 - x1), 0, w, min(y2 - y1, h)
    # index2 to bottom right part of image
    elif mosaic_index == 3:
        x1, y1, x2, y2 = xc, yc, min(xc + w, input_w * 2), min(input_h * 2, yc + h)  # noqa
        small_coord = 0, 0, min(w, x2 - x1), min(y2 - y1, h)
    return (x1, y1, x2, y2), small_coord


# ----------
input_h = 640
input_w = 640

# mosaic center
yc = int(random.uniform(0.5 * input_h, 1.5 * input_h))
xc = int(random.uniform(0.5 * input_w, 1.5 * input_w))
mosaic_img = np.full((input_h * 2, input_w * 2, c), 114, dtype=np.uint8)
print(mosaic_img.shape)

# image original shape
h0 = 1080
w0 = 1920
scale = min(1. * input_h / h0, 1. * input_w / w0)

# resized shape
h = int(h0 * scale)
w = int(w0 * scale)


# coordinate in mosaic: index = 0
x1, y1, x2, y2 = max(xc - w, 0), max(yc - h, 0), xc, yc
small_coord = w - (x2 - x1), h - (y2 - y1), w, h

# coordinate in mosaic: index = 1
x1, y1, x2, y2 = xc, max(yc - h, 0), min(xc + w, input_w * 2), yc
small_coord = 0, h - (y2 - y1), min(w, x2 - x1), h

# coordinate in mosaic: index = 2
x1, y1, x2, y2 = max(xc - w, 0), yc, xc, min(input_h * 2, yc + h)
small_coord = w - (x2 - x1), 0, w, min(y2 - y1, h)

# coordinate in mosaic: index = 3
x1, y1, x2, y2 = xc, yc, min(xc + w, input_w * 2), min(input_h * 2, yc + h)  # noqa
small_coord = 0, 0, min(w, x2 - x1), min(y2 - y1, h)


print(f'mosaic center: {(xc, yc)}')
print(f'resized: {(w, h)}')
print(f'mosaic coord: {(x1, y1, x2, y2)}')
print(f'small coord: {(small_coord)}')


# (l_x1, l_y1, l_x2, l_y2), (s_x1, s_y1, s_x2, s_y2) = get_mosaic_coordinate(
#     mosaic_img, i_mosaic, xc, yc, w, h, input_h, input_w
# )
# # ----------
# mosaic_img[l_y1:l_y2, l_x1:l_x2] = img_resized[s_y1:s_y2, s_x1:s_x2]


############################################################################################
# ------------------------------------------------------------------------------------------
# check:
# resize: multiscale range
# yolox.core.trainer  -->  yolox.exp.yolox_base.py
# ------------------------------------------------------------------------------------------

import torch

input_size_original = (640, 640)
multiscale_range = 5

for _ in range(10):
    tensor = torch.LongTensor(2).cuda()
    size_factor = input_size_original[1] * 1.0 / input_size_original[0]
    min_size = int(input_size_original[0] / 32) - multiscale_range
    max_size = int(input_size_original[0] / 32) + multiscale_range
    random_size = (min_size, max_size)
    size = random.randint(*random_size)
    size = (int(32 * size), 32 * int(size * size_factor))
    tensor[0] = size[0]
    tensor[1] = size[1]
    input_size = (tensor[0].item(), tensor[1].item())
    print(f'{_} : {input_size}')

