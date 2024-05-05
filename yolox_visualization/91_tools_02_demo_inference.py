
import os
import sys
import importlib

import torch
# import torch.nn as nn
# import torch.nn.functional as F
import torchvision

from yolox.models.yolo_pafpn import YOLOPAFPN
from yolox.models.yolo_head import YOLOXHead
from yolox.models.yolox import YOLOX

from yolox.utils import *

import numpy as np
import cv2
import PIL.Image

# ----------
base_dir = '/home/kswada/kw/yolox/YOLOX'


############################################################################################
# ------------------------------------------------------------------------------------------
# get experiment configuration
# ------------------------------------------------------------------------------------------

exp_file = os.path.join(base_dir, 'exps/default/yolox_s.py')

sys.path.append(os.path.dirname(exp_file))

print(os.path.basename(exp_file).split(".")[0])

current_exp = importlib.import_module(os.path.basename(exp_file).split(".")[0])

# ----------
exp = current_exp.Exp()

print(exp)
print(dir(exp))


# ------------------------------------------------------------------------------------------
# set up
# ------------------------------------------------------------------------------------------

cls_names = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)

_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)


num_classes = exp.num_classes
confthre = exp.test_conf
nmsthre = exp.nmsthre
test_size = exp.test_size


# ------------------------------------------------------------------------------------------
# get model from experiment configuration
# ------------------------------------------------------------------------------------------

in_channels = [256, 512, 1024]
in_features=("dark3", "dark4", "dark5")

backbone = YOLOPAFPN(
    depth=exp.depth,
    width=exp.width,
    in_features=("dark3", "dark4", "dark5"),
    in_channels=in_channels,
    act=exp.act
    )


strides = [8, 16, 32]
depthwise = False

head = YOLOXHead(
    num_classes=num_classes,
    width=exp.width,
    strides=strides,
    in_channels=in_channels,
    act=exp.act,
    depthwise=depthwise
    )

model = YOLOX(backbone, head)

def init_yolo(M):
    for m in M.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eps = 1e-3
            m.momentum = 0.03

model.apply(init_yolo)

model.head.initialize_biases(1e-2)

model.train()


# ----------
device = 'gpu'
fp16 = True

if device == "gpu":
    model.cuda()
    if fp16:
        model.half()

model.eval()


# ------------------------------------------------------------------------------------------
# load checkpoing
# ------------------------------------------------------------------------------------------

ckpt_file = os.path.join(base_dir, 'YOLOX_outputs/yolox_s/best_ckpt.pth')

ckpt = torch.load(ckpt_file, map_location='cuda:0')
# ckpt = torch.load(ckpt_file, map_location='cpu')

model.load_state_dict(ckpt["model"])


# ------------------------------------------------------------------------------------------
# preprocessor
# ------------------------------------------------------------------------------------------

def preproc(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114
    # ----------
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
    # ----------
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


# class ValTransform:
#     """
#     Defines the transformations that should be applied to test PIL image
#     for input into the network

#     dimension -> tensorize -> color adj

#     Arguments:
#         resize (int): input dimension to SSD
#         rgb_means ((int,int,int)): average RGB of the dataset
#             (104,117,123)
#         swap ((int,int,int)): final order of channels

#     Returns:
#         transform (transform) : callable transform to be applied to test/val
#         data
#     """

#     def __init__(self, swap=(2, 0, 1), legacy=False):
#         self.swap = swap
#         self.legacy = legacy

#     # assume input is cv2 img for now
#     def __call__(self, img, res, input_size):
#         img, _ = preproc(img, input_size, self.swap)
#         if self.legacy:
#             img = img[::-1, :, :].copy()
#             img /= 255.0
#             img -= np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
#             img /= np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
#         return img, np.zeros((1, 5))

# legacy = False
# preproc = ValTransform(legacy=legacy)


# ------------------------------------------------------------------------------------------
# inference
# ------------------------------------------------------------------------------------------

# img_dir = os.path.join(base_dir, 'assets')
img_dir = os.path.join(base_dir, 'sample_images')


# img_fname = 'dog.jpg'
img_fname = 'ADE_val_00000010.jpg'
img_fname = 'ADE_val_00000019.jpg'
img_fname = 'ADE_val_00000040.jpg'
img_fname = 'ADE_val_00000055.jpg'

img_path = os.path.join(img_dir, img_fname)


img_info = {"id": 0}
if isinstance(img_path, str):
    img_info["file_name"] = os.path.basename(img_path)

# ----------
img = cv2.imread(img_path)
print(img.shape)


# ----------
# preprocessing

height, width = img.shape[:2]

img_info["height"] = height
img_info["width"] = width
img_info["raw_img"] = img

ratio = min(test_size[0] / img.shape[0], test_size[1] / img.shape[1])
img_info["ratio"] = ratio


# padded image
img_, r = preproc(img=img, input_size=exp.input_size, swap=(2, 0, 1))
img_ = torch.from_numpy(img_).unsqueeze(0)
img_ = img_.float()


# ----------
# inference 

if device == "gpu":
    img_ = img_.cuda()
    if fp16:
        img_ = img_.half()


# confthre in test is very small(=0.01), so increase confthre to 0.6 or default 0.7
with torch.no_grad():
    outputs = model(img_)
    # ----------
    # nms
    # outputs = postprocess(outputs, num_classes, confthre, nmsthre, class_agnostic=True)
    outputs = postprocess(outputs, num_classes, 0.7, nmsthre, class_agnostic=True)


# (14, 7)
print(outputs[0].shape)

output = outputs[0]

# if output is None:
#     return img_info["raw_img"]

output = output.cpu()

bboxes = output[:, 0:4]

# preprocessing: resize
bboxes /= ratio

cls_ids = output[:, 6]
scores = output[:, 4] * output[:, 5]


# ------------------------------------------------------------------------------------------
# visualize
# ------------------------------------------------------------------------------------------

img_vis = img.copy()

for i in range(len(bboxes)):
    box = bboxes[i]
    cls_id = int(cls_ids[i])
    score = scores[i]
    if score < confthre:
        continue
    x0 = int(box[0])
    y0 = int(box[1])
    x1 = int(box[2])
    y1 = int(box[3])
    # ----------
    color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
    text = '{}:{:.1f}%'.format(cls_names[cls_id], score * 100)
    txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # ----------
    txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
    cv2.rectangle(img_vis, (x0, y0), (x1, y1), color, 2)
    # ----------
    txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
    cv2.rectangle(
        img_vis,
        (x0, y0 + 1),
        (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
        txt_bk_color,
        -1
    )
    cv2.putText(img_vis, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)


# ----------
# show

PIL.Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).show()
PIL.Image.fromarray(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)).show()



############################################################################################
# ------------------------------------------------------------------------------------------
# check inference + postprocess + visualization step by step
# ------------------------------------------------------------------------------------------

# img_dir = os.path.join(base_dir, 'assets')
img_dir = os.path.join(base_dir, 'sample_images')


# img_fname = 'dog.jpg'
img_fname = 'ADE_val_00000010.jpg'
img_fname = 'ADE_val_00000019.jpg'
img_fname = 'ADE_val_00000040.jpg'
img_fname = 'ADE_val_00000055.jpg'

img_path = os.path.join(img_dir, img_fname)


img_info = {"id": 0}
if isinstance(img_path, str):
    img_info["file_name"] = os.path.basename(img_path)

# ----------
img = cv2.imread(img_path)
print(img.shape)


##########
# preprocessing
##########

height, width = img.shape[:2]

img_info["height"] = height
img_info["width"] = width
img_info["raw_img"] = img

ratio = min(test_size[0] / img.shape[0], test_size[1] / img.shape[1])
img_info["ratio"] = ratio


# padded image
img_, r = preproc(img=img, input_size=exp.input_size, swap=(2, 0, 1))
img_ = torch.from_numpy(img_).unsqueeze(0)
img_ = img_.float()


# ----------
# inference 

if device == "gpu":
    img_ = img_.cuda()
    if fp16:
        img_ = img_.half()


# (1, 3, 640, 640)
print(img_.shape)


##########
# backbone: YOLOPAFPN
##########

fpn_outs = model.backbone(img_)

# ----------
# C3_p3
# (batch_size, 128, 80, 80)
print(fpn_outs[0].shape)

# ----------
# C3_n3
# (batch_size, 256, 40, 40)
print(fpn_outs[1].shape)

# ----------
# C3_n4
# (batch_size, 512, 20, 20)
print(fpn_outs[2].shape)
# -->
# total 8400 = 80*80 + 40*40 + 20*20


##########
# head: YOLOXHead
##########

outputs = []
origin_preds = []
x_shifts = []
y_shifts = []
expanded_strides = []

# ----------
for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
    zip(model.head.cls_convs, model.head.reg_convs, model.head.strides, fpn_outs)
):
    x = model.head.stems[k](x)
    cls_x = x
    reg_x = x
    # ----------
    cls_feat = cls_conv(cls_x)
    cls_output = model.head.cls_preds[k](cls_feat)
    # ----------
    reg_feat = reg_conv(reg_x)
    reg_output = model.head.reg_preds[k](reg_feat)
    obj_output = model.head.obj_preds[k](reg_feat)
    # ----------
    output = torch.cat(
        [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
    )
    # ----------
    outputs.append(output)

# ----------
# (1, 85, 80, 80)
print(outputs[0].shape)
# (1, 85, 40, 40)
print(outputs[1].shape)
# (1, 85, 20, 20)
print(outputs[2].shape)


# ----------
hw = [x.shape[-2:] for x in outputs]
print(hw)

# [batch, n_anchors_all, 85]
outputs = torch.cat(
    [x.flatten(start_dim=2) for x in outputs], dim=2
).permute(0, 2, 1)


# (1, 8400, 85)
print(outputs.shape)


##########
# decode (since model.head.decode_in_inference is True)
##########

dtype = fpn_outs[0].type()

grids = []
strides_ = []

for (hsize, wsize), stride in zip(hw, strides):
    yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
    grid = torch.stack((xv, yv), 2).view(1, -1, 2)
    grids.append(grid)
    shape = grid.shape[:2]
    strides_.append(torch.full((*shape, 1), stride))

grids = torch.cat(grids, dim=1).type(dtype)

strides_ = torch.cat(strides_, dim=1).type(dtype)

outputs = torch.cat([
    (outputs[..., 0:2] + grids) * strides_,
    torch.exp(outputs[..., 2:4]) * strides_,
    outputs[..., 4:]
], dim=-1)


# (1, 8400, 85)
print(outputs.shape)


##########
# postprocess
# outputs = postprocess(outputs, num_classes, 0.7, nmsthre, class_agnostic=True)
##########

confthre = 0.7
class_agnostic = True
prediction = outputs.clone()

box_corner = prediction.new(prediction.shape)
box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
prediction[:, :, :4] = box_corner[:, :, :4]
output = [None for _ in range(len(prediction))]

# ----------
for i, image_pred in enumerate(prediction):
    # ----------
    # If none are remaining => process next image
    if not image_pred.size(0):
        continue
    # Get score and class with highest confidence
    class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)
    # ----------
    conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= confthre).squeeze()
    # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
    detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
    detections = detections[conf_mask]
    if not detections.size(0):
        continue
    # ----------
    if class_agnostic:
        nms_out_index = torchvision.ops.nms(
            detections[:, :4],
            detections[:, 4] * detections[:, 5],
            nmsthre,
        )
    else:
        nms_out_index = torchvision.ops.batched_nms(
            detections[:, :4],
            detections[:, 4] * detections[:, 5],
            detections[:, 6],
            nmsthre,
        )
    # ----------
    detections = detections[nms_out_index]
    if output[i] is None:
        output[i] = detections
    else:
        output[i] = torch.cat((output[i], detections))


# 1: only 1 image
print(len(output))

# (14, 7)
print(output[0].shape)
print(output[0])


# ----------
output = output[0]

# if output is None:
#     return img_info["raw_img"]

output = output.cpu()

bboxes = output[:, 0:4]

# preprocessing: resize
bboxes /= ratio

cls_ids = output[:, 6]
scores = output[:, 4] * output[:, 5]


##########
# visualizaton

img_vis = img.copy()

for i in range(len(bboxes)):
    box = bboxes[i]
    cls_id = int(cls_ids[i])
    score = scores[i]
    if score < confthre:
        continue
    x0 = int(box[0])
    y0 = int(box[1])
    x1 = int(box[2])
    y1 = int(box[3])
    # ----------
    color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
    text = '{}:{:.1f}%'.format(cls_names[cls_id], score * 100)
    txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # ----------
    txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
    cv2.rectangle(img_vis, (x0, y0), (x1, y1), color, 2)
    # ----------
    txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
    cv2.rectangle(
        img_vis,
        (x0, y0 + 1),
        (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
        txt_bk_color,
        -1
    )
    cv2.putText(img_vis, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)


# ----------
# show

PIL.Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).show()
PIL.Image.fromarray(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)).show()

