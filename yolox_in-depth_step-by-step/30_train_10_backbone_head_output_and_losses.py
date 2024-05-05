
import os
import sys
import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolox.models.yolo_pafpn import YOLOPAFPN
from yolox.models.yolo_head import YOLOXHead
from yolox.models.yolox import YOLOX

from yolox.utils import *

from yolox.data import DataPrefetcher
# from yolox.core import Trainer

from yolox.models.losses import IOUloss

# ----------
base_dir = '/home/kswada/kw/yolox/YOLOX'


# REFERENCE but YOLO
# https://www.slideshare.net/ssuser07aa33/introduction-to-yolo-detection-model

# REFERENCE YOLOX
# https://medium.com/p/c01f6a8a0830
# https://medium.com/mlearning-ai/yolox-explanation-how-does-yolox-work-3e5c89f2bf78
# https://github.com/gmongaras/YOLOX_From_Scratch/tree/main


############################################################################################
# ------------------------------------------------------------------------------------------
# get experiment configuration
# ------------------------------------------------------------------------------------------

exp_file = os.path.join(base_dir, 'exps/default/yolox_s.py')


# ----------
sys.path.append(os.path.dirname(exp_file))


# ----------
print(os.path.basename(exp_file).split(".")[0])

current_exp = importlib.import_module(os.path.basename(exp_file).split(".")[0])

print(current_exp)


# ----------
exp = current_exp.Exp()

print(exp)
print(dir(exp))


# ------------------------------------------------------------------------------------------
# set up 
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
    num_classes=exp.num_classes,
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



# ------------------------------------------------------------------------------------------
# load checkpoing
# ------------------------------------------------------------------------------------------

torch.cuda.set_device(rank)

model.cuda(rank)

model.eval()


# ----------
ckpt_file = os.path.join(base_dir, 'YOLOX_outputs/yolox_s/best_ckpt.pth')

loc = "cuda:{}".format(rank)
ckpt = torch.load(ckpt_file, map_location=loc)

model.load_state_dict(ckpt["model"])


# ----------
fp16 = True
half = fp16

tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor

model = model.eval()

model.to(device)


# ------------------------------------------------------------------------------------------
# data loader
# ------------------------------------------------------------------------------------------

batch_size = 8


# ----------
# cache_img = None creates index
train_loader = exp.get_data_loader(
    batch_size=batch_size,
    is_distributed=is_distributed,
    no_aug=True,
    cache_img=None,
)

print(dir(train_loader))


# ----------
# next_input, next_target, a, b = next(iter(train_loader))

prefetcher = DataPrefetcher(train_loader)

max_iter = len(train_loader)

print(prefetcher)
print(max_iter)


# ------------------------------------------------------------------------------------------
# get one batch
# ------------------------------------------------------------------------------------------

inps, targets = prefetcher.next()

# (batch_size, 3, input_size[0], input_size[1])
print(inps.shape)

# (batch_size, 120, 5)
print(targets.shape)


# ----------
data_type = torch.float32

inps = inps.to(data_type)
targets = targets.to(data_type)


# ----------
# preprocess

targets.requires_grad = False

inps, targets = exp.preprocess(inps, targets, exp.input_size)
print(inps.shape)
print(targets.shape)


# ------------------------------------------------------------------------------------------
# backbone outputs
# ------------------------------------------------------------------------------------------

fpn_outs = model.backbone(inps)

# 3
print(len(fpn_outs))

# C3_p3
# (batch_size, 128, 80, 80)
print(fpn_outs[0].shape)

# C3_n3
# (batch_size, 256, 40, 40)
print(fpn_outs[1].shape)

# C3_n4
# (batch_size, 512, 20, 20)
print(fpn_outs[2].shape)


# -->
# total 8400 = 80*80 + 40*40 + 20*20


# ------------------------------------------------------------------------------------------
# head outputs
# ------------------------------------------------------------------------------------------

print(len(in_channels))

print(len(model.head.cls_convs))
print(len(model.head.reg_convs))
print(len(model.head.strides))


# ----------
use_l1 = False

outputs = []
origin_preds = []
x_shifts = []
y_shifts = []
expanded_strides = []

for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
    zip(model.head.cls_convs, model.head.reg_convs, model.head.strides, fpn_outs)
):
    # ----------
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
    ##########
    # training:
    output = torch.cat([reg_output, obj_output, cls_output], 1)
    output, grid = model.head.get_output_and_grid(
        output, k, stride_this_level, fpn_outs[0].type()
    )
    # ----------
    print(f'stride_this_level: {stride_this_level}')
    print(f'len grid: {len(grid[0])}')
    # ----------
    x_shifts.append(grid[:, :, 0])
    y_shifts.append(grid[:, :, 1])
    expanded_strides.append(
        torch.zeros(1, grid.shape[1])
        .fill_(stride_this_level)
        .type_as(fpn_outs[0])
    )
    # ----------
    if use_l1:
        batch_size = reg_output.shape[0]
        hsize, wsize = reg_output.shape[-2:]
        reg_output = reg_output.view(
            batch_size, 1, 4, hsize, wsize
        )
        reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
            batch_size, -1, 4
        )
        origin_preds.append(reg_output.clone())
    # ----------
    outputs.append(output)


# ----------
# note that number of all grids is 6400 + 1600 + 400 = 8400

print(len(expanded_strides[0][0]))
print(len(expanded_strides[1][0]))
print(len(expanded_strides[2][0]))

print(grid[0])

# ----------
print(len(x_shifts[0][0]))
print(x_shifts[1])
print(x_shifts[2])


# ----------
print(len(outputs))
# (batch_size, 6400, 85)
print(outputs[0].shape)
# (batch_size, 1600, 85)
print(outputs[1].shape)
# (batch_size, 400, 85)
print(outputs[2].shape)

# 1st image, 1st anchor out of 6400, 85 output
# 85 output = reg (cx, cy, h, w), obj, cls
print(outputs[0][0][0])
print(outputs[1][0][0])
print(outputs[2][0][0])

# fist 10 output
print(outputs[0][0][0][:10])
print(outputs[1][0][0][:10])
print(outputs[2][0][0][:10])


# ------------------------------------------------------------------------------------------
# check step by step:  head output
# ------------------------------------------------------------------------------------------

# outputs = []
# origin_preds = []
# x_shifts = []
# y_shifts = []
# expanded_strides = []

# use_l1 = False

# k = 0

# cls_conv = model.head.cls_convs[k]
# reg_conv = model.head.reg_convs[k]
# stride_this_level = model.head.strides[k]

# x = fpn_outs[k]
# stem = model.head.stems[k]

# print(stride_this_level)


# # ----------
# # stem:  cls_x and reg_x
# print(stem)
# x2 = stem(x)

# print(x.shape)
# print(x2.shape)

# cls_x = x2
# reg_x = x2

# # ----------
# # cls:  channel 128 to num_class (=80)
# print(cls_conv)
# print(model.head.cls_preds[k])

# cls_feat = cls_conv(cls_x)
# cls_output = model.head.cls_preds[k](cls_feat)

# print(cls_feat.shape)
# print(cls_output.shape)


# # ----------
# # reg:  channel 128 to 4
# # obj:  channel 128 to 1
# print(reg_conv)
# print(model.head.reg_preds[k])
# print(model.head.obj_preds[k])

# reg_feat = reg_conv(reg_x)
# reg_output = model.head.reg_preds[k](reg_feat)
# obj_output = model.head.obj_preds[k](reg_feat)

# print(reg_feat.shape)
# print(reg_output.shape)
# print(obj_output.shape)


# # ----------
# # concat all
# output = torch.cat([reg_output, obj_output, cls_output], 1)

# # (8, 85, 80, 80) = (batchsize, 80+4+1, 80, 80)
# print(output.shape)


# # ----------
# # output, grid = model.head.get_output_and_grid(
# #     output, k, stride_this_level, fpn_outs[0].type()
# # )

# grids = [torch.zeros(1)] * len(in_channels)
# grid = grids[k]
# print(grid)

# batch_size = output.shape[0]

# n_ch = 5 + exp.num_classes

# hsize, wsize = output.shape[-2:]

# dtype = fpn_outs[0].dtype

# if grid.shape[2:4] != output.shape[2:4]:
#     yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
#     grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)
#     grids[k] = grid

# print(grid.shape)

# output = output.view(batch_size, 1, n_ch, hsize, wsize)

# # (8, 1, 85, 80, 80)
# print(output.shape)

# output = output.permute(0, 1, 3, 4, 2).reshape(
#     batch_size, hsize * wsize, -1
# )

# # (8, 6400, 85)
# print(output.shape)

# grid = grid.view(1, -1, 2)
# # (1, 6400, 2)
# print(grid.shape)
# print(grid)

# # cx, cy
# output[..., :2] = (output[..., :2] + grid) * stride_this_level

# # height and width, requires exp !!
# output[..., 2:4] = torch.exp(output[..., 2:4]) * stride_this_level


# # ----------
# x_shifts.append(grid[:, :, 0])

# y_shifts.append(grid[:, :, 1])

# expanded_strides.append(
#     torch.zeros(1, grid.shape[1])
#     .fill_(stride_this_level)
#     .type_as(fpn_outs[0])
# )


# # ----------
# if use_l1:
#     batch_size = reg_output.shape[0]
#     hsize, wsize = reg_output.shape[-2:]
#     reg_output = reg_output.view(
#         batch_size, 1, 4, hsize, wsize
#     )
#     reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
#         batch_size, -1, 4
#     )
#     origin_preds.append(reg_output.clone())


# # ----------
# outputs.append(output)


############################################################################################
# ------------------------------------------------------------------------------------------
# get losses
# ------------------------------------------------------------------------------------------

outputs = torch.cat(outputs, 1)


# ----------
bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
obj_preds = outputs[:, :, 4:5]  # [batch, n_anchors_all, 1]
cls_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]


# ----------
# labels is targets and imgs is inps
labels = targets
imgs = inps


# calculate targets
nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

# -----------
total_num_anchors = outputs.shape[1]

x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]

y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]

expanded_strides = torch.cat(expanded_strides, 1)

if use_l1:
    origin_preds = torch.cat(origin_preds, 1)


# ----------
dtype = fpn_outs[0].dtype

cls_targets = []
reg_targets = []
l1_targets = []
obj_targets = []
fg_masks = []

num_fg = 0.0
num_gts = 0.0

## see later: check step by step
for batch_idx in range(outputs.shape[0]):
    num_gt = int(nlabel[batch_idx])
    num_gts += num_gt
    if num_gt == 0:
        cls_target = outputs.new_zeros((0, exp.num_classes))
        reg_target = outputs.new_zeros((0, 4))
        l1_target = outputs.new_zeros((0, 4))
        obj_target = outputs.new_zeros((total_num_anchors, 1))
        fg_mask = outputs.new_zeros(total_num_anchors).bool()
    else:
        gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]
        gt_classes = labels[batch_idx, :num_gt, 0]
        bboxes_preds_per_image = bbox_preds[batch_idx]
        # ----------
        try:
            (
                gt_matched_classes,
                fg_mask,
                pred_ious_this_matching,
                matched_gt_inds,
                num_fg_img,
            ) = model.head.get_assignments(  # noqa
                batch_idx,
                num_gt,
                gt_bboxes_per_image,
                gt_classes,
                bboxes_preds_per_image,
                expanded_strides,
                x_shifts,
                y_shifts,
                cls_preds,
                obj_preds,
            )
        except RuntimeError as e:
            # TODO: the string might change, consider a better way
            if "CUDA out of memory. " not in str(e):
                raise  # RuntimeError might not caused by CUDA OOM
            # ----------
            torch.cuda.empty_cache()
            (
                gt_matched_classes,
                fg_mask,
                pred_ious_this_matching,
                matched_gt_inds,
                num_fg_img,
            ) = model.head.get_assignments(  # noqa
                batch_idx,
                num_gt,
                gt_bboxes_per_image,
                gt_classes,
                bboxes_preds_per_image,
                expanded_strides,
                x_shifts,
                y_shifts,
                cls_preds,
                obj_preds,
                "cpu",
            )
        # ----------
        torch.cuda.empty_cache()
        num_fg += num_fg_img
        # ----------
        cls_target = F.one_hot(
            gt_matched_classes.to(torch.int64), exp.num_classes
        ) * pred_ious_this_matching.unsqueeze(-1)
        obj_target = fg_mask.unsqueeze(-1)
        reg_target = gt_bboxes_per_image[matched_gt_inds]
        # ----------
        if use_l1:
            l1_target = model.head.get_l1_target(
                outputs.new_zeros((num_fg_img, 4)),
                gt_bboxes_per_image[matched_gt_inds],
                expanded_strides[0][fg_mask],
                x_shifts=x_shifts[0][fg_mask],
                y_shifts=y_shifts[0][fg_mask],
            )
    # ----------
    cls_targets.append(cls_target)
    reg_targets.append(reg_target)
    obj_targets.append(obj_target.to(dtype))
    fg_masks.append(fg_mask)
    if use_l1:
        l1_targets.append(l1_target)


cls_targets = torch.cat(cls_targets, 0)
reg_targets = torch.cat(reg_targets, 0)
obj_targets = torch.cat(obj_targets, 0)
fg_masks = torch.cat(fg_masks, 0)

print(cls_targets.shape)
print(reg_targets.shape)
print(obj_targets.shape)
print(fg_masks.shape)


# ----------
if use_l1:
    l1_targets = torch.cat(l1_targets, 0)


# ----------
# total matched (assigned) anchors
num_fg = max(num_fg, 1)
print(num_fg)


# ----------
# iou_loss = IOUloss(reduction="none" ,loss_type="iou")

# loss_iou = (
#     iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)
# ).sum() / num_fg

pred = bbox_preds.view(-1, 4)[fg_masks]

target = reg_targets

pred = pred.view(-1, 4)

target = target.view(-1, 4)

# (num fg, 4)
print(pred.shape)

# (num fg, 4)
print(target.shape)


tl = torch.max(
    (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
)

br = torch.min(
    (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
)

area_p = torch.prod(pred[:, 2:], 1)
area_g = torch.prod(target[:, 2:], 1)

en = (tl < br).type(tl.type()).prod(dim=1)
area_i = torch.prod(br - tl, 1) * en
area_u = area_p + area_g - area_i
iou = (area_i) / (area_u + 1e-16)

# (376)
print(iou.shape)


loss_type = "giou"
reduction = "None"

if loss_type == "iou":
    loss = 1 - iou ** 2
elif loss_type == "giou":
    c_tl = torch.min(
        (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
    )
    c_br = torch.max(
        (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
    )
    area_c = torch.prod(c_br - c_tl, 1)
    giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
    loss = 1 - giou.clamp(min=-1.0, max=1.0)


if reduction == "mean":
    loss = loss.mean()
elif reduction == "sum":
    loss = loss.sum()

loss_iou = loss.sum() / num_fg


# ----------
bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")

loss_obj = (
    bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)
).sum() / num_fg

loss_cls = (
    bcewithlog_loss(
        cls_preds.view(-1, exp.num_classes)[fg_masks], cls_targets
    )
).sum() / num_fg


# ----------
l1_loss = nn.L1Loss(reduction="none")

if use_l1:
    loss_l1 = (
        l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)
    ).sum() / num_fg
else:
    loss_l1 = 0.0


# ----------
reg_weight = 5.0

loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1

# return (
#     loss,
#     reg_weight * loss_iou,
#     loss_obj,
#     loss_cls,
#     loss_l1,
#     num_fg / max(num_gts, 1),
# )


# ------------------------------------------------------------------------------------------
# check step by step:  process by batch
# ------------------------------------------------------------------------------------------

outputs = torch.cat(outputs, 1)


# ----------
bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
obj_preds = outputs[:, :, 4:5]  # [batch, n_anchors_all, 1]
cls_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]


# ----------
# labels is targets and imgs is inps
labels = targets
imgs = inps


# calculate targets
nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects
print(nlabel)


# -----------
total_num_anchors = outputs.shape[1]
print(f'total_num_anchors: {total_num_anchors}')


# -----------
# remove list
x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]

y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]

expanded_strides = torch.cat(expanded_strides, 1)

if use_l1:
    origin_preds = torch.cat(origin_preds, 1)


# ----------
dtype = fpn_outs[0].dtype

cls_targets = []
reg_targets = []
l1_targets = []
obj_targets = []
fg_masks = []

num_fg = 0.0
num_gts = 0.0


# ----------
# for batch_idx in range(outputs.shape[0]):

# now focus to batch_idx
batch_idx = 2

num_gt = int(nlabel[batch_idx])

num_gts += num_gt

gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]

gt_classes = labels[batch_idx, :num_gt, 0]

bboxes_preds_per_image = bbox_preds[batch_idx]


##### get_assignments
# (
#     gt_matched_classes,
#     fg_mask,
#     pred_ious_this_matching,
#     matched_gt_inds,
#     num_fg_img,
# ) = model.head.get_assignments(
#     batch_idx,
#     num_gt,
#     gt_bboxes_per_image,
#     gt_classes,
#     bboxes_preds_per_image,
#     expanded_strides,
#     x_shifts,
#     y_shifts,
#     cls_preds,
#     obj_preds,
# )

######
# 1. get_geometry_constraint:
#    Calculate whether the center of an object is located in a fixed range of
#    an anchor. This is used to avert inappropriate matching. It can also reduce
#    the number of candidate anchors so that the GPU memory is saved.

# note that 0 is applied
expanded_strides_per_image = expanded_strides[0]
x_centers_per_image = ((x_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0)
y_centers_per_image = ((y_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0)
print(x_centers_per_image)
print(y_centers_per_image)

# ----------
# in fixed center  --> 1.5 means 3 * 3 multi positives
center_radius = 1.5
center_dist = expanded_strides_per_image.unsqueeze(0) * center_radius
gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0:1]) - center_dist
gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0:1]) + center_dist
gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1:2]) - center_dist
gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1:2]) + center_dist

print(gt_bboxes_per_image_l.shape)

# ----------
c_l = x_centers_per_image - gt_bboxes_per_image_l
c_r = gt_bboxes_per_image_r - x_centers_per_image
c_t = y_centers_per_image - gt_bboxes_per_image_t
c_b = gt_bboxes_per_image_b - y_centers_per_image
center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
# (12, 8400, 4) = (num of gt, num all anchors, 4)
print(center_deltas.shape)

is_in_centers = center_deltas.min(dim=-1).values > 0.0
anchor_filter = is_in_centers.sum(dim=0) > 0
geometry_relation = is_in_centers[:, anchor_filter]

# (num of gt, 8400)
print(is_in_centers.shape)
# (8400)
print(anchor_filter.shape)
# (num of gt, filtered anchors)
print(geometry_relation.shape)

fg_mask = anchor_filter


######
# 2. mask only anchors with objectness

bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
cls_preds_ = cls_preds[batch_idx][fg_mask]
obj_preds_ = obj_preds[batch_idx][fg_mask]

num_in_boxes_anchor = bboxes_preds_per_image.shape[0]
print(num_in_boxes_anchor)


######
# 3. calculte bboxes_iou (pair_wise_ious) --> calculate pair wise iou loss

# pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)

bboxes_a = gt_bboxes_per_image
bboxes_b = bboxes_preds_per_image

tl = torch.max(
    (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
    (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
)

br = torch.min(
    (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
    (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
)

area_a = torch.prod(bboxes_a[:, 2:], 1)
area_b = torch.prod(bboxes_b[:, 2:], 1)
en = (tl < br).type(tl.type()).prod(dim=2)
area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())

pair_wise_ious = area_i / (area_a[:, None] + area_b - area_i)

# (num of gt, filtered anchors)
print(pair_wise_ious.shape)

# ----------
# iou loss
pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)
print(pair_wise_ious)
print(pair_wise_ious_loss)


######
# 4. calculate pair wise cls loss

gt_cls_per_image = (
    F.one_hot(gt_classes.to(torch.int64), exp.num_classes)
    .float()
)

# (12)
print(gt_classes.shape)

# (num of gt, num classes)
print(gt_cls_per_image.shape)


# cls_preds_ * obj_preds_  after sigmoid
with torch.cuda.amp.autocast(enabled=False):
    cls_preds_ = (
        cls_preds_.float().sigmoid_() * obj_preds_.float().sigmoid_()
    ).sqrt()
    # ----------
    # this is binary cross entropy
    pair_wise_cls_loss = F.binary_cross_entropy(
        cls_preds_.unsqueeze(0).repeat(num_gt, 1, 1),
        gt_cls_per_image.unsqueeze(1).repeat(1, num_in_boxes_anchor, 1),
        reduction="none"
    ).sum(-1)

# (num of gt, filtered anchor, num classes)
print(cls_preds_.unsqueeze(0).repeat(num_gt, 1, 1).shape)
# (num of gt, filtered anchor, num classes)
print(gt_cls_per_image.unsqueeze(1).repeat(1, num_in_boxes_anchor, 1).shape)
# (num of gt, filtered anchor)
print(pair_wise_cls_loss.shape)


######
# 5. cost = cls loss + 3.0 * iou loss + 1000000 * (negated) geometry_relation

# note that geometry_relation is negated
# geometry relation weight is VERY LARGE

cost = (
    pair_wise_cls_loss
    + 3.0 * pair_wise_ious_loss
    + float(1e6) * (~geometry_relation)
)

# (num of gt, filtered anchor)
print(cost.shape)


######
# 6. simOTA matching

# (
#     num_fg,
#     gt_matched_classes,
#     pred_ious_this_matching,
#     matched_gt_inds,
# ) = self.simota_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)

matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

# top 10
n_candidate_k = min(10, pair_wise_ious.size(1))

topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=1)

# (num of gt, 10)
print(topk_ious.shape)

dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
print(dynamic_ks)

for gt_idx in range(num_gt):
    _, pos_idx = torch.topk(
        cost[gt_idx], k=dynamic_ks[gt_idx], largest=False
    )
    matching_matrix[gt_idx][pos_idx] = 1

# del topk_ious, dynamic_ks, pos_idx

anchor_matching_gt = matching_matrix.sum(0)

# deal with the case that one anchor matches multiple ground-truths
if anchor_matching_gt.max() > 1:
    multiple_match_mask = anchor_matching_gt > 1
    _, cost_argmin = torch.min(cost[:, multiple_match_mask], dim=0)
    matching_matrix[:, multiple_match_mask] *= 0
    matching_matrix[cost_argmin, multiple_match_mask] = 1

fg_mask_inboxes = anchor_matching_gt > 0

num_fg = fg_mask_inboxes.sum().item()

fg_mask2 = fg_mask.clone()
fg_mask2[fg_mask2.clone()] = fg_mask_inboxes

matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
gt_matched_classes = gt_classes[matched_gt_inds]

pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
    fg_mask_inboxes
]

# return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds

# ----------
fg_mask = fg_mask2

num_fg = num_fg_img

##### END of get_assigments #####


# ----------
# 
print(gt_matched_classes)
print(pred_ious_this_matching)

cls_target = F.one_hot(
    gt_matched_classes.to(torch.int64), exp.num_classes
) * pred_ious_this_matching.unsqueeze(-1)

# (matched anchors, num classes)
print(cls_target.shape)

obj_target = fg_mask.unsqueeze(-1)

reg_target = gt_bboxes_per_image[matched_gt_inds]


# ----------
if use_l1:
    l1_target = get_l1_target(
        outputs.new_zeros((num_fg_img, 4)),
        gt_bboxes_per_image[matched_gt_inds],
        expanded_strides[0][fg_mask],
        x_shifts=x_shifts[0][fg_mask],
        y_shifts=y_shifts[0][fg_mask],
    )



# ----------
if use_l1:
    l1_targets = torch.cat(l1_targets, 0)


# ----------
num_fg = max(num_fg, 1)
print(num_fg)


# ----------
iou_loss = IOUloss(reduction="none" ,loss_type="iou")

loss_iou = (
    iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)
).sum() / num_fg


# ----------
bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")

loss_obj = (
    bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)
).sum() / num_fg

loss_cls = (
    bcewithlog_loss(
        cls_preds.view(-1, exp.num_classes)[fg_masks], cls_targets
    )
).sum() / num_fg


# ----------
l1_loss = nn.L1Loss(reduction="none")

if use_l1:
    loss_l1 = (
        l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)
    ).sum() / num_fg
else:
    loss_l1 = 0.0


# ----------
reg_weight = 5.0

loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1

# return (
#     loss,
#     reg_weight * loss_iou,
#     loss_obj,
#     loss_cls,
#     loss_l1,
#     num_fg / max(num_gts, 1),
# )
