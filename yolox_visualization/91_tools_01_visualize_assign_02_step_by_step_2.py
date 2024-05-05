
import os
# import sys
import importlib

import torch
import torch.nn as nn
import torchvision

from yolox.models.yolo_pafpn import YOLOPAFPN
from yolox.models.yolo_head import YOLOXHead
from yolox.models.yolox import YOLOX

from yolox.utils import *
from yolox.data import *

from yolox.data.datasets.coco_classes import COCO_CLASSES
from yolox.utils.visualize import _COLORS


# ----------
import shutil
import numpy as np
import cv2
import PIL.Image
import random


############################################################################################
# ------------------------------------------------------------------------------------------
# base setting
# ------------------------------------------------------------------------------------------

base_dir = '/home/kswada/kw/yolox/YOLOX'

# experiment base module
exp_base_module = 'exps.default.yolox_s'
current_exp = importlib.import_module(exp_base_module)
exp = current_exp.Exp()

# checkpoint
ckpt_file = os.path.join(base_dir, 'YOLOX_outputs/yolox_s/best_ckpt.pth')

# # data and tensor type
# data_type = torch.float32
# tensor_type = torch.cuda.HalfTensor

# ----------
batch_size = 8
max_batch = 10

cls_names = COCO_CLASSES

# ----------
# get experiment configuration
# exp_file = os.path.join(base_dir, 'exps/default/yolox_s.py')
# sys.path.append(os.path.dirname(exp_file))
# current_exp = importlib.import_module(os.path.basename(exp_file).split(".")[0])

exp.data_num_workers = 4
# ----------
# it seems to specify here, if not specify, depth and width will be set default 1.0
exp.depth = 0.33
exp.width = 0.50
# ----------
exp.multiscale_range = 0
# ----------
exp.test_conf = 0.6
exp.nmsthre = 0.65
class_agnostic = False
############################
# MIXUP
# apply mixup aug or not  -->  but mosaic is required
exp.enable_mixup = True
# exp.enable_mixup = False
exp.mixup_prob = 1.0
# exp.mixup_scale = (0.5, 1.5)
exp.mixup_scale = (0.95, 1.05)
############################
# MOSAIC, random affine  (yolox.data.datasets : mosaicdetection.py)
exp.enable_mosaic = True
# exp.enable_mosaic = False
exp.mosaic_prob = 1.0
# -----
# random affine:
# -----
exp.mosaic_scale = (0.1, 2)
# exp.mosaic_scale = (1.0, 1.0)
# -----
# rotation angle range, for example, if set to 2, the true range is (-2, 2)
exp.degrees = 10.0
# exp.degrees = 0.0
# -----
# translate range, for example, if set to 0.1, the true range is (-0.1, 0.1)
exp.translate = 0.1
# exp.translate = 0.0
# -----
# shear angle range, for example, if set to 2, the true range is (-2, 2)
exp.shear = 2.0
# exp.shear = 0.0

# ----------
max_labels = 50


# ----------
# save path
save_dir_vis_res = os.path.join(base_dir, 'YOLOX_outputs/vis_res')
save_prefix_vis_res = "assign_vis_"

# ----------
local_rank = get_local_rank()

device = "cuda:{}".format(get_local_rank())
torch.cuda.set_device(local_rank)


# ----------
print(exp)
print(dir(exp))


############################################################################################
# ------------------------------------------------------------------------------------------
# data set and data loader
# ------------------------------------------------------------------------------------------

no_aug = not exp.enable_mosaic

# load annotations into memory and create index
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


dataset_md = MosaicDetection(
    dataset=dataset,
    mosaic=not no_aug,
    img_size=exp.input_size,
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


sampler = InfiniteSampler(size=len(dataset_md), shuffle=True, seed=0, rank=0, world_size=1)


# here batch size is set
batch_sampler = YoloBatchSampler(
    sampler=sampler,
    batch_size=batch_size,
    drop_last=False,
    mosaic=not no_aug,
    # mosaic=False,
)

# list of (mosaic(bool), index)
# print(next(iter(batch_sampler)))


dataloader_kwargs = {"num_workers": exp.data_num_workers, "pin_memory": True}
dataloader_kwargs["batch_sampler"] = batch_sampler
dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

train_loader = DataLoader(dataset_md, **dataloader_kwargs)
iter_train_loader = iter(train_loader)


# ----------
# check image
inps, targets, img_info, img_id = next(iter_train_loader)
print(img_id)


# for i in range(len(inps)):
# # for i in range(3):
#     PIL.Image.fromarray(cv2.cvtColor(np.array(inps[i]).transpose(1,2,0).astype('uint8'), cv2.COLOR_BGR2RGB)).show()


############################################################################################
# ------------------------------------------------------------------------------------------
# load checkpoint
# ------------------------------------------------------------------------------------------

# ----------
# get model and load checkpoint

in_channels = [256, 512, 1024]

backbone = YOLOPAFPN(exp.depth, exp.width, in_channels=in_channels, act=exp.act)
head = YOLOXHead(exp.num_classes, exp.width, in_channels=in_channels, act=exp.act)
model = YOLOX(backbone, head)

def init_yolo(M):
    for m in M.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eps = 1e-3
            m.momentum = 0.03

model.apply(init_yolo)

prior_prob = 1e-2
model.head.initialize_biases(prior_prob)

model.train()


# ----------
# load checkpoint
model.eval()

ckpt = torch.load(ckpt_file, map_location=device)
model.load_state_dict(ckpt["model"])

model.to(device)


############################################################################################
# ------------------------------------------------------------------------------------------
# get batch to forward
#  1. train:  get assignments --> visualize
#  2. eval:   decode --> postprocess --> visualize
# ------------------------------------------------------------------------------------------

if os.path.exists(save_dir_vis_res):
    shutil.rmtree(save_dir_vis_res)

os.makedirs(save_dir_vis_res, exist_ok=True)


# ----------
for batch_ in range(max_batch):
    # ------------------------------------------------------------------------------------------
    # get one batch
    # ------------------------------------------------------------------------------------------
    inps, targets, img_info, img_id = next(iter_train_loader)
    # PIL.Image.fromarray(cv2.cvtColor(np.array(inps[0]).transpose(1,2,0).astype('uint8'), cv2.COLOR_BGR2RGB)).show()
    # ----------
    # targets.requires_grad = False
    inps, targets = exp.preprocess(inps, targets, exp.input_size)
    # print(inps.shape)
    # print(targets.shape)
    # PIL.Image.fromarray(cv2.cvtColor(np.array(inps[0]).transpose(1,2,0).astype('uint8'), cv2.COLOR_BGR2RGB)).show()
    # ------------------------------------------------------------------------------------------
    # train forward:  fpn_outputs
    # ------------------------------------------------------------------------------------------
    fpn_outs = model.backbone(inps.to(device))
    # ------------------------------------------------------------------------------------------
    # train and eval forward outputs
    # ------------------------------------------------------------------------------------------
    outputs, x_shifts, y_shifts, expanded_strides = [], [], [], []
    outputs_eval = []
    for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
        zip(model.head.cls_convs, model.head.reg_convs, model.head.strides, fpn_outs)
    ):
        x = model.head.stems[k](x)
        cls_x = x
        reg_x = x
        # ----------
        cls_feat = cls_conv(cls_x)
        cls_output = model.head.cls_preds[k](cls_feat)
        reg_feat = reg_conv(reg_x)
        reg_output = model.head.reg_preds[k](reg_feat)
        obj_output = model.head.obj_preds[k](reg_feat)
        # ----------
        output = torch.cat([reg_output, obj_output, cls_output], 1)
        output_eval = torch.cat([reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1)
        # ----------
        output, grid = model.head.get_output_and_grid(output, k, stride_this_level, fpn_outs[0].type())
        x_shifts.append(grid[:, :, 0])
        y_shifts.append(grid[:, :, 1])
        expanded_strides.append(
            torch.full((1, grid.shape[1]), stride_this_level).type_as(fpn_outs[0])
        )
        outputs.append(output)
        outputs_eval.append(output_eval)
    # ------------------------------------------------------------------------------------------
    # train outputs
    # ------------------------------------------------------------------------------------------
    outputs = torch.cat(outputs, 1)
    bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
    obj_preds = outputs[:, :, 4:5]  # [batch, n_anchors_all, 1]
    cls_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]
    # ----------
    # calculate targets
    total_num_anchors = outputs.shape[1]
    x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
    y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
    expanded_strides = torch.cat(expanded_strides, 1)
    bbox_preds = bbox_preds.to('cpu')
    obj_preds = obj_preds.to('cpu')
    cls_preds = cls_preds.to('cpu')
    x_shifts = x_shifts.to('cpu')
    y_shifts = y_shifts.to('cpu')
    expanded_strides = expanded_strides.to('cpu')
    # ------------------------------------------------------------------------------------------
    # eval outputs --> decode
    # ------------------------------------------------------------------------------------------
    hw = [x.shape[-2:] for x in outputs_eval]
    outputs_eval = torch.cat([x.flatten(start_dim=2) for x in outputs_eval], dim=2).permute(0, 2, 1)
    dtype_tmp = fpn_outs[0].type()
    grids, strides = [], []
    for (hsize, wsize), stride in zip(hw, model.head.strides):
        yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
        grid = torch.stack((xv, yv), 2).view(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        strides.append(torch.full((*shape, 1), stride))
    grids = torch.cat(grids, dim=1).type(dtype_tmp)
    strides = torch.cat(strides, dim=1).type(dtype_tmp)
    prediction = torch.cat([
        (outputs_eval[..., 0:2] + grids) * strides,
        torch.exp(outputs_eval[..., 2:4]) * strides,
        outputs_eval[..., 4:]
    ], dim=-1)
    prediction = prediction.to('cpu')
    ############################################################################################
    # ------------------------------------------------------------------------------------------
    # visual assignments
    # ------------------------------------------------------------------------------------------
    # labels is targets and imgs is inps
    labels = targets
    imgs = inps
    # number of objects
    nlabel = (labels.sum(dim=2) > 0).sum(dim=1)
    # ------------------------------------------------------------------------------------------
    # visual asssignments:  get assignments and convert bbox coord to xyxy
    # ------------------------------------------------------------------------------------------
    for batch_idx in range(batch_size):
        # batch_idx = 0
        save_name = save_prefix_vis_res + str(batch_).zfill(2) + "_" + str(batch_idx).zfill(2) + "_0res.png"
        # ----------
        # img = imgs[batch_idx]
        num_gt = nlabel[batch_idx]
        label = labels[batch_idx]
        # ----------
        img = imgs[batch_idx].permute(1, 2, 0).to(torch.uint8)
        num_gt = int(num_gt.cpu())
        # ----------
        if num_gt == 0:
            fg_mask = outputs.new_zeros(total_num_anchors).bool()
        else:
            # ----------
            # model.head.get_assignments --> bbox coord (xyxy)
            gt_bboxes_per_image = label[:num_gt, 1:5]
            gt_classes = label[:num_gt, 0]
            bboxes_preds_per_image = bbox_preds[batch_idx]
            _, fg_mask, _, matched_gt_inds, _ = model.head.get_assignments(
                batch_idx, num_gt, gt_bboxes_per_image, gt_classes,
                bboxes_preds_per_image, expanded_strides, x_shifts,
                y_shifts, cls_preds, obj_preds,
            )
            coords = torch.stack([
                ((x_shifts + 0.5) * expanded_strides).flatten()[fg_mask],
                ((y_shifts + 0.5) * expanded_strides).flatten()[fg_mask],
            ], 1)
            xyxy_boxes = cxcywh2xyxy(gt_bboxes_per_image)
        # ------------------------------------------------------------------------------------------
        # visualize_assign from yolox.utils : demo_utils.py
        # ------------------------------------------------------------------------------------------
        # copy is crucial here
        img_vis = img.cpu().numpy().copy()
        for box_id, box in enumerate(xyxy_boxes):
            x1, y1, x2, y2 = box
            color = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
            assign_coords = coords[matched_gt_inds == box_id]
            if assign_coords.numel() == 0:
                # unmatched boxes are red
                color = (0, 0, 255)
                img_vis = cv2.putText(img_vis, "unmatched", (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            else:
                for coord in assign_coords:
                    # draw assigned anchor
                    img_vis = cv2.circle(img_vis, (int(coord[0]), int(coord[1])), 3, color, -1)
            # ----------
            img_vis = cv2.rectangle(img_vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        # PIL.Image.fromarray(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)).show()
        cv2.imwrite(os.path.join(save_dir_vis_res, save_name), img_vis)
    ############################################################################################
    # ------------------------------------------------------------------------------------------
    # eval decoded outputs to postprocess
    # ------------------------------------------------------------------------------------------
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]
    pred_output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):
        image_pred = prediction[i]
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + exp.num_classes], 1, keepdim=True)
        # image_pred[:, 4] is object confidence score
        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= exp.test_conf).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue
        if class_agnostic:
            nms_out_index = torchvision.ops.nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                exp.nmsthre,
            )
        else:
            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                exp.nmsthre,
            )
        detections = detections[nms_out_index]
        if pred_output[i] is None:
            pred_output[i] = detections
        else:
            pred_output[i] = torch.cat((pred_output[i], detections))
    # ------------------------------------------------------------------------------------------
    # eval postprocessed output to visualization
    # ------------------------------------------------------------------------------------------
    font = cv2.FONT_HERSHEY_SIMPLEX
    ratio = 1.0
    for batch_idx in range(batch_size):
        # batch_idx = 0
        output = pred_output[batch_idx]
        # ----------
        save_name = save_prefix_vis_res + str(batch_).zfill(2) + "_" + str(batch_idx).zfill(2) + "_1pred.png"
        img_vis = imgs[batch_idx].permute(1, 2, 0).to(torch.uint8).cpu().numpy().copy()
        # ----------
        if output is None:
            img_vis = np.array(inps[batch_idx]).transpose(1,2,0).astype('uint8')
        else:
            # ----------
            output = output.cpu()
            bboxes, cls_ids, scores = output[:, 0:4]/ratio, output[:, 6], output[:, 4] * output[:, 5]
            # ----------
            # visualizaton
            for i in range(len(bboxes)):
                box, cls_id, score = bboxes[i], int(cls_ids[i]), scores[i]
                if score < exp.test_conf:
                    continue
                x0, y0, x1, y1 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                # ----------
                color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
                text = '{}:{:.1f}%'.format(cls_names[cls_id], score * 100)
                txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
                # ----------
                txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
                img_vis = cv2.rectangle(img_vis, (x0, y0), (x1, y1), color, 2)
                # ----------
                txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
                img_vis = cv2.rectangle(img_vis, (x0, y0 + 1), (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])), txt_bk_color, -1)
                img_vis = cv2.putText(img_vis, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)
        cv2.imwrite(os.path.join(save_dir_vis_res, save_name), img_vis)


