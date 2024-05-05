
import os
import sys
import importlib

import torch
import torch.nn as nn

from yolox.models.yolo_pafpn import YOLOPAFPN
from yolox.models.yolo_head import YOLOXHead
from yolox.models.yolox import YOLOX

from yolox.utils import *

from yolox.data import DataPrefetcher
from yolox.core import Trainer


# ----------
base_dir = '/home/kswada/kw/yolox/YOLOX'


############################################################################################
# ------------------------------------------------------------------------------------------
# tools : visualize_assign.py
# AssignVisualizer()
# ------------------------------------------------------------------------------------------

class AssignVisualizer(Trainer):

    def __init__(self, exp: Exp, args):
        super().__init__(exp, args)
        self.batch_cnt = 0
        self.vis_dir = os.path.join(self.file_name, "vis")
        os.makedirs(self.vis_dir, exist_ok=True)

    def train_one_iter(self):
        iter_start_time = time.time()

        inps, targets = self.prefetcher.next()
        inps = inps.to(self.data_type)
        targets = targets.to(self.data_type)
        targets.requires_grad = False
        inps, targets = self.exp.preprocess(inps, targets, self.input_size)
        data_end_time = time.time()

        with torch.cuda.amp.autocast(enabled=self.amp_training):
            path_prefix = os.path.join(self.vis_dir, f"assign_vis_{self.batch_cnt}_")
            self.model.visualize(inps, targets, path_prefix)

        if self.use_model_ema:
            self.ema_model.update(self.model)

        iter_end_time = time.time()
        self.meter.update(
            iter_time=iter_end_time - iter_start_time,
            data_time=data_end_time - iter_start_time,
        )
        self.batch_cnt += 1
        if self.batch_cnt >= self.args.max_batch:
            sys.exit(0)

    def after_train(self):
        logger.info("Finish visualize assignment, exit...")


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

depth = exp.depth
width = exp.width
act = exp.act
num_classes = exp.num_classes
input_size = exp.input_size

print(f'depth: {depth}  width: {width}  act: {act}  num_classes: {num_classes}  input_size: {input_size}')


# ----------
in_channels = [256, 512, 1024]

backbone = YOLOPAFPN(depth, width, in_channels=in_channels, act=act)

head = YOLOXHead(num_classes, width, in_channels=in_channels, act=act)

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


# ------------------------------------------------------------------------------------------
# before train
# ------------------------------------------------------------------------------------------

model.to(device)


# ----------
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


# ----------
evaluator = exp.get_evaluator(
    batch_size=batch_size, is_distributed=is_distributed
)


# ------------------------------------------------------------------------------------------
# train in epoch
#  - before epoch
# ------------------------------------------------------------------------------------------

train_loader.close_mosaic()

model.head.use_l1 = True

exp.eval_interval = 1


# ------------------------------------------------------------------------------------------
# train in epoch
#  - train in iter
#     - before iter --> pass
# ------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------
# train in epoch
#  - train in iter
#     - train one iter  --> this is from  class AssingVisualizer : train_one_iter
# ------------------------------------------------------------------------------------------

inps, targets = prefetcher.next()

# (batch_size, input_size[0], input_size[1])
print(inps.shape)

# (batch_size, 120, 5)
print(targets.shape)


# ----------
data_type = torch.float32

inps = inps.to(data_type)
targets = targets.to(data_type)


# ----------
targets.requires_grad = False

inps, targets = exp.preprocess(inps, targets, input_size)
print(inps.shape)
print(targets.shape)


# ------------------------------------------------------------------------------------------
# model.visualize()
# -->  yolox.models : yolox.py  class YOLOX : visualize()
# ------------------------------------------------------------------------------------------

##### self.model.visualize(inps, targets, path_prefix)

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


# -----------
# JUST FOR REFERENCE
# head
loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = model.head(fpn_outs, targets, inps)

outputs = {
    "total_loss": loss,
    "iou_loss": iou_loss,
    "l1_loss": l1_loss,
    "conf_loss": conf_loss,
    "cls_loss": cls_loss,
    "num_fg": num_fg,
}

print(outputs)


# ----------
##### self.head.visualize_assign_result(fpn_outs, targets, x, save_prefix)
# --> yolox.models : yolo_head.py : visualize_assign_result

# 3
print(len(model.head.cls_convs))
print(len(model.head.reg_convs))
print(len(model.head.strides))


# ----------
k = 0
cls_conv = model.head.cls_convs[0]
reg_conv = model.head.reg_convs[0]
stride_this_level = model.head.strides[0]
x = fpn_outs[0]

x = model.head.stems[k](x)
print(x.shape)
cls_x = x
reg_x = x

cls_feat = cls_conv(cls_x)
print(cls_feat.shape)

cls_output = model.head.cls_preds[k](cls_feat)
print(cls_output.shape)

reg_feat = reg_conv(reg_x)
print(reg_feat.shape)

reg_output = model.head.reg_preds[k](reg_feat)
obj_output = model.head.obj_preds[k](reg_feat)
print(reg_output.shape)
print(obj_output.shape)

output = torch.cat([reg_output, obj_output, cls_output], 1)
output, grid = model.head.get_output_and_grid(output, k, stride_this_level, fpn_outs[0].type())
print(output.shape)
print(grid.shape)

print(torch.full((1, grid.shape[1]), stride_this_level).type_as(fpn_outs[0]))
print(len(torch.full((1, grid.shape[1]), stride_this_level).type_as(fpn_outs[0])[0]))


# ----------
outputs, x_shifts, y_shifts, expanded_strides = [], [], [], []

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
    output, grid = model.head.get_output_and_grid(output, k, stride_this_level, fpn_outs[0].type())
    x_shifts.append(grid[:, :, 0])
    y_shifts.append(grid[:, :, 1])
    expanded_strides.append(
        torch.full((1, grid.shape[1]), stride_this_level).type_as(fpn_outs[0])
    )
    outputs.append(output)


# ----------
print(len(outputs))

outputs = torch.cat(outputs, 1)
print(len(outputs))

bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
obj_preds = outputs[:, :, 4:5]  # [batch, n_anchors_all, 1]
cls_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]

print(bbox_preds.shape)
print(obj_preds.shape)
print(cls_preds.shape)


# ----------
# calculate targets

total_num_anchors = outputs.shape[1]

x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]

y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]

expanded_strides = torch.cat(expanded_strides, 1)


# ----------
# labels is targets and imgs is inps
labels = targets
imgs = inps

# number of objects
nlabel = (labels.sum(dim=2) > 0).sum(dim=1)


# ----------
batch_idx = 0
img = imgs[0]
num_gt = nlabel[0]
label = labels[0]

img = imgs[batch_idx].permute(1, 2, 0).to(torch.uint8)
num_gt = int(num_gt)


gt_bboxes_per_image = label[:num_gt, 1:5]

gt_classes = label[:num_gt, 0]

bboxes_preds_per_image = bbox_preds[batch_idx]

_, fg_mask, _, matched_gt_inds, _ = model.head.get_assignments(
    batch_idx, num_gt, gt_bboxes_per_image, gt_classes,
    bboxes_preds_per_image, expanded_strides, x_shifts,
    y_shifts, cls_preds, obj_preds,
)


img = img.cpu().numpy().copy()  # copy is crucial here

coords = torch.stack([
    ((x_shifts + 0.5) * expanded_strides).flatten()[fg_mask],
    ((y_shifts + 0.5) * expanded_strides).flatten()[fg_mask],
], 1)

xyxy_boxes = cxcywh2xyxy(gt_bboxes_per_image)

save_prefix = "assign_vis_"
save_name = save_prefix + str(batch_idx) + ".png"


# visualize_assign from yolox.utils : demo_utils.py
img = visualize_assign(img, xyxy_boxes, coords, matched_gt_inds, save_name)


# ----------
for batch_idx, (img, num_gt, label) in enumerate(zip(imgs, nlabel, labels)):
    img = imgs[batch_idx].permute(1, 2, 0).to(torch.uint8)
    num_gt = int(num_gt)
    if num_gt == 0:
        fg_mask = outputs.new_zeros(total_num_anchors).bool()
    else:
        gt_bboxes_per_image = label[:num_gt, 1:5]
        gt_classes = label[:num_gt, 0]
        bboxes_preds_per_image = bbox_preds[batch_idx]
        _, fg_mask, _, matched_gt_inds, _ = model.head.get_assignments(  # noqa
            batch_idx, num_gt, gt_bboxes_per_image, gt_classes,
            bboxes_preds_per_image, expanded_strides, x_shifts,
            y_shifts, cls_preds, obj_preds,
        )

    img = img.cpu().numpy().copy()  # copy is crucial here
    coords = torch.stack([
        ((x_shifts + 0.5) * expanded_strides).flatten()[fg_mask],
        ((y_shifts + 0.5) * expanded_strides).flatten()[fg_mask],
    ], 1)

    xyxy_boxes = cxcywh2xyxy(gt_bboxes_per_image)
    save_name = save_prefix + str(batch_idx) + ".png"
    img = visualize_assign(img, xyxy_boxes, coords, matched_gt_inds, save_name)
    logger.info(f"save img to {save_name}")



# ------------------------------------------------------------------------------------------
# train in epoch
#  - train in iter
#     - after iter
# ------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------
# after epoch
# ------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------
# after train   --> this is from  class AssignVisualizer : after_train
# ------------------------------------------------------------------------------------------
