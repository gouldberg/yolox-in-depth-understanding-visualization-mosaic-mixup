
import os
import sys
import importlib
import random

import numpy as np

from copy import deepcopy
from collections import defaultdict
import itertools

import torch
import torch.nn as nn
import torchvision

from yolox.utils import *

from yolox.models.yolo_pafpn import YOLOPAFPN
from yolox.models.yolo_head import YOLOXHead
from yolox.models.yolox import YOLOX

from yolox.data import COCODataset, ValTransform
from yolox.evaluators import COCOEvaluator

# from yolox.exp import get_evaluater

import cv2
import PIL.Image

# import io
# import itertools
import json
import tempfile
from tabulate import tabulate


# ----------
# REFERENCE:
# https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py


############################################################################################
# ------------------------------------------------------------------------------------------
# tools : eval.py
# ------------------------------------------------------------------------------------------

##########
# ----------
# ORIGINAL SCRIPT
# ----------
# if args.conf is not None:
#     exp.test_conf = args.conf
# if args.nms is not None:
#     exp.nmsthre = args.nms
# if args.tsize is not None:
#     exp.test_size = (args.tsize, args.tsize)

# model = exp.get_model()
# logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
# logger.info("Model Structure:\n{}".format(str(model)))

# evaluator = exp.get_evaluator(args.batch_size, is_distributed, args.test, args.legacy)
# evaluator.per_class_AP = True
# evaluator.per_class_AR = True

# torch.cuda.set_device(rank)
# model.cuda(rank)
# model.eval()
##########


############################################################################################
# ------------------------------------------------------------------------------------------
# eval.py  step by step
#  - experiment config
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



# ------------------------------------------------------------------------------------------
# eval.py  step by step
#  - get model
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

model.head.initialize_biases(1e-2)

model.train()


# ------------------------------------------------------------------------------------------
# eval.py  step by step
#  - eval dataset
# ------------------------------------------------------------------------------------------

data_dir = os.path.join(base_dir, 'datasets/COCO')

val_ann = "instances_val2017.json"
test_ann = "instances_test2017.json"

testdev = False
legacy = False

# output image size during evaluation/test
test_size = (640, 640)


##### valdataset = self.get_eval_dataset(**kwargs)
# ---------->
valdataset = COCODataset(
            data_dir=data_dir,
            json_file=val_ann if not testdev else test_ann,
            name="val2017" if not testdev else "test2017",
            img_size=test_size,
            preproc=ValTransform(legacy=legacy),
        )


# ----------
# check
print(len(valdataset))

idx = random.randint(0, len(valdataset) - 1)

img, labels, img_info, img_id = valdataset.pull_item(idx)

print(img.shape)
print(labels.shape)

# height, width
print(img_info)

print(f'selected image: {idx}')
print(f'img_id: {img_id}')
print(f'len labels: {len(labels}')

for i in range(len(labels)):
    tmp_label = labels[i]
    cv2.rectangle(
        img=img,
        pt1=(int(tmp_label[0]), int(tmp_label[1])),
        pt2=(int(tmp_label[2]), int(tmp_label[3])),
        color=(255, 0, 0),
        thickness=3
    )

PIL.Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).show()



# ------------------------------------------------------------------------------------------
# eval.py  step by step
#  - eval dataloader
# ------------------------------------------------------------------------------------------

sampler = torch.utils.data.SequentialSampler(valdataset)
print(list(sampler))


# ----------
# set worker to 4 for shorter dataloader init time
# If your training process cost many memory, reduce this value.
data_num_workers = 4

dataloader_kwargs = {
    "num_workers": data_num_workers,
    "pin_memory": True,
    "sampler": sampler,
}

# batch_size = 8
batch_size = 64

dataloader_kwargs["batch_size"] = batch_size

val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)


# ------------------------------------------------------------------------------------------
# eval.py  step by step
#  - eval evaluator
# ------------------------------------------------------------------------------------------

# confidence threshold during evaluation/test,
# boxes whose scores are less than test_conf will be filtered
test_conf = 0.01

# nms threshold
nmsthre = 0.65

evaluator = COCOEvaluator(
            dataloader=val_loader,
            img_size=test_size,
            confthre=test_conf,
            nmsthre=nmsthre,
            num_classes=exp.num_classes,
            testdev=testdev,
        )

evaluator.per_class_AP = True
evaluator.per_class_AR = True


# ------------------------------------------------------------------------------------------
# yolox.evaluators : coco_evaluator.py  convert_to_coco_format
# ------------------------------------------------------------------------------------------

def convert_to_coco_format(outputs, info_imgs, ids, return_outputs=False):
    data_list = []
    image_wise_data = defaultdict(dict)
    for (output, img_h, img_w, img_id) in zip(
        outputs, info_imgs[0], info_imgs[1], ids
    ):
        if output is None:
            continue
        output = output.cpu()
        # ----------
        bboxes = output[:, 0:4]
        # ----------
        # preprocessing: resize
        scale = min(
            test_size[0] / float(img_h), test_size[1] / float(img_w)
        )
        bboxes /= scale
        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]
        # ----------
        image_wise_data.update({
            int(img_id): {
                "bboxes": [box.numpy().tolist() for box in bboxes],
                "scores": [score.numpy().item() for score in scores],
                "categories": [
                    val_loader.dataset.class_ids[int(cls[ind])]
                    for ind in range(bboxes.shape[0])
                ],
            }
        })
        # ----------
        bboxes = xyxy2xywh(bboxes)
        # ----------
        for ind in range(bboxes.shape[0]):
            label = val_loader.dataset.class_ids[int(cls[ind])]
            pred_data = {
                "image_id": int(img_id),
                "category_id": label,
                "bbox": bboxes[ind].numpy().tolist(),
                "score": scores[ind].numpy().item(),
                "segmentation": [],
            }  # COCO json format
            data_list.append(pred_data)
    # ----------
    if return_outputs:
        return data_list, image_wise_data
    return data_list



# ------------------------------------------------------------------------------------------
# eval.py  step by step
#  - model to eval and load checkpoint
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
trt_file = None

decoder = None


# ------------------------------------------------------------------------------------------
# eval.py  step by step
#  - evaluate  (in yolox.evaluators : coco_evaluater.py)
# ------------------------------------------------------------------------------------------

#####
# *_, summary = evaluator.evaluate(
#     model, is_distributed, args.fp16, trt_file, decoder, exp.test_size
# )
# ---------->
fp16 = True
half = fp16

tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor

model = model.eval()

model = model.half()

ids = []
data_list = []
output_data = defaultdict()
# progress_bar = tqdm if is_main_process() else iter

inference_time = 0
nms_time = 0
n_samples = max(len(val_loader) - 1, 1)


# ----------
cur_iter = 0

imgs, _, info_imgs, ids = next(iter(val_loader))

print(imgs.shape)
print(ids)


# ----------
with torch.no_grad():
    imgs = imgs.type(tensor_type)
    # ----------
    outputs = model(imgs)
    # ----------
    # (batch_size, 80*80, 85)
    # 85 = cx, cy, w, h, conf + 80 classes
    # outputs IS ALREADY DECODED
    print(outputs.shape)
    # ----------
    # utils.boxes postprocess()
    # note that bboxes is (x1, x2, y1, y2)
    # ----------
    # outputs = postprocess(
    #     outputs, exp.num_classes, test_conf, nmsthre
    # )
    # ----------
    # test_conf: 0.01 --> 0.7
    outputs = postprocess(
        outputs, exp.num_classes, 0.7, nmsthre
    )

# 8
print(len(outputs))

# (some numbers, 7)
print(outputs[0].shape)


# ----------
# convert to coco format

data_list_elem, image_wise_data = convert_to_coco_format(
    outputs, info_imgs, ids, return_outputs=True)

data_list.extend(data_list_elem)
output_data.update(image_wise_data)

print(len(data_list))
print(data_list[0])


# ----------
for k, v in output_data.items():
    print(k)

for k, v in output_data[ids[0].item()].items():
    print(k)


print(output_data[ids[0].item()]['bboxes'])
print(output_data[ids[0].item()]['scores'])
print(output_data[ids[0].item()]['categories'])


# ----------
# GT
for i in range(10):
    print(data_list[i])

# inference
for i in range(10):
    print(ids[i].item())
    print(output_data[ids[i].item()])


# ------------------------------------------------------------------------------------------
# check step by step
# yolox.evaluators : coco_evaluator.py  convert_to_coco_format
# ------------------------------------------------------------------------------------------

# data_list = []

# image_wise_data = defaultdict(dict)


# # ----------
# i = 0

# output = outputs[i]

# img_h = info_imgs[0][i]
# img_w = info_imgs[1][i]
# img_id = ids[i]

# output = output.cpu()

# bboxes = output[:, 0:4]

# scale = min(
#     test_size[0] / float(img_h), test_size[1] / float(img_w)
# )

# bboxes /= scale
# cls = output[:, 6]

# # score is the product
# scores = output[:, 4] * output[:, 5]


# # ----------
# image_wise_data.update({
#     int(img_id): {
#         "bboxes": [box.numpy().tolist() for box in bboxes],
#         "scores": [score.numpy().item() for score in scores],
#         "categories": [
#             val_loader.dataset.class_ids[int(cls[ind])]
#             for ind in range(bboxes.shape[0])
#         ],
#     }
# })

# print(image_wise_data)


# # ----------
# # now bboxes is converted to original (before postprocess) cx, cy, w, h
# bboxes = xyxy2xywh(bboxes)
# print(bboxes.shape)


# # ----------
# for ind in range(bboxes.shape[0]):
#     label = val_loader.dataset.class_ids[int(cls[ind])]
#     pred_data = {
#         "image_id": int(img_id),
#         "category_id": label,
#         "bbox": bboxes[ind].numpy().tolist(),
#         "score": scores[ind].numpy().item(),
#         "segmentation": [],
#     }  # COCO json format
#     data_list.append(pred_data)


# ------------------------------------------------------------------------------------------
# yolox.evaluators : coco_evaluator.py
# per_class_AR_table,  per_class_AP_table
# ------------------------------------------------------------------------------------------

from yolox.data.datasets import COCO_CLASSES

def per_class_AR_table(coco_eval, class_names=COCO_CLASSES, headers=["class", "AR"], colums=6):
    per_class_AR = {}
    recalls = coco_eval.eval["recall"]
    # dimension of recalls: [TxKxAxM]
    # recall has dims (iou, cls, area range, max dets)
    assert len(class_names) == recalls.shape[1]
    # ----------
    for idx, name in enumerate(class_names):
        recall = recalls[:, idx, 0, -1]
        recall = recall[recall > -1]
        ar = np.mean(recall) if recall.size else float("nan")
        per_class_AR[name] = float(ar * 100)
    # ----------
    num_cols = min(colums, len(per_class_AR) * len(headers))
    result_pair = [x for pair in per_class_AR.items() for x in pair]
    row_pair = itertools.zip_longest(*[result_pair[i::num_cols] for i in range(num_cols)])
    table_headers = headers * (num_cols // len(headers))
    table = tabulate(
        row_pair, tablefmt="pipe", floatfmt=".3f", headers=table_headers, numalign="left",
    )
    return table


def per_class_AP_table(coco_eval, class_names=COCO_CLASSES, headers=["class", "AP"], colums=6):
    per_class_AP = {}
    precisions = coco_eval.eval["precision"]
    # dimension of precisions: [TxRxKxAxM]
    # precision has dims (iou, recall, cls, area range, max dets)
    assert len(class_names) == precisions.shape[2]
    # ----------
    for idx, name in enumerate(class_names):
        # area range index 0: all area ranges
        # max dets index -1: typically 100 per image
        precision = precisions[:, :, idx, 0, -1]
        precision = precision[precision > -1]
        ap = np.mean(precision) if precision.size else float("nan")
        per_class_AP[name] = float(ap * 100)
    # ----------
    num_cols = min(colums, len(per_class_AP) * len(headers))
    result_pair = [x for pair in per_class_AP.items() for x in pair]
    row_pair = itertools.zip_longest(*[result_pair[i::num_cols] for i in range(num_cols)])
    table_headers = headers * (num_cols // len(headers))
    table = tabulate(
        row_pair, tablefmt="pipe", floatfmt=".3f", headers=table_headers, numalign="left",
    )
    return table


# ------------------------------------------------------------------------------------------
# eval.py  step by step
#  - evaluate_prediction
# ------------------------------------------------------------------------------------------

annType = ["segm", "bbox", "keypoints"]

# ----------
# Evaluate the Dt (detection) json comparing with the ground truth

# val_loader.dataset.coco is instance of
# from pycocotools.coco import COCO'
# self.coco = COCO(os.path.join(self.data_dir, "annotations", self.json_file))

# 0. annotation
cocoGt = val_loader.dataset.coco


# 1. result
# https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py
# loadRes: Load result file and return a result api object

if testdev:
    json.dump(data_list, open("./yolox_testdev_2017.json", "w"))
    cocoDt = cocoGt.loadRes("./yolox_testdev_2017.json")
else:
    # dump data_list
    _, tmp = tempfile.mkstemp()
    json.dump(data_list, open(tmp, "w"))
    cocoDt = cocoGt.loadRes(tmp)


# ----------
# now import original (modified) COCOeval
try:
    from yolox.layers import COCOeval_opt as COCOeval
except ImportError:
    from pycocotools.cocoeval import COCOeval


# ----------
cocoEval = COCOeval(cocoGt, cocoDt, annType[1])


# ----------
# https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py

# Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
cocoEval.evaluate()

# Accumulate per image evaluation results and store the result in self.eval
cocoEval.accumulate()

# NOTE now the gt includes all data, but prediction is only 1 batch
print(cocoEval.summarize())


# ----------
cat_ids = list(cocoGt.cats.keys())

cat_names = [cocoGt.cats[catId]['name'] for catId in sorted(cat_ids)]

if evaluator.per_class_AP:
    AP_table = per_class_AP_table(cocoEval, class_names=cat_names)

if evaluator.per_class_AR:
    AR_table = per_class_AR_table(cocoEval, class_names=cat_names)


# ----------
# AP IOU=0.50:0.95
print(cocoEval.stats[0])
# AP IOU=0.50
print(cocoEval.stats[1])

print(AP_table)

print(AR_table)

# return cocoEval.stats[0], cocoEval.stats[1], info



############################################################################################
# ------------------------------------------------------------------------------------------
# utils : boxes.py  postprocess()
# ------------------------------------------------------------------------------------------

imgs, _, info_imgs, ids = next(iter(val_loader))

print(imgs.shape)
print(ids)


# ----------
with torch.no_grad():
    imgs = imgs.type(tensor_type)
    # ----------
    outputs = model(imgs)
    # ----------
    # (batch_size, 80*80, 85)
    print(outputs.shape)


#####
# outputs = postprocess(
#     outputs, exp.num_classes, test_conf, nmsthre
# )
# ---------->
num_classes = exp.num_classes
conf_thre = test_conf
nms_thre = nmsthre
class_agnostic = False

prediction = outputs

# (batch_size, 60*60, 85)
print(prediction.shape)


# ----------
box_corner = prediction.new(prediction.shape)

box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
prediction[:, :, :4] = box_corner[:, :, :4]

output = [None for _ in range(len(prediction))]


# ----------
# for i, image_pred in enumerate(prediction):

i = 0
image_pred = prediction[i]

# (8400, 85)
print(image_pred.shape)

# If none are remaining => process next image
# if not image_pred.size(0):
#     continue

# Get score and class with highest confidence
class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)
print(class_conf.shape)
print(class_pred.shape)

# image_pred[:, 4] is object confidence score
conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()

# Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)

detections = detections[conf_mask]

if not detections.size(0):
    continue


# ----------
# https://pytorch.org/vision/main/generated/torchvision.ops.batched_nms.html
# torchvision.ops.batched_nms:
#   - Each index value correspond to a category,
#      and NMS will not be applied between elements of different categories.
#
# boxes (Tensor[N, 4]) – boxes where NMS will be performed. They are expected to be in (x1, y1, x2, y2) format with 0 <= x1 < x2 and 0 <= y1 < y2.
# scores (Tensor[N]) – scores for each one of the boxes
# idxs (Tensor[N]) – indices of the categories for each one of the boxes.
# iou_threshold (float) – discards all overlapping boxes with IoU > iou_threshold

# Returns: int64 tensor with the indices of the elements
# that have been kept by NMS, sorted in decreasing order of scores

if class_agnostic:
    nms_out_index = torchvision.ops.nms(
        detections[:, :4],
        detections[:, 4] * detections[:, 5],
        nms_thre,
    )
else:
    nms_out_index = torchvision.ops.batched_nms(
        detections[:, :4],
        detections[:, 4] * detections[:, 5],
        detections[:, 6],
        nms_thre,
    )

# -->
# obj_conf * class_conf

print(nms_out_index)

detections = detections[nms_out_index]

if output[i] is None:
    output[i] = detections
else:
    output[i] = torch.cat((output[i], detections))

# return output


############################################################################################
# ------------------------------------------------------------------------------------------
# coco eval  step by step
# ------------------------------------------------------------------------------------------

annType = ["segm", "bbox", "keypoints"]


# 0. annotation
# val_loader.dataset.coco is instance of
# from pycocotools.coco import COCO'
# self.coco = COCO(os.path.join(self.data_dir, "annotations", self.json_file))
cocoGt = val_loader.dataset.coco


# 1. result
# https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py
# loadRes: Load result file and return a result api object

if testdev:
    json.dump(data_list, open("./yolox_testdev_2017.json", "w"))
    cocoDt = cocoGt.loadRes("./yolox_testdev_2017.json")
else:
    # dump data_list
    _, tmp = tempfile.mkstemp()
    json.dump(data_list, open(tmp, "w"))
    cocoDt = cocoGt.loadRes(tmp)

print(len(data_list))
print(len(cocoDt.imgs))


# ----------
# now import original (modified) COCOeval
try:
    from yolox.layers import COCOeval_opt as COCOeval
except ImportError:
    from pycocotools.cocoeval import COCOeval


# ----------
cocoEval = COCOeval(cocoGt, cocoDt, annType[1])


# ----------
# only 1 image evaluation
idx = 0
imgId = ids[idx]

cocoEval.params.imgIds = imgId
cocoEval.evaluate()
cocoEval.accumulate()


# ----------
# result summarization
cocoEval.summarize()
cocoEval.stats


# ----------
# ious between all gts and dts
cocoEval.ious

# GT
data_list[idx]


# ----------
for k, v in cocoEval.eval.items():
    print(k)

# T R K A M
# T=10: IoU thresholds for evaluation
# R=101 recall thresholds for evaluation
# K cat ids to use evaluation
# A=4 object area ranges for evaluation
# M=3 threholds on max detections per image

print(cocoEval.eval['scores'].shape)
print(cocoEval.eval['precision'].shape)

# T K A M
print(cocoEval.eval['recall'].shape)


############################
# evaluate all loaded images
cocoEval.params.imgIds = ids
cocoEval.evaluate()
cocoEval.accumulate()

cocoEval.summarize()
cocoEval.stats


# ----------
# per_class_AR_table

cat_ids = list(cocoGt.cats.keys())
cat_names = [cocoGt.cats[catId]['name'] for catId in sorted(cat_ids)]
headers=['class', 'AR']
columns = 6

per_class_AR = {}

recalls = cocoEval.eval["recall"]
# dimension of recalls: [TxKxAxM]
# recall has dims (iou, cls, area range, max dets)

assert len(cat_names) == recalls.shape[1]

for idx, name in enumerate(cat_names):
    recall = recalls[:, idx, 0, -1]
    recall = recall[recall > -1]
    ar = np.mean(recall) if recall.size else float("nan")
    per_class_AR[name] = float(ar * 100)

print(per_class_AR)

num_cols = min(columns, len(per_class_AR) * len(headers))
result_pair = [x for pair in per_class_AR.items() for x in pair]
row_pair = itertools.zip_longest(*[result_pair[i::num_cols] for i in range(num_cols)])
table_headers = headers * (num_cols // len(headers))
table = tabulate(
    row_pair, tablefmt="pipe", floatfmt=".3f", headers=table_headers, numalign="left",
)

table
