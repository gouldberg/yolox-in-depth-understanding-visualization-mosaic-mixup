
import os
import sys
import importlib

import random
import numpy as np
import copy

from pycocotools.coco import COCO

from yolox.data import COCODataset, TrainTransform

import cv2
import PIL.Image


############################################################################################
# ------------------------------------------------------------------------------------------
# 0-1. yolox.data.datasets : coco.py  class COCODataset
#    __init__  (loading annotations)
# ------------------------------------------------------------------------------------------

base_dir = '/home/kswada/kw/yolox/YOLOX/'

yolox_datadir = os.path.join(base_dir, 'datasets/COCO')

# ----------
data_dir = yolox_datadir
json_file = 'instances_train2017.json'

# ----------
# load annotations into memory
coco = COCO(os.path.join(data_dir, "annotations", json_file))


# ----------
# Remove useless info in coco dataset.COCO object is modified inplace.
# This function is mainly used for saving memory(save about 30 %mem).
### remove_useless_info(self.coco)

# ---------->
dataset = coco.dataset
dataset.pop("info", None)
dataset.pop("licenses", None)

for img in dataset["images"]:
    img.pop("license", None)
    img.pop("coco_url", None)
    img.pop("date_captured", None)
    img.pop("flickr_url", None)

if "annotations" in coco.dataset:
    for anno in coco.dataset["annotations"]:
        anno.pop("segmentation", None)

# ----------
ids = coco.getImgIds()
# 118287 images
print(len(ids))
num_imgs = len(ids)

class_ids = sorted(coco.getCatIds())
cats = coco.loadCats(coco.getCatIds())

name = "train2017"
_classes = tuple([c["name"] for c in cats])

# (height, width)
img_size = (640, 640)

# currently set None
preproc = None


# ----------
# load annotions
annotations_list = []

for id_ in ids:
    im_ann = coco.loadImgs(id_)[0]
    width = im_ann["width"]
    height = im_ann["height"]
    # ----------
    # only iscrowd=False    
    anno_ids = coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
    annotations = coco.loadAnns(anno_ids)
    # print(annotations)
    # ----------
    # convert (xmin, ymin, width, height) --> (xmin, ymin, xmax, ymax)
    objs = []
    for obj in annotations:
        x1 = np.max((0, obj["bbox"][0]))
        y1 = np.max((0, obj["bbox"][1]))
        x2 = np.min((width, x1 + np.max((0, obj["bbox"][2]))))
        y2 = np.min((height, y1 + np.max((0, obj["bbox"][3]))))
        if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
            obj["clean_bbox"] = [x1, y1, x2, y2]
            objs.append(obj)
    # ----------
    num_objs = len(objs)
    # print(num_objs)
    res = np.zeros((num_objs, 5))
    for ix, obj in enumerate(objs):
        cls = class_ids.index(obj["category_id"])
        res[ix, 0:4] = obj["clean_bbox"]
        res[ix, 4] = cls
    # ----------
    # bbox is resized
    r = min(img_size[0] / height, img_size[1] / width)
    res[:, :4] *= r
    # ----------
    img_info = (height, width)
    resized_info = (int(height * r), int(width * r))
    # ----------
    file_name = (
        im_ann["file_name"]
        if "file_name" in im_ann
        else "{:012}".format(id_) + ".jpg"
    )
    # ----------
    annotations_list.append((res, img_info, resized_info, file_name))


# ----------
annotations = annotations_list
path_filename = [os.path.join(name, anno[3]) for anno in annotations]

# 118287
print(len(annotations))
print(len(path_filename))


# ----------
# now the bounding box is resized and (xmin, ymin, xmax, ymax)
print(annotations[1])


# ------------------------------------------------------------------------------------------
# 0-2. yolox.data.datasets : coco.py  class COCODataset
#    pull_item() by index  (<-- read_img() <-- load_resized_img() <-- load_img())
# ------------------------------------------------------------------------------------------

index = 0


# ----------
id_ = ids[index]
label, origin_image_size, _, _ = annotations[index]


# load_img()
file_name = annotations[index][3]
img_file = os.path.join(data_dir, name, file_name)
img = cv2.imread(img_file)


# load_resized_img()
r = min(img_size[0] / img.shape[0], img_size[1] / img.shape[1])

resized_img = cv2.resize(
    img,
    (int(img.shape[1] * r), int(img.shape[0] * r)),
    interpolation=cv2.INTER_LINEAR,
).astype(np.uint8)


# ----------
print(img.shape)
print(resized_img.shape)

# return resized_img, copy.deepcopy(label), origin_image_size, np.array([id_])


# ------------------------------------------------------------------------------------------
# 0-2. yolox.data.datasets : coco.py  class COCODataset
#    __getitem__()  ( <-- pull_item() )
#
#      yolox.data.datasets : coco.py  proproc
# ------------------------------------------------------------------------------------------

#################################
# HELPERS FUNCTIONS in yolox.data data_augment.py

def xyxy2cxcywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] * 0.5
    bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] * 0.5
    return bboxes

# ----------
def augment_hsv(img, hgain=5, sgain=30, vgain=30):
    hsv_augs = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain]  # random gains
    hsv_augs *= np.random.randint(0, 2, 3)  # random selection of h, s, v
    hsv_augs = hsv_augs.astype(np.int16)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
    # ----------
    img_hsv[..., 0] = (img_hsv[..., 0] + hsv_augs[0]) % 180
    img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_augs[1], 0, 255)
    img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_augs[2], 0, 255)
    # ----------
    cv2.cvtColor(img_hsv.astype(img.dtype), cv2.COLOR_HSV2BGR, dst=img)  # no return needed

# ----------
# _mirror handles bounding box
def _mirror(image, boxes, prob=0.5):
    _, width, _ = image.shape
    if random.random() < prob:
        image = image[:, ::-1]
        boxes[:, 0::2] = width - boxes[:, 2::-2]
    return image, boxes

# ----------
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
#################################

# img, target, img_info, img_id = self.pull_item(index)

# if self.preproc is not None:
#     img, target = self.preproc(img, target, self.input_dim)
# return img, target, img_info, img_id


##### pull_item(idx)
# ---------->

img = resized_img.copy()
target = copy.deepcopy(label)
img_info = origin_image_size
img_id = np.array([id_])


##### img, target = self.preproc(img, target, self.input_dim)
# for COCODataset, preproc is TrainTransform:

# yolox.exp : yolox_base.py
# preproc=TrainTransform(
#     max_labels=50,
#     flip_prob=self.flip_prob,
#     hsv_prob=self.hsv_prob
# )

# yolox.data : data_augment.py
# class TrainTransform:
#     def __init__(self, max_labels=50, flip_prob=0.5, hsv_prob=1.0):
#         self.max_labels = max_labels
#         self.flip_prob = flip_prob
#         self.hsv_prob = hsv_prob
#     def __call__(self, image, targets, input_dim):

max_labels = 50
flip_prob = 0.5
hsv_prob = 0.5

image = img.copy()
targets = target.copy()
input_dim = img_size

boxes = targets[:, :4].copy()
labels = targets[:, 4].copy()

# if len(boxes) == 0:
#     targets = np.zeros((max_labels, 5), dtype=np.float32)
#     image, r_o = preproc(image, input_dim)
#     return image, targets

image_o = image.copy()
targets_o = targets.copy()
height_o, width_o, _ = image_o.shape
boxes_o = targets_o[:, :4]
labels_o = targets_o[:, 4]

# ----------
# bbox_o: [xyxy] to [c_x,c_y,w,h]
boxes_o = xyxy2cxcywh(boxes_o)
print(boxes_o)

# ----------
if random.random() < hsv_prob:
    augment_hsv(image)

# ----------
# _mirror handles bounding box
image_t, boxes = _mirror(image, boxes, flip_prob)
print(boxes_o)
print(boxes)


# ----------
height, width, _ = image_t.shape


# ----------
# preprocess to flipped image
# padded + axis is swapped
image_t, r_ = preproc(image_t, input_dim)
print(image_t.shape)


# ----------
# boxes [xyxy] 2 [cx,cy,w,h]
# mirroed boxes to cx, cy, w, h and resized to fit to padded image
boxes = xyxy2cxcywh(boxes)
boxes *= r_
print(boxes)


# ----------
# 2: w  3: h
# check w, h > 1
mask_b = np.minimum(boxes[:, 2], boxes[:, 3]) > 1
boxes_t = boxes[mask_b]
labels_t = labels[mask_b]

# if len(boxes_t) == 0:
#     image_t, r_o = preproc(image_o, input_dim)
#     boxes_o *= r_o
#     boxes_t = boxes_o
#     labels_t = labels_o

labels_t = np.expand_dims(labels_t, 1)
print(labels_t)


# ----------
# class label and bounxing box after transformed
targets_t = np.hstack((labels_t, boxes_t))

padded_labels = np.zeros((max_labels, 5))

padded_labels[range(len(targets_t))[: max_labels]] = targets_t[: max_labels]

padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)

# (3, 640, 640)
print(image_t.shape)
# (num class labels, 5)  (5 = class label, cx, cy, w, h)
print(targets_t.shape)
# (max_labels, 5)
print(padded_labels.shape)

# this is return from TrainTransform (= preproc in COCOdataset)
# return image_t, padded_labels


# ----------
img = image_t.copy()
target = padded_labels.copy()

# this is return from COCOdataset __getitem__
# return img, target, img_info, img_id


############################################################################################
# ------------------------------------------------------------------------------------------
# just pull image from COCODataset.pull_item()
# ------------------------------------------------------------------------------------------

base_dir = '/home/kswada/kw/yolox/YOLOX'

exp_file = os.path.join(base_dir, 'exps/default/yolox_s.py')

sys.path.append(os.path.dirname(exp_file))

current_exp = importlib.import_module(os.path.basename(exp_file).split(".")[0])

print(current_exp)

exp = current_exp.Exp()

print(exp)
print(dir(exp))


# ----------
# creating index
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
# get one image
idx = random.randint(0, len(dataset) - 1)

# JUST PULL, not applyin TrainTransform
img, labels, img_info, img_id = dataset.pull_item(idx)

print(img.shape)
print(labels.shape)

# height, width
print(img_info)

print(f'selected image: {idx}')
print(f'img_id: {img_id}')
print(f'len labels: {len(labels)}')


# bounding box: (xmin, ymin, xmax, ymax), already resized by COCODataset.__init__
# show image and bbox
tmp_img = img.copy()
for i in range(len(labels)):
    tmp_label = labels[i]
    cv2.rectangle(
        img=tmp_img,
        pt1=(int(tmp_label[0]), int(tmp_label[1])),
        pt2=(int(tmp_label[2]), int(tmp_label[3])),
        color=(255, 0, 0),
        thickness=2
    )

PIL.Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).show()

