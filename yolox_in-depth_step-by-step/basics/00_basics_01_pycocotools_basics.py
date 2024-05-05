
import os

from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt

import cv2
import PIL.Image

from pprint import pprint

import random

import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)


# ----------
# REFERENCE
# https://gist.github.com/interactivetech/c2913317603b79c02ff49fa9824f1104

# COCO data format
# https://cocodataset.org/#format-data


##################################################################################################################
# ----------------------------------------------------------------------------------------------------------------
# Annotation Data:
#   coco.anns: dictionary of annotations
#   coco.imgs: dictionary of images
#   coco.cats: dictionary of categories
#   coco.catToImgs: dictionary that maps and informs which images have this category
#   coco.imgToAnns: dictionary that maps image_id to annotations
# ----------------------------------------------------------------------------------------------------------------

datDir = '/media/kswada/MyFiles/dataset/COCO'
# dataType = 'val2017'
dataType = 'train2017'

annFile = f'{datDir}/annotations/instances_{dataType}.json'


# ----------
# initialize COCO api for instance annotations
# loading annotations into memory
# 'creating index ... index created !' is shown.
coco = COCO(annFile)


# ----------------------------------------------------------------------------------------------------------------
# coco.ans: dictionary of annotations
#     area: size of an Annotation
#     bbox: bounding box of object in image
#     category_id: id of category that label of object (i.e. 25 corresponds to giraffe)
#     id: unique id of Annotation
#     iscrowd: ?
#     segmentation: array of 2D pixel coordinates (i.e. polygon) that defines segmentation mask
# ----------------------------------------------------------------------------------------------------------------

for key in coco.anns.keys():
    # pprint(key)
    pprint(coco.anns[key])
    break


# ----------------------------------------------------------------------------------------------------------------
# coco.cats: dictionary of categories
#     supercategory: supercategory that encapsulates category (i.e. giraffe is an Animal, supercategory = Animal)
#     id: integer that corresponds to category
#     name: english word that defines category (25 is giraffe)
# coco.getCatIds
# coco.getCatIds
# coco.loadCats
# ----------------------------------------------------------------------------------------------------------------

# get ids of categories
cat_all_ids = coco.getCatIds()
pprint(cat_all_ids)

for id in cat_all_ids:
    pprint(coco.cats[id])

# returns dictionary
cat_dic = coco.loadCats(coco.getCatIds())
pprint(cat_dic)

# display COCO categories and supercategories
nms = [cat['name'] for cat in cat_dic]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cat_dic])
print('COCO supercategories: \n{}'.format(' '.join(nms)))


# ----------------------------------------------------------------------------------------------------------------
# coco.imgs: dictionary of images
# ----------------------------------------------------------------------------------------------------------------

for key in coco.imgs.keys():
    pprint(coco.imgs[key])
    break


# ----------------------------------------------------------------------------------------------------------------
# coco.getImgIds
# ----------------------------------------------------------------------------------------------------------------

catNms = ['person', 'dog', 'skateboard']
supNms = ['vehicle']

catIds = coco.getCatIds(catNms=catNms)
supIds = coco.getCatIds(supNms=supNms)
print(catIds)
print(supIds)


for id in catIds:
    pprint(coco.cats[id])


# ----------
# getImgIds by catIds
imgIds = coco.getImgIds(catIds=catIds)
print(imgIds)


# ----------
# get img info
for id in imgIds:
    img = coco.loadImgs(id)
    pprint(img)


# ----------------------------------------------------------------------------------------------------------------
# read image from coco_url
# get annotation by image id
#     coco.loadImgs
#     coco.getAnnIds
#     coco.loadAnns
# ----------------------------------------------------------------------------------------------------------------

img_id = 350148
img_info = coco.loadImgs(img_id)
anno_ids = coco.getAnnIds(img_id)

pprint(img_info)
pprint(anno_ids)


# ----------
I = io.imread(img_info[0]['coco_url'])
plt.axis('off')
plt.imshow(I)
plt.show()


# only 1 annotation
annos = coco.loadAnns(anno_ids[0])
pprint(annos)


# ----------------------------------------------------------------------------------------------------------------
# read image from path
# visualize annotation
# ----------------------------------------------------------------------------------------------------------------

img_id = 350148
img_info,  = coco.loadImgs(img_id)

img_path = os.path.join(datDir, f"{dataType}/{img_info['file_name']}")
print(img_path)

I = io.imread(img_path)
plt.axis('off')
plt.imshow(I)
plt.show()


# ----------
anno_ids = coco.getAnnIds(img_id)
pprint(anno_ids)

annos = coco.loadAnns(anno_ids)


# ----------
img = plt.imread(img_path)
fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(img)
coco.showAnns(annos, draw_bbox=True)
for i, ann in enumerate(annos):
    ax.text(annos[i]['bbox'][0], annos[i]['bbox'][1], coco.cats[annos[i]['category_id']]['name'], style='italic',
            bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 5})
plt.show()


# ----------------------------------------------------------------------------------------------------------------
# visualize annotation  (bounding box)
# ----------------------------------------------------------------------------------------------------------------

coco = COCO(annFile)


img_id = 350148
img_info,  = coco.loadImgs(img_id)

img_path = os.path.join(datDir, f"{dataType}/{img_info['file_name']}")
print(img_path)

I = io.imread(img_path)
plt.axis('off')
plt.imshow(I)
plt.show()


# ----------
anno_ids = coco.getAnnIds(img_id)

# this image have 20 annotations
print(len(anno_ids))
pprint(anno_ids)

annos = coco.loadAnns(anno_ids)


### NOTE that bbox is (xmin, ymin, width, height)
# area: number of pixcels for segmentation
# segmentation is [x1, y1, x2, y2, ...]  (polygon)
pprint(annos[0])


img = cv2.imread(img_path)

for i in annos[:1]:
    [x, y, w, h] = i['bbox']
    cv2.rectangle(img, (int(x), int(y)), ((int(x+w), int(y+h))), (255, 0, 0), 5)

PIL.Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).show()


# ----------------------------------------------------------------------------------------------------------------
# iscrowd
# ----------------------------------------------------------------------------------------------------------------

# get all annotation ids
anno_ids = coco.getAnnIds()
# 36781
print(len(anno_ids))

# get all annotation ids with no crowd
anno_ids_nocrowd = coco.getAnnIds(iscrowd=False)
# 36335
print(len(anno_ids_nocrowd))


# get all annotations
annos = coco.loadAnns(anno_ids)
print(len(annos))


# ----------
for k in annos[0].keys():
    print(k)

annos_ids_iscrowd = []
for an in annos:
    if an['iscrowd']:
        annos_ids_iscrowd.append((an['id'], an['image_id']))

# 446 annotations is iscrowd
print(len(annos_ids_iscrowd))

# ----------
# iscrowd is bad annotations !!!

for (annos_id, img_id) in random.sample(annos_ids_iscrowd, 10):
    img_info,  = coco.loadImgs(img_id)
    img_path = os.path.join(datDir, f"{dataType}/{img_info['file_name']}")
    annos = coco.loadAnns(annos_id)
    img = cv2.imread(img_path)
    [x, y, w, h] = annos[0]['bbox']
    cv2.rectangle(img, (int(x), int(y)), ((int(x+w), int(y+h))), (255, 0, 0), 5)
    PIL.Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).show()

