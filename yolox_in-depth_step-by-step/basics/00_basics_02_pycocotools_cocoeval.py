
import os

try:
    from yolox.layers import COCOeval_opt as COCOeval
except ImportError:
    from pycocotools.cocoeval import COCOeval


cocoEval = COCOeval()


# ------------------------------------------------------------------------------------------
# parameter
# ------------------------------------------------------------------------------------------

# defauls in brackets

# [all] N img ids to use for evaluation
# now no loaded --> []
cocoEval.params.imgIds

# K cat ids to use for evaluation
# now no loaded --> []
cocoEval.params.catIds

# [.5:.05:.95] T=10 IoU thresholds for evaluation
cocoEval.params.iouThrs

# [0:.01:1] R=101 recall thresholds for evaluation
cocoEval.params.recThrs

# [...] A=4 object area ranges for evaluation
cocoEval.params.areaRng

# [1 10 100] M=3 thresholds on max detections per image
cocoEval.params.maxDets

# ['segm'] set iouType to 'segm', 'bbox' or 'keypoints'
# iouType replaced the now DEPRECATED useSegm parameter.
cocoEval.params.iouType
cocoEval.params.useSegm

# [1] if true use category labels for evaluation
cocoEval.params.useCats


# Note: if useCats=0 category labels are ignored as in proposal scoring.
# Note: multiple areaRngs [Ax2] and maxDets [Mx1] can be specified.

