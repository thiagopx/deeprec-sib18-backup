import sys
import os
import numpy as np
import cv2
from skimage import transform
from math import acos

sys.path.append('.')
from docrec.strips.strips import Strips

def features(points):
    feat = []
    for (x3, y3), (x2, y2), (x1, y1) in zip(points[2 :], points[1 : -1], points[ : -2]):
        val = (y3 - y2) * (y2 - y1) + (x3 - x2) * (x2 - x1)
        feat.append(val)
    return np.array(feat)

doc = 'datasets/D2/mechanical/D002'
i = 1
j = 2
strips = Strips(path=doc, filter_blanks=True)

si, sj = strips.strips[i], strips.strips[j]
hi, wi, _ = si.image.shape
hj, wj, _ = sj.image.shape
stride_feat = 5
radius_feat = 20
radius_search = 5
min_y = radius_search + radius_feat
max_y = min(hi, hj) - 1 - radius_search - radius_feat
stride = 3
smi = np.correlate(si.offsets_r, [0.05, 0.1, 0.7, 0.1, 0.05], mode='same')
smj = np.correlate(sj.offsets_l, [0.05, 0.1, 0.7, 0.1, 0.05], mode='same')
support = np.hstack([si.filled_image(), sj.filled_image()])
hs, ws, _ = support.shape
blank = np.zeros((hs, 31, 3), dtype=np.uint8)
for yi in range(min_y, max_y + 1, stride):
    draw = np.hstack([support, blank])
    xi = int(smi[yi])
    base_i = [(smi[y], y) for y in range(yi - radius_feat, yi + radius_feat + 1, stride_feat)]
    feat_i = features(base_i)

    best_yj = yi
    base_j = [(smj[y], y) for y in range(yi - radius_feat, yi + radius_feat + 1, stride_feat)]
    feat_j = features(base_j)
    min_cost = np.sum((feat_i - feat_j) ** 2)
    for yj in range(yi - radius_search, yi + radius_search + 1):
        base_j = [(smj[y], y) for y in range(yj - radius_feat, yj + radius_feat + 1, stride_feat)]
        feat_j = features(base_j)
        cost = np.sum((feat_i - feat_j) ** 2)
        #print('{} {}'.format(yi, yj), feat_i, feat_j, cost)
        if cost < min_cost:
            best_yj = yj
            min_cost = cost
            # print('=> Best {} {}'.format(yi, yj), feat_i, feat_j, cost)
    xj = int(smj[best_yj])
    xli, yli = xi - 15, yi - 15
    xri, yri = xi + 15, yi + 15
    xlj, ylj = xj - 15, best_yj - 15
    xrj, yrj = xj + 15, best_yj + 15
    cv2.rectangle(draw, (xli, yli), (xri, yri), (0, 255, 0), 3)
    cv2.rectangle(draw, (xlj + wi, ylj), (xrj + wi, yrj), (0, 255, 0), 3)
    input_image = 255 * np.ones((31, 31, 3), dtype=np.uint8)
    input_image[:, : 16] = si.filled_image()[yli : yli + 31, xi - 15 : xi + 1]
    input_image[:, 16 :] = sj.filled_image()[ylj : ylj + 31, xj : xj + 15]
    draw[yli : yli + 31, wi + wj : wi + wj + 31] = cv2.bitwise_or(draw[yli : yli + 31, wi + wj : wi + wj + 31], input_image)
    cv2.imshow('Slidind window', draw)
    cv2.waitKey(0)

