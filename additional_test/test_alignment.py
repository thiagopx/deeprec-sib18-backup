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
i = 21
j = 22
strips = Strips(path=doc, filter_blanks=True)
si, sj = strips.strips[i], strips.strips[j]
hi, wi, _ = si.image.shape
hj, wj, _ = sj.image.shape

offset = wi
stride_feat = 5
radius_feat = 20
radius_search = 5
num_points = 100
#num_feat = 2 * int(radius_feat / stride_feat)
min_y = radius_search + radius_feat
max_y = min(hi, hj) - 1 - radius_search - radius_feat
stride = int((max_y - min_y) / (num_points - 1))
smi = np.correlate(si.offsets_r, [0.05, 0.1, 0.7, 0.1, 0.05], mode='same')
smj = np.correlate(sj.offsets_l, [0.05, 0.1, 0.7, 0.1, 0.05], mode='same')
#smi = si.offsets_r
#smj = sj.offsets_l
si_pts, sj_pts = [], []
for yi in range(min_y, max_y + 1, stride):
    xi = int(smi[yi])
    si_pts.append((xi, yi))
    base_i = [(smi[y], y) for y in range(yi - radius_feat, yi + radius_feat + 1, stride_feat)]
    #print(base_i)
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
    xj = int(smj[best_yj] + offset)
    sj_pts.append((xj, best_yj))
    # print('{}->{} {}->{}'.format(xi, xj, yi, yj))
    #sys.exit()

points = np.hstack([si.filled_image(), sj.filled_image()])
for pi, pj in zip(si_pts, sj_pts):
    cv2.circle(points, pi, 5, (0, 255, 0), -1)
    cv2.circle(points, pj, 5, (0, 255, 0), -1)
    cv2.line(points, pi, pj, (255, 0, 0), 2)
cv2.imwrite('test/test_alignment/pair_points.jpg', points)

