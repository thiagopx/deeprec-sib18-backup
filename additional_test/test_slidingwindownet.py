from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os
import numpy as np
import cv2
import argparse
import tensorflow as tf
sys.path.append('.')
from docrec.strips.strips import Strips
from docrec.neural.models.squeezenet import squeezenet, load

# global params
input_size = 31
half = int(input_size / 2)
rem = input_size % 2
input_image = np.ones((3, input_size, input_size), dtype=np.float32)
#radius = int((input_size[0] - 1)/ 2)
stride_feat = 5
radius_feat = 20
radius_search = 10
stride = 10
pcont = 0.1
classes = ['negative', 'positive']

def features(points):
    feat = []
    for (x3, y3), (x2, y2), (x1, y1) in zip(points[2 :], points[1 : -1], points[ : -2]):
        val = (y3 - y2) * (y2 - y1) + (x3 - x2) * (x2 - x1)
        feat.append(val)
    return np.array(feat)


parser = argparse.ArgumentParser(description='Score.')
parser.add_argument(
    '-bs', '--bs', action='store', dest='bs', required=False, type=int,
    default=16, help='Batch size.'
)
parser.add_argument(
    '-d', '--d', action='store', dest='doc', required=False, type=str,
    default='datasets/D1/artificial/D001', help='Document.'
)
parser.add_argument(
    '-i', '--i', action='store', dest='i', required=False, type=int,
    default=2, help='Si.'
)
parser.add_argument(
    '-j', '--j', action='store', dest='j', required=False, type=int,
    default=3, help='Sj.'
)
args = parser.parse_args()

# model
images_ph = tf.placeholder(tf.float32, name='images_ph', shape=(None, 3, input_size, input_size)) # channels first
images_adjust_op = tf.image.convert_image_dtype(images_ph, tf.float32)
logits_op = squeezenet(images_ph, 'val', 2, channels_first=True)
probs_op = tf.nn.softmax(logits_op)
predictions_op = tf.argmax(logits_op, 1)

# pair
i, j = args.i, args.j
strips = Strips(path=args.doc, filter_blanks=True)
si, sj = strips.strips[i], strips.strips[j]
hi, wi, _ = si.image.shape
hj, wj, _ = sj.image.shape
min_y = radius_search + radius_feat
max_y = min(hi, hj) - 1 - radius_search - radius_feat
smi = np.correlate(si.offsets_r, [0.05, 0.1, 0.7, 0.1, 0.05], mode='same')
smj = np.correlate(sj.offsets_l, [0.05, 0.1, 0.7, 0.1, 0.05], mode='same')
support = np.hstack([si.filled_image(), sj.filled_image()])
hs, ws, _ = support.shape
blank = np.zeros((hs, 31, 3), dtype=np.uint8)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    params_fname = open('best_model.txt').read()
    load(params_fname, sess, model_scope='SqueezeNet')

    _, image_i = cv2.threshold(
            cv2.cvtColor(si.filled_image(), cv2.COLOR_RGB2GRAY), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    ) # range [0, 1]strips.pair(i, j, accurate=True, filled=True)
    hi, wi = image_i.shape
    temp = image_i.astype(np.float32)
    image3_i = np.stack([temp, temp, temp])

    _, image_j = cv2.threshold(
            cv2.cvtColor(sj.filled_image(), cv2.COLOR_RGB2GRAY), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    ) # range [0, 1]strips.pair(i, j, accurate=True, filled=True)
    hj, wj = image_j.shape
    temp = image_j.astype(np.float32)
    image3_j = np.stack([temp, temp, temp])

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

        yj = best_yj
        xj = int(smj[yj])

        xli = xi - half + 1 - rem
        xrj = xj + half - 1
        yli = yi - half + 1 - rem
        yri = yli + input_size - 1
        ylj = yj - half + 1
        yrj = ylj + input_size - 1

        input_image[:] = 1.0 # white
        input_image[:, :, : half + rem] = image3_i[:, yli : yri + 1, xli : xi + 1]
        input_image[:, :, half + rem :] = image3_j[:, ylj : yrj + 1, xj : xrj + 1]
        if (1 - input_image).sum() / input_image.size < pcont:
            color = (255, 0, 0) # blue
        else:
            batch_arr = input_image[np.newaxis, :] # 1 batch
            logits, probs, preds = sess.run(
                [logits_op, probs_op, predictions_op],
                feed_dict={images_ph: batch_arr}
            )
            color = (0, 255, 0) if preds[0] == 1 else (0, 0, 255)
        cv2.rectangle(draw, (xli, yli), (xi, yri), color, 3)
        cv2.rectangle(draw, (xj + wi, ylj), (xrj + wi, yrj), color, 3)
        cv2.rectangle(draw, (xj + wi, ylj), (xrj + wi, yrj), color, 3)
        cv2.circle(support, (xi, yi), 3, color, -1)
        cv2.circle(support, (xj + wi, yj), 3, color, -1)
        cv2.line(support, (xi, yi), (xj + wi, yj), color, 1)
        input_image_transp = (255 * np.transpose(input_image, axes=(1, 2, 0))).astype(np.uint8)
        draw[yli : yri + 1, wi + wj : wi + wj + input_size] = input_image_transp

        H = 500
        DH = int (yi / H)
        cv2.imshow('Slidind window', draw[DH * H : (DH + 1) * H])
        k = cv2.waitKey(0) # 33
        if k == 27:    # Esc key to stop
            cv2.destroyAllWindows()
            sys.exit()

