from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os
import numpy as np
import cv2
import math
import time
import argparse
import tensorflow as tf
sys.path.append('.')
from docrec.strips.strips import Strips
from docrec.neural.models.squeezenet import squeezenet, load

NUM_CLASSES = 2

def extract_features(strip, input_size_h, input_size_w, stride=5):
    _, thresh = cv2.threshold(
        cv2.cvtColor(strip.filled_image(), cv2.COLOR_RGB2GRAY), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    image_bin = np.stack([thresh, thresh, thresh]).astype(np.float32)

    wl = math.ceil(input_size_w / 2)
    wr = int(input_size_w / 2)
    h, w, _ = strip.image.shape
    offset = int((h - input_size_h) / 2)

    # left image
    left_border = strip.offsets_l
    left = np.ones((3, input_size_h, wl), dtype=np.float32)
    for y, x in enumerate(left_border[offset : offset + input_size_h]):
        w_new = min(wl, w - x)
        left[:, y, : w_new] = image_bin[:, y + offset, x : x + w_new]

    # right image
    right_border = strip.offsets_r
    right = np.ones((3, input_size_h, wr), dtype=np.float32)
    for y, x in enumerate(right_border[offset : offset + input_size_h]):
        w_new = min(wr, x + 1)
        right[:, y, : w_new] = image_bin[:, y + offset, x - w_new + 1: x + 1]

    return left, right


# global params
radius_search = 10

parser = argparse.ArgumentParser(description='Score.')
parser.add_argument(
    '-d', '--d', action='store', dest='doc', required=False, type=str,
    default='datasets/D1/artificial/D001', help='Document.'
)
args = parser.parse_args()

# load strips
print('Load strips ', end='')
t0 = time.time()
strips = Strips(path=args.doc, filter_blanks=True)
tstr = time.time() - t0
N = len(strips.strips)
scores = np.zeros((N, N), dtype=np.float32)
min_h = min([strip.image.shape[0] for strip in strips.strips])
print(':: elapsed time={:.2f} sec.'.format(tstr))

# features
input_size_h = min_h if min_h % 2 == 1 else min_h - 1 # should be odd
input_size_w = 31
wl = math.ceil(input_size_w / 2)
wr = int(input_size_w / 2)
print('Features ', end='')
t0 = time.time()
features = []
for i, strip in enumerate(strips.strips):
    left, right = extract_features(strip, input_size_h, input_size_w)
    features.append((left, right))
    #left = (255 * np.transpose(left, axes=(1, 2, 0))).astype(np.uint8)
    #right = (255 * np.transpose(right, axes=(1, 2, 0))).astype(np.uint8)
    #cv2.imwrite('test/test_globalscore3/{}_left.jpg'.format(i), left)
    #cv2.imwrite('test/test_globalscore3/{}_right.jpg'.format(i), (right)
    #if i > 0 and i < N - 1:
    #    stacked = np.hstack([last, left])#, axis=1)
    #    cv2.imwrite('test/test_globalscore3/{}-{}.jpg'.format(i, i + 1), stacked)
    #last = right
tfeat = time.time() - t0
print(':: elapsed time={:.2f} sec.'.format(tfeat))

# model
input_image = np.ones((3, input_size_h, input_size_w), dtype=np.float32)
images_ph = tf.placeholder(tf.float32, name='images_ph', shape=(None, 3, input_size_h, input_size_w)) # channels first
logits_op, conv10_op = squeezenet(images_ph, 'test', NUM_CLASSES, channels_first=True)
probs_op = tf.nn.softmax(logits_op)
predictions_op = tf.argmax(logits_op, 1)

with tf.Session() as sess:
    # preparing model
    sess.run(tf.global_variables_initializer())
    params_fname = open('best_model.txt').read()
    load(params_fname, sess, model_scope='SqueezeNet')

    t0_global = time.time()
    wl = math.ceil(input_size_w / 2)
    wr = int(input_size_w / 2)
    batch = np.ones((2 * radius_search + 1, 3, input_size_h, input_size_w), dtype=np.float32)
    for i in range(N):
        batch[:, :, :, : wr] = features[i][1]

        #si = strips.strips[i]
        #image3_i = si.image_bin
        '''
        for yi in range(min_y, max_y + 1, stride):
            xi = si.offsets_r[yi]
            xli = xi - half
            yli = yi - half
            yri = yli + input_size - 1
            best_disp = 0
            max_comp = 0
            match = False
        '''
        for j in range(N):
            if i == j:
                continue

            print('=> Scoring [{}][{}]='.format(i, j), end='')
            t0 = time.time()
            '''
            sj = strips.strips[j]
            hj, wj, _ = sj.image.shape
            image3_j = sj.image_bin
            '''
            feat_j = features[j][0]
            batch[radius_search, :, :, wr : ] = feat_j
            for r in range(1, radius_search + 1):
                batch[radius_search - r, :, : -r, wr : ] = feat_j[:, r :, :]
                batch[radius_search + r, :, r : , wr : ] = feat_j[:, : -r, :]

            logits, conv10, probs, preds = sess.run(
                [logits_op, conv10_op, probs_op, predictions_op],
                feed_dict={images_ph: batch}
            )
            scores[i, j] = probs[:, 1].max()
            best_disp = probs[:, 1].argmax() - radius_search
            print('{} [disp={}] :: elapsed time={:.2f} seconds'.format(scores[i, j], best_disp, time.time() - t0))

t_overall = tfeat + tstr + tfeat + time.time() - t0_global
print('Overall elapsed time={:.2f} sec.'.format(t_overall))

for i, line in enumerate(scores):
    print(' '.join('{:1.2f}'.format(x) for x in line), end='')
    if i < line.size - 1:
        print(' => Fail' if i + 1 != np.argmax(line) else '')
    else:
        print()
