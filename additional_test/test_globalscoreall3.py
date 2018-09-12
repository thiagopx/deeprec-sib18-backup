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

docs1 = ['datasets/D1/mechanical/D{:03}'.format(i) for i in range(1, 62) if i != 3]
docs2 = ['datasets/D2/mechanical/D{:03}'.format(i) for i in range(1, 21)]
docs = docs1 + docs2
log = {doc: 0 for doc in docs}

# global params
radius_search = 10
input_size_h = 3000
input_size_w = 31
wl = math.ceil(input_size_w / 2)
wr = int(input_size_w / 2)

# model
images_ph = tf.placeholder(tf.float32, name='images_ph', shape=(None, 3, input_size_h, input_size_w)) # channels first
logits_op, _ = squeezenet(images_ph, 'test', NUM_CLASSES, channels_first=True)
probs_op = tf.nn.softmax(logits_op)

batch = np.ones((2 * radius_search + 1, 3, input_size_h, input_size_w), dtype=np.float32)
with tf.Session() as sess:

    # preparing model
    sess.run(tf.global_variables_initializer())
    params_fname = open('best_model.txt').read()
    load(params_fname, sess, model_scope='SqueezeNet')

    for doc in docs:

        print('Processing {}= '.format(doc), end='')
        sys.stdout.flush()
        strips = Strips(path=doc, filter_blanks=True)
        N = len(strips.strips)
        scores = np.zeros((N, N), dtype=np.float32)

        # features
        features = []
        for strip in strips.strips:
            left, right = extract_features(strip, input_size_h, input_size_w)
            features.append((left, right))

        for i in range(N):
            batch[:, :, :, : wr] = features[i][1]

            for j in range(N):
                if i == j:
                    continue

                feat_j = features[j][0]
                batch[radius_search, :, :, wr : ] = feat_j
                for r in range(1, radius_search + 1):
                    batch[radius_search - r, :, : -r, wr : ] = feat_j[:, r :, :]
                    batch[radius_search + r, :, r : , wr : ] = feat_j[:, : -r, :]

                probs = sess.run(probs_op, feed_dict={images_ph: batch})
                scores[i, j] = probs[:, 1].max()

        for i, line in enumerate(scores):
            if i < line.size - 1:
                if i + 1 != np.argmax(line):
                    log[doc] += 1
        print('{}'.format(log[doc]))
        sys.stdout.flush()

# dump
with open('test/test_globalscoreall3/summary.txt', 'w') as fh:
    lines = ['{}: {}\n'.format(doc, log[doc]) for doc in docs]
    fh.writelines(lines)
