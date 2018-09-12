from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import matplotlib.pyplot as plt
import time
import random
import os, shutil
import math
import argparse
import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix

sys.path.append('.')
from docrec.strips.stripstext import StripsText
from docrec.neural.models.squeezenet import squeezenet, load

# PYTHONPATH=. python test/test_score.py -bs 16 -d datasets/D1/artificial/D001

parser = argparse.ArgumentParser(
    description='Score.'
)
parser.add_argument(
    '-bs', '--bs', action='store', dest='bs', required=False, type=int,
    default=16, help='Batch size.'
)
parser.add_argument(
    '-d', '--d', action='store', dest='doc', required=False, type=str,
    default='datasets/D1/artificial/D001', help='Document.'
)
args = parser.parse_args()

# model
input_size = (31, 31)
radius = int((input_size[0] - 1)/ 2)
images_ph = tf.placeholder(tf.float32, name='images_ph', shape=(None, 3, input_size[0], input_size[1])) # channels first
images_adjust_op = tf.image.convert_image_dtype(images_ph, tf.float32)
logits_op = squeezenet(images_ph, 'val', 2, channels_first=True)
probs_op = tf.nn.softmax(logits_op)
predictions_op = tf.argmax(logits_op, 1)

# segmentation
print('Processing document {}'.format(args.doc))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    params_fname = open('best_model.txt').read()
    load(params_fname, sess, model_scope='SqueezeNet')
    classes = ['negative', 'positive']

    t0_global = time.time()
    print('=> Segmentation')
    t0 = time.time()
    strips = StripsText(path=args.doc, filter_blanks=True)
    print('Segmentation elapsed time={:.2f} seconds'.format(time.time() - t0))

    N = len(strips.strips)
    pcont = 0.2
    scores = np.zeros((N, N), dtype=np.int32)
    for i in range(N):
        border = strips.strips[i].offsets_r
        for j in range(N):
            if i == j:
                continue

            print('=> Scoring [{}][{}]='.format(i, j), end='')
            t0 = time.time()

            image = strips.pair(i, j)
            h, w, _ = image.shape
            _, image_bin = cv2.threshold(
                cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            ) # range [0, 1]
            temp = image_bin.astype(np.float32)
            image3 = np.stack([temp, temp, temp])
            # image3 = np.transpose(image, axes=(2, 0, 1)) / 255.

            batch = []
            total_inferences = 0
            for y in range(radius, h - radius):
                x = border[y]
                # crop_bin = image_bin[y - radius : y + radius + 1, x - radius : x + radius + 1]
                # if (crop_bin == 0).sum() / crop_bin.size < pcont:
                    # continue
                crop = image3[:, y - radius : y + radius + 1, x - radius : x + radius + 1]
                if (crop == 0).sum() / crop.size >= pcont:
                    batch.append(crop)

                # Batch full or last image?
                if (len(batch) == args.bs) or (y == h - radius - 1 and len(batch) > 0):
                    batch_arr = np.stack(batch, axis=0)
                    logits, probs, preds = sess.run(
                        [logits_op, probs_op, predictions_op],
                        feed_dict={images_ph: batch_arr}
                    )
                    scores[i, j] += preds.sum()
                    total_inferences += len(batch)
                    batch[:] = [] # clear

            tf = time.time()
            print('{} :: {} inferences :: elapsed time={:.2f} seconds'.format(scores[i, j], total_inferences, tf - t0))
            #sys.exit()

tf_global = time.time()

for i, line in enumerate(scores):
    print(' '.join('{:3}'.format(x) for x in line), end='')
    if i < line.size - 1:
        print(' => Fail' if i + 1 != np.argmax(line) else '')
    else:
        print()

print('Overall elapsed time={:.2f} min.'.format((tf_global - t0_global) / 60))