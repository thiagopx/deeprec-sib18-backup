from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import matplotlib as mpl
mpl.use('Agg')
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
from docrec.strips.strips import Strips
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
    #strips = StripsText(path=args.doc, filter_blanks=True)
    strips = Strips(path=args.doc, filter_blanks=True)
    print('Segmentation elapsed time={:.2f} seconds'.format(time.time() - t0))

    N = len(strips.strips)
    pcont = 0.2
    scores = np.zeros((N, N), dtype=np.int32)
    for i in range(N - 1):
        border = strips.strips[i].offsets_r
        j = i + 1

        print('=> Scoring [{}][{}]'.format(i, j))
        image = strips.pair(i, j, accurate=True, filled=True)
        h, w, _ = image.shape
        _, image_bin = cv2.threshold(
            cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        ) # range [0, 1]
        temp = image_bin.astype(np.float32)
        image3 = np.stack([temp, temp, temp])

        batch = []
        total_inferences = 0
        yx = []
        labels = []
        for y in range(radius, h - radius):
            x = border[y]
            crop = image3[:, y - radius : y + radius + 1, x - radius : x + radius + 1]
            if (crop == 0).sum() / crop.size >= pcont:
                batch.append(crop)
                yx.append((y, x))

            # Batch full or last image?
            #print(y, h - radius - 1)
            if (len(batch) == args.bs) or (y == (h - radius - 1) and len(batch) > 0):
                batch_arr = np.stack(batch)
                logits, probs, preds = sess.run(
                    [logits_op, probs_op, predictions_op],
                    feed_dict={images_ph: batch_arr}
                )
                scores[i, j] += preds.sum()
                total_inferences += len(batch)
                labels.extend(preds.tolist())
                batch = [] # clear


        colors = [[255, 0, 0], [0, 255, 0]]
        #print(len(labels), len(yx), total_inferences)
        for k, (y, x) in enumerate(yx):
            image[y, x, :] = colors[labels[k]]
        plt.imshow(image)
        plt.axis('off')
        plt.savefig('test/test_heatmap/{}_{}.jpg'.format(i, j), dpi=300)