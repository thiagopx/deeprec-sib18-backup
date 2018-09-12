from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os
import numpy as np
import cv2
import time
import argparse
import tensorflow as tf
sys.path.append('.')
from docrec.strips.strips import Strips
from docrec.neural.models.squeezenet import squeezenet, load

# global params
input_size = 31 # should be odd
half = int(input_size / 2)
input_image = np.ones((3, input_size, input_size), dtype=np.float32)
radius_search = 10
stride = 10
pcont = 0.15
classes = ['negative', 'positive']

parser = argparse.ArgumentParser(description='Score.')
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
images_ph = tf.placeholder(tf.float32, name='images_ph', shape=(None, 3, input_size, input_size)) # channels first
images_adjust_op = tf.image.convert_image_dtype(images_ph, tf.float32)
logits_op = squeezenet(images_ph, 'val', 2, channels_first=True)
probs_op = tf.nn.softmax(logits_op)
predictions_op = tf.argmax(logits_op, 1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    params_fname = open('best_model.txt').read()
    load(params_fname, sess, model_scope='SqueezeNet')
    t0_global = time.time()
    strips = Strips(path=args.doc, filter_blanks=True)
    N = len(strips.strips)
    scores = np.zeros((N, N), dtype=np.int32)
    min_y = half + radius_search
    for i in range(N):
        si = strips.strips[i]
        hi, wi, _ = si.image.shape
        _, image_i = cv2.threshold(
            cv2.cvtColor(si.filled_image(), cv2.COLOR_RGB2GRAY), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        temp = image_i.astype(np.float32)
        image3_i = np.stack([temp, temp, temp])
        for j in range(N):
            if i == j:
                continue

            print('=> Scoring [{}][{}]='.format(i, j), end='')
            t0 = time.time()

            sj = strips.strips[j]
            hj, wj, _ = sj.image.shape
            _, image_j = cv2.threshold(
                cv2.cvtColor(sj.filled_image(), cv2.COLOR_RGB2GRAY), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            ) # range [0, 1]strips.pair(i, j, accurate=True, filled=True)
            temp = image_j.astype(np.float32)
            image3_j = np.stack([temp, temp, temp])

            max_y = min(hi, hj) - half - 2 - radius_search
            count = {disp: 0 for disp in range(-radius_search, radius_search + 1)}
            for yi in range(min_y, max_y + 1, stride):
                xi = si.offsets_r[yi]
                xli = xi - half
                yli = yi - half
                yri = yli + input_size - 1
                best_disp = 0
                max_comp = 0
                match = False

                for yj in range(yi - radius_search, yi + radius_search + 1):
                    disp = yj - yi
                    xj = sj.offsets_l[yj]
                    xrj = xj + half - 1
                    ylj = yj - half
                    yrj = ylj + input_size - 1

                    input_image[:] = 1.0 # white
                    input_image[:, :, : half + 1] = image3_i[:, yli : yri + 1, xli : xi + 1]
                    input_image[:, :, half + 1 :] = image3_j[:, ylj : yrj + 1, xj : xrj + 1]

                    if (1 - input_image).sum() / input_image.size < pcont:
                        continue

                    batch_arr = input_image[np.newaxis, :] # 1 batch
                    logits, probs, preds = sess.run(
                        [logits_op, probs_op, predictions_op],
                        feed_dict={images_ph: batch_arr}
                    )

                    if preds[0] == 1 and probs[0][1] > max_comp:
                        max_comp = probs[0][1]
                        best_disp = disp
                        match = True

                if match:
                    count[best_disp] += 1

            scores[i, j] = max(count.values())
            best_disps = [disp for disp in count if count[disp] == scores[i, j]]
            tf = time.time()
            print('{} [disp={}] :: elapsed time={:.2f} seconds'.format(scores[i, j], str(best_disp), tf - t0))

tf_global = time.time()
print('Overall elapsed time={:.2f} min.'.format((tf_global - t0_global) / 60))