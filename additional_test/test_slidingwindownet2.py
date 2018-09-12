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
min_y = half + radius_search
max_y = min(hi, hj) - half - 2 - radius_search
support = np.hstack([si.filled_image(), sj.filled_image()])
hs, ws, _ = support.shape
blank = np.zeros((hs, 31, 3), dtype=np.uint8)
score = {disp: 0 for disp in range(-radius_search, radius_search + 1)}
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    params_fname = open('best_model.txt').read()
    load(params_fname, sess, model_scope='SqueezeNet')

    _, image_i = cv2.threshold(
            cv2.cvtColor(si.filled_image(), cv2.COLOR_RGB2GRAY), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    hi, wi = image_i.shape
    temp = image_i.astype(np.float32)
    image3_i = np.stack([temp, temp, temp])

    _, image_j = cv2.threshold(
            cv2.cvtColor(sj.filled_image(), cv2.COLOR_RGB2GRAY), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    ) # range [0, 1]strips.pair(i, j, accurate=True, filled=True)
    hj, wj = image_j.shape
    temp = image_j.astype(np.float32)
    image3_j = np.stack([temp, temp, temp])
    frame = np.hstack([support, blank])
    draw = frame.copy()
    for yi in range(min_y, max_y + 1, stride):

        H = 500
        DH = int(yi / H)

        xi = si.offsets_r[yi]
        xli = xi - half
        yli = yi - half
        yri = yli + input_size - 1
        best_yj = yi
        max_comp = 0
        decision = (255, 0, 0)
        for yj in range(yi - radius_search, yi + radius_search + 1):
            xj = sj.offsets_l[yj]
            xrj = xj + half - 1
            ylj = yj - half
            yrj = ylj + input_size - 1
            # print(yi, yj)

            color = (255, 0, 0) # blue
            input_image[:] = 1.0 # white
            input_image[:, :, : half + 1] = image3_i[:, yli : yri + 1, xli : xi + 1]
            input_image[:, :, half + 1 :] = image3_j[:, ylj : yrj + 1, xj : xrj + 1]

            if (1 - input_image).sum() / input_image.size >= pcont:
                batch_arr = input_image[np.newaxis, :] # 1 batch
                logits, probs, preds = sess.run(
                    [logits_op, probs_op, predictions_op],
                    feed_dict={images_ph: batch_arr}
                )
                color = (0, 255, 0) if preds[0] == 1 else (0, 0, 255)
                if preds[0] == 1 and probs[0][1] > max_comp:
                    max_comp = probs[0][1]
                    decision = (0, 255, 0)
                    best_yj = yj
                elif decision == (255, 0, 0) and preds[0] == 0:
                    decision = (0, 0, 255)

            draw[:] = frame
            cv2.rectangle(draw, (xli, yli), (xi, yri), color, 3)
            cv2.rectangle(draw, (xj + wi, ylj), (xrj + wi, yrj), color, 3)

            input_image_transp = (255 * np.transpose(input_image, axes=(1, 2, 0))).astype(np.uint8)
            draw[yli : yri + 1, wi + wj : wi + wj + input_size] = input_image_transp


            cv2.imshow('Slidind window', draw[DH * H : (DH + 1) * H])
            k = cv2.waitKey(0) # 33
            if k == 27:    # Esc key to stop
                cv2.destroyAllWindows()
                sys.exit()

        yj = best_yj
        if decision == (0, 255, 0):
            score[yj - yi] += 1

        xj = sj.offsets_l[yj]
        xrj = xj + half - 1
        ylj = yj - half
        yrj = ylj + input_size - 1

        draw[:] = frame
        cv2.rectangle(draw, (xli, yli), (xi, yri), decision, 3)
        cv2.rectangle(draw, (xj + wi, ylj), (xrj + wi, yrj), decision, 3)
        cv2.circle(frame, (xi, yi), 3, decision, -1)
        cv2.circle(frame, (xj + wi, yj), 3, decision, -1)
        cv2.line(frame, (xi, yi), (xj + wi, yj), decision, 1)
        input_image[:] = 1.0 # white
        input_image[:, :, : half + 1] = image3_i[:, yli : yri + 1, xli : xi + 1]
        input_image[:, :, half + 1 :] = image3_j[:, ylj : yrj + 1, xj : xrj + 1]
        input_image_transp = (255 * np.transpose(input_image, axes=(1, 2, 0))).astype(np.uint8)
        draw[yli : yri + 1, wi + wj : wi + wj + input_size] = input_image_transp

        H = 500
        DH = int (yi / H)
        cv2.imshow('Slidind window', draw[DH * H : (DH + 1) * H])
        k = cv2.waitKey(0) # 33
        if k == 27:    # Esc key to stop
            cv2.destroyAllWindows()
            sys.exit()

        max_score = max(score.values())
        best_disp = [disp for disp in score if score[disp] == max_score]
        print(score, best_disp, max_score)
