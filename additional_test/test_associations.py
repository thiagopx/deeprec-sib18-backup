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


parser = argparse.ArgumentParser(
    description='Dataset patches assembling.'
)
parser.add_argument(
    '-i', '--i', action='store', dest='i', required=False, type=int,
    default=2, help='Si.'
)
parser.add_argument(
    '-j', '--j', action='store', dest='j', required=False, type=int,
    default=3, help='Sj.'
)
parser.add_argument(
    '-bs', '--bs', action='store', dest='bs', required=False, type=int,
    default=32, help='Batch size.'
)
parser.add_argument(
    '-d', '--d', action='store', dest='display', required=False, type=bool,
    default=True, help='Display.'
)
args = parser.parse_args()
i, j = args.i, args.j

# model
input_size = (31, 31)
radius = int((input_size[0] - 1)/ 2)
images_ph = tf.placeholder(tf.float32, name='images_ph', shape=(None, 3, input_size[0], input_size[1])) # channels first
images_adjust_op = tf.image.convert_image_dtype(images_ph, tf.float32)
logits_op = squeezenet(images_ph, 'val', 2, channels_first=True)
probs_op = tf.nn.softmax(logits_op)
predictions_op = tf.argmax(logits_op, 1)

# segmentation
print('=> Segmentation')
t0 = time.time()
strips = StripsText(path='datasets/D1/artificial/D061', filter_blanks=True)
print('Segmentation elapsed time={:.2f} seconds'.format(time.time() - t0))
N = len(strips.strips)

# positives
#combs = [(i, i +1) for i in range(N - 1)]
#random.shuffle(combs)
#i, j = random.choice(combs)

print('=> Pairing')
t0 = time.time()
image = strips.pair(i, j)
_, image_bin = cv2.threshold(
    cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)
image = cv2.cvtColor(image_bin, cv2.COLOR_GRAY2RGB)
print('Pairing time={:.2f} seconds'.format(time.time() - t0))

h, w, _ = image.shape
#cv2.imshow('pair', image)
#cv2.waitKey(0)

classes = ['negative', 'positive']#, 'neutral']
# for category in categories:
#     if os.path.exists('/tmp/{}'.format(category)):
#         shutil.rmtree('/tmp/{}'.format(category))
#     os.makedirs('/tmp/{}'.format(category))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    params_fname = open('best_model.txt').read()
    load(params_fname, sess, model_scope='SqueezeNet')

    print('=> Score')
    t0 = time.time()
    border = strips.strips[i].offsets_r
    # crops = []
    # for y in range(radius, 400):#border.size - radius):
    #     x = border[y]
    #     crop = image[y - radius : y + radius + 1, x - radius : x + radius + 1]
    #     crops.append(crop)

    num_samples = h - 2 * radius
    num_batches = math.ceil(num_samples / args.bs)

    # sys.exit()
    logits_dic = {}
    probs_dic = {}
    preds_dic = {}
    pcont = 0.3
    valid_batches = []
    for b in range(num_batches):
        batch = []
        for y in range(b * args.bs + radius , min((b + 1) * args.bs + radius, num_samples)):
            x = border[y]
            # print(x, y)
            crop = image[y - radius : y + radius + 1, x - radius : x + radius + 1]
            # print((crop != 255).sum() / crop.size)
            # cv2.imshow('pair', crop)
            # cv2.waitKey(0)
            if (crop != 255).sum() / crop.size >= pcont:
                batch.append(crop)
            #print(crop.shape)
        #batch = crops[b * args.bs : (b + 1) * args.bs]
        if not batch:
            continue

        crops_arr = np.transpose(np.stack(batch, axis=0), axes=(0, 3, 1, 2))
        crops_arr = crops_arr.astype(np.float32) / 255.
        logits, probs, preds = sess.run(
            [logits_op, probs_op, predictions_op],
            feed_dict={images_ph: crops_arr}
        )
        logits_dic[b] = logits
        probs_dic[b] = probs
        preds_dic[b] = preds
    print('Scoring time={:.2f} seconds'.format(time.time() - t0))

    for b in preds_dic.keys():
        logits, probs, preds = logits_dic[b], probs_dic[b], preds_dic[b]
        k = 0
        for y in range(b * args.bs + radius , min((b + 1) * args.bs + radius, num_samples)):
            # if preds[k] == 2:
            #     continue
            x = border[y]
            crop = image[y - radius : y + radius + 1, x - radius : x + radius + 1]
            if (crop != 255).sum() / crop.size < pcont:
                continue

            cv2.imshow('{} :: [neg={:.2f} pos={:.2f}]'.format(classes[preds[k]], probs[k, 0], probs[k, 1]), crop)
            # cv2.imshow('{} :: [neg={:.2f} pos={:.2f} neu={:.2f}]'.format(classes[preds[k]], probs[k, 0], probs[k, 1], probs[k, 2]), crop)
            cv2.waitKey(0)

            # if cv2.waitKey(33) == 27:
            #     sys.exit()
            cv2.destroyAllWindows()
            print('{} :: [neg={:.2f} pos={:.2f}]'.format(classes[preds[k]], probs[k, 0], probs[k ,1]))
            # print('{} :: [neg={:.2f} pos={:.2f} neu={:.2f}]'.format(classes[preds[k]], probs[
            #
            # k, 0], probs[k ,1], probs[k, 2]))
            k += 1
