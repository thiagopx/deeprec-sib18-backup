from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import numpy as np
import cv2
import math
import argparse
import tensorflow as tf
sys.path.append('.')
from docrec.strips.strips import Strips
from docrec.neural.models.squeezenet import squeezenet, load

NUM_CLASSES = 2

input_size_w = 31
input_size_h = 3000

radius_search = 10

def extract_features(strip):
    ''' Extract image around the border. '''

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


parser = argparse.ArgumentParser(description='Score.')
parser.add_argument(
    '-d', '--d', action='store', dest='doc', required=False, type=str,
    default='datasets/D1/artificial/D001', help='Document.'
)
parser.add_argument(
    '-i', '--i', action='store', dest='i', required=False, type=int,
    default=0, help='Strip i'
)
parser.add_argument(
    '-j', '--j', action='store', dest='j', required=False, type=int,
    default=1, help='Strip j'
)
args = parser.parse_args()

# images_ph = tf.placeholder(tf.float32, name='images_ph', shape=(None, 3, 800, 600))
images_ph = tf.placeholder(tf.float32, name='images_ph', shape=(None, 3, input_size_h, input_size_w))
batch = np.ones((2 * radius_search + 1, 3, input_size_h, input_size_w), dtype=np.float32)
logits_op, conv10_op = squeezenet(images_ph, 'test', NUM_CLASSES, channels_first=True)
probs_op = tf.nn.softmax(logits_op)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #path_model = '{}/{}'.format(os.getcwd(), open('best_model.txt').read())
    #path_model = '{}/{}'.format(os.getcwd(), '3.npy')
    path_model = 'model.npy'
    load(path_model, sess, model_scope='SqueezeNet')

    # load strips
    strips = Strips(path=args.doc, filter_blanks=True)
    _, feat_i = extract_features(strips.strips[args.i])
    feat_j, _ = extract_features(strips.strips[args.j])

    batch[:, :, :, : 15] = feat_i
    batch[radius_search, :, :, 15 : ] = feat_j # radius zero
    for r in range(1, radius_search + 1):
        batch[radius_search - r, :, : -r, 15 : ] = feat_j[:, r :, :]
        batch[radius_search + r, :, r : , 15 : ] = feat_j[:, : -r, :]

    probs, conv10 = sess.run([probs_op, conv10_op], feed_dict={images_ph: batch})
    print('Conv10 dimensions: ', conv10.shape)
    alpha = 0.5
    for k in range(2 * radius_search + 1):
        pos = conv10[k, 1] # positive
        neg = conv10[k, 0] # negative
        print('neg={} pos={}'.format(probs[k, 0], probs[k, 1]))
        print(neg[:50], pos[:50])
        map_ = np.stack([neg, pos, np.zeros_like(neg)])
        overlay = (255 * np.transpose(map_ / map_.max(), axes=(1, 2, 0))).astype(np.uint8)
        overlay = cv2.resize(overlay, dsize=(31, 3000), interpolation=cv2.INTER_CUBIC)
        output = (255 * np.transpose(batch[k], axes=(1, 2, 0))).astype(np.uint8)
        output = cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0)
        cv2.imshow('Output', output[:, :, (2, 1, 0)])
        cv2.waitKey(0)
        cv2.imwrite('ignore/map_{}.jpg'.format(k), output[:, :, (2, 1, 0)])