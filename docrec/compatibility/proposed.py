from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from time import clock
import cv2
import numpy as np
import math
import tensorflow as tf

from .algorithm import Algorithm
from ..neural.models.squeezenet import squeezenet, load


class Proposed(Algorithm):
    '''  Proposed algorithm. '''

    def __init__(self, path_model, radius_search, input_size, num_classes=2):

        # preparing model
        self.radius_search = radius_search
        self.input_size_h, self.input_size_w = input_size

        self.images_ph = tf.placeholder(
            tf.float32, name='images_ph', shape=(None, 3, self.input_size_h, self.input_size_w)
        ) # channels first
        self.batch = np.ones((2 * radius_search + 1, 3, self.input_size_h, self.input_size_w), dtype=np.float32)

        logits_op, _ = squeezenet(self.images_ph, 'test', num_classes, channels_first=True)
        probs_op = tf.nn.softmax(logits_op)
        self.comp_op = tf.reduce_max(probs_op[:, 1])
        self.disp_op = tf.argmax(probs_op[:, 1]) - radius_search

        # result
        self.compatibilities = None
        self.displacements = None

        # init model
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        load(path_model, self.sess, model_scope='SqueezeNet')


    def _extract_features(self, strip):
        ''' Extract image around the border. '''

        _, thresh = cv2.threshold(
            cv2.cvtColor(strip.filled_image(), cv2.COLOR_RGB2GRAY), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        image_bin = np.stack([thresh, thresh, thresh]).astype(np.float32)

        wl = math.ceil(self.input_size_w / 2)
        wr = int(self.input_size_w / 2)
        h, w, _ = strip.image.shape
        offset = int((h - self.input_size_h) / 2)

        # left image
        left_border = strip.offsets_l
        left = np.ones((3, self.input_size_h, wl), dtype=np.float32)
        for y, x in enumerate(left_border[offset : offset + self.input_size_h]):
            w_new = min(wl, w - x)
            left[:, y, : w_new] = image_bin[:, y + offset, x : x + w_new]

        # right image
        right_border = strip.offsets_r
        right = np.ones((3, self.input_size_h, wr), dtype=np.float32)
        for y, x in enumerate(right_border[offset : offset + self.input_size_h]):
            w_new = min(wr, x + 1)
            right[:, y, : w_new] = image_bin[:, y + offset, x - w_new + 1: x + 1]

        return left, right


    def run(self, strips, d=0): # d is not used at this moment
        ''' Run algorithm. '''

        N = len(strips.strips)
        compatibilities = np.zeros((N, N), dtype=np.float32)
        displacements = np.zeros((N, N), dtype=np.int32)
        wr = int(self.input_size_w / 2)

        # features
        features = []
        for strip in strips.strips:
            left, right = self._extract_features(strip)
            features.append((left, right))

        for i in range(N):
            self.batch[:, :, :, : wr] = features[i][1]

            for j in range(N):
                if i == j:
                    continue

                feat_j = features[j][0]
                self.batch[self.radius_search, :, :, wr : ] = feat_j
                for r in range(1, self.radius_search + 1):
                    self.batch[self.radius_search - r, :, : -r, wr : ] = feat_j[:, r :, :]
                    self.batch[self.radius_search + r, :, r : , wr : ] = feat_j[:, : -r, :]

                comp, disp = self.sess.run([self.comp_op, self.disp_op], feed_dict={self.images_ph: self.batch})
                compatibilities[i, j] = comp
                displacements[i, j] = disp

        self.compatibilities = compatibilities
        self.displacements = displacements
        return self


    def name(self):
        ''' Method name. '''

        return 'proposed'
