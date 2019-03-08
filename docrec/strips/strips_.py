import os
import re
import numpy as np
import cv2
from random import shuffle
import matplotlib.pyplot as plt
from skimage import transform

from .strip import Strip


class Strips(object):
    ''' Strips operations manager.'''

    def __init__(self, path=None, obj=None, filter_blanks=True, blank_tresh=127):
        ''' Strips constructor.

        @path: path to strips (in case of load real strips)
        '''

        assert (path is not None) or (obj is not None)

        self.strips = []
        self.artificial_mask = False
        if path is not None:
            assert os.path.exists(path)
            self._load_data(path)
        else:
            self.strips = [strip.copy() for strip in obj.strips]

        if filter_blanks:
            self.strips = [strip for strip in self.strips if not strip.is_blank(blank_tresh)]


    def __call__(self, i):
        ''' Returns the i-th strip. '''

        return self.strips[i]


    def _load_data(self, path, regex_str='.*\d\d\d\d\d\.*'):
        ''' Stack strips horizontally.

        Strips are images with same basename (and extension) placed in a common
        directory. Example:

        basename="D001" and extension=".jpg" => strips D00101.jpg, ..., D00130.jpg.
        '''

        path_images = '{}/strips'.format(path)
        path_masks = '{}/masks'.format(path)
        regex = re.compile(regex_str)

        # loading images
        fnames = sorted([fname for fname in  os.listdir(path_images) if regex.match(fname)])
        images = []
        for fname in fnames:
            image = cv2.cvtColor(
                cv2.imread('{}/{}'.format(path_images, fname)),
                cv2.COLOR_BGR2RGB
            )
            images.append(image)

        # load masks
        masks = []
        if os.path.exists(path_masks):
            for fname in fnames:
                mask = np.load('{}/{}.npy'.format(path_masks, os.path.splitext(fname)[0]))
                masks.append(mask)
        else:
            masks = len(images) * [None]
            self.artificial_mask = True

        for position, (image, mask) in enumerate(zip(images, masks), 1):
            strip = Strip(image, position, mask)
            self.strips.append(strip)


    def trim(self, left=0, right=0):
        ''' Trim borders from strips. '''

        n = len(self.strips)
        self.strips = self.strips[left : n - right]
        return self


    def image(self, order=None, displacements=None, filled=False):
        ''' Return the reconstruction image in a specific order . '''

        N = len(self.strips)
        if order is None:
            order = list(range(N))
        if displacements is None:
            displacements = np.zeros((N, N), dtype=np.int32)

        prev = order[0]
        image = self.strips[prev].copy()
        for curr in order[1 :]:
            disp = displacements[prev, curr]
            image.stack(self.strips[curr].copy().shift(disp), filled=filled)
            prev = curr
        return image.image

    # def _matching_transform(self, i, j, num_points=30, radius_search=3, radius_feat=10):
    #     ''' Matching points for alignment. '''

    #     if self.artificial_mask:
    #         return np.eye(3, 3)

    #     si, sj = self.strips[i], self.strips[j]
    #     offset = si.image.shape[1]
    #     hi, wi, _ = si.image.shape
    #     hj, wj, _ = sj.image.shape
    #     min_y = radius_search + radius_feat
    #     max_y = min(hi, hj) - 1 - radius_search - radius_feat
    #     stride = int((max_y - min_y) / (num_points - 1))
    #     diff_si, diff_sj = np.diff(si.offsets_r), np.diff(sj.offsets_l)
    #     si_pts, sj_pts = [], []
    #     for yi in range(min_y, max_y + 1, stride):
    #         xi = si.offsets_r[yi]
    #         si_pts.append((xi, yi))
    #         feat_i = diff_si[yi - radius_feat : yi + radius_feat + 1]
    #         min_cost = float('inf')
    #         best_yj = -1
    #         for yj in range(yi - radius_search, yi + radius_search + 1):
    #             feat_j = diff_sj[yj - radius_feat : yj + radius_feat + 1]
    #             cost = np.sum((feat_i - feat_j) ** 2)
    #             if cost < min_cost:
    #                 best_yj = yj
    #                 min_cost = cost
    #         xj = sj.offsets_l[best_yj] + offset
    #         sj_pts.append((xj, best_yj))
    #         #print((xi, yi), (xj, best_yj))

    #     T = transform.EuclideanTransform()
    #     T.estimate(np.array(si_pts), np.array(sj_pts))
    #     return T.params


    # def _align(self, i, j):
    #     '''Align a pair of strips. '''

    #     si, sj = self.strips[i], self.strips[j]

    #     # align sj -> si
    #     hi, wi, _ = si.image.shape
    #     hj, wj, _ = sj.image.shape
    #     h = max(hi, hj)
    #     support = 255 * np.ones((h, wi + wj, 3), dtype=np.uint8)
    #     M = self._matching_transform(i, j)
    #     print(M)
    #     support[:, wi : wi + wj] = sj.filled_image()
    #     support = (255 * transform.warp(support, M, cval=1)).astype(np.uint8)
    #     support[: hi, : wi] = cv2.bitwise_and(support[: hi, : wi], si.filled_image())
    #     return support


    def pair(self, i, j, filled=False, accurate=False):
        ''' Return a single image with two paired strips. '''

        if accurate:
            return self._align(i, j) # filled not used

        return self.strips[i].copy().stack(self.strips[j], filled).image

    def plot(self, size=(8, 8), fontsize=6, ax=None, show_lines=False):
        ''' Plot strips given the current order. '''

        assert len(self.strips) > 0
        if ax is None:
            fig = plt.figure(figsize=size, dpi=150)
            ax = fig.add_axes([0, 0, 1, 1])
        else:
            fig = None

        shapes = [[strip.h, strip.w] for strip in self.strips]
        max_h, max_w = np.max(shapes, axis=0)
        sum_h, sum_w = np.sum(shapes, axis=0)

        # Background
        offsets = [0]
        background = self.strips[0].copy()
        for strip in self.strips[1 :]:
            offset = background.stack(strip).image.shape[1]
            print(offset, strip.image.shape[0])
            offsets.append(offset)

        ax.imshow(background.image)
        ax.axis('off')

        for strip, offset in zip(self.strips, offsets):
            d = strip.w / 2
            ax.text(
                offset + d, 50, str(strip.position), color='blue',
                fontsize=fontsize, horizontalalignment='center'
            )
        if show_lines:
            ax.vlines(
                offsets[1 :], 0, max_h, linestyles='dotted', color='red',
                linewidth=0.5
            )
        return fig, ax, offsets