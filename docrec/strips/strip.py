import numpy as np
import cv2
import copy
import matplotlib.pyplot as plt

from ..ndarray.utils import first_nonzero, last_nonzero

class Strip(object):
    ''' Strip image.'''


    def __init__(self, image, position, mask=None):

        h, w = image.shape[: 2]
        if mask is None:
            mask = 255 * np.ones((h, w), dtype=np.uint8)

        self.h = h
        self.w = w
        self.image = cv2.bitwise_and(image, image, mask=mask)
        self.position = position
        self.mask = mask

        self.offsets_l = np.apply_along_axis(first_nonzero, 1, self.mask) # left border (hor.) offsets
        self.offsets_r = np.apply_along_axis(last_nonzero, 1, self.mask)   # right border (hor.) offsets
        self.approx_width = int(np.mean(self.offsets_r - self.offsets_l + 1))


    def copy(self):
        ''' Copy object. '''

        return copy.deepcopy(self)


    def shift(self, disp):
        ''' shift strip vertically. '''

        M = np.float32([[1, 0, 0], [0, 1, disp]])
        self.image = cv2.warpAffine(self.image, M, (self.w, self.h))
        self.mask = cv2.warpAffine(self.mask, M, (self.w, self.h))
        self.offsets_l = np.apply_along_axis(first_nonzero, 1, self.mask) # left border (hor.) offsets
        self.offsets_r = np.apply_along_axis(last_nonzero, 1, self.mask)   # right border (hor.) offsets
        self.approx_width = int(np.mean(self.offsets_r - self.offsets_l + 1))
        return self


    def filled_image(self):
        ''' Return image with masked-out areas in white. '''

        return cv2.bitwise_or(
            self.image, cv2.cvtColor(
                cv2.bitwise_not(self.mask), cv2.COLOR_GRAY2RGB
            )
        )

    def is_blank(self, blank_tresh=127):
        ''' Check if is a blank strip. '''

        blurred = cv2.GaussianBlur(
            cv2.cvtColor(self.filled_image(), cv2.COLOR_RGB2GRAY), (5, 5), 0
        )
        return (blurred < blank_tresh).sum() == 0


    def stack(self, other, disp=0, filled=False):
        ''' Stack horizontally with other strip. '''

        min_h, max_h = min(self.h, other.h), max(self.h, other.h)

        # borders coordinates
        r1 = self.offsets_r[: min_h]
        l2 = other.offsets_l[: min_h]

        offset = self.w - np.min(l2 + self.w - r1) + 1

        # offset
        temp_image = np.zeros((max_h,  offset + other.w, 3), dtype=np.uint8)
        temp_image[: self.h, : self.w] = self.image
        temp_image[: other.h, offset :] += other.image

        temp_mask = np.zeros((max_h, offset + other.w), dtype=np.uint8)
        temp_mask[: self.h, : self.w] = self.mask
        temp_mask[: other.h, offset :] += other.mask

        self.h, self.w = temp_mask.shape
        self.image = temp_image
        self.mask = temp_mask
        self.offsets_r =np.apply_along_axis(last_nonzero, 1, self.mask)
        if filled:
            self.image = self.filled_image()
        return self