import cv2
import numpy as np
from numba import jit

from .algorithm import Algorithm
from docrec.ndarray.utils import transitions


class StripsLines:

    def __init__(self, left_lines, right_lines, left_border, right_border):
        self.right_lines = right_lines
        self.left_lines = left_lines
        self.right_border = right_border
        self.left_border = left_border
        self.right_black = right_border.sum()
        self.left_black = left_border.sum()


class Lines:

    def __init__(self, locations, heights, mask):
        self.locations = locations
        self.heights = heights
        self.mask = mask
        self.length = locations.size if locations is not None else 0


@jit(nopython=True)
def _process_lines_mask(mask, starts, ends, h):
    ''' Remove small lines from mask. '''

    assert (mask.size > 0)

    # Threshold (h% of the average line height)
    tau_h = h * (ends - starts + 1).mean()

    # Search the first /last line not connected to the top / bottom edgess
    s = 0 if starts[0] else 1
    e = ends.size if ends[-1] else ends.size - 1

    # Clear small lines

    for i in range(s, e):
        if ends[i] - starts[i] + 1 < tau_h:
            mask[starts[i]:ends[i] + 1] = False


class Sleit(Algorithm):
    '''
    Algorithm Sleit (CostX)

    Sleit et al, An alternative clustering approach for reconstructing cross cut shredded text documents.
    (2011) Tellecomun. Systems.
    '''

    def __init__(self, t=0.15, h=0.25, p=0.33, linesth=1, blackth=2):
        ''' Constructor. '''

        self.t = t
        self.h = h
        self.p = p
        self.linesth = linesth
        self.blackth = blackth

        # result
        self.compatibilities = None


    def _lines_mask(self, img, h):
        ''' Detect lines of an image as a boolean mask array.'''

        # Assert img as a 2D-binary image
        assert (img.ndim == 2)

        # Horizontal projection histogram (HPH) (accounts black pixels)
        hph = img.sum(axis=1)

        # Lines are detected calculated by thresholdind the HPH
        mask = (hph > 0)
        if not mask.any():
            return mask
        starts, ends = transitions(mask, edges_val=0)

        # Remove small lines
        _process_lines_mask(mask, starts, ends, h)
        return mask


    def _lines(self, img, t, h):
        ''' Detect lines of a given part of si. '''

        img = 1 - img # negative version

        # Lines mask
        mask = self._lines_mask(img, h)
        if not mask.any():
            return Lines(None, None, None)

        starts, ends = transitions(mask, edges_val=0)

        # 1) Location
        centroid = lambda img: \
            int(
                float(
                    (img.sum(axis=1) * np.arange(img.shape[0])).sum()
                ) / img.sum() #+ 0.5
            )
        imgs = [img[starts[i] : ends[i] + 1] for i in range(starts.size)]
        locations = []
        for img in imgs:
            location = centroid(img)
            locations.append(location)

        locations = np.array(locations) + starts

        # 2) Heights
        heights = ends - starts + 1
        return Lines(locations, heights, mask)


    def _number_of_compatible_lines(self, A, B, pixelth):
        ''' Counts the number of compatible lines between the right part of si and the left part of sj. '''

        i = 0
        j = 0
        n_lines = 0

        while i < A.right_lines.length and j < B.left_lines.length:
            pos_difference = abs(A.right_lines.locations[i] - B.left_lines.locations[j])
            # print pos_difference, pixelth
            if pos_difference < pixelth:
                n_lines += 1
                i += 1
                j += 1
            else:
                if A.right_lines.locations[i] > B.left_lines.locations[j]:
                    j += 1
                else:
                    i += 1
        return n_lines


    def _lines_compatibility_x(self, A, B, pixelth, linesth):
        ''' Returns whether the right lines of A are compatible with the left lines of B. '''

        if A.right_lines.length == 0 or B.left_lines.length == 0:
            return False

        lines_difference = abs(A.right_lines.length - B.left_lines.length)
        number_of_compatible_lines = self._number_of_compatible_lines(A, B, pixelth)
        return lines_difference <= linesth and number_of_compatible_lines > 0


    def _pixel_compatibility_x_left_side(self, A, B, y):
        ''' Pixel-level compatibily function (left side). '''

        if A.right_border[y]: # white
            return 0

        c1 = -0.75 if B.left_border[y - 1] else 1.5
        c2 = -5 if B.left_border[y] else 4
        c3 = -0.75 if B.left_border[y + 1] else 1.5
        return c1 + c2 + c3


    def _pixel_compatibility_x_right_side(self, A, B, y):
        ''' Pixel-level compatibily function (right side). '''

        if B.left_border[y]: # white
            return 0

        c1 = -0.75 if A.right_border[y - 1] else 1.5
        c2 = -5 if A.right_border[y] else 4
        c3 = -0.75 if A.right_border[y + 1] else 1.5
        return c1 + c2 + c3


    def _all_pixel_compatibility(self, A, B):

        pixel_compatibility = lambda y: \
            self._pixel_compatibility_x_left_side(A, B, y) + \
            self._pixel_compatibility_x_right_side(A, B, y)
        h = A.right_border.size
        total = 0
        for y in range(1, h - 1):
            total += pixel_compatibility(y)
        return total


    def _cost_x(self, A, B, pixelth, linesth, blackth):
        ''' Cost function. '''

        h = A.right_border.size
        inf = (5 + 0.75 + 0.75) * (h - 2) * 2

        if A.right_black < blackth or B.left_black < blackth:
            return inf
        lines_compatibility_x = lambda A, B : \
            self._lines_compatibility_x(A, B, pixelth, linesth)
        return -self._all_pixel_compatibility(A, B) \
            if lines_compatibility_x(A, B) else inf


    def _extract_features(self, strips, t, d):
        ''' Features '''

        convert = lambda image : \
            cv2.threshold(
                cv2.cvtColor(image, cv2.COLOR_RGB2GRAY),
                0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )[1]
        images = [convert(strip.image) for strip in strips.strips]

        min_h = min([strip.h for strip in strips.strips])
        first = lambda strip: strip.offsets_l[: min_h]
        last = lambda strip: strip.offsets_r[: min_h]

        w = np.array(
            [strip.approx_width for strip in strips.strips]
        )
        E = np.round(t * w).astype(np.int32).tolist()
        L = list(map(first, strips.strips))
        R = list(map(last, strips.strips))

        idx = np.arange(min_h)
        text_left_features = []
        text_right_features = []
        pixel_left_features = []
        pixel_right_features = []
        for l, r, e, image in zip(L, R, E, images):
            temp1 = []
            temp2 = []
            for i in range(e):
                temp1.append(image[idx, l + d + i][:, np.newaxis])
                temp2.append(image[idx, r - d - i][:, np.newaxis])
            text_left_features.append(np.hstack(temp1))
            text_right_features.append(np.hstack(temp2))
            pixel_left_features.append(image[idx, l + d])
            pixel_right_features.append(image[idx, r - d])

        return text_left_features, text_right_features, pixel_left_features, \
            pixel_right_features


    def _compute_matrix(self, strips, d=0):
        ''' Compute matrix. '''

        tl, tr, pl, pr = self._extract_features(strips, self.t, d)

        # lines
        left_lines = lambda img: self._lines(img, self.t, self.h)
        right_lines = lambda img: self._lines(img, self.t, self.h)
        strips_lines = [ \
            StripsLines(
                left_lines(tl[i]), right_lines(tr[i]), pl[i], pr[i]
            )
        for i in range(len(strips.strips))]

        # global average line height
        values = []
        for s_lines in strips_lines:
            if s_lines.right_lines.heights is not None:
                values += s_lines.right_lines.heights.tolist()

            if s_lines.left_lines.heights is not None:
                values += s_lines.left_lines.heights.tolist()

        # adjust pixel threshold
        pixelth = self.p * np.mean(values)

        # Compute pairwise distances
        dist = lambda A, B: \
            self._cost_x(
                A, B, pixelth, self.linesth, self.blackth
            )
        N = len(strips.strips)
        matrix = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            for j in range(N):
                if i != j:
                    matrix[i, j] = dist(strips_lines[i], strips_lines[j])
        matrix = matrix - matrix.min() # shift for positive values
        np.fill_diagonal(matrix, 1e7)
        return matrix


    def run(self, strips, d=0):

        self.compatibilities = self._compute_matrix(strips, d)
        return self


    def name(self):
        return 'sleit'
