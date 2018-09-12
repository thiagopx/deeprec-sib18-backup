import numpy as np
import cv2
from skimage import measure

from docrec.ndarray import utils


def extract_text(image, min_height, max_height, max_separation, max_extent, pbline=0.66):
    ''' Extract text regions from image.

    text = [(box, patch_binary, patch_rgb), ...]
    '''

    # Thresholded image
    _, thresh = cv2.threshold(
        cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Text candidates
    dx = int(2 * max_separation)
    dy = int(dx / 4)
    dx_dy = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dx, dy))
    dilated = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, dx_dy)
    labels = measure.label(dilated)
    props = measure.regionprops(labels)

    # Filtering
    labels_to_remove = set([])
    for region in props:
        yt_min, _, yt_max, _ = region.bbox
        ht, wt = region.image.shape

        if (region.label in labels_to_remove) or (ht > max_height):
            labels_to_remove.update(
                set(labels[yt_min : yt_max, : ].flatten())
            )
        elif (ht < min_height) or (wt < 2):
            labels_to_remove.update([region.label])

    text = []
    for region in props:
        if region.label not in labels_to_remove:
            y_min, x_min, y_max, x_max = region.bbox # dilated region

            # trim patch
            patch_bin = thresh[y_min : y_max, x_min : x_max] # cut binary patch
            hph = patch_bin.sum(axis=1) # horizontal proj. histogram
            y_min_, y_max_ = utils.first_nonzero(hph), utils.last_nonzero(hph) + 1 # local y
            if y_max_ - y_min_ < min_height:
                continue

            vph = patch_bin.sum(axis=0) # vertical proj. histogram
            x_min_, x_max_ = utils.first_nonzero(vph), utils.last_nonzero(vph) + 1 # local x
            if x_max_ - x_min_ < 3:
                continue
 
            h, w = (y_max_ - y_min_, x_max_ - x_min_)
            y_min += y_min_
            y_max = y_min + h
            x_min += x_min_
            x_max = x_min + w
            patch_bin = thresh[y_min : y_max, x_min : x_max] # cut binary patch
            patch_rgb = image[y_min : y_max, x_min : x_max] # cut rgb patch

            # bounding box
            box = (x_min, y_min, w, h)

            # baseline
            hph = patch_bin.sum(axis=1)[int(pbline * h) :] # down half
            if hph.size < 3:
                baseline = y_max - 1
            else:
                baseline = np.correlate(hph, [0.25, 0.5, 0.25], 'same').argmax() + int(pbline * h) + y_min
            text.append((box, patch_bin, patch_rgb, baseline))
    return text
