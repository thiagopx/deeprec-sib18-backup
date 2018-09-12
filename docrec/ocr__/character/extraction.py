import numpy as np
from skimage import measure
import matplotlib.pyplot as plt


def extract_characters(text, max_width=None, min_overlap = 0.5, invert=False):
    ''' Extract characters from a text box.

    chars = [(box, patch), ...], where patch is in RGB format.
    '''

    box, patch_bin, patch_rgb = text
    offset_x, offset_y, _, _ = box

    # Char box width range
    _, w = patch_bin.shape
    if max_width is None:
        max_width = w

    # Filtering
    labels = measure.label(patch_bin)
    props = measure.regionprops(labels)
    props = [region for region in props if region.image.shape[1] < max_width]

    # Merge characters
    chars = []
    remaining = set(range(len(props)))
    while remaining:
        i = min(remaining)
        remaining.remove(i)
        merged = [i]
        region_i = props[i]
        yi, xi, _, _ = region_i.bbox
        hi, wi = region_i.image.shape
        for j in remaining:
            region_j = props[j]
            yj, xj, _, _ = region_j.bbox
            hj, wj = region_j.image.shape
            intersection =  min(xi + wi - 1, xj + wj - 1) - max(xi, xj)
            overlap = float(intersection) / min(wi, wj)
            if overlap >= min_overlap:
                merged.append(j)
                xm = min(xi, xj)
                ym = min(yi, yj)
                wm = max(xi + wi, xj + wj) - xm
                hm = max(yi + hi, yj + hj) - ym
                xi, yi, wi, hi = xm, ym, wm, hm
        remaining -= set(merged)
        box = (offset_x + xi, offset_y + yi, wi, hi)
        patch = patch_rgb[yi : yi + hi, xi : xi + wi]
        if invert:
            patch = 255 - patch
        chars.append((box, patch))
    return chars