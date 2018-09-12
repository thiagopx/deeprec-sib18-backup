import cv2
import numpy as np
import json

from docrec.strips.strips import Strips
from docrec.metrics.solution import accuracy
from docrec.metrics.matrix import precision_mc

results = json.load(open('results/results_proposed.json', 'r'))

perfect_kbh = 0
perfect_ls = 0
for doc in results:
    _, dataset, _, doc_ = doc.split('/')
    solution = results[doc]['solution']
    compatibilities = np.asarray(results[doc]['compatibilities'])
    if accuracy(solution) == 1:
        perfect_ls += 1
    if precision_mc(compatibilities, True, True) == 1:
        perfect_kbh += 1

print('#perfect_kbh={} #perfect_ls={}'.format(perfect_kbh, perfect_ls))
