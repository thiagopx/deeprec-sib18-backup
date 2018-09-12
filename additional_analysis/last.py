import cv2
import numpy as np
import json

from docrec.strips.strips import Strips
from docrec.metrics.solution import accuracy
from docrec.metrics.matrix import precision_mc

results = json.load(open('results/results_proposed.json', 'r'))

last = 0
for doc in results:
    _, dataset, _, doc_ = doc.split('/')
    solution = results[doc]['solution']
    N = len(solution)
    for i in range(N - 1):
        if solution[i] == N - 1 and solution[i + 1] == 0:
            last += 1

print('#last={}'.format(last))
