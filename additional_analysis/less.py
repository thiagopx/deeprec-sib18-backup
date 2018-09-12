import cv2
import numpy as np
import json

from docrec.strips.strips import Strips
from docrec.metrics.solution import accuracy
from docrec.metrics.matrix import precision_mc

results = json.load(open('results/results_proposed.json', 'r'))

total = 0
for doc in results:
    _, dataset, _, doc_ = doc.split('/')
    solution = results[doc]['solution']
    N = len(solution)
    last = False
    for i in range(N - 1):
        if solution[i] == N - 1 and solution[i + 1] == 0:
            last = True
    mistakes = 0
    for i in range(N - 1):
        if solution[i] + 1 != solution[i + 1]:
            mistakes += 1

    if (mistakes == 1) and last:
        total += 1
    else:
        print(doc, N, '{:.2f}'.format(100 * accuracy(solution)))


print('#last={}'.format(total))
