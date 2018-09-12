import cv2
import numpy as np
import json

from docrec.strips.strips import Strips
from docrec.metrics.solution import accuracy
from docrec.metrics.matrix import precision_mc, perfect_matchings

results = json.load(open('results/results_proposed.json', 'r'))

worse = ''
worse_acc= 1.1
for doc in results:
    _, dataset, _, doc_ = doc.split('/')
    solution = results[doc]['solution']
    acc = accuracy(solution)
    if acc < worse_acc:
        worse = doc
        worse_acc = acc

qc = perfect_matchings(np.array(results[worse]['compatibilities']), True, True)
kbh = precision_mc(np.array(results[worse]['compatibilities']), True, True)
print('worse={} Qc={:.2f} KBH={:.2f} acc={:.2f}'.format(worse, 100 * qc, 100 * kbh, 100 * worse_acc))

