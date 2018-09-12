import cv2
import numpy as np
import json

from docrec.strips.strips import Strips
from docrec.metrics.solution import accuracy
from docrec.metrics.matrix import perfect_matchings, precision_mc
import glob

results = json.load(open('results/results_proposed.json', 'r'))

for doc in results:
    _, dataset, _, doc_ = doc.split('/')
    solution = results[doc]['solution']
    displacements = np.asarray(results[doc]['displacements'])
    compatibilities = np.asarray(results[doc]['compatibilities'])
    strips = Strips(path=doc, filter_blanks=True)
    reconstruction = strips.image(solution, displacements, True)
    print('Document {} :: quality={:.2f}%  KBH={:.2f} LS={:.2f}%'.format(
        doc, 100 * perfect_matchings(compatibilities, True, True),
        100 * precision_mc(compatibilities, True, True), 100 * accuracy(solution)))
    #cv2.imshow('Reconstruction', recontruction)
    #cv2.waitKey(0)
    cv2.imwrite('illustration/reconstruction/{}-{}_{:.2f}.jpg'.format(dataset, doc_, 100 * accuracy(solution)),  reconstruction)
