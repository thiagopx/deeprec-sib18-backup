import cv2
import numpy as np
import json

from docrec.strips.strips import Strips
from docrec.metrics.solution import accuracy

results_prop = json.load(open('results/results_proposed.json', 'r'))
results_marques = json.load(open('results/results_best_marques.json', 'r'))

for dataset in ['D1', 'D2']:
    avg_prop = 0
    avg_marques = 0
    perfect_ls = 0
    cnt = 0
    for doc in results_prop:
        _, dataset_, _, doc_ = doc.split('/')
        if dataset == dataset_:
            solution_prop = results_prop[doc]['solution']
            solution_marques = results_marques[doc]['solution']
            avg_prop += accuracy(solution_prop)
            avg_marques += accuracy(solution_marques)
            cnt += 1

    avg_prop /= cnt
    avg_marques /= cnt
    print('{}: avg_prop={} - avg_marques={} => {} '.format(dataset, 100 * avg_prop, 100 * avg_marques, 100 * (avg_prop - avg_marques)))
