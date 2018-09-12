import numpy as np


def accuracy(solution, ground_truth=None, method='neighbor'):
    ''' Accuracy by direct  / neighbor comparison. '''

    assert len(solution) > 0
    assert method in ('direct', 'neighbor')

    N = len(solution)
    if ground_truth is None:
        ground_truth = list(range(N))

    num_correct = 0
    if method == 'direct':
        for i in range(N):
            if solution[i] == ground_truth[i]:
                num_correct += 1
        return num_correct / N

    neighbors = {ground_truth[i]: ground_truth[i + 1] for i in range(N - 1)}
    neighbors[ground_truth[N - 1]] = -1 # no neighbor for the last
    for i in range(N - 1):
        if neighbors[solution[i]] == solution[i + 1]:
            num_correct += 1
    return num_correct / (N - 1)