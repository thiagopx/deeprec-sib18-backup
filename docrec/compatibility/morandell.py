import cv2
import numpy as np
from numba import jit

from .algorithm import Algorithm


@jit(nopython=True)
def _delta_x(X, Y, X_dist, Y_closest, epsilon):
    ''' Calculate delta_x values. '''

    for x in range(epsilon, X.size - epsilon):
        X_dist[x - epsilon] = epsilon + 1

        if not X[x]:
            X_dist[x - epsilon] = 0
            continue

        # Check zero distance
        if Y[x]:
            X_dist[x - epsilon] = 0
            Y_closest[x - epsilon] = 1
            continue

        # Check distances less or equal then epsilon
        for d in range(1, epsilon + 1):
            if Y[x - d]:
                X_dist[x - epsilon] = d
                Y_closest[x - d - epsilon] = 1
                break

            if Y[x + d]:
                X_dist[x - epsilon] = d
                Y_closest[x + d - epsilon] = 1
                break

@jit(nopython=True)
def _delta_y(Y, X, Y_dist, epsilon):
    ''' Calculate delta_x values. '''

    for y in range(epsilon, Y.size - epsilon):
        Y_dist[y - epsilon] = epsilon + 1

        if not Y[y]:
            Y_dist[y - epsilon] = 0
            continue

        # Check zero distance
        if X[y]:
            Y_dist[y - epsilon] = 0
            continue

        # Check distances less or equal then epsilon
        for d in range(1, epsilon + 1):
            if X[y - d] or X[y + d]:
                Y_dist[y - epsilon] = d
                break


class Morandell(Algorithm):
    '''
    Algorithm Morandell

    Evaluation and Reconstruction of Strip-Shredded Text Documents (2008).
    '''

    def __init__(self, epsilon=10, h=1, pi=0, phi=0):
        ''' Constructor. '''

        self.epsilon = epsilon
        self.h = h
        self.pi = pi
        self.phi = phi

        # result 
        self.compatibilities = None


    def _cost(self, X, Y, epsilon, h, pi, phi):

        # Invert and pad image
        X_ext = np.pad(X, (epsilon, epsilon), mode='constant', constant_values=False)
        Y_ext = np.pad(Y, (epsilon, epsilon), mode='constant', constant_values=False)

        X_dist = np.empty(X.size, dtype=np.int32)
        Y_dist = np.empty(Y.size, dtype=np.int32)

        # X -> Y cost
        Y_closest = np.zeros(Y.size, dtype=np.uint8)
        _delta_x(X_ext, Y_ext, X_dist, Y_closest, epsilon)

        # Exclude Y* pixels from analysis
        Y_star = np.where(Y_closest == 1)[0]
        Y_ext[Y_star + epsilon] = False

        # Y -> X cost
        _delta_y(Y_ext, X_ext, Y_dist, epsilon)

        correction = np.vectorize(lambda d: pi if d == 0 else (d**h if d <= epsilon else epsilon**h + phi))
        X_dist = np.apply_along_axis(correction, 0, X_dist)
        Y_dist = np.apply_along_axis(correction, 0, Y_dist)

        return X_dist.sum() + Y_dist.sum()


    def _compute_matrix(self, strips, d):
        ''' Compute cost matrix. '''
        
        # Distance computation
        dist = lambda x, y : self._cost(x, y, self.epsilon, self.h, self.pi, self.phi)
        
        min_h = min([strip.h for strip in strips.strips])

        features = []
        for strip in strips.strips:
            features.append(self._extract_features(strip, d, min_h))

        N = len(strips.strips)
        matrix = np.zeros((N, N), dtype=np.float32)
        l, r = list(zip(*features))
        for i in range(N):
            for j in range(N):
                if i != j:
                    matrix[i, j] = dist(r[i], l[j])

        np.fill_diagonal(matrix, 1e7)
        return matrix

    
    def _extract_features(self, strip, d, size):
        ''' Features. '''
    
        # inverted thresholded image
        _, image_bin = cv2.threshold(
            cv2.cvtColor(strip.image, cv2.COLOR_RGB2GRAY),
            0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
                
        # borders
        l = strip.offsets_l[: size] + d
        r = strip.offsets_r[: size] - d

        # features
        idx = np.arange(size)
        left = image_bin[idx, l + d]
        right = image_bin[idx, r - d]
        return left, right
    
    
    def run(self, strips, d=0):
        
        self.compatibilities = self._compute_matrix(strips, d)
        return self


    def name(self):

        return 'morandell'
