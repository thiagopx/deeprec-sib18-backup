import cv2
import numpy as np
from .algorithm import Algorithm


class Marques(Algorithm):
    ''' Algorithm Marques 2013. '''

    def __init__(self):
        
        # result 
        self.compatibilities = None
   

    def _compute_matrix(self, strips, d):
        ''' Compute cost matrix. '''

        # distances computation
        dist = lambda x, y: np.sqrt(np.sum((x - y) ** 2))

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
    
        # value channel
        V = cv2.cvtColor(strip.image, cv2.COLOR_RGB2HSV)[:, :, 2]

        # borders
        l = strip.offsets_l[: size] + d
        r = strip.offsets_r[: size] - d

        # features
        idx = np.arange(size)
        left = V[idx, l + d]
        right = V[idx, r - d]
        return left, right
                
    
    def run(self, strips, d=3):
        ''' Run algorithm. '''
        
        self.compatibilities = self._compute_matrix(strips, d)
        return self

    
    def name(self):
        
        return 'marques'
