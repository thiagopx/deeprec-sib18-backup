import numpy as np

import libjigsaw
from .algorithm import Algorithm


class Andalo(Algorithm):
    ''' Algorithm Andalo 2017. '''

    def __init__(self, p=1.0, q=0.3):
      
        self.p = p
        self.q = q
        
        self.compatibilities = None

   
    def _compute_matrix(self, strips, d):
        ''' Compute cost matrix. '''
        
        min_h = min([strip.h for strip in strips.strips])
        
        features = []
        for strip in strips.strips:
            features.append(self._extract_features(strip, d, min_h))

        N = len(strips.strips)
        matrix = libjigsaw.compatibility(np.hstack(features), N, self.p, self.q)
        matrix = matrix - matrix.min()
        np.fill_diagonal(matrix, 1e7)        
        return matrix


    def _extract_features(self, strip, d, size):
        ''' Features. '''

        # borders
        l = strip.offsets_l[: size] + d
        r = strip.offsets_r[: size] - d
        
        # Extract borders
        idx = np.arange(size)
        features = []
        temp = np.zeros((size, 4 , 3), dtype=np.uint8)
        temp[:, 0] = strip.image[idx, l + d]
        temp[:, 1] = strip.image[idx, l + d + 1]
        temp[:, 2] = strip.image[idx, r - d - 1]
        temp[:, 3] = strip.image[idx, r - d]
        return temp

    
    def run(self, strips, d=0):
        ''' Run algorithm. '''
        
        self.compatibilities = self._compute_matrix(strips, d)
        return self

    
    def name(self):
        
        return 'andalo'
