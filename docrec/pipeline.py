import numpy as np
from time import time

from .strips.strips import Strips
#from .solver.solverls import SolverLS


class Pipeline:

    def __init__(self, algorithm, solver):

        self.algorithm = algorithm
        self.solver = solver


    def run(self, strips, d=0):

        self.algorithm.run(strips, d)
        self.solver.solve(self.algorithm.compatibilities)

        if self.algorithm.name() in ['proposed', 'proposed-mn']:
            return self.solver.solution, self.algorithm.compatibilities, self.algorithm.displacements

        return self.solver.solution, self.algorithm.compatibilities
