import os
import argparse
import matplotlib.pyplot as plt
from docrec.metrics.solution import accuracy
from docrec.strips.strips import Strips
from docrec.compatibility.proposed import Proposed
from docrec.solver.solverlocal import SolverLS
from docrec.pipeline import Pipeline

#path_model = '{}/{}'.format(os.getcwd(), open('best_model.txt').read())
path_model = 'model.npy'
# reconstruction pipeline (compatibility algorithm + solver)
algorithm = Proposed(path_model, 10, (3000, 31), num_classes=2)
solver = SolverLS(maximize=True)
pipeline = Pipeline(algorithm, solver)

parser = argparse.ArgumentParser(description='Score.')
parser.add_argument(
    '-d', '--d', action='store', dest='doc', required=False, type=str,
    default='datasets/D1/artificial/D001', help='Document.'
)
args = parser.parse_args()
strips = Strips(path=args.doc, filter_blanks=True)
solution, compatibilities, displacements = pipeline.run(strips)
print(compatibilities, displacements, solution)
print(accuracy(solution))