import time
import json

from docrec.strips.strips import Strips
from docrec.compatibility.andalo import Andalo
from docrec.compatibility.marques import Marques
from docrec.compatibility.balme import Balme
from docrec.compatibility.morandell import Morandell
from docrec.compatibility.sleit import Sleit
from docrec.solver.solverlocal import SolverLS
from docrec.pipeline import Pipeline

t0_ = time.time()

solver = SolverLS(maximize=False)
algorithms = [Andalo(p=1.0, q=3.0),
              Marques(),
              Balme(tau=0.1),
              Morandell(epsilon=10, h=1, pi=0, phi=0),
              Sleit(t=0.15, h=0.25, p=0.33, linesth=1, blackth=2)]
pipelines = [Pipeline(algorithm, solver) for algorithm in algorithms]

# reconstruction instances
docs1 = ['datasets/D1/mechanical/D{:03}'.format(i) for i in range(1, 62) if i != 3]
docs2 = ['datasets/D2/mechanical/D{:03}'.format(i) for i in range(1, 21)]
docs = docs1 + docs2

processed = 0
total = len(docs) * len(pipelines)
for pipeline in pipelines:
    results = dict()
    d = 2 if pipeline.algorithm.name() == 'marques' else 0
    for doc in docs:
        print('[{:.2f}] Processing document {} :: alg. {}'.format(100 * processed / total, doc, pipeline.algorithm.name()))
        processed += 1
        t0 = time.time()
        strips = Strips(path=doc, filter_blanks=True)
        solution, compatibilities = pipeline.run(strips, d)
        results[doc] = {
            'solution': solution,
            'compatibilities': compatibilities.tolist(),
            'time': time.time() - t0
        }
    json.dump(results, open('results_{}.json'.format(pipeline.algorithm.name()), 'w'))

print('Elapsed time={:.2f} sec.'.format(time.time() - t0_))
