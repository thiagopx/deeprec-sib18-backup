import sys
import json
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
rc('font',**{'family':'serif','serif':['Times']})
import pandas as pd

from docrec.metrics.solution import accuracy
from docrec.metrics.matrix import precision_mc

import seaborn as sns
sns.set(
    context='paper', style='whitegrid', palette='deep', font_scale=1.5,# {}
)

# accuracy
df = pd.DataFrame(
    columns=('Network/Dataset', 'Document', 'Solver', 'Accuracy (\\%)')
)

index = 0
for net in ['SN', 'MN']:
    comp = '-mn' if net == 'MN' else ''
    results = json.load(open('results/results_proposed{}.json'.format(comp), 'r'))
    for doc in results:
        _, dataset, _, doc_ = doc.split('/')
        matrix = results[doc]['compatibilities']
        solution = results[doc]['solution']
        acc = 100 * accuracy(solution, None, 'neighbor')
        bl = 100 * precision_mc(matrix, True, True)
        df.loc[index] = ['{}/{}'.format(net, dataset), int(doc_[1: ]), 'KBH',  bl]
        index += 1
        df.loc[index] = ['{}/{}'.format(net, dataset), int(doc_[1: ]), 'LS',  acc]
        index += 1

meanlineprops = dict(linestyle='--', linewidth=1, color='red')

fp = sns.catplot(
    x='Network/Dataset', y='Accuracy (\\%)', data=df,
    hue='Solver', kind='box', hue_order=['KBH','LS'], height=3, aspect=2,
    margin_titles=True, fliersize=1.0, width=0.6, linewidth=1,
    legend=True, showmeans=True, meanline=True, meanprops=meanlineprops
)

fp.despine(left=True)
path = 'graphs'
if len(sys.argv) > 1:
    path = sys.argv[1]
plt.savefig('{}/accuracy_proposed.pdf'.format(path), bbox_inches='tight')
