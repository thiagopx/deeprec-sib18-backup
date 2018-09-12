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
from docrec.metrics.matrix import perfect_matchings, precision_mc

import seaborn as sns
sns.set(
    context='paper', style='whitegrid', font_scale=1.5,# {}
)
colors = sns.color_palette('deep')
order = [5, 0, 1, 9, 3, 8, 2, 4, 6, 7]
pallete = [colors[i] for i in order]
sns.set_palette(pallete)

df = pd.DataFrame(
    columns=('method', 'legend', 'Dataset', 'document', 'pf', 'pm', 'direct', 'Accuracy (\\%)')
)
methods = ['proposed', 'proposed-mn', 'marques', 'morandell', 'andalo', 'balme', 'sleit']
legends = ['\\textbf{Proposed-SN}', '\\textbf{Proposed-MN}', 'Marques', 'Morandell', 'Andal\\\'o', 'Balme', 'Sleit']
proposed = [json.load(open('results/results_proposed.json', 'r'))]
proposed_mn = [json.load(open('results/results_proposed-mn.json', 'r'))]
results = proposed + proposed_mn + [json.load(open('results/results_best_{}.json'.format(method), 'r')) for method in methods[2 :]]

index = 0
for legend, method, result in zip(legends, methods, results):
    for doc in result:
        _, dataset, _, doc_ = doc.split('/')
        matrix = result[doc]['compatibilities']
        solution = result[doc]['solution']
        pf = perfect_matchings(matrix, pre_process=(method == 'proposed'), normalized=False)
        pm = precision_mc(matrix, pre_process=(method == 'proposed'), normalized=False)
        acc_dir = 100 * accuracy(solution, None, 'direct')
        acc_nei = 100 * accuracy(solution, None, 'neighbor')
        df.loc[index] = [method, legend, dataset, doc_, pf, pm, acc_dir, acc_nei]
        index += 1

df_d1 = df[df.Dataset == 'D1'].copy()
df_d2 = df[df.Dataset == 'D2'].copy()
df_d1_d2 = df.copy()
df_d1_d2['Dataset'] = 'D1 $\cup$ D2'
df_glob = pd.concat([df_d1_d2, df_d1, df_d2])

fp = sns.catplot(
    x='Dataset', y='Accuracy (\\%)', data=df_glob,
    hue='legend', kind='box', height=3, aspect=2.5,
    margin_titles=True, fliersize=1, width=0.8, linewidth=1.5,
    legend=False,
)
fp.despine(left=True)
plt.legend(loc='upper left', bbox_to_anchor=(0.0, -0.3), ncol=3)
path = 'graphs'
if len(sys.argv) > 1:
    path = sys.argv[1]
plt.savefig('{}/accuracy_best.pdf'.format(path), bbox_inches='tight')
