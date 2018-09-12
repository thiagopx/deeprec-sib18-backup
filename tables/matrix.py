import json
import pandas as pd

from docrec.metrics.matrix import perfect_matchings

template = """
\\begin{table}[b]
	\\centering
	\\caption{Performance of the compatibility scoring methods: $Q_{\mathbf{C}} \pm \sigma$ (\\%%).}
	\\label{tab:matrix}
	\\begin{tabular}{lccc} \\toprule
		\\textbf{Method} & D1 $\cup$ D2 & D1 & D2\\\\
		\\midrule
        %s
		\\bottomrule
	\\end{tabular}
\\end{table}
"""
#       \multirow{3}{*}{\\textbf{Method}} & D1 $\cup$ D2 & D1 & D2\\\\
#		\cmidrule(l){2-2}  \cmidrule(l){3-3}  \cmidrule(l){4-4}
#		& $Q_{\mathbf{C}} \pm \sigma$ (\\%%) & $Q_{\mathbf{C}} \pm \sigma$ (\\%%) & $Q_{\mathbf{C}} \pm \sigma$ (\\%%) \\\\

# accuracy
df = pd.DataFrame(
    columns=('method', 'legend', 'dataset', 'doc', 'pm')
)
methods = ['proposed', 'proposed-mn', 'andalo', 'morandell', 'balme', 'sleit', 'marques']
legends = ['\\textbf{Proposed-SN}', '\\textbf{Proposed-MN}', 'Andaló', 'Morandell',  'Balme', 'Sleit', 'Marques']
#proposed = [json.load(open('results.json', 'r'))] # <= change
proposed = [json.load(open('results/results_proposed.json', 'r'))] # <= change
proposed_mn = [json.load(open('results/results_proposed-mn.json', 'r'))] # <= change
results = proposed + proposed_mn + [json.load(open('results/results_best_{}.json'.format(method), 'r')) for method in methods[2 :]]

index = 0
for legend, method, result in zip(legends, methods, results):
    for doc in result:
        _, dataset, _, doc_ = doc.split('/')
        matrix = result[doc]['compatibilities']
        solution = result[doc]['solution']
        flag = (method in ['proposed', 'proposed-mn'])
        pm = 100 * perfect_matchings(matrix, pre_process=flag, normalized=True)
        df.loc[index] = [method, legend, dataset, doc_, pm]
        index += 1

body = ''
for legend, method in zip(legends, methods):
    flag = (method in ['proposed', 'proposed-mn'])
    num_str = '\\textbf{{{:.2f} $\pm$ {:.2f}}}' if flag else '{:.2f} $\pm$ {:.2f}'
    df_ = df[df.method == method]
    pm = df_['pm'].mean()
    pm_s = df_['pm'].std()
    df_d1 = df_[df_.dataset == 'D1']
    df_d2 = df_[df_.dataset == 'D2']
    pm_d1 = df_d1['pm'].mean()
    pm_d1_s = df_d1['pm'].std()
    pm_d2 = df_d2['pm'].mean()
    pm_d2_s = df_d2['pm'].std()
    values = [(pm, pm_s), (pm_d1, pm_d1_s), (pm_d2, pm_d2_s)]
    body += '{} & {}\\\\ \n '.format(legend, ' & '.join([num_str.format(v, s) for v, s in values]))

# change path to save the latex table in other place
open('../../Dropbox/pubs/sibgrapi2018/latex/tables/matrix.tex', 'w').write(template % body)