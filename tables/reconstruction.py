import json
import pandas as pd

from docrec.metrics.solution import accuracy
from docrec.metrics.matrix import precision_mc

template = """
\\begin{table*}[htb]
	\\centering
	\\caption{Full reconstruction performance (original compatibility methods + our ATSP-based solver): $Acc_\pi \pm \sigma $ (\\%%).}
	\\label{tab:rec}
	\\begin{tabular}{lcccccc} \\toprule
		\multirow{3}{*}{\\textbf{Method}} & \multicolumn{2}{c}{D1 $\cup$ D2} & \multicolumn{2}{c}{D1} & \multicolumn{2}{c}{D2}\\\\
		\cmidrule(l){2-3}  \cmidrule(l){4-5}  \cmidrule(l){6-7}
		& KBH & LS & KBH & LS & KBH & LS \\\\
		\\midrule
        %s
		\\bottomrule
	\\end{tabular}
\\end{table*}
"""

# accuracy
df = pd.DataFrame(
    columns=('method', 'legend', 'dataset', 'doc', 'bl', 'ls')
)
methods = ['proposed', 'proposed-mn', 'marques', 'morandell', 'andalo', 'balme', 'sleit']
legends = ['\\textbf{Proposed-SN}', '\\textbf{Proposed-MN}', 'Marques', 'Morandell', 'Andaló', 'Balme', 'Sleit']
proposed = [json.load(open('results/results_proposed.json', 'r'))]
proposed_mn = [json.load(open('results/results_proposed-mn.json', 'r'))]
results = proposed + proposed_mn + [json.load(open('results/results_best_{}.json'.format(method), 'r')) for method in methods[2 :]]

index = 0
for legend, method, result in zip(legends, methods, results):
    for doc in result:
        _, dataset, _, doc_ = doc.split('/')
        matrix = result[doc]['compatibilities']
        solution = result[doc]['solution']
        flag = (method in ['proposed', 'proposed-mn'])
        bl = 100 * precision_mc(matrix, pre_process=flag, normalized=True)
        ls = 100 * accuracy(solution, None, 'neighbor')
        df.loc[index] = [method, legend, dataset, doc_, bl, ls]
        index += 1

body = ''
for legend, method in zip(legends, methods):
    flag = (method in ['proposed', 'proposed-mn'])
    num_str = '\\textbf{{{:.2f} $\pm$ {:.2f}}}' if flag else '{:.2f} $\pm$ {:.2f}'
    df_ = df[df.method == method]
    bl = df_['bl'].mean()
    bl_s = df_['bl'].std()
    ls = df_['ls'].mean()
    ls_s = df_['ls'].std()
    df_d1 = df_[df_.dataset == 'D1']
    df_d2 = df_[df_.dataset == 'D2']
    bl_d1 = df_d1['bl'].mean()
    bl_d1_s = df_d1['bl'].std()
    ls_d1 = df_d1['ls'].mean()
    ls_d1_s = df_d1['ls'].std()
    bl_d2 = df_d2['bl'].mean()
    bl_d2_s = df_d2['bl'].std()
    ls_d2 = df_d2['ls'].mean()
    ls_d2_s = df_d2['ls'].std()
    values = [(bl, bl_s), (ls, ls_s), (bl_d1, bl_d1_s), (ls_d1, ls_d1_s), (bl_d2, bl_d2_s), (ls_d2, ls_d2_s)]

    body += '{} & {}\\\\ \n '.format(legend, ' & '.join([num_str.format(v, s) for v, s in values]))

open('../../Dropbox/pubs/sibgrapi2018/latex/tables/reconstruction.tex', 'w').write(template % body)