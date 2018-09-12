import os
import json
import shutil

from docrec.metrics.solution import accuracy


# dataset 1 (category info)
categories = json.load(open('categories_D1.json', 'r'))
doc_category_map = {}
for category, docs in categories.items():
    for doc in docs:
        doc_category_map[doc] = category.upper()
print('#documents in D1 (per categories): to = {}, lg = {} fg = {}'.format(
    len(categories['to']), len(categories['lg']), len(categories['fg'])
))

# data to be analyzed
results = json.load(open('results/results_proposed.json', 'r'))
for doc in results:
    _, dataset, _, doc_ = doc.split('/')
    solution = results[doc]['solution']
    comp = results[doc]['compatibilities']
    if dataset == 'D2':
        print(comp)
        continue
    acc = accuracy(solution)
    print('{}-{} {:.2f}% {}'.format(dataset, doc_, 100 * acc, doc_category_map[doc_] if dataset == 'D1' else ''))
    if acc < 0.8:
        if dataset == 'D1':
            src1 = '{}/{}.jpg'.format(doc, doc_)
        else:
            src1 = 'datasets/D2/integral/{}.TIF'.format(doc_)
        shutil.copy(src1, 'ignore/{}-{}'.format(dataset, os.path.basename(src1)))

        if os.path.exists('ignore/{}-{}'.format(dataset, doc_)):
            shutil.rmtree('ignore/{}-{}'.format(dataset, doc_))
        shutil.copytree(doc, 'ignore/{}-{}'.format(dataset, doc_))