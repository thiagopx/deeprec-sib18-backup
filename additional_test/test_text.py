import sys
import matplotlib.pyplot as plt
import time

sys.path.append('.')
from docrec.strips.stripstext import StripsText

# segmentation
print('=> Segmentation')
t0 = time.time()
strips = StripsText(path='datasets/D2/mechanical/D002', filter_blanks=True)
print('Segmentation elapsed time={:.2f} seconds'.format(time.time() - t0))
N = len(strips.strips)
fig = plt.figure(figsize=(8, 8), dpi=150)
for i in range(N):
    for j in range(N):
        if i + 1 == j:
            t0 = time.time()
            image = strips.pair(i, j)
            print('Pairing time={:.2f} seconds'.format(time.time() - t0))
            plt.clf()
            plt.imshow(image)
            #plt.savefig('/home/thiagopx/temo/{}-{}.pdf'.format(i, j))
            plt.show()
strips.plot()
plt.show()
