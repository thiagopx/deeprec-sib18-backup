import sys
import matplotlib.pyplot as plt
import time

sys.path.append('.')
from docrec.strips.strips import Strips

# segmentation
print('=> Segmentation')
t0 = time.time()
strips = Strips(path='datasets/D2/mechanical/D002', filter_blanks=True)
strips.plot()
plt.show()
print('Strips elapsed time={:.2f} seconds'.format(time.time() - t0))
N = len(strips.strips)
fig = plt.figure(figsize=(8, 8), dpi=150)
for i in range(N):
    for j in range(N):
        if i + 1 == j:
            t0 = time.time()
            print(i, j, N)
            image = strips.pair(i, j, filled=True)
            print('Pairing time={:.2f} seconds'.format(time.time() - t0))
            plt.clf()
            plt.imshow(image)
            #plt.savefig('/home/thiagopx/temo/{}-{}.pdf'.format(i, j))
            plt.show()
#strips.plot()

