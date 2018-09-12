import sys
import os
import cv2

sys.path.append('.')
from docrec.strips.strips import Strips

docs = ['datasets/D2/mechanical/D{:03}'.format(i) for i in range(1, 21)][:1]
for doc in docs:
    print('Processing document {}'.format(doc))
    strips = Strips(path=doc, filter_blanks=True)
    image = strips.reconstruction_image()
    cv2.imwrite('test/test_recimage/{}.jpg'.format(os.path.basename(doc)), image)