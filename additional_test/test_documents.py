import os
import numpy as np
import cv2
import shutil
import random
import glob

from docrec.image import utils
from docrec.strips.strips import Strips


'''
https://code.google.com/archive/p/isri-ocr-evaluation-tools
'''
ISRI_DATASET_DIR = 'datasets/isri-ocr'
ignore_images = ['9461_009.3B', '8509_001.3B', '8519_001.3B', '8520_003.3B', '9415_015.3B',
                 '8541_001.3B', '8541_001.3B', '8541_001.3B', '8541_001.3B', '8541_001.3B',
                 '9421_005.3B', '9421_005.3B', '9421_005.3B', '9462_056.3B', '8502_001.3B',
                 '8518_001.3B', '8535_001.3B', '9413_018.3B', '8505_001.3B', '9462_096.3B']
TEMP_DIR = '/tmp/doc'


def create_directory(dir):
    ''' Create a directory if it does no exist. '''

    if not os.path.exists(dir):
        os.makedirs(dir)


disp_noise = 2
docs = glob.glob('{}/**/*.tif'.format(ISRI_DATASET_DIR), recursive=True)
for f, fname in enumerate(docs, 1):
    print('Processing document {}/{}'.format(f, len(docs)))
    if os.path.basename(fname).replace('.tif', '') in ignore_images:
        print('{} is no considered to compose the dataset.'.format(fname))
        continue

    # generate temporary strips
    print('     => Shredding')
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs('{}/strips'.format(TEMP_DIR))
    image = cv2.imread(fname)
    h, w, c = image.shape
    acc = 0
    for i in range(30):
        dw = int((w - acc) / (30 - i))
        strip = image[:, acc : acc + dw]
        noise_left = np.random.randint(0, 255, (h, disp_noise)).astype(np.uint8)
        noise_right = np.random.randint(0, 255, (h, disp_noise)).astype(np.uint8)
        for j in range(c): # for each channel
            strip[:, : disp_noise, j] = cv2.add(strip[:, : disp_noise, j], noise_left)
            strip[:, -disp_noise :, j] = cv2.add(strip[:, -disp_noise :, j], noise_right)
        cv2.imwrite('{}/strips/D001{:02}.jpg'.format(TEMP_DIR, i + 1), strip)
        acc += dw

    print('     => Load strips object')
    strips = Strips(path=TEMP_DIR, filter_blanks=True)
