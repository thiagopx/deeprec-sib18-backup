import os
import numpy as np
import cv2
import argparse
import time
import json
import shutil
import random
import glob

from docrec.image import utils
from docrec.strips.strips import Strips

''' Command line
$ python dataset.py -v 0.1 -r 1.0 -d 3 -nd 250 -s 'datasets/patches -ns 30
'''

# sudo apt-get install tesseract-ocr libtesseract-dev libleptonica-dev
# CC=/usr/bin/gcc-5 CXX=/usr/bin/g++-5 pip install tesserocr
# pip install Pillow
# conda install numba matplotlib scikit-image

'''
https://code.google.com/archive/p/isri-ocr-evaluation-tools
'''
ISRI_DATASET_DIR = '../datasets/isri-ocr'
ignore_images = ['9461_009.3B', '8509_001.3B', '8519_001.3B', '8520_003.3B', '9415_015.3B',
                 '8541_001.3B', '8541_001.3B', '8541_001.3B', '8541_001.3B', '8541_001.3B',
                 '9421_005.3B', '9421_005.3B', '9421_005.3B', '9462_056.3B', '8502_001.3B',
                 '8518_001.3B', '8535_001.3B', '9413_018.3B', '8505_001.3B', '9462_096.3B']
TEMP_DIR = '/tmp/doc'


def create_directory(dir):
    ''' Create a directory if it does no exist. '''

    if not os.path.exists(dir):
        os.makedirs(dir)


def init(args):
    ''' Initial setup. '''

    # build tree directory
    create_directory('{}/positives'.format(args.savedir))
    create_directory('{}/negatives'.format(args.savedir))
    #create_directory('{}/neutral'.format(args.savedir))


def test(args, radius=15, hdisp=3, disp_noise=2, pcont=0.3):

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
        for i in range(args.num_strips):
            dw = int((w - acc) / (args.num_strips - i))
            strip = image[:, acc : acc + dw]
            noise_left = np.random.randint(0, 255, (h, disp_noise)).astype(np.uint8)
            noise_right = np.random.randint(0, 255, (h, disp_noise)).astype(np.uint8)
            for j in range(c): # for each channel
                strip[:, : disp_noise, j] = cv2.add(strip[:, : disp_noise, j], noise_left)
                strip[:, -disp_noise :, j] = cv2.add(strip[:, -disp_noise :, j], noise_right)
            cv2.imwrite('{}/strips/D001{:02}.jpg'.format(TEMP_DIR, i + 1), strip)
            acc += dw

        print('     => Load strips object', end='')
        strips = Strips(path=TEMP_DIR, filter_blanks=True)
        print('done!')

def main():

    np.random.seed(0)
    random.seed(0)
    parser = argparse.ArgumentParser(
        description='Dataset patches assembling.'
    )
    parser.add_argument(
        '-ns', '--nstrips', action='store', dest='num_strips', required=False, type=int,
        default=5, help='Number of strips (simulated shredding).'
    )
    args = parser.parse_args()

    test(args)
    

if __name__ == '__main__':
    t0 = time.time()
    main()
    print('Elapsed time={:.2f} minutes'.format((time.time() - t0) / 60.0))
