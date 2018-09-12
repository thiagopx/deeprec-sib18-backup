import os
import numpy as np
import cv2
import argparse
import math
import random
import glob
import time


''' Command line
$ python dataset.py
Elapsed time=9.93 minutes (596.038068627 seconds)
'''

# sudo apt-get install tesseract-ocr libtesseract-dev libleptonica-dev
# CC=/usr/bin/gcc-5 CXX=/usr/bin/g++-5 pip install tesserocr
# pip install Pillow
# conda install numba matplotlib scikit-image

'''
https://code.google.com/archive/p/isri-ocr-evaluation-tools
'''
ISRI_DATASET_DIR = 'datasets/isri-ocr'
ignore_images = ['9461_009.3B', '8509_001.3B', '8519_001.3B', '8520_003.3B', '9415_015.3B',
                 '8541_001.3B', '8541_001.3B', '8541_001.3B', '8541_001.3B', '8541_001.3B',
                 '9421_005.3B', '9421_005.3B', '9421_005.3B', '9462_056.3B', '8502_001.3B',
                 '8518_001.3B', '8535_001.3B', '9413_018.3B', '8505_001.3B', '9462_096.3B']


def create_directory(dir):
    ''' Create a directory if it does no exist. '''

    if not os.path.exists(dir):
        os.makedirs(dir)


def init(args):
    ''' Initial setup. '''

    # build tree directory
    create_directory('{}/positives/train'.format(args.savedir))
    create_directory('{}/positives/val'.format(args.savedir))
    create_directory('{}/negatives/train'.format(args.savedir))
    create_directory('{}/negatives/val'.format(args.savedir))
    #create_directory('{}/neutral/train'.format(args.savedir))
    #create_directory('{}/neutral/val'.format(args.savedir))


def generate_samples(args, input_size=31, pcont=0.2, disp_noise=2):
    ''' Sampling process. '''

    if glob.glob('{}/**/*.jpg'.format(args.savedir)):
        print('Sampling already done!')
        return

    docs = glob.glob('{}/**/*.tif'.format(ISRI_DATASET_DIR), recursive=True)

    # filter documents in ignore list
    docs = [doc for doc in docs if os.path.basename(doc).replace('.tif', '') not in ignore_images]
    random.shuffle(docs)
    if args.num_docs is not None:
        docs = docs[ : args.num_docs]

    # split train and val sets
    num_docs = len(docs)
    docs_train = docs[int(args.pval * num_docs) :]
    docs_val = docs[ : int(args.pval * num_docs)]

    processed = 0
    size_right = math.ceil(input_size / 2)
    size_left = input_size - size_right
    for mode, docs in zip(['train', 'val'], [docs_train, docs_val]):
        count = {'positives': 0, 'negatives': 0}#, 'neutral': 0}
        for doc in docs:
            max_per_doc = 0

            print('Processing document {}/{}[mode={}]'.format(processed + 1, num_docs, mode))
            processed += 1

            # shredding
            print('     => Shredding')
            image = cv2.imread(doc)
            h, w, c = image.shape
            acc = 0
            strips = []
            for i in range(args.num_strips):
                dw = int((w - acc) / (args.num_strips - i))
                strip = image[:, acc : acc + dw]
                noise_left = np.random.randint(0, 255, (h, disp_noise)).astype(np.uint8)
                noise_right = np.random.randint(0, 255, (h, disp_noise)).astype(np.uint8)
                for j in range(c): # for each channel
                    strip[:, : disp_noise, j] = cv2.add(strip[:, : disp_noise, j], noise_left)
                    strip[:, -disp_noise :, j] = cv2.add(strip[:, -disp_noise :, j], noise_right)
                strips.append(strip)
                acc += dw

            # positives
            print('     => Positive samples')
            N = len(strips)
            combs = [(i, i + 1) for i in range(N - 1)]
            random.shuffle(combs)
            stop = False
            for i, j in combs:
                print('[{}][{}] :: total={}'.format(i, j, count['positives']))
                image = np.hstack([strips[i][:, -size_left :], strips[j][:, : size_right]])
                for y in range(0, h - input_size, args.stride):
                    crop = image[y : y + input_size]
                    if (crop != 255).sum() / crop.size >= pcont:
                        count['positives'] += 1
                        max_per_doc += 1
                        cv2.imwrite('{}/positives/{}/{}.jpg'.format(args.savedir, mode, count['positives']), crop)
                        if max_per_doc == args.max_pos:
                            stop = True
                            break
                if stop:
                    break


            print('     => Negative samples')
            # negatives
            combs = [(i, j) for i in range(N) for j in range(N) if (i != j) and (i + 1 != j)]
            random.shuffle(combs)
            stop = False
            for i, j in combs:
                print('[{}][{}] :: total={}'.format(i, j, count['negatives']))
                image = np.hstack([strips[i][:, -size_left :], strips[j][:, : size_right]])
                for y in range(0, h - input_size, args.stride):
                    crop = image[y : y + input_size]
                    if (crop != 255).sum() / crop.size >= pcont:
                        count['negatives'] += 1
                        cv2.imwrite('{}/negatives/{}/{}.jpg'.format(args.savedir, mode, count['negatives']), crop)
                        if count['negatives'] >= int(args.rneg * count['positives']):
                            stop = True
                            break
                if stop:
                    break
'''
            print('     => Neutral samples')
            combs = [(i, j) for i in range(N) for j in range(N) if i != j]
            random.shuffle(combs)
            stop = False
            for i, j in combs:
                print('[{}][{}] :: total={}'.format(i, j, count['neutral']))
                image = np.hstack([strips[i][:, -size_left :], strips[j][:, : size_right]])
                for y in range(0, h - input_size, args.stride):
                    crop = image[y : y + input_size]
                    if (crop != 255).sum() / crop.size < pcont:
                        count['neutral'] += 1
                        cv2.imwrite('{}/neutral/{}/{}.jpg'.format(args.savedir, mode, count['neutral']), crop)
                        if count['neutral'] >= int(args.rneu * count['positives']):
                            stop = True
                            break
                if stop:
                    break

            print('Total: neg {} pos {} neu {}'.format(count['negatives'], count['positives'], count['neutral']))
'''
def generate_files(args):
    ''' Generate train.txt and val.txt. '''

    docs_neg_train = glob.glob('{}/negatives/train/*.jpg'.format(args.savedir))
    docs_neg_val = glob.glob('{}/negatives/val/*.jpg'.format(args.savedir))
    docs_pos_train = glob.glob('{}/positives/train/*.jpg'.format(args.savedir))
    docs_pos_val = glob.glob('{}/positives/val/*.jpg'.format(args.savedir))
    #docs_neu_train = glob.glob('{}/neutral/train/*.jpg'.format(args.savedir))
    #docs_neu_val = glob.glob('{}/neutral/val*.jpg'.format(args.savedir))

    neg_train = ['{} 0'.format(doc) for doc in docs_neg_train]
    pos_train = ['{} 1'.format(doc) for doc in docs_pos_train]
    #neu_train = ['{} 2'.format(doc) for doc in docs_neu_train] # <= neutral as negative
    neg_val = ['{} 0'.format(doc) for doc in docs_neg_val]
    pos_val = ['{} 1'.format(doc) for doc in docs_pos_val]
    #neu_val = ['{} 2'.format(doc) for doc in docs_neu_val] # <= neutral as negative

    train = neg_train + pos_train# + neu_train
    val = neg_val + pos_val# + neu_val
    random.shuffle(train)
    random.shuffle(val)

    # select 10%
    #train = train[: int(0.1 * len(train))]
    #val = val[: int(0.1 * len(val))]
    # save
    open('{}/train.txt'.format(args.savedir), 'w').write('\n'.join(train))
    open('{}/val.txt'.format(args.savedir), 'w').write('\n'.join(val))


def main():

    random.seed(0)
    parser = argparse.ArgumentParser(
        description='Dataset patches assembling.'
    )
    parser.add_argument(
        '-v', '--pval', action='store', dest='pval', required=False, type=float,
        default=0.1, help='Percentage of samples reserved for validation.'
    )
    parser.add_argument(
        '-m', '--mpos', action='store', dest='max_pos', required=False, type=int,
        default=1000, help='Max. positives per document.'
    )
    parser.add_argument(
        '-rneg', '--rneg', action='store', dest='rneg', required=False, type=float,
        default=1.0, help='Ratio between number of negative samples and positives.'
    )
    parser.add_argument(
        '-rneu', '--rneu', action='store', dest='rneu', required=False, type=float,
        default=1.0, help='Ratio between number of neutral samples and positives.'
    )
    parser.add_argument(
        '-d', '--ndocs', action='store', dest='num_docs', required=False, type=int,
        default=None, help='Number of documents.'
    )
    parser.add_argument(
        '-ns', '--nstrips', action='store', dest='num_strips', required=False, type=int,
        default=30, help='Number of strips (simulated shredding).'
    )
    parser.add_argument(
        '-sd', '--sdir', action='store', dest='savedir', required=False, type=str,
        default='datasets/patches', help='Path where samples will be placed.'
    )
    parser.add_argument(
        '-st', '--str', action='store', dest='stride', required=False, type=int,
        default=2, help='Stride.'
    )
    args = parser.parse_args()

    init(args)
    print('Extracting characters')
    generate_samples(args)
    print('Generate files.')
    generate_files(args)


if __name__ == '__main__':
    t0 = time.time()
    main()
    t1 = time.time()
    print('Elapsed time={:.2f} minutes ({} seconds)'.format((t1 - t0) / 60.0, t1 - t0))

