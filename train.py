
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import math
import cv2
import numpy as np
import tensorflow as tf
from docrec.neural.models.squeezenet import squeezenet, load, save
import time
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


'''
epochs 5
python train.py -e 5 -bs 128 -lr 0.0001 -s 0.33 -d datasets/patches
'''
tf.logging.set_verbosity(tf.logging.INFO)

NUM_CLASSES = 2
# NUM_RUNS = 10
# Elapsed time=173.50 minutes (10409.99261974 seconds)

# https://github.com/aymericdamien/TensorFlow-Examples
# https://cs230-stanford.github.io/tensorflow-input-data.html
# https://stackoverflow.com/questions/46772685/how-to-accumulate-gradients-in-tensorflow?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa


def get_dataset_info(args, mode='train', max_size=None):
    ''' Returns: filenames and labels. '''

    assert mode in ['train', 'val']

    txt_file = '{}/{}.txt'.format(args.datasetdir, mode)
    lines = open(txt_file).readlines()
    if max_size is not None:
        lines = lines [ : int(max_size * len(lines))]
    filenames = []
    labels = []
    for line in lines:
        filename, label = line.split()
        filenames.append(filename)
        labels.append(int(label))
    return filenames, labels


def input_fn(args, mode='train', num_channels=3):
    ''' Dataset load function.'''

    def _parse_function(filename, label):
        ''' Parse function. '''

        image_string = tf.read_file(filename)
        image = tf.image.decode_jpeg(image_string, channels=num_channels, dct_method='INTEGER_ACCURATE') # works with png too
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.transpose(image, perm=[2, 0, 1]) # channels first
        if mode == 'train':
            return image, tf.one_hot(label, NUM_CLASSES)
        return image, label

    assert mode in ['train', 'val']

    filenames, labels = get_dataset_info(args, mode)

    # TF pipeline
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    if mode == 'train':
        dataset = dataset.shuffle(len(filenames))
    dataset = dataset.map(_parse_function, num_parallel_calls=4)
    dataset = dataset.batch(args.batch_size).repeat(args.num_epochs)
    return dataset.make_one_shot_iterator().get_next()


def train(args):

    tf.set_random_seed(0) # <= change this in case of multiple runs

    ''' Training stage. '''
    filenames, _ = get_dataset_info(args, mode='train')
    num_samples = len(filenames)

    # network input size
    input_size = cv2.imread(filenames[0]).shape
    H, W, C = input_size

    # general variables and ops
    global_step_var = tf.Variable(1, trainable=False, name='global_step')
    inc_global_step_op = global_step_var.assign_add(1)

    # placeholders
    images_ph = tf.placeholder(tf.float32, name='images_ph', shape=(None, C, H, W))     # channels first
    labels_ph = tf.placeholder(tf.float32, name='labels_ph', shape=(None, NUM_CLASSES)) # one-hot enconding

    # dataset iterator
    next_batch_op = input_fn(args, 'train', num_channels=C)

    # model
    logits_op = squeezenet(images_ph, 'train', NUM_CLASSES, channels_first=True, seed=None)
    # loss function
    loss_op = tf.losses.softmax_cross_entropy(onehot_labels=labels_ph, logits=logits_op)

    # learning rate definition
    num_steps_per_epoch = math.ceil(num_samples / args.batch_size)
    total_steps = args.num_epochs * num_steps_per_epoch
    decay_steps = math.ceil(args.step_size * total_steps)
    learning_rate_op = tf.train.exponential_decay(
        args.learning_rate, global_step_var, decay_steps, 0.1, staircase=True
    )

    # optimizer (adam method)
    optimizer = tf.train.AdamOptimizer(learning_rate_op)
    # training step operation
    train_op = optimizer.minimize(loss_op)

    # summary
    #tf.summary.scalar('loss', loss_op)
    #tf.summary.scalar('learning_rate', learning_rate_op)
    #summary_op = tf.summary.merge_all()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        #writer = tf.summary.FileWriter('traindata/tensorboard', sess.graph)
        # (re)load weights/biases of the squeezet pretrained on imagenet
        load(
            'docrec/neural/models/imagenet.npy', sess, model_scope='SqueezeNet', first_layer='conv1',
            ignore_layers=['conv10'], BGR=True, ignore_missing=False
        ) # <= comment this part to train from scratch
        # training loop
        start = time.time()
        losses_group = []
        losses = []
        steps = []
        for epoch in range(1, args.num_epochs + 1):
            for step in range(1, num_steps_per_epoch + 1):
                # global step
                global_step = sess.run(global_step_var)
                # batch data
                images, labels = sess.run(next_batch_op)
                # train
                learning_rate, loss, x = sess.run(
                    [learning_rate_op, loss_op, train_op],
                    feed_dict={images_ph: images, labels_ph: labels}
                )
                #learning_rate, summary, loss, x = sess.run(
                #    [learning_rate_op, summary_op, loss_op, train_op],
                #    feed_dict={images_ph: images, labels_ph: labels}
                #)
                # update summary
                #writer.add_summary(summary, global_step)
                # show training status
                losses_group.append(loss)
                if (step % 10 == 0) or (step == num_steps_per_epoch):
                    losses.append(np.mean(losses_group))
                    steps.append(global_step)
                    elapsed = time.time() - start
                    remaining = elapsed * (total_steps - global_step) / global_step
                    print('[{:.2f}%] step={}/{} epoch={} loss={:.3f} :: {:.2f}/{:.2f} seconds lr={}'.format(
                        100 * global_step / total_steps, global_step, total_steps, epoch,
                        np.mean(losses_group), elapsed, remaining, learning_rate
                    ))
                    losses_group = []

                # increment global step
                sess.run(inc_global_step_op)

            # save epoch model
            save('traindata/model/{}.npy'.format(epoch), sess)
        plt.plot(steps, losses)
        plt.savefig('traindata/loss.png')


def validate(args):
    ''' Validate and select the best model. '''

    filenames, _ = get_dataset_info(args, mode='val')
    num_samples = len(filenames)

    # network input size
    input_size = cv2.imread(filenames[0]).shape
    H, W, C = input_size

    # placeholders
    images_ph = tf.placeholder(tf.float32, name='images_ph', shape=(None, C, H, W)) # channels first
    labels_ph = tf.placeholder(tf.float32, name='labels_ph', shape=(None,))         # normal enconding

    # dataset iterator
    next_batch_op = input_fn(args, 'val', num_channels=C)
    # model
    logits_op = squeezenet(images_ph, 'val', NUM_CLASSES, channels_first=True)
    # predictions
    predictions_op = tf.argmax(logits_op, 1)

    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        num_steps_per_epoch = math.ceil(num_samples / args.batch_size)
        best_epoch = 0
        best_accuracy = 0.0
        accuracies = []
        for epoch in range(1, args.num_epochs + 1):
		    # load epoch model
            load('traindata/model/{}.npy'.format(epoch), sess, model_scope='SqueezeNet')
            total_correct = 0
            for step in range(1, num_steps_per_epoch + 1):
                images, labels = sess.run(next_batch_op)
                batch_size = images.shape[0]
                logits, predictions = sess.run(
                    [logits_op, predictions_op],
                    feed_dict={images_ph: images, labels_ph: labels}
                )
                num_correct = np.sum(predictions==labels)
                total_correct += num_correct
                if (step % 10 == 0) or (step == num_steps_per_epoch):
                    print('step={} accuracy={:.2f}'.format(step, 100 * num_correct / batch_size))
            # epoch average accuracy
            accuracy = 100.0 * total_correct / num_samples
            accuracies.append(accuracy)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_epoch = epoch
            print('-------------------------------------------------')
            print('epoch={} (best={}) accuracy={:.2f} (best={:.2f})'.format(epoch, best_epoch, accuracy, best_accuracy))
            print('-------------------------------------------------')

        open('best_model.txt', 'w').write('traindata/model/{}.npy'.format(best_epoch))
        print('best_epoch={} accuracy={:.2f}'.format(best_epoch, best_accuracy))
    plt.clf()
    plt.plot(list(range(1, args.num_epochs + 1)), accuracies)
    plt.savefig('traindata/accuracies.png')


def main():
    parser = argparse.ArgumentParser(description='Training the network.')
    parser.add_argument(
        '-lr', '--learning-rate', action='store', dest='learning_rate', required=False, type=float,
        default=0.0001, help='Learning rate.'
    )
    parser.add_argument(
        '-bs', '--batch-size', action='store', dest='batch_size', required=False, type=int,
        default=1, help='Batch size.'
    )
    parser.add_argument(
        '-e', '--epochs', action='store', dest='num_epochs', required=False, type=int,
        default=5, help='Number of training epochs.'
    )
    parser.add_argument(
        '-s', '--step-size', action='store', dest='step_size', required=False, type=float,
        default=0.33, help='Step size for learning with step-down policy.'
    )
    parser.add_argument(
        '-d', '--ddir', action='store', dest='datasetdir', required=False, type=str,
        default='datasets/patches', help='Path where samples will be placed.'
    )
    args = parser.parse_args()

    # training stage
    train(args)

    # validation
    validate(args)


if __name__ == '__main__':
    t0 = time.time()
    main()
    t1 = time.time()
    print('Elapsed time={:.2f} minutes ({} seconds)'.format((t1 - t0) / 60.0, t1 - t0))
