
import os
import numpy as np
import tensorflow as tf


# https://github.com/vonclites/squeezenet/blob/master/networks/squeezenet.py
# https://github.com/ethereon/caffe-tensorflow
# python convert.py --caffemodel <path-caffemodel> <path-deploy_prototxt> --data-output-path <path_weights> <path_model>py>

def load(data_path, session, model_scope='SqueezeNet', first_layer='conv1', ignore_layers=[], BGR=False, ignore_missing=False):
    ''' Load network weights and biases (format caffe-tensorflow).

        data_path: path to the numpy-serialized network weights.
        session: current TensorFlow session.
        first_layer: model fisrt layer will be changed in case of BGR data.
        ignore_layers: layers whose parameters must be ignored.
        BGR: if data is BGR, convert weights from the first layer to RGB.
        ignore_missing: if true, serialized weights for missing layers are ignored.
    '''

    data_dict = np.load(data_path, encoding='latin1').item()
    for layer in data_dict:
        if layer in ignore_layers:
            continue
        for param_name, data in data_dict[layer].items():
            param_name = param_name.replace('weights', 'kernel').replace('biases', 'bias')
            try:
                scope = '{}/{}'.format(model_scope, layer) if model_scope else layer
                with tf.variable_scope(scope, reuse=True):
                    var = tf.get_variable(param_name)
                    if (layer == first_layer) and BGR and (param_name == 'kernel'):
                        data = data[:, :, [2, 1, 0], :] # BGR => RGB
                    session.run(var.assign(data))
            except ValueError:
                if not ignore_missing:
                    raise


def save(data_path, session, ignore_layers=[]):
    ''' Load network weights and biases (format caffe-tensorflow).

        data_path: path to the numpy-serialized network weights.
        session: current TensorFlow session.
        ignore_layers: layers whose parameters must be ignored.
    '''

    data_dict = {}
    for var in tf.trainable_variables():
        layer, param_name = var.op.name.split('/')[-2 :] # excluce scope if existing
        if layer in ignore_layers:
            continue
        data = session.run(var)
        try:
            data_dict[layer][param_name] = data
        except KeyError:
            data_dict[layer]= {param_name: data}

    # ckeck directory path
    if not os.path.exists(os.path.dirname(data_path)):
        os.makedirs(os.path.dirname(data_path))
    np.save(data_path, np.array(data_dict))


def squeezenet(input_layer, mode='train', num_classes=1000, channels_first=False, seed=None):

    ''' SqueezeNet v1.1

    Adpated from the Caffe original implementation: https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1

    Reference:
    @article{SqueezeNet,
    Author = {Forrest N. Iandola and Song Han and Matthew W. Moskewicz and Khalid Ashraf and William J. Dally and Kurt Keutzer},
    Title = {SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and $<$0.5MB model size},
    Journal = {arXiv:1602.07360},
    Year = {2016}
    '''

    assert mode in ['train', 'val', 'test', 'view']

    data_format = 'channels_first' if channels_first else 'channels_last'
    concat_axis = 1 if channels_first else 3
    with tf.variable_scope('SqueezeNet', reuse=tf.AUTO_REUSE):
        conv1 = tf.layers.conv2d(input_layer, 64, 3, 2, padding='valid', activation=tf.nn.relu, data_format=data_format, name='conv1')
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=3, strides=2, data_format=data_format)
        fire2_squeeze1x1 = tf.layers.conv2d(pool1, 16, 1, activation=tf.nn.relu, data_format=data_format, name='fire2_squeeze1x1')
        fire2_expand1x1 = tf.layers.conv2d(fire2_squeeze1x1, 64, 1, activation=tf.nn.relu, data_format=data_format, name='fire2_expand1x1')
        fire2_expand3x3 = tf.layers.conv2d(fire2_squeeze1x1, 64, 3, padding='same', activation=tf.nn.relu, data_format=data_format, name='fire2_expand3x3')
        fire2_concat = tf.concat([fire2_expand1x1, fire2_expand3x3], axis=concat_axis)
        fire3_squeeze1x1 = tf.layers.conv2d(fire2_concat, 16, 1, activation=tf.nn.relu, data_format=data_format, name='fire3_squeeze1x1')
        fire3_expand1x1 = tf.layers.conv2d(fire3_squeeze1x1, 64, 1, activation=tf.nn.relu, data_format=data_format, name='fire3_expand1x1')
        fire3_expand3x3 = tf.layers.conv2d(fire3_squeeze1x1, 64, 3, padding='same', activation=tf.nn.relu, data_format=data_format, name='fire3_expand3x3')
        fire3_concat = tf.concat([fire3_expand1x1, fire3_expand3x3], axis=concat_axis)
        pool3 = tf.layers.max_pooling2d(fire3_concat, 3, 2, data_format=data_format)
        fire4_squeeze1x1 = tf.layers.conv2d(pool3, 32, 1, activation=tf.nn.relu, data_format=data_format, name='fire4_squeeze1x1')
        fire4_expand1x1 = tf.layers.conv2d(fire4_squeeze1x1, 128, 1, activation=tf.nn.relu, data_format=data_format, name='fire4_expand1x1')
        fire4_expand3x3 = tf.layers.conv2d(fire4_squeeze1x1, 128, 3, padding='same', activation=tf.nn.relu, data_format=data_format, name='fire4_expand3x3')
        fire4_concat = tf.concat([fire4_expand1x1, fire4_expand3x3], axis=concat_axis)
        fire5_squeeze1x1 = tf.layers.conv2d(fire4_concat, 32, 1, activation=tf.nn.relu, data_format=data_format, name='fire5_squeeze1x1')
        fire5_expand1x1 = tf.layers.conv2d(fire5_squeeze1x1, 128, 1, activation=tf.nn.relu, data_format=data_format, name='fire5_expand1x1')
        fire5_expand3x3 = tf.layers.conv2d(fire5_squeeze1x1, 128, 3, padding='same', activation=tf.nn.relu, data_format=data_format, name='fire5_expand3x3')
        fire5_concat = tf.concat([fire5_expand1x1, fire5_expand3x3], axis=concat_axis)
        pool5 = tf.layers.max_pooling2d(fire5_concat, 3, 2, data_format=data_format)
        fire6_squeeze1x1 = tf.layers.conv2d(pool5, 48, 1, activation=tf.nn.relu, data_format=data_format, name='fire6_squeeze1x1')
        fire6_expand1x1 = tf.layers.conv2d(fire6_squeeze1x1, 192, 1, activation=tf.nn.relu, data_format=data_format, name='fire6_expand1x1')
        fire6_expand3x3 = tf.layers.conv2d(fire6_squeeze1x1, 192, 3, padding='same', activation=tf.nn.relu, data_format=data_format, name='fire6_expand3x3')
        fire6_concat = tf.concat([fire6_expand1x1, fire6_expand3x3], axis=concat_axis)
        fire7_squeeze1x1 = tf.layers.conv2d(fire6_concat, 48, 1, activation=tf.nn.relu, data_format=data_format, name='fire7_squeeze1x1')
        fire7_expand1x1 = tf.layers.conv2d(fire7_squeeze1x1, 192, 1, activation=tf.nn.relu, data_format=data_format, name='fire7_expand1x1')
        fire7_expand3x3 = tf.layers.conv2d(fire7_squeeze1x1, 192, 3, padding='same', activation=tf.nn.relu, data_format=data_format, name='fire7_expand3x3')
        fire7_concat = tf.concat([fire7_expand1x1, fire7_expand3x3], axis=concat_axis)
        fire8_squeeze1x1 = tf.layers.conv2d(fire7_concat, 64, 1, activation=tf.nn.relu, data_format=data_format, name='fire8_squeeze1x1')
        fire8_expand1x1 = tf.layers.conv2d(fire8_squeeze1x1, 256, 1, activation=tf.nn.relu, data_format=data_format, name= 'fire8_expand1x1')
        fire8_expand3x3 = tf.layers.conv2d(fire8_squeeze1x1, 256, 3, padding='same', activation=tf.nn.relu, data_format=data_format, name='fire8_expand3x3')
        fire8_concat = tf.concat([fire8_expand1x1, fire8_expand3x3], axis=concat_axis)
        fire9_squeeze1x1 = tf.layers.conv2d(fire8_concat, 64, 1, activation=tf.nn.relu, data_format=data_format, name='fire9_squeeze1x1')
        fire9_expand1x1 = tf.layers.conv2d(fire9_squeeze1x1, 256, 1, activation=tf.nn.relu, data_format=data_format, name='fire9_expand1x1')
        fire9_expand3x3 = tf.layers.conv2d(fire9_squeeze1x1, 256, 3, padding='same', activation=tf.nn.relu, data_format=data_format, name='fire9_expand3x3')
        fire9_concat = tf.concat([fire9_expand1x1, fire9_expand3x3], axis=concat_axis)
        drop9 = tf.layers.dropout(fire9_concat, 0.5, training=(mode=='train'), seed=seed)

        conv10 = tf.layers.conv2d(
            drop9, num_classes, 1, kernel_initializer=tf.random_normal_initializer(0.0, 0.01),
            activation=tf.nn.relu, data_format=data_format, name='conv10'
        ) # discarded in case of finetuning with less than 1000 classes
        #pool10 = tf.layers.average_pooling2d(conv10, pool10_window[(H, W)], 1, padding= 'valid')
        #logits = tf.squeeze(pool10, [1, 2], name='logits')
        axes = [2, 3] if channels_first else [1, 2]
        logits = tf.reduce_mean(conv10, axes, keepdims=False, name='pool10')

    if mode == 'test':
        return logits, conv10

    return logits