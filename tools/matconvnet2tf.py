from __future__ import absolute_import, division, print_function

import h5py as h5
import numpy as np
import tensorflow as tf


def read_mat_dset(neth5, dataset_name):
    refs = [key for key in neth5[dataset_name]]
    return [np.array(neth5[ref[0]]) for ref in refs]


def read_mat_strs(neth5, dataset_name):
    arr_names = read_mat_dset(neth5, dataset_name)
    return [''.join(chr(c) for c in arr_name) for arr_name in arr_names]


def read_mat_group(neth5, dataset_name):
    refs = [key for key in neth5[dataset_name]]
    return [neth5[ref[0]] for ref in refs]


def read_layers(neth5):
    p_names = read_mat_strs(neth5, '/net/params/name/')
    p_values = read_mat_dset(neth5, '/net/params/value/')
    params = dict(zip(p_names, p_values))

    l_names = read_mat_strs(neth5, '/net/layers/name/')
    l_types = read_mat_strs(neth5, '/net/layers/type/')
    l_par_refs = read_mat_dset(neth5, '/net/layers/params/')
    l_par_names = []
    for par_refs in l_par_refs:
        par_names = []
        for par_ref in par_refs:
            if bool(par_ref):
                par_name = ''.join(chr(c) for c in neth5[par_ref[0]])
                par_names.append(par_name)
        l_par_names.append(par_names)
    l_blocks = read_mat_group(neth5, '/net/layers/block/')
    layers = zip(l_names, l_types, l_par_names, l_blocks)

    return layers, params


def build_bn(layer, params, inputs, trainable, reuse):
    par_names = layer[2]
    num_channels = int(layer[3]['numChannels'][0, 0])

    for par_name in par_names:
        if par_name[-1] == 'f':
            gamma = np.array(params[par_name])
            gamma_initializer = tf.constant_initializer(gamma, tf.float32)
        elif par_name[-1] == 'b':
            beta = np.array(params[par_name])
            beta_initializer = tf.constant_initializer(beta, tf.float32)
        elif par_name[-1] == 'm':
            moments = np.array(params[par_name])
            moving_mean_initializer = tf.constant_initializer(moments[0], tf.float32)
            moving_variance_initializer = tf.constant_initializer(moments[1], tf.float32)
        else:
            print('Warning, parameter ignored:', par_name)

    return tf.layers.batch_normalization(inputs, axis=-1,
                                         momentum=0.99, epsilon=0.001, center=True, scale=True,
                                         beta_initializer=beta_initializer, gamma_initializer=gamma_initializer,
                                         moving_mean_initializer=moving_mean_initializer,
                                         moving_variance_initializer=moving_variance_initializer,
                                         beta_regularizer=None, gamma_regularizer=None,
                                         training=False, trainable=trainable, name='mcn_%s' % layer[0], reuse=reuse)


def build_conv(layer, params, inputs, trainable, reuse):
    par_names = layer[2]
    use_bias = bool(layer[3]['hasBias'][0, 0])
    kernel_size = [int(num) for num in layer[3]['size'][:, 0]][:2]
    filters = [int(num) for num in layer[3]['size'][:, 0]][3]
    assert kernel_size[0] == kernel_size[1]
    pad = [int(num) for num in layer[3]['pad'][:, 0]]
    strides = [int(num) for num in layer[3]['stride'][:, 0]]

    print(kernel_size, pad, strides)
    kernel_initializer = bias_initializer = None
    for par_name in par_names:
        if par_name[-1] == 'f':
            matrix = np.array(params[par_name])
            if len(np.shape(matrix)) == 3:
                matrix = np.expand_dims(matrix, axis=0)
            matrix = np.transpose(matrix, [3, 2, 1, 0])
            kernel_initializer = tf.constant_initializer(matrix, tf.float32)
        elif par_name[-1] == 'b':
            bias = np.array(params[par_name])
            bias_initializer = tf.constant_initializer(bias, tf.float32)
        else:
            print('Warning, parameter ignored:', par_name)

    inputs = tf.pad(inputs, [[0, 0], [pad[0], pad[1]], [pad[2], pad[3]], [0, 0]], mode='CONSTANT')
    return tf.layers.conv2d(inputs, filters, kernel_size,
                            strides=strides, padding='valid', use_bias=use_bias,
                            kernel_initializer=kernel_initializer,
                            bias_initializer=bias_initializer,
                            trainable=trainable, name='mcn_%s' % layer[0], reuse=reuse)


def build_conv_transpose(layer, params, inputs, trainable, reuse):
    par_names = layer[2]
    use_bias = bool(layer[3]['hasBias'][0, 0])
    kernel_size = [int(num) for num in layer[3]['size'][:, 0]][:2]
    filters = [int(num) for num in layer[3]['size'][:, 0]][3]
    crop = [int(num) for num in layer[3]['crop'][:, 0]]
    strides = [int(num) for num in layer[3]['upsample'][:, 0]]
    num_groups = int(layer[3]['numGroups'][0, 0])
    assert num_groups == 1

    print(kernel_size, num_groups, crop, strides)
    kernel_initializer = bias_initializer = None
    for par_name in par_names:
        if par_name[-1] == 'f':
            matrix = np.transpose(np.array(params[par_name]), [3, 2, 1, 0])
            kernel_initializer = tf.constant_initializer(matrix, tf.float32)
        elif par_name[-1] == 'b':
            bias = np.array(params[par_name])
            bias_initializer = tf.constant_initializer(bias, tf.float32)
        else:
            print('Warning, parameter ignored:', par_name)

    inputs = tf.layers.conv2d_transpose(inputs, filters, kernel_size,
                                        strides=strides, padding='valid', use_bias=use_bias,
                                        kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                                        trainable=trainable, name='mcn_%s' % layer[0], reuse=reuse)
    return inputs[:, crop[0]:-crop[1], crop[2]:-crop[3], :]


def build_relu(layer, params, inputs):
    return tf.nn.relu(inputs, name='mcn_%s' % layer[0])


def build_pooling(layer, params, inputs):
    method = ''.join(chr(c) for c in layer[3]['method'])
    if method == 'max':
        pooling = tf.layers.max_pooling2d
    elif method == 'mean':
        pooling = tf.layers.average_pooling2d
    else:
        raise Exception('unknown pooling algorithm')
    pool_size = [int(num) for num in layer[3]['poolSize'][:, 0]]
    assert pool_size[0] == pool_size[1]
    pad = [int(num) for num in layer[3]['pad'][:, 0]]
    strides = [int(num) for num in layer[3]['stride'][:, 0]]

    print(pad, strides)
    inputs = tf.pad(inputs, [[0, 0], [pad[0], pad[0]], [pad[1], pad[1]], [0, 0]], mode='CONSTANT')
    return pooling(inputs, pool_size, strides, padding='valid', name='mcn_%s' % layer[0])


def build_dropout(layer, params, inputs):
    frozen = bool(layer[3]['frozen'][0, 0])
    rate = float(layer[3]['rate'][0, 0])
    return tf.layers.dropout(inputs, rate=rate, noise_shape=None, training=False, name='mcn_%s' % layer[0])


def build_make_symmetric(layer, params, inputs):
    return (inputs + tf.transpose(inputs, [0, 2, 1, 3])) / 2


def import_net(netfile, inputs, trainable=False, reuse=None, verbose=False):
    with tf.variable_scope("imported_model", reuse=reuse):
        if verbose:
            print('importing net...')
        neth5 = h5.File(netfile)
        layers, params = read_layers(neth5)
        for layer in layers:
            if verbose:
                print('layer:', layer[0], 'shape:', inputs.get_shape())
            if layer[1] == 'dagnn.BatchNorm':
                outputs = build_bn(layer, params, inputs, trainable, reuse)
            elif layer[1] == 'dagnn.Conv':
                outputs = build_conv(layer, params, inputs, trainable, reuse)
            elif layer[1] == 'dagnn.ConvTranspose':
                outputs = build_conv_transpose(layer, params, inputs, trainable, reuse)
            elif layer[1] == 'dagnn.ReLU':
                outputs = build_relu(layer, params, inputs)
            elif layer[1] == 'dagnn.Pooling':
                outputs = build_pooling(layer, params, inputs)
            elif layer[1] == 'dagnn.DropOut':
                outputs = build_dropout(layer, params, inputs)
            elif layer[1] == 'dagnn.MakeSymmetric':
                outputs = build_make_symmetric(layer, params, inputs)
            elif layer[1] == 'dagnn.LossMatrixRegressionSym':
                pass
            else:
                raise Exception('unknown layer type: %s' % layer[1])
            inputs = outputs
        return outputs

if __name__ == '__main__':
    import_net('../data/net-epoch-1486-v73.mat', tf.zeros([10,14,14,1]), False, None, True)
