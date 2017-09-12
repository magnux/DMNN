from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import layers as tfl
import tensorflow.contrib.layers as tcl

__all__ = ['resnet', 'resnext']

_weights_initializer = tcl.variance_scaling_initializer()
_weights_regularizer = tcl.l2_regularizer(5e-4)


def _conv3d(inputs, num_outputs, kernel_size, stride):
    return tfl.conv3d(inputs,
                      num_outputs,
                      kernel_size,
                      stride,
                      padding='same',
                      use_bias=False,
                      kernel_initializer=_weights_initializer,
                      kernel_regularizer=_weights_regularizer)


def _conv2d(inputs, num_outputs, kernel_size, stride):
    return tfl.conv2d(inputs,
                      num_outputs,
                      kernel_size,
                      stride,
                      padding='same',
                      use_bias=False,
                      kernel_initializer=_weights_initializer,
                      kernel_regularizer=_weights_regularizer)


def _batch_norm(inputs, is_training):
    return tcl.batch_norm(inputs,
                          center=True,
                          scale=False,
                          decay=0.9,
                          activation_fn=tf.nn.relu,
                          updates_collections=None,
                          fused=False,
                          is_training=is_training)


def _preact_conv(inputs, num_outputs, kernel_size, stride, is_training,
                 scope=None, groups=None, conv_fn=_conv3d):
    with tf.variable_scope(scope, 'conv', [inputs]):
        out = _batch_norm(inputs, is_training)
        if groups is None:
            out = conv_fn(out, num_outputs, kernel_size, stride)
        else:
            num_outputs = num_outputs // groups
            split_axis = len(out.get_shape()) - 1
            outs = [conv_fn(s, num_outputs, kernel_size, stride) for s in tf.split(out, groups, split_axis)]
            out = tf.concat(outs, -1)
    return out


def _resnet_block(inputs, num_in, num_out, kernel_size, stride, is_training,
                  scope=None, conv_fn=_conv3d):
    with tf.variable_scope(scope, 'res', [inputs]):
        if num_in != num_out or stride > 1:
            shortcut = _preact_conv(
                inputs, num_out, 1, stride, is_training, 'shortcut',
                conv_fn=conv_fn)
        else:
            shortcut = inputs
        
        out = _preact_conv(
            inputs, num_out, kernel_size, stride, is_training, 'conv1', conv_fn=conv_fn)
        out = _preact_conv(
            out, num_out, kernel_size, 1, is_training, 'conv2', conv_fn=conv_fn)
        
        return out + shortcut


def _resnext_block(inputs, num_in, bottleneck, cardinality, num_out, kernel_size,
                   stride, is_training, scope=None, conv_fn=_conv3d):
    with tf.variable_scope(scope, 'res', [inputs]):
        if num_in != num_out or stride > 1:
            shortcut = _preact_conv(
                inputs, num_out, 1, stride, is_training, 'shortcut',
                conv_fn=conv_fn)
        else:
            shortcut = inputs
        
        out = _preact_conv(
            inputs, bottleneck, 1, 1, is_training, 'conv1', conv_fn=conv_fn)
        out = _preact_conv(
            out, bottleneck, kernel_size, stride, is_training, 'conv2',
            groups=cardinality, conv_fn=conv_fn)
        out = _preact_conv(
            out, num_out, 1, 1, is_training, 'conv3', conv_fn=conv_fn)
        
        return out + shortcut


def resnet(inputs, n_res, config, is_training, mode='3D'):
    mode = mode.upper()
    if mode == '3D':
        conv_fn = _conv3d
        pool_fn = lambda inputs: tf.reduce_mean(inputs, [1, 2, 3])
    elif mode == '2D':
        conv_fn = _conv2d
        pool_fn = lambda inputs: tf.reduce_mean(inputs, [1, 2])
    else:
        raise ValueError('mode must be on of "3D" or "2D"')
    
    with tf.variable_scope('block0', values=[inputs]):
        out = conv_fn(inputs, config[0]['size'], 3, config[0]['stride'])
    
    for b in range(1, len(config)):
        with tf.variable_scope('block%d' % b, values=[out]):
            for r in range(n_res):
                num_in = config[b - 1]['size'] if r == 0 else config[b]['size']
                num_out = config[b]['size']
                stride = config[b]['stride'] if r == 0 else 1
                out = _resnet_block(
                    out, num_in, num_out, 3, stride, is_training, 'res%d' % r,
                    conv_fn)
    
    with tf.variable_scope('pooling', values=[out]):
        out = pool_fn(out)
        out = _batch_norm(out, is_training)
    return out


def resnext(inputs, n_res, config, is_training, mode='3D'):
    mode = mode.upper()
    if mode == '3D':
        conv_fn = _conv3d
        pool_fn = lambda inputs: tf.reduce_mean(inputs, [1, 2, 3])
    elif mode == '2D':
        conv_fn = _conv2d
        pool_fn = lambda inputs: tf.reduce_mean(inputs, [1, 2])
    else:
        raise ValueError('mode must be on of "3D" or "2D"')
    
    with tf.variable_scope('block0', values=[inputs]):
        out = conv_fn(inputs, config[0]['size'], 3, config[0]['stride'])
    
    for b in range(1, len(config)):
        with tf.variable_scope('block%d' % b, values=[out]):
            for r in range(n_res):
                num_in = config[b - 1]['size'] if r == 0 else config[b]['size']
                bottleneck = config[b]['bottleneck']
                cardinality = config[b]['cardinality']
                num_out = config[b]['size']
                stride = config[b]['stride'] if r == 0 else 1
                out = _resnext_block(
                    out, num_in, bottleneck, cardinality, num_out, 3, stride,
                    is_training, 'res%d' % r, conv_fn)
    
    with tf.variable_scope('pooling', values=[out]):
        out = pool_fn(out)
        out = _batch_norm(out, is_training)
    return out
