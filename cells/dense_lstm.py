from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import random_ops
from tensorflow.contrib.rnn import RNNCell
from layers.ops import *
from layers.base import *

class LSTMCell(RNNCell):
    '''Vanilla LSTM implemented with same initializations as BN-LSTM'''
    def __init__(self, num_units):
        self.num_units = num_units

    @property
    def state_size(self):
        return (self.num_units, self.num_units)

    @property
    def output_size(self):
        return self.num_units

    def __call__(self, x, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            c, h = state

            # Keep W_xh and W_hh separate here as well to reuse initialization methods
            x_size = x.get_shape().as_list()[1]
            W_xh = tf.get_variable('W_xh',
                [x_size, 4 * self.num_units],
                initializer=orthogonal_initializer())
            W_hh = tf.get_variable('W_hh',
                [self.num_units, 4 * self.num_units],
                initializer=bn_lstm_identity_initializer(0.95))
            bias = tf.get_variable('bias', [4 * self.num_units])

            # hidden = tf.matmul(x, W_xh) + tf.matmul(h, W_hh) + bias
            # improve speed by concat.
            concat = tf.concat([x, h], axis=1)
            W_both = tf.concat([W_xh, W_hh], axis=0)
            hidden = tf.matmul(concat, W_both) + bias

            i, j, f, o = tf.split(hidden, num_or_size_splits=4, axis=1)

            new_c = c * tf.sigmoid(f) + tf.sigmoid(i) * tf.tanh(j)
            new_h = tf.tanh(new_c) * tf.sigmoid(o)

            return new_h, (new_c, new_h)

class BNLSTMCell(RNNCell):
    '''Batch normalized LSTM as described in arxiv.org/abs/1603.09025'''
    def __init__(self, num_units, max_length, is_training):
        self.num_units = num_units
        self.is_training = is_training
        self.max_length = max_length

    @property
    def state_size(self):
        return (self.num_units, self.num_units, 1)

    @property
    def output_size(self):
        return self.num_units

    def __call__(self, x, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            c, h, step = state
            step_int = tf.cast(tf.reshape(step[0],[]), tf.int32)

            x_size = x.get_shape().as_list()[1]
            W_xh = tf.get_variable('W_xh',
                [x_size, 4 * self.num_units],
                initializer=orthogonal_initializer())
            W_hh = tf.get_variable('W_hh',
                [self.num_units, 4 * self.num_units],
                initializer=bn_lstm_identity_initializer(0.95))

            bias = tf.get_variable('bias', [4 * self.num_units])

            xh = tf.matmul(x, W_xh)
            hh = tf.matmul(h, W_hh)

            bn_xh = batch_norm_layer_in_time(xh, self.max_length, step_int, self.is_training, scope='xh')
            bn_hh = batch_norm_layer_in_time(hh, self.max_length, step_int, self.is_training, scope='hh')

            hidden = bn_xh + bn_hh + bias

            i, j, f, o = tf.split(hidden, 4, axis=1)

            new_c = c * tf.sigmoid(f) + tf.sigmoid(i) * tf.tanh(j)
            bn_new_c = batch_norm_layer_in_time(new_c, self.max_length, step_int, self.is_training, scope='new_c')

            new_h = tf.tanh(bn_new_c) * tf.sigmoid(o)

            return new_h, (new_c, new_h, step + 1)

class PhasedLSTMCell(RNNCell):
    '''Phased LSTM'''
    def __init__(self, num_units, is_training, alpha = 1e-3, tau_init = 6, r_on_init = 5e-2, peephole=False):
        self.num_units = num_units
        self.is_training = is_training
        self.alpha = alpha
        self.tau_init = tau_init
        self.r_on_init = r_on_init
        self.peephole = peephole

    @property
    def state_size(self):
        return (self.num_units, self.num_units)

    @property
    def output_size(self):
        return self.num_units*2 # because we are passing the time along with the input

    def __call__(self, x, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            c, h = state
            time = x[:,-self.num_units:]
            x = x[:,:-self.num_units]

            if self.is_training:
                alpha = tf.constant(self.alpha, dtype=tf.float32)
            else:
                alpha = tf.constant(0, dtype=tf.float32)

            # Keep W_xh and W_hh separate here as well to reuse initialization methods
            x_size = x.get_shape().as_list()[1]
            W_xh = tf.get_variable('W_xh',
                [x_size, 4 * self.num_units],
                initializer=orthogonal_initializer()
            )
            W_hh = tf.get_variable('W_hh',
                [self.num_units, 4 * self.num_units],
                initializer=bn_lstm_identity_initializer(0.95)
            )
            if self.peephole:
                W_pi = tf.get_variable('W_pi',
                    [self.num_units],
                    initializer=tf.truncated_normal_initializer(
                        stddev=1e-2, dtype=tf.float32),
                )
                W_pf = tf.get_variable('W_pf',
                    [self.num_units],
                    initializer=tf.truncated_normal_initializer(
                        stddev=1e-2, dtype=tf.float32),
                )
                W_po = tf.get_variable('W_po',
                    [self.num_units],
                    initializer=tf.truncated_normal_initializer(
                        stddev=1e-2, dtype=tf.float32),
                )
            bias = tf.get_variable('bias', [4 * self.num_units])

            tau = tf.get_variable("tau",
                [self.num_units],
                initializer=random_exp_initializer(0, self.tau_init),
                dtype=tf.float32
            )
            s = tf.get_variable("s",
                [self.num_units],
                initializer=init_ops.random_uniform_initializer(0., tau.initialized_value()),
                dtype=tf.float32
            )
            r_on = tf.get_variable("r_on",
                [self.num_units],
                initializer=init_ops.constant_initializer(self.r_on_init),
                dtype=tf.float32,
                trainable=False
            )

            batch_size = x.get_shape().as_list()[0]
            tau = tf.tile(
                tf.expand_dims(tau, 0),
                [batch_size, 1]
            )
            s = tf.tile(
                tf.expand_dims(s, 0),
                [batch_size, 1]
            )
            r_on = tf.tile(
                tf.expand_dims(r_on, 0),
                [batch_size, 1]
            )

            phi = dk_mod(dk_mod((time - s), tau) + tau, tau) / tau

            is_up = tf.less(phi, (r_on * 0.5))
            is_down = tf.logical_and(tf.less(phi, r_on), tf.logical_not(is_up))

            k = tf.where(is_up, 2. * (phi / r_on),
                         tf.where(is_down, 2. - 2. * (phi / r_on), alpha * phi))

            concat = tf.concat([x, h], axis=1)
            W_both = tf.concat([W_xh, W_hh], axis=0)
            hidden = tf.matmul(concat, W_both) + bias

            i, j, f, o = tf.split(hidden, num_or_size_splits=4, axis=1)

            if self.peephole:
                i += W_pi * c
                f += W_pf * c

            new_c = c * tf.sigmoid(f) + tf.sigmoid(i) * tf.tanh(j)

            phased_new_c = k * new_c + (1 - k) * c

            if self.peephole:
                o += W_po * c

            new_h = tf.tanh(new_c) * tf.sigmoid(o)
            phased_new_h = k * new_h + (1 - k) * h

            return tf.concat([phased_new_h,time], axis=1), (phased_new_c, phased_new_h)

def orthogonal(shape):
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    return q.reshape(shape)

def bn_lstm_identity_initializer(scale):
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        '''Ugly cause LSTM params calculated in one matrix multiply'''
        size = shape[0]
        # gate (j) is identity
        t = np.zeros(shape)
        t[:, size:size * 2] = np.identity(size) * scale
        t[:, :size] = orthogonal([size, size])
        t[:, size * 2:size * 3] = orthogonal([size, size])
        t[:, size * 3:] = orthogonal([size, size])
        return tf.constant(t, dtype)

    return _initializer

def orthogonal_initializer():
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        return tf.constant(orthogonal(shape), dtype)
    return _initializer

def random_exp_initializer(minval=0, maxval=None, seed=None,
                               dtype=tf.float32):
    """Returns an initializer that generates tensors with an exponential distribution.
    Args:
      minval: A python scalar or a scalar tensor. Lower bound of the range
        of random values to generate.
      maxval: A python scalar or a scalar tensor. Upper bound of the range
        of random values to generate.  Defaults to 1 for float types.
      seed: A Python integer. Used to create random seeds. See
        [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
        for behavior.
      dtype: The data type.
    Returns:
      An initializer that generates tensors with an exponential distribution.
    """

    def _initializer(shape, dtype=dtype, partition_info=None):
        return tf.exp(random_ops.random_uniform(shape, minval, maxval, dtype, seed=seed))

    return _initializer
