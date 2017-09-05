from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import random_ops
from tensorflow.contrib.rnn import RNNCell
from layers.ops import *
from layers.base import *

class ConvLSTMCell(RNNCell):
    """Conv LSTM recurrent network cell"""

    def __init__(self, in_shape, filter_size, num_in_ch, num_out_ch, max_pool=False, activation=tf.nn.tanh,
                    batch_norm=False, pres_ident=False, is_training=False, max_length=0, keep_prob=1.0, new_pool=False):
        """Initialize the basic Conv LSTM cell.
        Args:
            shape: int tuple thats the height and width of the cell
            filter_size: int tuple thats the height and width of the filter
            num_in_ch: number of input channels to the cell
            num_out_ch: number of output channels to the cell
            max_pool: perform max pooling on output
            activation: Activation function of the inner states.
            batch_norm: Apply batch normalization
            is_training: Training switch
            max_length: Max length of the sequence, required for batch_norm
        """
        self.in_shape = (int(in_shape[0]) , int(in_shape[1]))
        self.filter_size = filter_size
        self.num_in_ch = int(num_in_ch)
        self.num_out_ch = int(num_out_ch)
        self.max_pool = max_pool
        self.activation = activation
        self.num_units =  self.in_shape[0] * self.in_shape[1] * self.num_out_ch
        self.batch_norm = batch_norm
        self.pres_ident = pres_ident
        self.is_training = is_training
        self.max_length = max_length
        self.keep_prob = keep_prob
        self.new_pool = new_pool

    @property
    def out_shape(self):
        out_height = np.ceil(self.in_shape[0] / 2) if self.new_pool else self.in_shape[1] // 2
        out_height = out_height if self.max_pool else self.in_shape[0]
        out_width = np.ceil(self.in_shape[1] / 2) if self.new_pool else self.in_shape[0] // 2
        out_width = out_width if self.max_pool else self.in_shape[1]
        return (out_height, out_width)

    @property
    def state_size(self):
        return (self.num_units, self.num_units, 1)

    @property
    def output_size(self):
        return self.out_shape[0] * self.out_shape[1] * self.num_out_ch

    def shape_in(self, a, is_input=False):
            return tf.reshape(a, [tf.shape(a)[0], self.in_shape[0], self.in_shape[1],
                self.num_in_ch if is_input else self.num_out_ch])

    def shape_out(self, a, pool=False):
            return tf.reshape(a, [tf.shape(a)[0], self.output_size if pool else self.num_units])

    def __call__(self, inputs, state, scope=None):
        """Long short-term memory cell (LSTM)."""
        with tf.variable_scope(scope or type(self).__name__):  # "BasicLSTMCell"
            # Parameters of gates are concatenated into one multiply for efficiency.
            c, h, step = state
            step_int = tf.cast(tf.reshape(step[0],[]), tf.int32)

            c = self.shape_in(c)
            h = self.shape_in(h)
            inputs = self.shape_in(inputs, True)

            if self.keep_prob < 1:
                inputs = tf.nn.dropout(
                    inputs, self.keep_prob,
                    noise_shape=[tf.shape(inputs)[0], self.in_shape[0], self.in_shape[1], 1]
                )

            if self.batch_norm:
                    xh = conv_linear([inputs], self.filter_size, self.num_out_ch * 4, False,
                        scope='xh', initializer=conv_orthogonal_initializer, init_param=None)
                    hh = conv_linear([h], self.filter_size, self.num_out_ch * 4, False,
                        scope='hh', initializer=conv_identity_initializer, init_param=0.95)
                    bn_xh = batch_norm_layer_in_time(xh, self.max_length, step_int, self.is_training, scope='xh')
                    bn_hh = batch_norm_layer_in_time(hh, self.max_length, step_int, self.is_training, scope='hh')

                    bias = tf.get_variable("bias", [self.num_out_ch * 4])
                    hidden = bn_xh + bn_hh + bias

                    i, j, f, o = tf.split(hidden, 4, axis=3)
            else:
                    concat = conv_linear([inputs, h], self.filter_size, self.num_out_ch * 4, True)
                    i, j, f, o = tf.split(concat, 4, axis=3)

            new_c = c * tf.nn.sigmoid(f) + tf.nn.sigmoid(i) * self.activation(j)

            if self.batch_norm:
                    new_c2h = batch_norm_layer_in_time(new_c, self.max_length, step_int, self.is_training, scope='new_c')
                    if self.pres_ident:
                        def cum_erf(x):
                            return 0.5 * tf.erfc(-x/np.sqrt(2))
                        keep_prob = cum_erf(new_c2h + 1) - cum_erf(new_c2h - 1)
                        def train_pres():
                            keep_mask = tf.greater(keep_prob, tf.random_uniform(tf.shape(new_c2h), dtype=tf.float32))
                            return tf.where(keep_mask, new_c2h, c)
                        def val_pres():
                            return (new_c2h * keep_prob) + (c * (1 - keep_prob))
                        if self.is_training:
                            new_c2h = train_pres()
                        else:
                            new_c2h = val_pres()
            else:
                    new_c2h = new_c
            new_h = self.activation(new_c2h) * tf.nn.sigmoid(o)

            out_h = new_h
            if self.max_pool:
                out_h = tf.nn.max_pool(out_h, [1,2,2,1], [1,2,2,1], padding=("SAME" if self.new_pool else "VALID"))

            new_c = self.shape_out(new_c)
            new_h = self.shape_out(new_h)
            out_h = self.shape_out(out_h, self.max_pool)


            return out_h, (new_c, new_h, step+1)

class ConvPhasedLSTMCell(RNNCell):
    '''Phased LSTM'''
    def __init__(self, shape, filter_size, num_in_ch, num_out_ch, is_training,
                 alpha = 1e-3, tau_init = 6, r_on_init = 5e-2, activation=tf.nn.tanh):
        self.shape = (int(shape[0]) , int(shape[1]))
        self.filter_size = filter_size
        self.num_in_ch = int(num_in_ch)
        self.num_out_ch = int(num_out_ch)
        self.num_units = self.shape[0] * self.shape[1] * self.num_out_ch
        self.is_training = is_training
        self.alpha = alpha
        self.tau_init = tau_init
        self.r_on_init = r_on_init
        self.activation = activation

    @property
    def state_size(self):
        return (self.num_units, self.num_units, 1)

    @property
    def output_size(self):
        return self.num_units

    def shape_in(self, a, is_input=False):
            return tf.reshape(a, [tf.shape(a)[0], self.shape[0], self.shape[1],
                self.num_in_ch if is_input else self.num_out_ch])

    def shape_out(self, a):
            return tf.reshape(a, [tf.shape(a)[0], self.num_units])


    def __call__(self, x, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            c, h, step = state
            time = tf.tile(step[0], [self.num_out_ch])
            batch_size = x.get_shape().as_list()[0]

            c = self.shape_in(c)
            h = self.shape_in(h)
            x = self.shape_in(x, True)

            if self.is_training:
                alpha = tf.constant(self.alpha, dtype=tf.float32)
            else:
                alpha = tf.constant(0, dtype=tf.float32)

            bias = tf.get_variable('bias', [4 * self.num_out_ch])

            tau = tf.get_variable("tau",
                [self.num_out_ch],
                initializer=random_exp_initializer(0, self.tau_init),
                dtype=tf.float32
            )
            s = tf.get_variable("s",
                [self.num_out_ch],
                initializer=init_ops.random_uniform_initializer(0., tau.initialized_value()),
                dtype=tf.float32
            )
            r_on = tf.get_variable("r_on",
                [self.num_out_ch],
                initializer=init_ops.constant_initializer(self.r_on_init),
                dtype=tf.float32,
                trainable=False
            )

            phi = dk_mod(dk_mod((time - s), tau) + tau, tau) / tau

            is_up = tf.less(phi, (r_on * 0.5))
            is_down = tf.logical_and(tf.less(phi, r_on), tf.logical_not(is_up))

            k = tf.where(is_up, 2. * (phi / r_on),
                         tf.where(is_down, 2. - 2. * (phi / r_on), alpha * phi))

            k = tf.reshape(k, [1, 1, 1, self.num_out_ch])

            xh = conv_linear([x], self.filter_size, self.num_out_ch * 4, False, scope='xh',
                             initializer=conv_orthogonal_initializer, init_param=None)
            hh = conv_linear([h], self.filter_size, self.num_out_ch * 4, False, scope='hh',
                             initializer=conv_identity_initializer, init_param=0.95)

            hidden = xh + hh + bias

            i, j, f, o = tf.split(hidden, 4, axis=3)

            new_c = c * tf.nn.sigmoid(f) + tf.nn.sigmoid(i) * self.activation(j)

            phased_new_c = k * new_c + (1 - k) * c

            new_h = tf.nn.tanh(new_c) * tf.nn.sigmoid(o)
            phased_new_h = k * new_h + (1 - k) * h

            phased_new_c = self.shape_out(phased_new_c)
            phased_new_h = self.shape_out(phased_new_h)

            return phased_new_h, (phased_new_c, phased_new_h, step+1)

def conv_linear(args, filter_size, num_out_ch, bias, bias_start=0.0,
    scope="linear", initializer=tf.truncated_normal_initializer, init_param=1e-2):
    """convolution:
    Args:
        args: a 4D Tensor or a list of 4D, batch x n, Tensors.
        filter_size: int tuple of filter height and width.
        num_out_ch: int, number of features.
        bias_start: starting value to initialize the bias; 0 by default.
        scope: VariableScope for the created subgraph; defaults to "Linear".
    Returns:
        A 4D Tensor with shape [batch h w num_out_ch]
    Raises:
        ValueError: if some of the arguments has unspecified or wrong shape.
    """

    # Calculate the total size of arguments on dimension 1.
    total_arg_size_depth = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 4:
            raise ValueError("Linear is expecting 4D arguments: %s" % str(shapes))
        if not shape[3]:
            raise ValueError("Linear expects shape[4] of arguments: %s" % str(shapes))
        else:
            total_arg_size_depth += shape[3]

    dtype = [a.dtype for a in args][0]

    # Now the computation.
    with tf.variable_scope('conv_'+scope):
        matrix = tf.get_variable(
                "matrix", [filter_size[0], filter_size[1], total_arg_size_depth, num_out_ch], dtype=dtype,
                initializer=initializer(init_param, dtype=dtype) if init_param !=None else initializer(dtype=dtype))
        # tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(5.0e-4), [matrix])
        if len(args) == 1:
            res = tf.nn.conv2d(args[0], matrix, strides=[1,1,1,1], padding='SAME')
        else:
            res = tf.nn.conv2d(tf.concat(args, axis=3), matrix, strides=[1,1,1,1], padding='SAME')
        if not bias:
            return res
        bias_term = tf.get_variable(
                "Bias", [num_out_ch],
                dtype=dtype,
                initializer=tf.constant_initializer(
                        bias_start, dtype=dtype))
    return res + bias_term

def conv_orthogonal_filter(shape):
    if shape[3] > shape[2]*shape[0]*shape[1]:
        a = np.random.normal(0.0, 1.0, (shape[2]*shape[0]*shape[1], shape[3]))
    else:
        a = np.random.normal(0.0, 1.0, (shape[3], shape[2]*shape[0]*shape[1]))
    u, _, v = np.linalg.svd(a, full_matrices=False)
    if shape[3] > shape[2]*shape[0]*shape[1]:
        v = v.transpose()
    v = v.reshape((shape[3], shape[2], shape[0], shape[1]))
    return v.transpose([2,3,1,0])

def conv_identity_initializer(scale, dtype=tf.float32):
    def _initializer(shape, dtype=dtype, partition_info=None):
        filter_shape = [shape[0],shape[1],shape[2],shape[2]]
        filter_j = np.random.normal(scale=1e-2, size=filter_shape)
        id0 = shape[0]//2
        id1 = shape[1]//2
        for dim in range(shape[2]):
            filter_j[id0,id1,dim,dim] += (1*scale)
        filter_i = conv_orthogonal_filter(filter_shape)
        filter_f = conv_orthogonal_filter(filter_shape)
        filter_o = conv_orthogonal_filter(filter_shape)
        filter_concat = np.concatenate((filter_i,filter_j,filter_f,filter_o), axis=3)
        return tf.constant(filter_concat, dtype)

    return _initializer

def conv_orthogonal_initializer(dtype=tf.float32):
    def _initializer(shape, dtype=dtype, partition_info=None):
        return tf.constant(conv_orthogonal_filter(shape), dtype)
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
