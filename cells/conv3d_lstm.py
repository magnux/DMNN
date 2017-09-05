from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import random_ops
from tensorflow.contrib.rnn import RNNCell
from layers.ops import *
from layers.base import *


class Conv3dLSTMCell(RNNCell):
    """Conv LSTM recurrent network cell"""

    def __init__(self, in_shape, filter_size, num_in_ch, num_out_ch, max_pool=False, activation=tf.nn.tanh,
                    batch_norm=False, pres_ident=False, is_training=False, max_length=0, keep_prob=1.0):
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
        self.in_shape = (int(in_shape[0]), int(in_shape[1]), int(in_shape[2]))
        self.filter_size = filter_size
        self.num_in_ch = int(num_in_ch)
        self.num_out_ch = int(num_out_ch)
        self.max_pool = max_pool
        self.activation = activation
        self.num_units = self.in_shape[0] * self.in_shape[1] * self.in_shape[2] * self.num_out_ch
        self.batch_norm = batch_norm
        self.pres_ident = pres_ident
        self.is_training = is_training
        self.max_length = max_length
        self.keep_prob = keep_prob

    @property
    def out_shape(self):
        out_depth = np.ceil(self.in_shape[0] / 2)
        out_depth = out_depth if self.max_pool else self.in_shape[0]
        out_height = np.ceil(self.in_shape[1] / 2)
        out_height = out_height if self.max_pool else self.in_shape[1]
        out_width = np.ceil(self.in_shape[2] / 2)
        out_width = out_width if self.max_pool else self.in_shape[2]
        return (out_depth, out_height, out_width)

    @property
    def state_size(self):
        return (self.num_units, self.num_units, 1)

    @property
    def output_size(self):
        return self.out_shape[0] * self.out_shape[1] * self.out_shape[2] * self.num_out_ch

    def shape_in(self, a, is_input=False):
            return tf.reshape(a, [tf.shape(a)[0], self.in_shape[0], self.in_shape[1], self.in_shape[2],
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
                    xh = conv_linear_3d([inputs], self.filter_size, self.num_out_ch * 4, False,
                        scope='xh', initializer=conv3d_orthogonal_initializer, init_param=None)
                    hh = conv_linear_3d([h], self.filter_size, self.num_out_ch * 4, False,
                        scope='hh', initializer=conv3d_identity_initializer, init_param=0.95)
                    bn_xh = batch_norm_layer_in_time(xh, self.max_length, step_int, self.is_training, scope='xh')
                    bn_hh = batch_norm_layer_in_time(hh, self.max_length, step_int, self.is_training, scope='hh')

                    bias = tf.get_variable("bias", [self.num_out_ch * 4])
                    hidden = bn_xh + bn_hh + bias

                    i, j, f, o = tf.split(hidden, 4, axis=4)
            else:
                    concat = conv_linear_3d([inputs, h], self.filter_size, self.num_out_ch * 4, True)
                    i, j, f, o = tf.split(concat, 4, axis=4)

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
                out_h = tf.nn.max_pool3d(out_h, [1,2,2,2,1], [1,2,2,2,1], padding="SAME")

            new_c = self.shape_out(new_c)
            new_h = self.shape_out(new_h)
            out_h = self.shape_out(out_h, self.max_pool)


            return out_h, (new_c, new_h, step+1)


def conv_linear_3d(args, filter_size, num_out_ch, bias, bias_start=0.0,
    scope="linear", initializer=tf.truncated_normal_initializer, init_param=1e-2):
    """convolution:
    Args:
        args: a 5D Tensor or a list of 5D, batch x n, Tensors.
        filter_size: int tuple of filter height and width.
        num_out_ch: int, number of features.
        bias_start: starting value to initialize the bias; 0 by default.
        scope: VariableScope for the created subgraph; defaults to "Linear".
    Returns:
        A 5D Tensor with shape [batch d h w num_out_ch]
    Raises:
        ValueError: if some of the arguments has unspecified or wrong shape.
    """

    # Calculate the total size of arguments on dimension 1.
    total_arg_size_depth = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 5 or not shape[4]:
            raise ValueError("Linear is expecting 5D arguments: %s" % str(shapes))
        else:
            total_arg_size_depth += shape[4]

    dtype = [a.dtype for a in args][0]

    # Now the computation.
    with tf.variable_scope('conv3d_'+scope):
        matrix = tf.get_variable(
                "matrix", [filter_size[0], filter_size[1], filter_size[2], total_arg_size_depth, num_out_ch], dtype=dtype,
                initializer=initializer(init_param, dtype=dtype) if init_param is not None else initializer(dtype=dtype))
        # tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(5.0e-4), [matrix])
        if len(args) == 1:
            res = tf.nn.conv3d(args[0], matrix, strides=[1, 1, 1, 1, 1], padding='SAME')
        else:
            res = tf.nn.conv3d(tf.concat(args, axis=4), matrix, strides=[1, 1, 1, 1, 1], padding='SAME')
        if not bias:
            return res
        bias_term = tf.get_variable(
                "Bias", [num_out_ch],
                dtype=dtype,
                initializer=tf.constant_initializer(
                        bias_start, dtype=dtype))
    return res + bias_term


def conv3d_orthogonal_filter(shape):
    if shape[4] > shape[3]*shape[2]*shape[0]*shape[1]:
        a = np.random.normal(0.0, 1.0, (shape[3]*shape[2]*shape[0]*shape[1], shape[4]))
    else:
        a = np.random.normal(0.0, 1.0, (shape[4], shape[3]*shape[2]*shape[0]*shape[1]))
    u, _, v = np.linalg.svd(a, full_matrices=False)
    if shape[3] > shape[3]*shape[2]*shape[0]*shape[1]:
        v = v.transpose()
    v = v.reshape((shape[4], shape[3], shape[2], shape[0], shape[1]))
    return v.transpose([3,4,2,1,0])


def conv3d_identity_initializer(scale, dtype=tf.float32):
    def _initializer(shape, dtype=dtype, partition_info=None):
        filter_shape = [shape[0],shape[1],shape[2],shape[3],shape[3]]
        filter_j = np.random.normal(scale=1e-2, size=filter_shape)
        id0 = shape[0]//2
        id1 = shape[1]//2
        id2 = shape[2]//2
        for dim in range(shape[2]):
            filter_j[id0,id1,id2,dim,dim] += (1*scale)
        filter_i = conv3d_orthogonal_filter(filter_shape)
        filter_f = conv3d_orthogonal_filter(filter_shape)
        filter_o = conv3d_orthogonal_filter(filter_shape)
        filter_concat = np.concatenate((filter_i,filter_j,filter_f,filter_o), axis=4)
        return tf.constant(filter_concat, dtype)

    return _initializer


def conv3d_orthogonal_initializer(dtype=tf.float32):
    def _initializer(shape, dtype=dtype, partition_info=None):
        return tf.constant(conv3d_orthogonal_filter(shape), dtype)
    return _initializer
