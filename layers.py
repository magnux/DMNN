from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np


def linear_layer(x, input_size, output_size, scope="linear_layer", collections=None):
    with tf.variable_scope(scope):
        linear_w = tf.get_variable(
            "weights", [input_size, output_size],
            initializer=tf.truncated_normal_initializer(
                stddev=1e-2, dtype=tf.float32),
            dtype=tf.float32
        )
        linear_b = tf.get_variable(
            "biases", [output_size],
            initializer=tf.truncated_normal_initializer(
                stddev=1e-4, dtype=tf.float32),
            dtype=tf.float32
        )
        if collections is not None:
            tf.add_to_collection(collections, linear_w)
            tf.add_to_collection(collections, linear_b)
        return tf.matmul(x, linear_w) + linear_b


def conv_layer(x, kernel_shape, conv_strides, conv_padding, scope="conv_layer"):
    with tf.variable_scope(scope):
        kernel = tf.get_variable(
            'kernel',
            kernel_shape,
            initializer=tf.truncated_normal_initializer(
                stddev=1e-2, dtype=tf.float32),
            dtype=tf.float32
        )
        conv = tf.nn.conv2d(
            x, kernel, conv_strides, conv_padding
        )
        biases = tf.get_variable(
            'biases',
            [kernel_shape[3]],
            initializer=tf.truncated_normal_initializer(
                stddev=1e-4, dtype=tf.float32),
            dtype=tf.float32
        )
        return tf.nn.bias_add(conv, biases)


def batch_norm_layer(x, is_training, epsilon=1e-3, decay=0.99, scope="layer"):
    '''Assume 2d [batch, values] 3d [batch, width, values] or 4d [batch, width, height, values] tensor'''
    with tf.variable_scope('bn_'+scope):
        dim_x = len(x.get_shape().as_list())
        size = x.get_shape().as_list()[dim_x-1]

        scale = tf.get_variable('scale', [size], initializer=tf.constant_initializer(0.1))
        offset = tf.get_variable('offset', [size])

        pop_mean = tf.get_variable('pop_mean', [size], initializer=tf.zeros_initializer(), trainable=False)
        pop_var = tf.get_variable('pop_var', [size], initializer=tf.ones_initializer(), trainable=False)
        batch_mean, batch_var = tf.nn.moments(x, [i for i in range(dim_x-1)])

        train_mean_op = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var_op = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))

        def batch_statistics():
            with tf.control_dependencies([train_mean_op, train_var_op]):
                return tf.nn.batch_normalization(x, batch_mean, batch_var, offset, scale, epsilon)

        def population_statistics():
            return tf.nn.batch_normalization(x, pop_mean, pop_var, offset, scale, epsilon)

        if is_training:
            return batch_statistics()
        else:
            return population_statistics()


def batch_norm_layer_in_time(x,  max_length, step, is_training, epsilon=1e-3, decay=0.99, scope="layer"):
    '''Assume 2d [batch, values] 3d [batch, width, values] or 4d [batch, width, height, values] tensor'''
    with tf.variable_scope('bn_'+scope):
        dim_x = len(x.get_shape().as_list())
        size = x.get_shape().as_list()[dim_x-1]

        step_idcs = tf.range(step*size, (step+1)*size)

        scale_var = tf.get_variable('scale', [size * max_length], initializer=tf.constant_initializer(0.1))
        scale = tf.gather(scale_var, step_idcs)
        offset_var = tf.get_variable('offset', [size * max_length])
        offset = tf.gather(offset_var, step_idcs)

        pop_mean_var = tf.get_variable('pop_mean', [size * max_length], initializer=tf.zeros_initializer(), trainable=False)
        pop_mean = tf.gather(pop_mean_var, step_idcs)
        pop_var_var = tf.get_variable('pop_var', [size * max_length], initializer=tf.ones_initializer(), trainable=False)
        pop_var = tf.gather(pop_var_var, step_idcs)
        batch_mean, batch_var = tf.nn.moments(x, [i for i in range(dim_x-1)])

        train_mean_op = tf.scatter_update(pop_mean_var, step_idcs, pop_mean * decay + batch_mean * (1 - decay))
        train_var_op = tf.scatter_update(pop_var_var, step_idcs, pop_var * decay + batch_var * (1 - decay))

        def batch_statistics():
            with tf.control_dependencies([train_mean_op, train_var_op]):
                return tf.nn.batch_normalization(x, batch_mean, batch_var, offset, scale, epsilon)

        def population_statistics():
            return tf.nn.batch_normalization(x, pop_mean, pop_var, offset, scale, epsilon)

        if is_training:
            return batch_statistics()
        else:
            return population_statistics()


def distance_matrix(x, y, scope=None):
    with tf.variable_scope(scope, 'distance_matrix', [x, y]):
        x = tf.expand_dims(x, axis=2)
        y = tf.expand_dims(y, axis=3)
        return tf.reduce_sum(tf.square(x - y), axis=-1, keep_dims=True)


def distance_matrix_embedding(x, consider_conf=False, scope="distance_matrix_embedding"):
    '''Assume 4d [batch, len, joints, dims(x,y,z ...)] or 5d [batch, len, bodysplit, joints, dims(x,y,z ...)]tensor'''
    with tf.variable_scope(scope):
        def td_dist(x_td):
            if consider_conf:
                conf_x = x_td[:, :, :, 2]
                x_td = x_td[:, :, :, :2]
            x_tile = tf.tile(tf.expand_dims(x_td, axis=3), [1, 1, 1, tf.shape(x_td)[2], 1])
            x_sub = x_tile - tf.transpose(x_tile, [0, 1, 3, 2, 4])
            x_dist = tf.reduce_sum(tf.square(x_sub), axis=4, keep_dims=True)
            if consider_conf:
                conf_mask = tf.cast(tf.greater(conf_x, 0), tf.float32)
                conf_mask = tf.reshape(conf_mask, [tf.shape(conf_mask)[0],
                                                   tf.shape(conf_mask)[1],
                                                   tf.shape(conf_mask)[2], 1, 1])
                x_dist = x_dist * conf_mask
                conf_mask = tf.transpose(conf_mask, [0, 1, 3, 2, 4])
                x_dist = x_dist * conf_mask
            return x_dist

        def td_plus_rot_dist(x):
            x_td = tf.slice(x, [0, 0, 0, 0], [-1, -1, -1, 3])
            x_dist = td_dist(x_td)

            x_rot = tf.slice(x, [0, 0, 0, 3], [-1, -1, -1, -1])
            x_tile = tf.tile(tf.expand_dims(x_rot, axis=3), [1, 1, 1, tf.shape(x)[2], 1])
            x_sub = x_tile * tf.transpose(x_tile, [0, 1, 3, 2, 4])
            rot_dist = 1 - tf.abs(tf.reduce_sum(x_sub, axis=4, keep_dims=True))
            return tf.concat([x_dist, rot_dist], axis=4)

        if len(x.get_shape().as_list()) == 4:
            if x.get_shape()[3] > 3:
                x_dist = td_plus_rot_dist(x)
            else:
                x_dist = td_dist(x)

            return x_dist
        elif len(x.get_shape().as_list()) == 5:
            x_dist_l = []
            for s in range(x.get_shape().as_list()[2]):
                x_dist_l.append(td_dist(x[:, :, s, :, :]))
            return tf.concat(x_dist_l, axis=4)


def triu_layer(x, scope="triu_layer"):
    '''Assume 5d [batch, len, height, width, matrixes] tensor'''
    with tf.variable_scope(scope):
        x_shape = x.get_shape().as_list()
        np_mask = np.triu(np.ones((x_shape[2], x_shape[3]), dtype=np.bool_), 1)
        np_mask = np.reshape(np_mask, (1, 1, x_shape[2], x_shape[3], 1))
        if x_shape[0] is not None and x_shape[1] is not None:  # Known and fixed batch size and sequence len
            np_mask = np.tile(np_mask, (x_shape[0], x_shape[1], 1, 1, x_shape[4]))
            tf_mask = tf.constant(np_mask, name="triu_layer_mask")
        else:
            tf_mask = tf.constant(np_mask, name="triu_layer_mask")
            tf_mask = tf.tile(tf_mask, [tf.shape(x)[0], tf.shape(x)[1], 1, 1, x_shape[4]])
        return tf.boolean_mask(x, tf_mask)
