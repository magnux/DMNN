from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np

COMB_MATRIX_COLLECTION = 'COMB_MATRIX'

def learn_comb_matrix(is_training, config, poses, dm_shape, batch_size, max_length, n_dims, _float_type=tf.float32):
    with tf.variable_scope("learn_comb", reuse=None if is_training else True):
        comb_matrix = tf.get_variable(
            "matrix", [dm_shape[0], dm_shape[1]],
            initializer=identity_initializer(0 if config.learn_comb_orth or config.learn_comb_orth_rmsprop else 0.01, dtype=_float_type),
            dtype=_float_type, trainable=True
        )
        tf.add_to_collection(COMB_MATRIX_COLLECTION, comb_matrix)

        comb_matrix_update = None
        if config.learn_comb:
            norm_comb_matrix = comb_matrix / tf.reduce_sum(comb_matrix, axis=0, keep_dims=True)
        elif config.learn_comb_sm:
            norm_comb_matrix = tf.nn.softmax(comb_matrix, dim=0)
            # comb_matrix_image = norm_comb_matrix * 255.0
        elif config.learn_comb_orth:
            # Special update code
            def update_comb_mat(grad, lr):
                A = tf.matmul(tf.transpose(grad), comb_matrix) - tf.matmul(tf.transpose(comb_matrix), grad)
                I = tf.constant(np.eye(dm_shape[0]), dtype=_float_type)
                t1 = I + lr / 2 * A
                t2 = I - lr / 2 * A
                Y = tf.matmul(tf.matmul(tf.matrix_inverse(t1), t2), comb_matrix)
                return tf.assign(comb_matrix, Y)

            comb_matrix_update = update_comb_mat
            norm_comb_matrix = comb_matrix
        elif config.learn_comb_orth_rmsprop:
            comb_matrix_m = tf.get_variable(
                "matrix_momentum", [dm_shape[0], dm_shape[1]],
                initializer=tf.zeros_initializer(),
                dtype=_float_type, trainable=False
            )

            # Special update code
            def update_comb_mat(grad, lr):
                I = tf.constant(np.eye(dm_shape[0]), dtype=_float_type)

                # Momentum update
                momentum_op = tf.assign(comb_matrix_m, comb_matrix_m * 0.99 + (1 - 0.99) * tf.square(grad))

                with tf.control_dependencies([momentum_op]):
                    # Matrix update
                    scaled_grad = lr * grad / tf.sqrt(comb_matrix_m + 1.e-5)
                    A = tf.matmul(tf.transpose(scaled_grad), comb_matrix) - \
                        tf.matmul(tf.transpose(comb_matrix), scaled_grad)
                    t1 = I + 0.5 * A
                    t2 = I - 0.5 * A
                    Y = tf.matmul(tf.matmul(tf.matrix_inverse(t1), t2), comb_matrix)
                    return tf.assign(comb_matrix, Y)

            comb_matrix_update = update_comb_mat
            norm_comb_matrix = comb_matrix
        elif config.learn_comb_unc:
            norm_comb_matrix = comb_matrix
        elif config.learn_comb_centered:
            def center_poses(poses):
                return poses - tf.reduce_mean(poses, axis=2, keep_dims=True)
            poses = [center_poses(poses_i) for poses_i in poses]
            norm_comb_matrix = comb_matrix

        def comb_poses(poses):
            poses = tf.transpose(poses, [0, 1, 3, 2])
            poses = tf.reshape(poses, [batch_size * max_length * n_dims, dm_shape[0]])
            poses = tf.matmul(poses, norm_comb_matrix)
            poses = tf.reshape(poses, [batch_size, max_length, n_dims, dm_shape[0]])
            poses = tf.transpose(poses, [0, 1, 3, 2])
            poses = tf.reshape(poses, [batch_size, max_length, dm_shape[0], n_dims])

            # Alternate code, more elegant but slower, has to be retested with version 1.3+
            # poses = tf.tensordot(poses, norm_comb_matrix, [[2], [1]])
            # poses = tf.transpose(poses, [0, 1, 3, 2])

            return poses

        poses = [comb_poses(poses_i) for poses_i in poses]

        cb_min = tf.reduce_min(norm_comb_matrix)
        cb_max = tf.reduce_max(norm_comb_matrix)
        comb_matrix_image = (norm_comb_matrix - cb_min) / (cb_max - cb_min) * 255.0
        comb_matrix_image = tf.cast(comb_matrix_image, tf.uint8)
        comb_matrix_image = tf.reshape(comb_matrix_image, [1, dm_shape[0], dm_shape[1], 1])

    return poses, comb_matrix_image, comb_matrix_update


def identity_initializer(scale, dtype):
    def _initializer(shape, dtype=dtype, partition_info=None):
        return tf.constant((np.eye(shape[0])) + np.random.randn(shape[0], shape[1]) * scale, dtype)

    return _initializer


# TODO: Remove the following learn_comb_* functions after testing the new code.
def learn_comb(poses, dm_shape, batch_size, max_length, n_dims, reuse=None, _float_type=tf.float32):
    with tf.variable_scope("learn_comb", reuse=reuse):
        comb_matrix = tf.get_variable(
            "matrix", [dm_shape[0], dm_shape[1]],
            initializer=identity_initializer(0.01),
            dtype=_float_type, trainable=True
        )
        norm_comb_matrix = comb_matrix / tf.reduce_sum(comb_matrix, axis=0, keep_dims=True)

        poses = tf.transpose(poses, [0, 1, 3, 2])
        poses = tf.reshape(poses, [batch_size * max_length * n_dims, dm_shape[0]])
        poses = tf.matmul(poses, norm_comb_matrix)
        poses = tf.reshape(poses, [batch_size, max_length, n_dims, dm_shape[0]])
        poses = tf.transpose(poses, [0, 1, 3, 2])
        poses = tf.reshape(poses, [batch_size, max_length, dm_shape[0], n_dims])

        cb_min = tf.reduce_min(norm_comb_matrix)
        cb_max = tf.reduce_max(norm_comb_matrix)
        comb_matrix_image = (norm_comb_matrix - cb_min) / (cb_max - cb_min) * 255.0
        comb_matrix_image = tf.cast(comb_matrix_image, tf.uint8)
        comb_matrix_image = tf.reshape(comb_matrix_image, [1, dm_shape[0], dm_shape[1], 1])
        return poses, comb_matrix_image


def learn_comb_sm(poses, dm_shape, batch_size, max_length, n_dims, reuse=None, _float_type=tf.float32):
    with tf.variable_scope("learn_comb", reuse=reuse):
        comb_matrix = tf.get_variable(
            "matrix", [dm_shape[0], dm_shape[1]],
            initializer=identity_initializer(0.01),
            dtype=_float_type, trainable=True
        )
        norm_comb_matrix = tf.nn.softmax(comb_matrix, dim=0)

        poses = tf.transpose(poses, [0, 1, 3, 2])
        poses = tf.reshape(poses, [batch_size * max_length * n_dims, dm_shape[0]])
        poses = tf.matmul(poses, norm_comb_matrix)
        poses = tf.reshape(poses, [batch_size, max_length, n_dims, dm_shape[0]])
        poses = tf.transpose(poses, [0, 1, 3, 2])
        poses = tf.reshape(poses, [batch_size, max_length, dm_shape[0], n_dims])

        comb_matrix_image = norm_comb_matrix * 255.0
        comb_matrix_image = tf.cast(comb_matrix_image, tf.uint8)
        comb_matrix_image = tf.reshape(comb_matrix_image, [1, dm_shape[0], dm_shape[1], 1])
        return poses, comb_matrix_image


def learn_comb_orth(poses, dm_shape, reuse=None, _float_type=tf.float32):
    with tf.variable_scope("learn_comb", reuse=reuse):
        comb_matrix = tf.get_variable(
            "matrix", [dm_shape[0], dm_shape[1]],
            initializer=identity_initializer(0),
            dtype=_float_type, trainable=False
        )
        tf.add_to_collection(COMB_MATRIX_COLLECTION, comb_matrix)
        poses = tf.tensordot(poses, comb_matrix, [[2], [1]])
        poses = tf.transpose(poses, [0, 1, 3, 2])

        # Special update code
        def update_comb_mat(grad, lr):
            A = tf.matmul(tf.transpose(grad), comb_matrix) - \
                tf.matmul(tf.transpose(comb_matrix), grad)
            I = tf.constant(np.eye(dm_shape[0]), dtype=_float_type)
            t1 = I + lr / 2 * A
            t2 = I - lr / 2 * A
            Y = tf.matmul(tf.matmul(tf.matrix_inverse(t1), t2), comb_matrix)
            return tf.assign(comb_matrix, Y)

        # Visualization
        cb_min = tf.reduce_min(comb_matrix)
        cb_max = tf.reduce_max(comb_matrix)
        comb_matrix_image = (comb_matrix - cb_min) / (cb_max - cb_min) * 255.0
        comb_matrix_image = tf.cast(comb_matrix_image, tf.uint8)
        comb_matrix_image = tf.reshape(comb_matrix_image, [1, dm_shape[0], dm_shape[1], 1])

        return poses, comb_matrix_image, update_comb_mat


def learn_comb_orth_rmsprop(poses, dm_shape, reuse=None, _float_type=tf.float32):
    with tf.variable_scope("learn_comb", reuse=reuse):
        comb_matrix = tf.get_variable(
            "matrix", [dm_shape[0], dm_shape[1]],
            initializer=identity_initializer(0),
            dtype=_float_type, trainable=False
        )
        comb_matrix_m = tf.get_variable(
            "matrix_momentum", [dm_shape[0], dm_shape[1]],
            initializer=tf.zeros_initializer(),
            dtype=_float_type, trainable=False
        )
        tf.add_to_collection(COMB_MATRIX_COLLECTION, comb_matrix)
        poses = tf.tensordot(poses, comb_matrix, [[2], [1]])
        poses = tf.transpose(poses, [0, 1, 3, 2])

        # Special update code
        def update_comb_mat(grad, lr):
            I = tf.constant(np.eye(dm_shape[0]), dtype=_float_type)

            # Momentum update
            momentum_op = tf.assign(comb_matrix_m,
                                    comb_matrix_m * 0.99 + (1 - 0.99) * tf.square(grad))

            with tf.control_dependencies([momentum_op]):
                # Matrix update
                scaled_grad = lr * grad / tf.sqrt(comb_matrix_m + 1.e-5)
                A = tf.matmul(tf.transpose(scaled_grad), comb_matrix) - \
                    tf.matmul(tf.transpose(comb_matrix), scaled_grad)
                t1 = I + 0.5 * A
                t2 = I - 0.5 * A
                Y = tf.matmul(tf.matmul(tf.matrix_inverse(t1), t2), comb_matrix)
                return tf.assign(comb_matrix, Y)

        # Visualization
        cb_min = tf.reduce_min(comb_matrix)
        cb_max = tf.reduce_max(comb_matrix)
        comb_matrix_image = (comb_matrix - cb_min) / (cb_max - cb_min) * 255.0
        comb_matrix_image = tf.cast(comb_matrix_image, tf.uint8)
        comb_matrix_image = tf.reshape(comb_matrix_image, [1, dm_shape[0], dm_shape[1], 1])

        return poses, comb_matrix_image, update_comb_mat


def learn_comb_unc(poses, dm_shape, batch_size, max_length, n_dims, reuse=None, _float_type=tf.float32):
    with tf.variable_scope("learn_comb", reuse=reuse):
        comb_matrix = tf.get_variable(
            "matrix", [dm_shape[0], dm_shape[1]],
            initializer=identity_initializer(0.01),
            dtype=_float_type, trainable=True
        )

        poses = tf.transpose(poses, [0, 1, 3, 2])
        poses = tf.reshape(poses, [batch_size * max_length * n_dims, dm_shape[0]])
        poses = tf.matmul(poses, comb_matrix)
        poses = tf.reshape(poses, [batch_size, max_length, n_dims, dm_shape[0]])
        poses = tf.transpose(poses, [0, 1, 3, 2])
        poses = tf.reshape(poses, [batch_size, max_length, dm_shape[0], n_dims])

        cb_min = tf.reduce_min(comb_matrix)
        cb_max = tf.reduce_max(comb_matrix)
        comb_matrix_image = (comb_matrix - cb_min) / (cb_max - cb_min) * 255.0
        comb_matrix_image = tf.cast(comb_matrix_image, tf.uint8)
        comb_matrix_image = tf.reshape(comb_matrix_image, [1, dm_shape[0], dm_shape[1], 1])
        return poses, comb_matrix_image


def learn_comb_centered(poses, dm_shape, batch_size, max_length, n_dims, reuse=None, _float_type=tf.float32):
    with tf.variable_scope("learn_comb", reuse=reuse):
        comb_matrix = tf.get_variable(
            "matrix", [dm_shape[0], dm_shape[1]],
            initializer=identity_initializer(0.01),
            dtype=_float_type, trainable=True
        )

        pcenter = tf.reduce_mean(poses, axis=2, keep_dims=True)
        poses = poses - pcenter

        poses = tf.transpose(poses, [0, 1, 3, 2])
        poses = tf.reshape(poses, [batch_size * max_length * n_dims, dm_shape[0]])
        poses = tf.matmul(poses, comb_matrix)
        poses = tf.reshape(poses, [batch_size, max_length, n_dims, dm_shape[0]])
        poses = tf.transpose(poses, [0, 1, 3, 2])
        poses = tf.reshape(poses, [batch_size, max_length, dm_shape[0], n_dims])

        cb_min = tf.reduce_min(comb_matrix)
        cb_max = tf.reduce_max(comb_matrix)
        comb_matrix_image = (comb_matrix - cb_min) / (cb_max - cb_min) * 255.0
        comb_matrix_image = tf.cast(comb_matrix_image, tf.uint8)
        comb_matrix_image = tf.reshape(comb_matrix_image, [1, dm_shape[0], dm_shape[1], 1])
        return poses, comb_matrix_image
