from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
from tools import matconvnet2tf
from .pre_process_poses import load_arr
from layers.base import triu_layer


from layers.base import distance_matrix_embedding


def dm_embedding(is_training, config, poses, dm_shape, dm_size, batch_size, max_length, hidden_size, consider_conf, _float_type=tf.float32):
    # Distance Matrix Embedding
    with tf.name_scope("dm_emb"):

        embedding = [dm_emb(poses_i, consider_conf, config.dm_transform, dm_shape, batch_size, max_length,
                            None if p == 0 and is_training else True, _float_type) for p, poses_i in enumerate(poses)]

        if config.norm_dms_ch:
            assert not config.split_bod, print('norm_dms_ch incompatible with split_bod')
            embedding = [norm_dms_ch(emb) for emb in embedding]
        elif config.norm_dms_px:
            assert not config.split_bod, print('norm_dms_px incompatible with split_bod')
            embedding = [norm_dms_px(emb) for emb in embedding]
        elif config.norm_dms_bn:
            embedding = [tf.contrib.layers.batch_norm(emb, center=False, scale=False, updates_collections=None,
                                                       reuse=None if (e == 0) and is_training else True,
                                                       fused=False, scope='dm_bn', is_training=is_training)
                          for e, emb in enumerate(embedding)]

        if config.inference_model != 'siamese':
            embedding = tf.concat(embedding, axis=4)

        def num_elems_triu(n):
            return (((n + 1) * n) / 2) - n

        triu_size = int(num_elems_triu(dm_shape[0]))

        if config.inference_model[0:4] != 'conv':
            # warning, gotcha: inference_model != cell_model. See base_config for more info
            if config.cell_model[0:4] == 'conv':
                """ Reshaping the distance matrices to 1d vectors so they can work with TF standard rnn,
                    they will be internally reshaped back to matrices in the conv cells """
                if config.inference_model == 'siamese':
                    # Siamese model uses split dms for the two skeletons
                    embedding = [tf.reshape(emb, [batch_size, max_length, dm_size//2]) for emb in embedding]
                else:
                    embedding = tf.reshape(embedding, [batch_size, max_length, dm_size])
            else:
                """ If the cells are linear, we are better of just taking the upper triangular portion of the dm
                    discarding redundant information """
                embedding = tf.reshape(triu_layer(embedding), [batch_size, max_length, triu_size * dm_shape[2]])
                if config.cell_model == 'phlstm':
                    times = tf.cast(tf.range(0, max_length), _float_type)
                    times = tf.expand_dims(tf.expand_dims(times, 1), 0)
                    times = tf.tile(times, [batch_size, 1, hidden_size])
                    embedding = tf.concat([embedding, times], axis=2)
        elif config.inference_model[0:6] == 'conv2d':
            embedding = tf.reshape(embedding, [batch_size, max_length, dm_shape[0] * dm_shape[1], dm_shape[2]])
            # This option was discarded definitely because it had suboptimal accuracy, left only for documentation
            # embedding = tf.reshape(triu_layer(embedding), [batch_size, max_length, triu_size, dm_shape[2]])

        return embedding


def dm_emb(poses, consider_conf, dm_transform, dm_shape, batch_size, max_length, reuse, _float_type=tf.float32):
    embedding = distance_matrix_embedding(poses, consider_conf)
    if dm_transform:
        emb = tf.reshape(embedding, [batch_size * max_length, dm_shape[0], dm_shape[1], 1])
        emb_mean = tf.constant(
            np.reshape(np.load('data/net-epoch-922-v73_mean.npy'), [1, dm_shape[0], dm_shape[1], 1]),
            _float_type)
        emb_mult = tf.constant(np.load('data/net-epoch-922-v73_range_multiplier.npy'), _float_type)
        emb = (tf.sqrt(emb) - emb_mean) * emb_mult
        emb = matconvnet2tf.import_net('data/net-epoch-1486-v73.mat', emb, reuse=reuse)
        embedding = tf.reshape(emb, [batch_size, max_length, dm_shape[0], dm_shape[1], 1])
    return embedding


def norm_dms_ch(emb):
    mean_ch_dist = load_arr('mean_ch_dist')
    std_ch_dist = load_arr('std_ch_dist')
    emb_0 = tf.expand_dims((emb[:, :, :, :, 0] - mean_ch_dist) / std_ch_dist, 4)
    if emb.get_shape().as_list()[4] == 1:
        return emb_0
    else:
        mean_ch_rot = load_arr('mean_ch_rot')
        std_ch_rot = load_arr('std_ch_rot')
        emb_1 = tf.expand_dims((emb[:, :, :, :, 1] - mean_ch_rot) / std_ch_rot, 4)
        return tf.concat([emb_0, emb_1], axis=4)


def norm_dms_px(emb, dm_shape):
    def shape_mat(x):
        return tf.reshape(x, [1, 1, dm_shape[0], dm_shape[1]])

    mean_px_dist = shape_mat(load_arr('mean_px_dist'))
    std_px_dist = shape_mat(load_arr('std_px_dist'))
    emb_0 = tf.expand_dims((emb[:, :, :, :, 0] - mean_px_dist) / (std_px_dist + 1e-8), 4)
    if emb.get_shape().as_list()[4] == 1:
        return emb_0
    else:
        mean_px_rot = shape_mat(load_arr('mean_px_rot'))
        std_px_rot = shape_mat(load_arr('std_px_rot'))
        emb_1 = tf.expand_dims((emb[:, :, :, :, 1] - mean_px_rot) / (std_px_rot + 1e-8), 4)
        return tf.concat([emb_0, emb_1], axis=4)
