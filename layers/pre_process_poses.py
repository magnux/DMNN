from __future__ import absolute_import, division, print_function

import os
import tensorflow as tf
import numpy as np


def pre_process_poses(is_training, config, poses, actions):

    """Dataset specific preprocessing"""
    if config.data_set == 'NTURGBD':
        poses = tf.transpose(poses, [0, 3, 1, 2])
        dm_shape = (25, 25, 2 * 2) if not config.only_3dpos else (25, 25, 1 * 2)

        joints_right_arm = [23, 24, 11, 10, 9, 8]
        joints_left_arm = [4, 5, 6, 7, 21, 22]
        joints_head = [20, 2, 3]
        joints_torso = [16, 12, 0, 1, 8, 4]
        joints_torso_s = [0, 1]
        joints_right_leg = [19, 18, 17, 16]
        joints_left_leg = [12, 13, 14, 15]

        joints_right_up = joints_right_arm + joints_head
        joints_left_up = joints_left_arm + joints_head
        joints_down = joints_right_leg + [0] + joints_left_leg
        joints_center = joints_torso + joints_head
        joints_broad = [18, 16, 12, 14, 11, 8, 4, 7, 3]

        body_splits = [joints_right_up, joints_left_up, joints_down, joints_center, joints_broad]

        # Hand picked perms for different models
        joints_perms = [
            np.arange(25),
            np.array(
                joints_left_arm + joints_head + joints_right_arm + joints_torso_s + joints_left_leg + joints_right_leg),
            np.array(
                joints_right_leg + joints_torso_s + joints_left_leg + joints_head + joints_right_arm + joints_left_arm),
        ]

        poses = list(tf.split(poses, 2, axis=2))
    elif config.data_set == 'SBU_inter':
        poses = tf.transpose(poses, [0, 3, 1, 2])
        # dm_shape = (15, 15, 2)
        dm_shape = (30, 30, 1)

        poses = list(tf.split(poses, 2, axis=2))
        para_sort = [0, 15, 1, 16, 2, 17, 3, 18, 4, 19, 5, 20, 6, 21, 7, 22, 8, 23, 9, 24, 10, 25, 11, 26, 12, 27, 13, 28, 14, 29]
        mirror_sort = [0, 15, 1, 16, 2, 17, 3, 21, 4, 22, 5, 23, 6, 18, 7, 19, 8, 20, 9, 27, 10, 28, 11, 29, 12, 24, 13, 25, 14, 26]

        joints_left_arm = [3, 4, 5]
        joints_right_arm = [8, 7, 6]
        joints_head = [1, 0]
        joints_torso = [2]
        joints_left_leg = [9, 10, 11]
        joints_right_leg = [14, 13, 12]

        body_sort = joints_left_arm + joints_head + joints_right_arm + joints_torso + joints_left_leg + joints_right_leg
        body_sort_2 = [b + 15 for b in body_sort]
        body_sort_22 = [element for tupl in zip(body_sort, body_sort_2) for element in tupl]

        body_sort_x = joints_left_arm + joints_left_leg + joints_head + joints_torso + joints_right_arm + joints_right_leg
        body_sort_2x = [b + 15 for b in body_sort_x]
        body_sort_22x = [element for tupl in zip(body_sort_x, body_sort_2x) for element in tupl]

        # body_splits = [joints_left_arm+joints_left_leg,joints_right_arm+joints_right_leg]
        # joints_perms = [np.arange(15), body_sort, body_sort_x]
        body_splits = [joints_left_arm + joints_left_leg, joints_right_arm + joints_right_leg]
        joints_perms = [mirror_sort, body_sort_22, body_sort_22x]

        if config.skel_transform:
            def trans_skel(poses):
                idcs = np.array([10, 2, 1, 0, 3, 4, 4, 5, 6, 7, 7, 8, 9, 10, 10, 11, 12, 13, 13, 14, 3, 5, 5, 8, 8])
                facts = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
                idcs_comb = np.array(
                    [11, 2, 1, 0, 3, 4, 5, 5, 6, 7, 8, 8, 9, 10, 11, 11, 12, 13, 14, 14, 6, 5, 5, 8, 8])
                facts_comb = np.array([1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 1, 0, 0, 0, 0])
                poses = tf.transpose(poses, [2, 1, 0, 3])
                new_poses = tf.gather_nd(poses, np.reshape(idcs, [25, 1])) * np.reshape(facts, [25, 1, 1, 1])
                new_poses += tf.gather_nd(poses, np.reshape(idcs_comb, [25, 1])) * np.reshape(facts_comb,
                                                                                              [25, 1, 1, 1])
                new_poses /= (np.reshape(facts, [25, 1, 1, 1]) + np.reshape(facts_comb, [25, 1, 1, 1]))
                return tf.transpose(new_poses, [2, 1, 0, 3])

            poses = [trans_skel(poses_i) for poses_i in poses]

            dm_shape = (25, 25, 1 * 2)

            joints_right_arm = [23, 24, 11, 10, 9, 8]
            joints_left_arm = [4, 5, 6, 7, 21, 22]
            joints_head = [20, 2, 3]
            joints_torso = [16, 12, 0, 1, 8, 4]
            joints_torso_s = [0, 1]
            joints_right_leg = [19, 18, 17, 16]
            joints_left_leg = [12, 13, 14, 15]

            joints_right_up = joints_right_arm + joints_head
            joints_left_up = joints_left_arm + joints_head
            joints_down = joints_right_leg + [0] + joints_left_leg
            joints_center = joints_torso + joints_head
            joints_broad = [18, 16, 12, 14, 11, 8, 4, 7, 3]

            body_splits = [joints_right_up, joints_left_up, joints_down, joints_center, joints_broad]

            joints_perms = [
                np.arange(25),
                np.array(
                    joints_left_arm + joints_head + joints_right_arm + joints_torso_s + joints_left_leg + joints_right_leg),
                np.array(
                    joints_right_leg + joints_torso_s + joints_left_leg + joints_head + joints_right_arm + joints_left_arm),
            ]
    elif config.data_set == 'UWA3DII':
        poses = [tf.transpose(poses, [0, 3, 1, 2])]
        dm_shape = (15, 15, 1)
    
        joints_left_arm = [3, 4, 5]
        joints_right_arm = [8, 7, 6]
        joints_head = [1, 0]
        joints_torso = [2]
        joints_left_leg = [9, 10, 11]
        joints_right_leg = [14, 13, 12]
    
        body_sort = joints_left_arm + joints_head + joints_right_arm + joints_torso + joints_left_leg + joints_right_leg
        body_sort_x = joints_left_arm + joints_left_leg + joints_head + joints_torso + joints_right_arm + joints_right_leg
    
        body_splits = [joints_left_arm + joints_left_leg, joints_right_arm + joints_right_leg]
        joints_perms = [np.arange(15), body_sort, body_sort_x]
    elif config.data_set == 'NUCLA' or config.data_set == 'MSRC12':
        poses = [tf.transpose(poses, [0, 3, 1, 2])]
        dm_shape = (20, 20, 1)
    
        joints_left_arm = [4, 5, 6, 7]
        joints_right_arm = [11, 10, 9, 8]
        joints_head = [2, 3]
        joints_torso = [0, 1]
        joints_left_leg = [12, 13, 14, 15]
        joints_right_leg = [19, 18, 17, 16]
    
        body_sort = joints_left_arm + joints_head + joints_right_arm + joints_torso + joints_left_leg + joints_right_leg
        body_sort_x = joints_left_arm + joints_left_leg + joints_head + joints_torso + joints_right_arm + joints_right_leg
    
        body_splits = [joints_left_arm + joints_left_leg, joints_right_arm + joints_right_leg]
        joints_perms = [np.arange(20), body_sort, body_sort_x]


    """General pre processing"""
    if config.single_input:
        poses.pop(1)
        dm_shape = (dm_shape[0], dm_shape[1], dm_shape[2] // 2)

    if config.split_bod:
        for b in range(1, len(body_splits)):
            assert len(body_splits[0]) == len(body_splits[b]), print('all body splits have to have the same size')

        assert config.only_3dpos, print('split_bod only working with only_3dpos enabled')
        dm_shape = (len(body_splits[0]), len(body_splits[0]), len(body_splits) * dm_shape[2])

    num_perms = len(joints_perms)

    dm_size = dm_shape[0] * dm_shape[1] * dm_shape[2]

    # These two can be dynamic
    batch_size = tf.shape(poses[0])[0]
    max_length = tf.shape(poses[0])[1]

    # This is known at launch time
    n_dims = poses[0].get_shape().as_list()[3]
    fixed_length = config.pick_num if (config.random_pick or config.inference_model == 'staged') \
                                   else config.crop_len if config.random_crop else None

    if config.seq_smooth:
        poses = [seq_smooth(poses_i, n_dims) for poses_i in poses]

    if config.dup_input and len(poses) > 1:
        if config.data_set == 'NTURGBD':
            to_select = tf.greater(actions, 48)
            poses[1] = tf.where(to_select, poses[1], poses[0])


    """Data Augmentation"""
    if is_training:
        if config.swap_input and len(poses) > 1:
            to_select = tf.cast(tf.round(tf.random_uniform([batch_size])), tf.bool)
            new_poses_0 = tf.where(to_select, poses[0], poses[1])
            new_poses_1 = tf.where(to_select, poses[1], poses[0])
            poses[0] = new_poses_0
            poses[1] = new_poses_1

        if config.norm_skel:
            poses = [norm_skel(poses_i) for poses_i in poses]

        if config.jitter_height:
            poses = [jitter_height(poses_i, batch_size) for poses_i in poses]

        if config.sim_occlusions:
            poses = [sim_occlusions(poses_i, dm_shape, batch_size, max_length, n_dims, body_splits) for poses_i in poses]

        if config.sim_translations:
            if len(poses) > 1:
                poses = [sim_translations(poses_i, dm_shape, batch_size, max_length, n_dims) for poses_i in poses]
            else:
                raise Exception('sim translations only make sense with two skels')

    if config.data_set == 'SBU_inter':
        poses = tf.concat(poses, axis=2)

    if config.restore_pretrained and dm_shape[2] == 1:
        poses.append(poses[0])
        dm_shape = (dm_shape[0], dm_shape[1], 2)
        dm_size = dm_shape[0] * dm_shape[1] * dm_shape[2]

    if not config.joint_permutation is None:
        # permutation = tf.constant(config.joint_permutation, dtype=self._int_type, shape=[dm_shape[0], 1])
        permutation = tf.constant(joints_perms[config.joint_permutation], dtype=tf.int32, shape=[dm_shape[0], 1])
        poses = [permute(poses_i, permutation) for poses_i in poses]

    if config.split_bod:
        poses = [split_bod(poses_i) for poses_i in poses]

    return poses, dm_shape, dm_size, body_splits, joints_perms, num_perms, batch_size, max_length, fixed_length, n_dims


def load_arr(name, config):
    data_set = 'NTURGBD' if config.skel_transform else config.data_set
    data_set_version = 'v1' if config.skel_transform else config.data_set_version
    file_path = os.path.join(config.data_path, data_set + data_set_version + '_' + name + '.npy')
    arr = None
    with open(file_path, 'rb') as f:
        arr = np.load(f)
        arr[np.isnan(arr)] = 0
    return tf.constant(arr, name=name)


def one_gaussian(x, mu, sig):
    vals = 1. / (np.sqrt(2. * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.) / 2)
    return vals / np.sum(vals)


def seq_smooth(poses, n_dims, _float_type=tf.float32):
    num_frames = 5
    poses_filter = one_gaussian(np.linspace(-1, 1, num_frames), 0, 0.3)
    poses_filter = np.tile(np.reshape(poses_filter, [num_frames, 1, 1, 1]), [1, 1, n_dims, 1])
    tf_poses_filter = tf.constant(poses_filter, dtype=_float_type)
    poses = tf.nn.depthwise_conv2d(poses, tf_poses_filter, [1, 1, 1, 1], 'SAME')
    return poses


def norm_skel(poses):
    skel_mean = load_arr('skel_mean')
    skel_std = load_arr('skel_std')
    # return (poses - skel_mean) / skel_std
    return poses / 1000


def jitter_height(poses, batch_size, _float_type=tf.float32):
    jitter_y = poses[:, :, :, 1] * tf.random_uniform([batch_size, 1, 1], minval=0.7, maxval=1.3, dtype=_float_type)
    new_poses = tf.concat(
        [tf.expand_dims(poses[:, :, :, 0], 3),
         tf.expand_dims(jitter_y, 3),
         poses[:, :, :, 2:]], axis=3)
    return new_poses


def sim_occlusions(poses, dm_shape, batch_size, max_length, n_dims, body_splits, _int_type=tf.int32, _float_type=tf.float32):
    def occluded_poses():
        body_splits_tf = tf.constant(body_splits, dtype=_int_type)
        occ_idcs = tf.random_uniform([batch_size, 1], minval=0, maxval=len(body_splits), dtype=_int_type)
        occ_idcs = tf.gather_nd(body_splits_tf, occ_idcs)
        noise_mask = tf.tile(
            tf.reshape(
                tf.cast(tf.reduce_sum(tf.one_hot(occ_idcs, dm_shape[0]), axis=1), dtype=tf.bool),
                [batch_size, 1, dm_shape[0], 1]),
            [1, max_length, 1, n_dims])
        noisy_poses = poses * tf.random_uniform([batch_size, max_length, 1, n_dims], minval=0.8, maxval=1.2, dtype=_float_type)
        return tf.where(noise_mask, noisy_poses, poses)

    occlude_rate = 0.5
    return tf.cond(tf.cast(tf.round(tf.random_uniform([], minval=-0.5, maxval=0.5) + occlude_rate), tf.bool),
                   occluded_poses, lambda: poses)


def sim_translations(poses, dm_shape, batch_size, max_length, n_dims, _float_type=tf.float32):
    trans_factor = 1 / 6

    def translated_poses():
        translation = tf.random_uniform([batch_size, 1, 1, n_dims], minval=1.0 - trans_factor,
                                        maxval=1.0 + trans_factor, dtype=_float_type)
        translation = tf.tile(translation, [1, max_length, dm_shape[0], 1])
        return poses + translation

    translate_rate = 0.75
    return tf.cond(tf.cast(tf.round(tf.random_uniform([], minval=-0.5, maxval=0.5) + translate_rate), tf.bool),
                   translated_poses, lambda: poses)


def split_bod(poses, body_splits, _int_type=tf.int32):
    poses = tf.transpose(poses, [2, 1, 0, 3])
    bodysplit_l = []
    for b in range(len(body_splits)):
        indcs = tf.reshape(tf.constant(body_splits[b], dtype=_int_type), [len(body_splits[b]), 1])
        split_poses = tf.gather_nd(poses, indcs)
        split_poses = tf.transpose(split_poses, [2, 1, 0, 3])
        bodysplit_l.append(split_poses)
    return tf.stack(bodysplit_l, axis=2)


def permute(poses, permutation):
    poses = tf.transpose(poses, [2, 1, 0, 3])
    shuffled_poses = tf.gather_nd(poses, permutation)
    shuffled_poses = tf.transpose(shuffled_poses, [2, 1, 0, 3])
    return shuffled_poses


def poses_perms(poses_l, joints_perms, num_perms, dm_shape, _int_type=tf.int32):
    poses_perms_l_l = []
    for n in range(num_perms):
        shuffle_idcs = np.reshape(joints_perms[n], [dm_shape[0], 1])
        tf_shuffle_idcs = tf.get_variable(
            "shuffle_idcs_%d" % n, [dm_shape[0], 1],
            initializer=tf.constant_initializer(shuffle_idcs),
            dtype=_int_type, trainable=False
        )
        poses_perms_l = []
        for poses in poses_l:
            poses_perms_l.append(permute(poses, tf_shuffle_idcs))
        poses_perms_l_l.append(poses_perms_l)
    return poses_perms_l_l