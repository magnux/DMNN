from __future__ import absolute_import, division, print_function

import os
from lstm import *
from conv_lstm import *
from conv3d_lstm import *
from tools import matconvnet2tf
from layers import *
import resnet


class PoseSeqModel(object):
    """The PoseSeq model."""

    def __init__(self, is_training, config, input_):
        self._int_type = tf.int16 if config.use_type16 else tf.int32
        self._float_type = tf.float16 if config.use_type16 else tf.float32

        self._is_training = is_training
        splitname = 'train' if self._is_training else 'val'

        hidden_size = config.hidden_size

        self._inputs = input_.generate_batch(self._is_training if not config.only_val else False)
        idxs, subjects, actions, poses, plens = self._inputs
        actions = actions
        subjects = subjects

        poses_0 = poses_1 = joints_perms = None
        consider_conf = False
        
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

            bodysplits = [joints_right_up, joints_left_up, joints_down, joints_center, joints_broad]

            # Hand picked perms for different models
            joints_perms = [
                np.arange(25),
                np.array(joints_left_arm+joints_head+joints_right_arm+joints_torso_s+joints_left_leg+joints_right_leg),
                np.array(joints_right_leg+joints_torso_s+joints_left_leg+joints_head+joints_right_arm+joints_left_arm),
            ]

            # Random perms for the conv3d cell
            if config.cell_model == 'conv3d_variant':
                joints_perms = []
                num_perms = 16
                for _ in range(num_perms):
                    joints_perms.append(np.reshape(np.random.permutation(dm_shape[0]), [dm_shape[0], 1]))

            poses_0, poses_1 = tf.split(poses, 2, axis=2)
        elif config.data_set == 'SBU_inter':
            poses = tf.transpose(poses,[0,3,1,2])
            # dm_shape = (15, 15, 2)
            dm_shape = (30, 30, 1)

            poses_0, poses_1 = tf.split(poses, 2, axis=2)
            para_sort = [0,15,1,16,2,17,3,18,4,19,5,20,6,21,7,22,8,23,9,24,10,25,11,26,12,27,13,28,14,29]
            mirror_sort = [0,15,1,16,2,17,3,21,4,22,5,23,6,18,7,19,8,20,9,27,10,28,11,29,12,24,13,25,14,26]

            joints_left_arm = [3, 4, 5]
            joints_right_arm = [8, 7, 6]
            joints_head = [1, 0]
            joints_torso = [2]
            joints_left_leg = [9, 10, 11]
            joints_right_leg = [14, 13, 12]

            body_sort = joints_left_arm+joints_head+joints_right_arm+joints_torso+joints_left_leg+joints_right_leg
            body_sort_2 = [b+15 for b in body_sort]
            body_sort_22 = [element for tupl in zip(body_sort, body_sort_2) for element in tupl]

            body_sort_x = joints_left_arm+joints_left_leg+joints_head+joints_torso+joints_right_arm+joints_right_leg
            body_sort_2x = [b+15 for b in body_sort_x]
            body_sort_22x = [element for tupl in zip(body_sort_x, body_sort_2x) for element in tupl]

            # bodysplits = [joints_left_arm+joints_left_leg,joints_right_arm+joints_right_leg]
            # joints_perms = [np.arange(15), body_sort, body_sort_x]
            bodysplits = [joints_left_arm + joints_left_leg, joints_right_arm + joints_right_leg]
            joints_perms = [mirror_sort, body_sort_22, body_sort_22x]

            if config.skel_transform:
                def trans_skel(poses):
                    idcs = np.array([10, 2, 1, 0, 3, 4, 4, 5, 6, 7, 7, 8, 9, 10, 10, 11, 12, 13, 13, 14, 3, 5, 5, 8, 8])
                    facts = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
                    idcs_comb = np.array([11, 2, 1, 0, 3, 4, 5, 5, 6, 7, 8, 8, 9, 10, 11, 11, 12, 13, 14, 14, 6, 5, 5, 8, 8])
                    facts_comb = np.array([1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 1, 0, 0, 0, 0])
                    poses = tf.transpose(poses, [2, 1, 0, 3])
                    new_poses = tf.gather_nd(poses, np.reshape(idcs, [25, 1])) * np.reshape(facts, [25, 1, 1, 1])
                    new_poses += tf.gather_nd(poses, np.reshape(idcs_comb, [25, 1])) * np.reshape(facts_comb,
                                                                                                  [25, 1, 1, 1])
                    new_poses /= (np.reshape(facts, [25, 1, 1, 1]) + np.reshape(facts_comb, [25, 1, 1, 1]))
                    return tf.transpose(new_poses, [2, 1, 0, 3])

                poses_0 = trans_skel(poses_0)
                poses_1 = trans_skel(poses_1)

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

                bodysplits = [joints_right_up, joints_left_up, joints_down, joints_center, joints_broad]

                joints_perms = [
                    np.arange(25),
                    np.array(joints_left_arm+joints_head+joints_right_arm+joints_torso_s+joints_left_leg+joints_right_leg),
                    np.array(joints_right_leg+joints_torso_s+joints_left_leg+joints_head+joints_right_arm+joints_left_arm),
                ]

        elif config.data_set == 'UWA3DII':
            poses = tf.transpose(poses, [0, 3, 1, 2])
            dm_shape = (15, 15, 1)

            joints_left_arm = [3, 4, 5]
            joints_right_arm = [8, 7, 6]
            joints_head = [1, 0]
            joints_torso = [2]
            joints_left_leg = [9, 10, 11]
            joints_right_leg = [14, 13, 12]

            body_sort = joints_left_arm+joints_head+joints_right_arm+joints_torso+joints_left_leg+joints_right_leg
            body_sort_x = joints_left_arm+joints_left_leg+joints_head+joints_torso+joints_right_arm+joints_right_leg

            bodysplits = [joints_left_arm + joints_left_leg, joints_right_arm + joints_right_leg]
            joints_perms = [np.arange(15), body_sort, body_sort_x]
        elif config.data_set == 'NUCLA' or config.data_set == 'MSRC12':
            poses = tf.transpose(poses,[0,3,1,2])
            dm_shape = (20, 20, 1)

            joints_left_arm = [4, 5, 6, 7]
            joints_right_arm = [11, 10, 9, 8]
            joints_head = [2, 3]
            joints_torso = [0, 1]
            joints_left_leg = [12, 13, 14, 15]
            joints_right_leg = [19, 18, 17, 16]

            body_sort = joints_left_arm+joints_head+joints_right_arm+joints_torso+joints_left_leg+joints_right_leg
            body_sort_x = joints_left_arm+joints_left_leg+joints_head+joints_torso+joints_right_arm+joints_right_leg

            bodysplits = [joints_left_arm + joints_left_leg, joints_right_arm + joints_right_leg]
            joints_perms = [np.arange(20), body_sort, body_sort_x]

        if config.single_input:
            poses = poses_0
            poses_0 = None
            poses_1 = None
            dm_shape = (dm_shape[0], dm_shape[1], dm_shape[2]//2)

        if config.split_bod:
            for b in range(1, len(bodysplits)):
                assert len(bodysplits[0]) == len(bodysplits[b]), print('all body splits have to have the same size')

            assert config.only_3dpos, print('split_bod only working with only_3dpos enabled')
            dm_shape = (len(bodysplits[0]), len(bodysplits[0]), len(bodysplits) * dm_shape[2])

        if joints_perms is not None:
            num_perms = len(joints_perms)
        dm_size = dm_shape[0] * dm_shape[1] * dm_shape[2]
        if config.cell_model == 'conv3d_variant':
            dm_size *= num_perms

        batch_size = tf.shape(poses)[0]
        max_length = tf.shape(poses)[1]
        n_dims = poses.get_shape().as_list()[3]
        fixed_length = config.pick_num if (config.random_pick or config.rnn_model == 'staged') else \
                       config.crop_len if config.random_crop else None

        def num_elems_triu(n):
            return (((n + 1) * n) / 2) - n

        triu_size = int(num_elems_triu(dm_shape[0]))

        def load_arr(name):
            data_set = 'NTURGBD' if config.skel_transform else config.data_set
            data_set_version = '' if config.skel_transform else config.data_set_version
            file_path = os.path.join(config.data_path, data_set + data_set_version + '_' + name + '.npy')
            arr = None
            with open(file_path, 'rb') as f:
                arr = np.load(f)
                arr[np.isnan(arr)] = 0
            return tf.constant(arr, name=name)

        def one_gaussian(x, mu, sig):
            vals = 1. / (np.sqrt(2. * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.) / 2)
            return vals / np.sum(vals)

        def seq_smooth(poses):
            num_frames = 5
            poses_filter = one_gaussian(np.linspace(-1, 1, num_frames), 0, 0.3)
            poses_filter = np.tile(np.reshape(poses_filter, [num_frames, 1, 1, 1]), [1, 1, n_dims, 1])
            tf_poses_filter = tf.constant(poses_filter, dtype=self._float_type)
            poses = tf.nn.depthwise_conv2d(poses, tf_poses_filter, [1, 1, 1, 1], 'SAME')
            return poses

        if config.seq_smooth:
            if poses_0 is not None and poses_1 is not None:
                poses_0 = seq_smooth(poses_0)
                poses_1 = seq_smooth(poses_1)
            else:
                poses = seq_smooth(poses)

        if config.dup_input and poses_0 is not None and poses_1 is not None:
            if config.data_set == 'NTURGBD' or config.data_set == 'NTURGBD_rtpose':
                to_select = tf.greater(actions, 48)
                poses_1 = tf.where(to_select, poses_1, poses_0)

        if config.swap_input and poses_0 is not None and poses_1 is not None:
            to_select = tf.cast(tf.round(tf.random_uniform([batch_size])),tf.bool)
            new_poses_0 = tf.where(to_select, poses_0, poses_1)
            new_poses_1 = tf.where(to_select, poses_1, poses_0)
            poses_0 = new_poses_0
            poses_1 = new_poses_1

        def identity_initializer(scale, dtype=self._float_type):
            def _initializer(shape, dtype=dtype, partition_info=None):
                return tf.constant((np.eye(shape[0]))+np.random.randn(shape[0],shape[1])*scale, dtype)
            return _initializer

        def norm_skel(poses):
            skel_mean = load_arr('skel_mean')
            skel_std = load_arr('skel_std')
            # return (poses - skel_mean) / skel_std
            return poses / 1000

        if config.norm_skel:
            if poses_0 is not None and poses_1 is not None:
                poses_0 = norm_skel(poses_0)
                poses_1 = norm_skel(poses_1)
            else:
                poses = norm_skel(poses)

        def jitter_height(poses):
            jitter_y = poses[:, :, :, 1] * tf.random_uniform([batch_size, 1, 1], minval=0.7, maxval=1.3, dtype=self._float_type)
            new_poses = tf.concat(
                [tf.expand_dims(poses[:, :, :, 0], 3),
                 tf.expand_dims(jitter_y, 3),
                 poses[:, :, :, 2:]], axis=3)
            return new_poses

        if config.jitter_height and self._is_training:
            if poses_0 is not None and poses_1 is not None:
                poses_0 = jitter_height(poses_0)
                poses_1 = jitter_height(poses_1)
            else:
                poses = jitter_height(poses)

        def sim_occlusions(poses):
            def occluded_poses():
                bodysplits_tf = tf.constant(bodysplits, dtype=self._int_type)
                occ_idcs = tf.random_uniform([batch_size, 1], minval=0, maxval=len(bodysplits), dtype=self._int_type)
                occ_idcs = tf.gather_nd(bodysplits_tf, occ_idcs)
                noise_mask = tf.tile(
                    tf.reshape(
                        tf.cast(tf.reduce_sum(tf.one_hot(occ_idcs, dm_shape[0]), axis=1), dtype=tf.bool),
                        [batch_size, 1, dm_shape[0], 1]),
                    [1, max_length, 1, n_dims])
                noisy_poses = poses * tf.random_uniform([batch_size, max_length, 1, n_dims], minval=0.8, maxval=1.2, dtype=self._float_type)
                return tf.where(noise_mask, noisy_poses, poses)

            occlude_rate = 0.5
            return tf.cond(tf.cast(tf.round(tf.random_uniform([], minval=-0.5, maxval=0.5) + occlude_rate), tf.bool),
                           occluded_poses, lambda: poses)

        if config.sim_occlusions and self._is_training:
            if poses_0 is not None and poses_1 is not None:
                poses_0 = sim_occlusions(poses_0)
                poses_1 = sim_occlusions(poses_1)
            else:
                poses = sim_occlusions(poses)

        def sim_translations(poses):
            trans_factor = 1 / 6

            def translated_poses():
                translation = tf.random_uniform([batch_size, 1, 1, n_dims], minval=1.0 - trans_factor,
                                                maxval=1.0 + trans_factor, dtype=self._float_type)
                translation = tf.tile(translation, [1, max_length, dm_shape[0], 1])
                return poses + translation

            translate_rate = 0.75
            return tf.cond(tf.cast(tf.round(tf.random_uniform([], minval=-0.5, maxval=0.5) + translate_rate), tf.bool),
                           translated_poses, lambda: poses)

        if config.sim_translations and self._is_training:
            if poses_0 is not None and poses_1 is not None:
                poses_0 = sim_translations(poses_0)
                poses_1 = sim_translations(poses_1)
            else:
                raise Exception('sim translations only make sense with two skels')
        
        if config.data_set == 'SBU_inter':
            poses = tf.concat([poses_0, poses_1], axis=2)
            poses_0 = poses_1 = None

        if config.restore_pretrained and dm_shape[2] == 1:
            poses_0 = poses_1 = poses
            dm_shape = (dm_shape[0], dm_shape[1], 2)
            dm_size = dm_shape[0] * dm_shape[1] * dm_shape[2]

        def split_bod(poses):
            poses = tf.transpose(poses, [2, 1, 0, 3])
            bodysplit_l = []
            for b in range(len(bodysplits)):
                indcs = tf.reshape(tf.constant(bodysplits[b], dtype=self._int_type), [len(bodysplits[b]), 1])
                split_poses = tf.gather_nd(poses, indcs)
                split_poses = tf.transpose(split_poses, [2, 1, 0, 3])
                bodysplit_l.append(split_poses)
            return tf.stack(bodysplit_l, axis=2)

        if config.split_bod:
            if poses_0 is not None and poses_1 is not None:
                poses_0 = split_bod(poses_0)
                poses_1 = split_bod(poses_1)
            else:
                poses = split_bod(poses)

        def poses_perms(poses_l, num_perms):
            poses_perms_l_l = []
            for n in range(num_perms):
                shuffle_idcs = np.reshape(joints_perms[n], [dm_shape[0], 1])
                tf_shuffle_idcs = tf.get_variable(
                    "shuffle_idcs_%d" % n, [dm_shape[0], 1],
                    initializer=tf.constant_initializer(shuffle_idcs),
                    dtype=self._int_type, trainable=False
                )
                poses_perms_l = []
                for poses in poses_l:
                    poses = tf.transpose(poses, [2, 1, 0, 3])
                    shuffled_poses = tf.gather_nd(poses, tf_shuffle_idcs)
                    shuffled_poses = tf.transpose(shuffled_poses, [2, 1, 0, 3])
                    poses_perms_l.append(shuffled_poses)
                poses_perms_l_l.append(poses_perms_l)
            return poses_perms_l_l
        
        if not config.joint_permutation is None:
            # permutation = tf.constant(config.joint_permutation, dtype=self._int_type, shape=[dm_shape[0], 1])
            permutation = tf.constant(joints_perms[config.joint_permutation], dtype=self._int_type, shape=[dm_shape[0], 1])

            def permute(poses):
                poses = tf.transpose(poses, [2, 1, 0, 3])
                shuffled_poses = tf.gather_nd(poses, permutation)
                shuffled_poses = tf.transpose(shuffled_poses, [2, 1, 0, 3])
                return shuffled_poses

            if poses_0 is not None and poses_1 is not None:
                poses_0 = permute(poses_0)
                poses_1 = permute(poses_1)
            else:
                poses = permute(poses)
        
        COMB_MATRIX_COLLECTION = 'COMB_MATRIX'
        def learn_comb(poses, reuse=None):
            with tf.variable_scope("learn_comb", reuse=reuse):
                comb_matrix = tf.get_variable(
                    "matrix", [dm_shape[0],dm_shape[1]],
                    initializer=identity_initializer(0.01),
                    dtype=self._float_type, trainable=True
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
                self._comb_matrix_image = (norm_comb_matrix - cb_min) / (cb_max - cb_min) * 255.0
                self._comb_matrix_image = tf.cast(self._comb_matrix_image, tf.uint8)
                self._comb_matrix_image = tf.reshape(self._comb_matrix_image, [1, dm_shape[0], dm_shape[1], 1])
                return poses
        
        def learn_comb_sm(poses, reuse=None):
            with tf.variable_scope("learn_comb", reuse=reuse):
                comb_matrix = tf.get_variable(
                    "matrix", [dm_shape[0],dm_shape[1]],
                    initializer=identity_initializer(0.01),
                    dtype=self._float_type, trainable=True
                )
                norm_comb_matrix = tf.nn.softmax(comb_matrix, dim=0)

                poses = tf.transpose(poses, [0, 1, 3, 2])
                poses = tf.reshape(poses, [batch_size * max_length * n_dims, dm_shape[0]])
                poses = tf.matmul(poses, norm_comb_matrix)
                poses = tf.reshape(poses, [batch_size, max_length, n_dims, dm_shape[0]])
                poses = tf.transpose(poses, [0, 1, 3, 2])
                poses = tf.reshape(poses, [batch_size, max_length, dm_shape[0], n_dims])
                
                self._comb_matrix_image = norm_comb_matrix * 255.0
                self._comb_matrix_image = tf.cast(self._comb_matrix_image, tf.uint8)
                self._comb_matrix_image = tf.reshape(self._comb_matrix_image, [1, dm_shape[0], dm_shape[1], 1])
                return poses
        
        def learn_comb_orth(poses, reuse=None):
            with tf.variable_scope("learn_comb", reuse=reuse):
                comb_matrix = tf.get_variable(
                    "matrix", [dm_shape[0],dm_shape[1]],
                    initializer=identity_initializer(0),
                    dtype=self._float_type, trainable=False
                )
                tf.add_to_collection(COMB_MATRIX_COLLECTION, comb_matrix)
                poses = tf.tensordot(poses, comb_matrix, [[2], [1]])
                poses = tf.transpose(poses, [0, 1, 3, 2])
                
                # Special update code
                def update_comb_mat(grad, lr):
                    A = tf.matmul(tf.transpose(grad), comb_matrix) - \
                        tf.matmul(tf.transpose(comb_matrix), grad)
                    I = tf.constant(np.eye(dm_shape[0]), dtype=self._float_type)
                    t1 = I + lr / 2 * A
                    t2 = I - lr / 2 * A
                    Y = tf.matmul(tf.matmul(tf.matrix_inverse(t1), t2), comb_matrix)
                    return tf.assign(comb_matrix, Y)                
                self._comb_matrix_update = update_comb_mat
                
                # Visualization
                cb_min = tf.reduce_min(comb_matrix)
                cb_max = tf.reduce_max(comb_matrix)
                self._comb_matrix_image = (comb_matrix - cb_min) / (cb_max - cb_min) * 255.0
                self._comb_matrix_image = tf.cast(self._comb_matrix_image, tf.uint8)
                self._comb_matrix_image = tf.reshape(self._comb_matrix_image, [1, dm_shape[0], dm_shape[1], 1])
                
                return poses
        
        def learn_comb_orth_rmsprop(poses, reuse=None):
            with tf.variable_scope("learn_comb", reuse=reuse):
                comb_matrix = tf.get_variable(
                    "matrix", [dm_shape[0], dm_shape[1]],
                    initializer=identity_initializer(0),
                    dtype=self._float_type, trainable=False
                )
                comb_matrix_m = tf.get_variable(
                    "matrix_momentum", [dm_shape[0], dm_shape[1]],
                    initializer=tf.zeros_initializer(),
                    dtype=self._float_type, trainable=False
                )
                tf.add_to_collection(COMB_MATRIX_COLLECTION, comb_matrix)
                poses = tf.tensordot(poses, comb_matrix, [[2], [1]])
                poses = tf.transpose(poses, [0, 1, 3, 2])
                
                # Special update code
                def update_comb_mat(grad, lr):
                    I = tf.constant(np.eye(dm_shape[0]), dtype=self._float_type)
                    
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
                self._comb_matrix_update = update_comb_mat
                
                # Visualization
                cb_min = tf.reduce_min(comb_matrix)
                cb_max = tf.reduce_max(comb_matrix)
                self._comb_matrix_image = (comb_matrix - cb_min) / (cb_max - cb_min) * 255.0
                self._comb_matrix_image = tf.cast(self._comb_matrix_image, tf.uint8)
                self._comb_matrix_image = tf.reshape(self._comb_matrix_image, [1, dm_shape[0], dm_shape[1], 1])
                
                return poses

        def learn_comb_unc(poses, reuse=None):
            with tf.variable_scope("learn_comb", reuse=reuse):
                comb_matrix = tf.get_variable(
                    "matrix", [dm_shape[0],dm_shape[1]],
                    initializer=identity_initializer(0.01),
                    dtype=self._float_type, trainable=True
                )

                poses = tf.transpose(poses, [0, 1, 3, 2])
                poses = tf.reshape(poses, [batch_size * max_length * n_dims, dm_shape[0]])
                poses = tf.matmul(poses, comb_matrix)
                poses = tf.reshape(poses, [batch_size, max_length, n_dims, dm_shape[0]])
                poses = tf.transpose(poses, [0, 1, 3, 2])
                poses = tf.reshape(poses, [batch_size, max_length, dm_shape[0], n_dims])
                
                cb_min = tf.reduce_min(comb_matrix)
                cb_max = tf.reduce_max(comb_matrix)
                self._comb_matrix_image = (comb_matrix - cb_min) / (cb_max - cb_min) * 255.0
                self._comb_matrix_image = tf.cast(self._comb_matrix_image, tf.uint8)
                self._comb_matrix_image = tf.reshape(self._comb_matrix_image, [1, dm_shape[0], dm_shape[1], 1])
                return poses

        def learn_comb_centered(poses, reuse=None):
            with tf.variable_scope("learn_comb", reuse=reuse):
                comb_matrix = tf.get_variable(
                    "matrix", [dm_shape[0],dm_shape[1]],
                    initializer=identity_initializer(0.01),
                    dtype=self._float_type, trainable=True
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
                self._comb_matrix_image = (comb_matrix - cb_min) / (cb_max - cb_min) * 255.0
                self._comb_matrix_image = tf.cast(self._comb_matrix_image, tf.uint8)
                self._comb_matrix_image = tf.reshape(self._comb_matrix_image, [1, dm_shape[0], dm_shape[1], 1])
                return poses

        if config.learn_comb:
            if poses_0 is not None and poses_1 is not None:
                poses_0 = learn_comb(poses_0, None if self._is_training else True)
                poses_1 = learn_comb(poses_1, True)
            else:
                poses = learn_comb(poses, None if self._is_training else True)
        elif config.learn_comb_sm:
            if poses_0 is not None and poses_1 is not None:
                poses_0 = learn_comb_sm(poses_0, None if self._is_training else True)
                poses_1 = learn_comb_sm(poses_1, True)
            else:
                poses = learn_comb_sm(poses, None if self._is_training else True)
        elif config.learn_comb_orth:
            if poses_0 is not None and poses_1 is not None:
                poses_0 = learn_comb_orth(poses_0, None if self._is_training else True)
                poses_1 = learn_comb_orth(poses_1, True)
            else:
                poses = learn_comb_orth(poses, None if self._is_training else True)
        elif config.learn_comb_orth_rmsprop:
            if poses_0 is not None and poses_1 is not None:
                poses_0 = learn_comb_orth_rmsprop(poses_0, None if self._is_training else True)
                poses_1 = learn_comb_orth_rmsprop(poses_1, True)
            else:
                poses = learn_comb_orth_rmsprop(poses, None if self._is_training else True)
        elif config.learn_comb_unc:
            if poses_0 is not None and poses_1 is not None:
                poses_0 = learn_comb_unc(poses_0, None if self._is_training else True)
                poses_1 = learn_comb_unc(poses_1, True)
            else:
                poses = learn_comb_unc(poses, None if self._is_training else True)
        elif config.learn_comb_centered:
            if poses_0 is not None and poses_1 is not None:
                poses_0 = learn_comb_centered(poses_0, None if self._is_training else True)
                poses_1 = learn_comb_centered(poses_1, True)
            else:
                poses = learn_comb_centered(poses, None if self._is_training else True)

        embedding = embeddings = embeddings_e = None
        ## Distance Matrix Embedding
        with tf.name_scope("dm_emb"):
            def dm_emb(poses, consider_conf, init_vars=True):
                embedding = distance_matrix_embedding(poses, consider_conf)
                if config.dm_transform:
                    emb = tf.reshape(embedding, [batch_size * max_length, dm_shape[0], dm_shape[1], 1])
                    emb_mean = tf.constant(np.reshape(np.load('data/net-epoch-922-v73_mean.npy'),[1,dm_shape[0],dm_shape[1],1]), self._float_type)
                    emb_mult = tf.constant(np.load('data/net-epoch-922-v73_range_multiplier.npy'), self._float_type)
                    emb = (tf.sqrt(emb) - emb_mean) * emb_mult
                    reuse = None if self._is_training and init_vars else True
                    emb = matconvnet2tf.import_net('data/net-epoch-1486-v73.mat', emb, reuse=reuse)
                    embedding = tf.reshape(emb, [batch_size, max_length, dm_shape[0], dm_shape[1], 1])
                return embedding

            if config.cell_model == 'conv3d_variant':
                embeddings_e = []
                if poses_0 is not None and poses_1 is not None:
                    poses_perms = poses_perms([poses_0, poses_1], num_perms)
                else:
                    poses_perms = poses_perms([poses], num_perms)
                for p in range(num_perms):
                    embeddings = []
                    embeddings.append(dm_emb(poses_perms[p][0], consider_conf))
                    if len(poses_perms[p]) > 1:
                        embeddings.append(dm_emb(poses_perms[p][1], consider_conf, False))
                    embeddings_e.append(embeddings)
            elif poses_0 is not None and poses_1 is not None:
                embeddings = []
                embeddings.append(dm_emb(poses_0, consider_conf))
                embeddings.append(dm_emb(poses_1, consider_conf, False))
            else:
                embedding = dm_emb(poses, consider_conf)

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

            def norm_dms_px(emb):
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

            if config.norm_dms_ch:
                assert not config.split_bod, print('norm_dms_ch incompatible with split_bod')
                if embedding is None:
                    embeddings[0] = norm_dms_ch(embeddings[0])
                    embeddings[1] = norm_dms_ch(embeddings[1])
                else:
                    embedding = norm_dms_ch(embedding)
            elif config.norm_dms_px:
                assert not config.split_bod, print('norm_dms_px incompatible with split_bod')
                if embedding is None:
                    embeddings[0] = norm_dms_px(embeddings[0])
                    embeddings[1] = norm_dms_px(embeddings[1])
                else:
                    embedding = norm_dms_px(embedding)
            elif config.norm_dms_bn:
                if embedding is None:
                    embeddings[0] = tf.contrib.layers.batch_norm(
                        embeddings[0], center=False, scale=False,
                        updates_collections=None, fused=False,
                        scope='dm_bn', is_training=self._is_training)
                    embeddings[1] = tf.contrib.layers.batch_norm(
                        embeddings[1], center=False, scale=False,
                        updates_collections=None, reuse=True,
                        fused=False, scope='dm_bn', is_training=self._is_training)
                else:
                    embedding = tf.contrib.layers.batch_norm(
                        embedding, center=False, scale=False,
                        updates_collections=None, fused=False,
                        scope='dm_bn', is_training=self._is_training)

            if embedding is None and config.rnn_model != 'siamese':
                if config.cell_model == 'conv3d_variant' and len(embeddings_e[0]) > 1:
                    embeddings_tmp = embeddings_e
                    embeddings_e = []
                    for emb in embeddings_tmp:
                        embeddings_e.append(tf.concat(emb, axis=4))
                embedding = tf.concat(embeddings, axis=4)

            if not (config.rnn_model[0:6] == 'conv3d' or config.rnn_model[0:6] == 'conv2d'):
                # warning, gotcha: rnn_model != cell_model. See base_config for more info
                if config.cell_model[0:4] == 'conv':
                    """ Reshaping the distance matrices to 1d vectors so they can work with TF standard rnn,
                        they will be internally reshaped back to matrices in the conv cells """
                    if config.rnn_model == 'siamese':
                        # Siamese model uses split dms for the two skeletons
                        embeddings[0] = tf.reshape(embeddings[0], [batch_size, max_length, dm_size//2])
                        embeddings[1] = tf.reshape(embeddings[1], [batch_size, max_length, dm_size//2])
                    else:
                        if config.cell_model == 'conv3d_variant':
                            # conv3d cell requires stacking by the different permutations... dangerous
                            print(embeddings_e[0].get_shape())
                            embedding = tf.stack(embeddings_e, axis=2)
                            print(embedding.get_shape())
                        embedding = tf.reshape(embedding, [batch_size, max_length, dm_size])
                else:
                    """ If the cells are linear, we are better of just taking the upper triangular portion of the dm
                        discarding redundant information """
                    embedding = tf.reshape(triu_layer(embedding), [batch_size, max_length, triu_size * dm_shape[2]])
                    if config.cell_model == 'phlstm':
                        times = tf.cast(tf.range(0, max_length), self._float_type)
                        times = tf.expand_dims(tf.expand_dims(times, 1), 0)
                        times = tf.tile(times, [batch_size, 1, hidden_size])
                        embedding = tf.concat([embedding, times], axis=2)
            elif config.rnn_model[0:6] == 'conv2d':
                embedding = tf.reshape(embedding, [batch_size, max_length, dm_shape[0] * dm_shape[1], dm_shape[2]])
                # embedding = tf.reshape(triu_layer(embedding), [batch_size, max_length, triu_size, dm_shape[2]])

        # TODO: IMPROVE THIS UGLY WORKAROUND
        if config.no_dm:
            embedding = tf.reshape(poses, [batch_size, max_length, dm_shape[0] * n_dims])

        def create_cell(dm_size):
            lstm_units = hidden_size
            cell = None
            if config.cell_model == 'lstm':
                lstm_cell = tf.contrib.rnn.LSTMCell(lstm_units)
            elif config.cell_model == 'bnlstm':
                lstm_cell = BNLSTMCell(lstm_units, fixed_length, self._is_training)
            elif config.cell_model == 'phlstm':
                lstm_cell = PhasedLSTMCell(lstm_units, self._is_training)
            elif config.cell_model == 'convlstm':
                cell = []
                num_in = dm_shape[2] // 2 if config.rnn_model == 'siamese' else dm_shape[2]
                for c in range(config.num_layers):
                    conv_cell = ConvLSTMCell(
                        in_shape=(dm_shape[0], dm_shape[1]),
                        filter_size=[3, 3],
                        num_in_ch=num_in,
                        num_out_ch=dm_shape[2],
                        max_pool=False,
                        activation=tf.nn.tanh,
                        batch_norm=config.batch_norm,
                        pres_ident=config.pres_ident,
                        is_training=self._is_training,
                        max_length=fixed_length,
                        # keep_prob=config.keep_prob if self._is_training else 1.0
                    )
                    num_in = dm_shape[2]
                    cell.append(conv_cell)
            elif config.cell_model == 'convphlstm':
                cell = []
                num_in = dm_shape[2] // 2 if config.rnn_model == 'siamese' else dm_shape[2]
                for c in range(config.num_layers):
                    conv_cell = ConvPhasedLSTMCell(
                        shape=(dm_shape[0], dm_shape[1]),
                        filter_size=[3, 3],
                        num_in_ch=num_in,
                        num_out_ch=dm_shape[2],
                        is_training=self._is_training,
                        activation=tf.nn.tanh,
                        tau_init=np.log(input_.max_plen / (2 * config.pick_num)),
                        r_on_init=1e-1
                    )
                    num_in = dm_shape[2]
                    cell.append(conv_cell)
            elif config.cell_model == 'conv_variant':
                cell = []
                conv_specs = config.conv_specs
                conv_shape = (dm_shape[0],dm_shape[1])
                num_in = dm_shape[2] // 2 if config.rnn_model == 'siamese' else dm_shape[2]
                for c in range(len(conv_specs)):
                    conv_cell = ConvLSTMCell(
                        in_shape=conv_shape,
                        filter_size=[3, 3],
                        num_in_ch=num_in,
                        num_out_ch=conv_specs[c]['num_out'],
                        max_pool=conv_specs[c]['max_pool'],
                        activation=tf.nn.tanh,
                        batch_norm=config.batch_norm,
                        pres_ident=config.pres_ident,
                        is_training=self._is_training,
                        max_length=fixed_length,
                        # keep_prob=config.keep_prob if self._is_training else 1.0
                        new_pool=config.new_pool
                    )
                    num_in = conv_specs[c]['num_out']
                    cell.append(conv_cell)
                    conv_shape = conv_cell.out_shape
                    dm_size = conv_cell.output_size
            elif config.cell_model == 'conv3d_variant':
                cell = []
                conv_specs = config.conv_specs
                conv_shape = (num_perms, dm_shape[0], dm_shape[1])
                num_in = dm_shape[2] // 2 if config.rnn_model == 'siamese' else dm_shape[2]
                for c in range(len(conv_specs)):
                    conv_cell = Conv3dLSTMCell(
                        in_shape=conv_shape,
                        filter_size=[3, 3, 3],
                        num_in_ch=num_in,
                        num_out_ch=conv_specs[c]['num_out'],
                        max_pool=conv_specs[c]['max_pool'],
                        activation=tf.nn.tanh,
                        batch_norm=config.batch_norm,
                        pres_ident=config.pres_ident,
                        is_training=self._is_training,
                        max_length=fixed_length
                    )
                    num_in = conv_specs[c]['num_out']
                    cell.append(conv_cell)
                    conv_shape = conv_cell.out_shape
                    dm_size = conv_cell.output_size

            if cell is None:
                cell = [lstm_cell] * config.num_layers

            cell = tf.contrib.rnn.MultiRNNCell(
                cell, state_is_tuple=True
            )
            initial_state = cell.zero_state(batch_size, self._float_type)
            return cell, initial_state, dm_size

        def conv_to_linear(outputs, max_length, dm_size, scope):
            if (config.keep_prob < 1.0) and self._is_training:
                outputs = tf.nn.dropout(outputs, config.keep_prob)
            r_outputs = tf.reshape(outputs, [batch_size * max_length, dm_size])
            # l_outputs = linear_layer(r_outputs, dm_size, hidden_size, scope="conv_to_linear"+scope)
            l_outputs = tf.layers.dense(r_outputs, hidden_size,
                kernel_regularizer=tf.contrib.layers.l2_regularizer(5.0e-4),
                kernel_initializer=tf.contrib.layers.xavier_initializer(False))
            return tf.reshape(tf.nn.elu(l_outputs), [batch_size, max_length, hidden_size])

        outputs = state = None
        if config.rnn_model == 'standard':
            with tf.variable_scope('standard_rnn') as scope:
                cell, self._initial_state, dm_size = create_cell(dm_size)
                outputs, state = tf.nn.dynamic_rnn(
                    cell,
                    embedding,
                    sequence_length=plens,
                    initial_state=self._initial_state,
                    parallel_iterations=4
                )
                self._rnn_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)
        elif config.rnn_model == 'bidirectional':
            with tf.variable_scope('bidirectional_rnn') as scope:
                cell_fw, initial_state_fw, dm_size = create_cell(dm_size)
                cell_bw, initial_state_bw, dm_size = create_cell(dm_size)
                outputs, state = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw,
                    embedding,
                    sequence_length=plens,
                    initial_state_fw=initial_state_fw, initial_state_bw=initial_state_bw,
                    parallel_iterations=4
                )
                outputs = tf.reduce_mean(tf.concat([tf.expand_dims(outputs[0], 3), tf.expand_dims(outputs[1], 3)], axis=3), axis=3)
                self._rnn_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)
        elif config.rnn_model == 'siamese':
            with tf.variable_scope('siamese_rnn') as scope:
                cell, self._initial_state, dm_size = create_cell(dm_size)
                outputs_s = []
                state_s = []
                for embedding in embeddings:
                    if self._is_training and len(outputs_s) == 1:
                        scope.reuse_variables()
                    outputs, state = tf.nn.dynamic_rnn(
                        cell,
                        embedding,
                        sequence_length=plens,
                        initial_state=self._initial_state,
                        parallel_iterations=4
                    )
                    outputs_s.append(outputs)
                    state_s.append(state)
                outputs = tf.concat(outputs_s, axis=2)
                state = state_s
                dm_size = dm_size*2
                self._rnn_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)
        elif config.rnn_model == 'staged':
            with tf.variable_scope('staged_rnn') as scope:
                cell = ConvLSTMCell(
                    in_shape=(dm_shape[0], dm_shape[1]),
                    filter_size=[3, 3],
                    num_in_ch=dm_shape[2],
                    num_out_ch=dm_shape[2],
                    max_pool=False,
                    activation=tf.nn.tanh,
                    batch_norm=config.batch_norm,
                    pres_ident=False,
                    is_training=self._is_training,
                    max_length=0,
                    keep_prob=config.keep_prob if self._is_training else 1.0
                )
                initial_state = cell.zero_state(batch_size, self._float_type)
                outputs, _ = tf.nn.dynamic_rnn(
                    cell,
                    embedding,
                    sequence_length=plens,
                    initial_state=initial_state,
                    parallel_iterations=4
                )
                outputs_l = tf.split(outputs, config.pick_num, axis=1)
                inputs_pool_l = []
                for outputs_split in outputs_l:
                    inputs_pool_l.append(tf.reduce_mean(outputs_split, axis=1))
                inputs = tf.stack(inputs_pool_l, axis=1)
                max_length = config.pick_num
                cell, self._initial_state, dm_size = create_cell(dm_size)
                outputs, state = tf.nn.dynamic_rnn(
                    cell,
                    inputs,
                    sequence_length=plens,
                    initial_state=self._initial_state,
                    parallel_iterations=4
                )
                self._rnn_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)
        elif config.rnn_model == 'conv3d_resnet':
            with tf.variable_scope('conv3d_resnet') as scope:
                outputs = resnet.resnet(
                    embedding, config.num_layers, config.resnet_blocks,
                    self._is_training, '3D')
                max_length = 1
                self._rnn_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)
        elif config.rnn_model == 'conv3d_resnext':
            with tf.variable_scope('conv3d_resnext') as scope:
                outputs = resnet.resnext(
                    embedding, config.num_layers, config.resnet_blocks,
                    self._is_training, '3D')
                max_length = 1
                self._rnn_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)
        elif config.rnn_model == 'conv2d_resnext':
            with tf.variable_scope('conv2d_resnext') as scope:
                outputs = resnet.resnext(
                    embedding, config.num_layers, config.resnet_blocks,
                    self._is_training, '2D')
                max_length = 1
                self._rnn_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)

        if config.cell_model == 'phlstm':
            outputs = outputs[:, :, :hidden_size]
        elif config.cell_model == 'convlstm' or config.cell_model == 'convphlstm' or config.cell_model == 'conv_variant' or config.cell_model == 'conv3d_variant':
            outputs = conv_to_linear(outputs, max_length, dm_size, "")

        self._outputs = outputs
        self._final_state = state
        key_logits_act = "KEY_LOGITS_ACT"
        key_logits_sub = "KEY_LOGITS_SUB"
        plens = tf.cast(plens, self._float_type)
        if config.loss_model == 'mean_pool':
            mean_pool = None
            if config.rnn_model[:4] == 'conv':
                if config.keep_prob < 1:
                    outputs = tf.layers.dropout(outputs, rate=(1 - config.keep_prob), training=self._is_training)
                logits = tf.layers.dense(outputs, config.num_actions,
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(5.0e-4),
                                         kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
                self.logits = logits
                self.labels = actions
                self.idxs = idxs
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits,
                    labels=actions
                )
            else:
                mean_pool = tf.reduce_sum(outputs, 1) / tf.expand_dims(plens, 1)
                if (config.keep_prob < 1.0) and self._is_training:
                    mean_pool = tf.nn.dropout(mean_pool, config.keep_prob)
                # logits = linear_layer(
                #     mean_pool, hidden_size,
                #     config.num_actions, scope="logits", collections=key_logits_act
                # )
                logits = tf.layers.dense(mean_pool, config.num_actions,
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(5.0e-4),
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(False))
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits,
                    labels=actions
                )
            self._prediction = tf.cast(tf.argmax(tf.nn.softmax(logits), 1), self._int_type)
            self._mistakes = tf.logical_not(
                tf.equal(self._prediction, actions)
            )
            if config.sub_loss:
                logits_sub = linear_layer(
                    mean_pool, hidden_size,
                    config.num_subjects, scope="logits_sub", collections=key_logits_sub
                )
                loss_sub = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits_sub,
                    labels=subjects
                )
        elif config.loss_model == 'class_per_frame':
            logits = linear_layer(
                tf.reshape(outputs, [-1, hidden_size]), hidden_size,
                config.num_actions, scope="logits"
            )
            logits = tf.reshape(
                logits, [batch_size, max_length, config.num_actions]
            )
            actions_tiled = tf.tile(
                tf.expand_dims(actions, 1), [1, max_length]
            )
            range_v = tf.tile(
                tf.expand_dims(tf.range(max_length), 0),
                [batch_size, 1]
            )
            plens_tiled = tf.tile(
                tf.expand_dims(plens, 1),
                [1, max_length]
            )
            mask = tf.cast(
                tf.greater(
                    plens_tiled,
                    tf.cast(range_v, self._float_type),
                    name="mask"
                ),
                self._int_type
            )
            actions_tiled = actions_tiled * mask
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits,
                labels=actions_tiled
            )
            loss = tf.reduce_sum(loss, 1) / plens
            m_softmax = tf.reduce_sum(tf.nn.softmax(logits), 1) / tf.expand_dims(plens, 1)
            self._prediction = tf.cast(
                    tf.argmax(m_softmax, 1),
                    self._int_type
            )
            self._mistakes = tf.logical_not(
                tf.equal(self._prediction, actions)
            )
        elif config.loss_model == 'last_n_outs':
            mean_pool = tf.reduce_sum(outputs[:,-32:], 1) / tf.expand_dims(plens, 1)
            logits = linear_layer(
                mean_pool, hidden_size,
                config.num_actions, scope="logits"
            )
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits,
                labels=actions
            )
            self._prediction = tf.cast(tf.argmax(tf.nn.softmax(logits),1), self._int_type)
            self._mistakes = tf.logical_not(
                tf.equal(self._prediction, actions)
            )

        self._loss = tf.reduce_mean(loss)
        self._accuracy = tf.contrib.metrics.accuracy(self._prediction, actions)

        confusion_labels = np.arange(config.num_actions)
        _, _, count = tf.unique_with_counts(
            tf.concat([tf.constant(confusion_labels, dtype=self._int_type),
                       tf.boolean_mask(actions, self._mistakes)], axis=0))
        self._mistakes_per_class = tf.reshape(
            tf.cast(((count - 1) * 255) // tf.reduce_max(count), tf.uint8),
            [1, 1, config.num_actions, 1])

        conf_mat = tf.confusion_matrix(actions, self._prediction, num_classes=config.num_actions)

        self._confusion_matrix = tf.reshape(
            tf.cast((conf_mat * 255) // tf.reduce_max(conf_mat), tf.uint8),
            [1, config.num_actions, config.num_actions, 1])

        self._mistake_labs = tf.concat(
            [tf.expand_dims(tf.boolean_mask(idxs, self._mistakes), 1),
             tf.expand_dims(tf.boolean_mask(subjects, self._mistakes), 1),
             tf.expand_dims(tf.boolean_mask(actions, self._mistakes), 1),
             tf.expand_dims(tf.boolean_mask(tf.cast(plens, self._int_type), self._mistakes), 1)], axis=1)

        if not self._is_training:
            return

        if config.custom_lr:
            g_step = tf.contrib.framework.get_or_create_global_step()
            self._lr = tf.constant(config.learning_rate)
            for step, lr in config.custom_lr_list:
                self._lr = tf.cond(g_step >= step, lambda: tf.constant(lr), lambda: self._lr)
        elif config.lr_decay:
            self._lr = tf.train.exponential_decay(
                config.learning_rate, tf.contrib.framework.get_or_create_global_step(),
                config.decay_steps, config.decay_rate, staircase=True)
        else:
            self._lr = tf.constant(config.learning_rate)

        if config.sub_loss:
            tvars_act = tf.get_collection(key_logits_act)
            grads_act = tf.gradients(loss, [mean_pool] + tvars_act)

            tvars_sub = tf.get_collection(key_logits_sub)
            grads_sub = tf.gradients(loss_sub, [mean_pool] + tvars_sub)

            tvars = list(set(tf.trainable_variables()) - set(tvars_act) - set(tvars_sub))

            if config.grad_clipping:
                grads_act, _ = tf.clip_by_global_norm(
                    grads_act, config.max_grad_norm
                )
                grads_sub, _ = tf.clip_by_global_norm(
                    grads_sub, config.max_grad_norm
                )
            a_fact = 0.5
            grads = tf.gradients(mean_pool, tvars, grad_ys=(grads_act[0]-(grads_sub[0]*a_fact)))
        else:
            if config.restore_pretrained:
                rnn_vars_names = set([var.name for var in self._rnn_vars])
                tvars_low_lr = [var for var in tf.trainable_variables() if var.name in rnn_vars_names]
                tvars_high_lr = [var for var in tf.trainable_variables() if var.name not in rnn_vars_names]

                if config.learn_comb_orth or config.learn_comb_orth_rmsprop:
                    cb = tf.get_collection(COMB_MATRIX_COLLECTION)
                    tvars_high_lr = cb + tvars_high_lr

                grads = tf.gradients(loss, tvars_low_lr + tvars_high_lr)
                grads_low_lr = grads[:len(tvars_low_lr)]
                grads_high_lr = grads[len(tvars_low_lr):]

                if config.learn_comb_orth or config.learn_comb_orth_rmsprop:
                    cb_grad = grads_high_lr[0]
                    grads_high_lr = grads_high_lr[1:]
                    tvars_high_lr = tvars_high_lr[1:]
            else:
                tvars = tf.trainable_variables()
                if config.learn_comb_orth or config.learn_comb_orth_rmsprop:
                    cb = tf.get_collection(COMB_MATRIX_COLLECTION)
                    tvars = cb + tvars

                if config.curriculum_l:
                    w_loss = input_.weight_losses(idxs, loss)
                    grads = tf.gradients(w_loss, tvars)
                else:
                    grads = tf.gradients([loss] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), tvars)

                if config.learn_comb_orth or config.learn_comb_orth_rmsprop:
                    cb_grad = grads[0]
                    grads = grads[1:]
                    tvars = tvars[1:]

        if config.grad_clipping:
            grads, _ = tf.clip_by_global_norm(
                grads, config.max_grad_norm
            )
        if config.restore_pretrained:
            optimizer_llr = tf.train.AdamOptimizer(self._lr * 1e-1)
            optimizer_hlr = tf.train.AdamOptimizer(self._lr)
        else:
            optimizer = tf.train.AdamOptimizer(self._lr)
        if config.sub_loss:
            train_op = optimizer.apply_gradients(
                zip(grads + grads_act[1:] + grads_sub[1:], tvars + tvars_act + tvars_sub),
                global_step=tf.contrib.framework.get_or_create_global_step()
            )
            self._train_op = train_op
        else:
            if config.restore_pretrained:
                train_op_llr = optimizer_llr.apply_gradients(
                    zip(grads_low_lr, tvars_low_lr),
                    global_step=None
                )
                train_op_hlr = optimizer_hlr.apply_gradients(
                    zip(grads_high_lr, tvars_high_lr),
                    global_step=tf.contrib.framework.get_or_create_global_step()
                )
                train_op = [train_op_llr, train_op_hlr]
            else:
                train_op = optimizer.apply_gradients(
                    zip(grads, tvars),
                    global_step=tf.contrib.framework.get_or_create_global_step()
                )

            if config.curriculum_l:
                with tf.control_dependencies(
                        [input_.update_diff(
                            self._accuracy, idxs, self._loss, plens
                        )]
                ):
                    self._train_op = train_op
            else:
                if config.learn_comb_orth or config.learn_comb_orth_rmsprop:
                    train_op = [train_op, self._comb_matrix_update(cb_grad, self._lr)]
                self._train_op = train_op

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def final_state(self):
        return self._final_state

    @property
    def loss(self):
        return self._loss

    @property
    def predictions(self):
        return self._predictions

    @property
    def mistakes(self):
        return self._mistakes

    @property
    def mistakes_per_class(self):
        return self._mistakes_per_class

    @property
    def mistake_labs(self):
        return self._mistake_labs

    @property
    def confusion_matrix(self):
        return self._confusion_matrix

    @property
    def accuracy(self):
        return self._accuracy

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op

    @property
    def rnn_vars(self):
        return self._rnn_vars

    @property
    def comb_matrix_image(self):
        return self._comb_matrix_image
