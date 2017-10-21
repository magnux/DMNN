from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np
import h5py as h5
import os
from glob import glob
from tqdm import trange


class DmnnInput(object):
    """The input data."""
    def __init__(self, config):
        self.data_path = config.data_path
        self.data_set = config.data_set
        self.batch_size = config.batch_size
        self.random_crop = config.random_crop
        self.crop_len = config.crop_len
        self.random_pick = config.random_pick
        self.pick_num = config.pick_num
        self.data_source = config.data_source
        self.curriculum_l = config.curriculum_l
        self.only_val = config.only_val
        self.only_3dpos = config.only_3dpos
        self.data_set_version = config.data_set_version
        self._int_type = tf.int16 if config.use_type16 else tf.int32
        self._float_type = tf.float16 if config.use_type16 else tf.float32

        if self.data_source == 'tfrecord':
            self.train_keys = [file for file in
                               glob(os.path.join(self.data_path, self.data_set + "_train_shard*.tf"))]
            self.val_keys = [file for file in
                             glob(os.path.join(self.data_path, self.data_set + "_val_shard*.tf"))]

            def num_records(keys):
                r = 0
                for fn in keys:
                    for _ in tf.python_io.tf_record_iterator(fn):
                        r += 1
                return r

            self.len_train_keys = num_records(self.train_keys)
            self.len_val_keys = num_records(self.val_keys)

        else:
            file_path = os.path.join(self.data_path, self.data_set + self.data_set_version + '.h5')
            self.h5file = h5.File(file_path, 'r')
            self.train_keys = [self.data_set + '/Train/' + k
                               for k in self.h5file.get(self.data_set + '/Train').keys()]
            self.val_keys = [self.data_set + '/Validate/' + k
                             for k in self.h5file.get(self.data_set + '/Validate').keys()]

            # if self.data_source == 'ram':
            #     self.train_keys = self.train_keys[0:int(len(self.train_keys)/4)]
            #     self.val_keys = self.val_keys[0:int(len(self.val_keys)/4)]

            self.len_train_keys = len(self.train_keys)
            self.len_val_keys = len(self.val_keys)

        self.train_epoch_size = (self.len_train_keys // self.batch_size) + 1
        self.val_epoch_size = (self.len_val_keys // self.batch_size) + 1

        self.pshape = [config.njoints, 3 if (config.data_set != 'NTURGBD')
                                             or self.only_3dpos else 7, None]
        self.max_plen = config.max_plen

        self.pshape[2] = self.pick_num if self.random_pick else (self.crop_len if self.random_crop else None)

        if self.curriculum_l:
            with tf.variable_scope("Input"):
                self.seq_entropy = tf.Variable(
                                        np.ones(self.len_train_keys),
                                        name="seq_entropy",
                                        trainable=False,
                                        dtype=self._float_type
                                    )
                self.acc_coef = tf.Variable(
                                        0.0,
                                        name="time_coef",
                                        trainable=False,
                                        dtype=self._float_type
                                    )

        if self.data_source == 'ram':
            if not self.only_val:
                self.train_batches = self.pre_comp_batches(True)
            self.val_batches = self.pre_comp_batches(False)

    def pre_comp_batches(self, is_training):
        epoch_size = self.train_epoch_size if is_training else self.val_epoch_size
        labs, poses = self.load_to_ram(is_training)

        batches = []
        for slice_idx in range(epoch_size):
            slice_start = slice_idx * self.batch_size
            slice_len = min(slice_start + self.batch_size, np.shape(labs)[0])
            labs_batch = labs[slice_start:slice_len, :]
            poses_batch = poses[slice_start:slice_len, :, :, :]
            batches.append((labs_batch, poses_batch))

        del labs
        del poses

        return batches

    def load_to_ram(self, is_training):
        len_keys = self.len_train_keys if is_training else self.len_val_keys
        labs = np.empty([len_keys, 4], dtype=np.int32)
        poses = np.empty([len_keys,self.pshape[0],self.pshape[1],self.max_plen], dtype=np.float32)
        random_crop_bkp = self.random_crop
        random_pick_bkp = self.random_pick
        self.random_crop = False
        self.random_pick = False
        splitname = 'train' if is_training else 'val'
        print('Loading "%s" data to ram...' % splitname)
        t = trange(len_keys, dynamic_ncols=True)
        for k in t:
            key_idx, subject, action, pose, plen = self.read_h5_data(k, is_training)
            pose = pose[:, :, :self.max_plen] if plen > self.max_plen else pose
            plen = self.max_plen if plen > self.max_plen else plen
            labs[k, :] = [key_idx, subject, action, plen]
            poses[k, :, :, :plen] = pose
        self.random_crop = random_crop_bkp
        self.random_pick = random_pick_bkp

        return labs, poses

    def generate_batch(self, is_training):
        splitname = 'train' if is_training else 'val'
        keys = self.train_keys if is_training else self.val_keys
        len_keys = self.len_train_keys if is_training else self.len_val_keys
        epoch_size = self.train_epoch_size if is_training else self.val_epoch_size
        num_readers = 6 if is_training else 2
        with tf.variable_scope('input_'+splitname):
            if self.data_source == 'hdf5':
                producer = tf.train.range_input_producer(
                    len_keys,
                    num_epochs=None if not self.only_val else 1,
                    shuffle=True if not self.only_val else False,
                    capacity=self.batch_size
                )

                key_idx = producer.dequeue()

                data = tf.py_func(
                    self.read_h5_data,
                    [key_idx, is_training],
                    [self._int_type, self._int_type, self._int_type, self._float_type, self._int_type]
                )

                return tf.train.batch(
                    tensors=data,
                    batch_size=self.batch_size,
                    capacity=self.batch_size * 4,
                    dynamic_pad=not (self.random_crop or self.random_pick),
                    shapes=([], [], [], self.pshape, []),
                    num_threads=1
                )

            elif self.data_source == 'tfrecord':
                producer = tf.train.string_input_producer(
                    keys,
                    num_epochs=None,
                    shuffle=True,
                    seed=None,
                    capacity=self.batch_size
                )
                features = {
                    "key_idx": tf.FixedLenFeature([], dtype=tf.int64),
                    "subject": tf.FixedLenFeature([], dtype=tf.int64),
                    "action": tf.FixedLenFeature([], dtype=tf.int64),
                    "plen": tf.FixedLenFeature([], dtype=tf.int64),
                    "pose": tf.VarLenFeature(dtype=self._float_type),
                }

                enqueue_ops = []
                examples_queue = tf.FIFOQueue(capacity=self.batch_size * num_readers, dtypes=[tf.string])
                for _ in range(num_readers):
                    reader = tf.TFRecordReader(name=None)
                    _, value = reader.read(producer)
                    enqueue_ops.append(examples_queue.enqueue([value]))

                tf.train.queue_runner.add_queue_runner(
                    tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))

                example = tf.parse_single_example(
                    examples_queue.dequeue(), features
                )
                data_l = []
                for _ in range(num_readers):
                    pose = tf.reshape(tf.sparse_tensor_to_dense(example["pose"]),
                                      shape=[self.pshape[0], self.pshape[1], -1])
                    plen = tf.cast(example["plen"], self._int_type)
                    pose, plen = self.process_pose_tf(pose, plen)
                    data = (tf.cast(example["key_idx"], self._int_type),
                            tf.cast(example["subject"] - 1, self._int_type),  # Small hack to reindex the classes from 0
                            tf.cast(example["action"] - 1, self._int_type),  # Small hack to reindex the classes from 0
                            pose, plen)
                    data_l.append(data)

                return tf.train.batch_join(
                    tensors_list=data_l,
                    batch_size=self.batch_size,
                    capacity=self.batch_size * num_readers,
                    dynamic_pad=not (self.random_crop or self.random_pick),
                    shapes=([], [], [], self.pshape, [])
                )

            elif self.data_source == 'ram':
                producer = tf.train.range_input_producer(
                    epoch_size,
                    num_epochs=None,
                    shuffle=not self.only_val,
                    capacity=epoch_size
                )

                slice_idx = producer.dequeue()
                labs_batch, poses_batch = tf.py_func(
                    self.get_ram_batch,
                    [slice_idx, is_training],
                    [self._int_type, self._float_type]
                )

                examples_queue = tf.FIFOQueue(capacity=self.batch_size * num_readers, dtypes=[self._int_type, self._float_type])
                enqueue_op = examples_queue.enqueue_many([labs_batch, poses_batch])

                tf.train.queue_runner.add_queue_runner(
                    tf.train.queue_runner.QueueRunner(examples_queue, [enqueue_op]))

                lab, pose = examples_queue.dequeue()

                key_idx = lab[0]
                subject = lab[1]
                action = lab[2]
                plen = lab[3]
                pose, plen = self.process_pose_tf(pose, plen)
                data = (key_idx, subject, action, pose, plen)

                dynamic_pad_needed = not (self.random_crop or self.random_pick)
                if self.only_val or dynamic_pad_needed:
                    return tf.train.batch(
                        tensors=data,
                        batch_size=self.batch_size,
                        capacity=self.batch_size * num_readers,
                        dynamic_pad=dynamic_pad_needed,
                        shapes=([], [], [], self.pshape, []),
                        num_threads=num_readers,
                        allow_smaller_final_batch=self.only_val
                    )
                else:
                    return tf.train.shuffle_batch(
                        tensors=data,
                        batch_size=self.batch_size,
                        capacity=self.batch_size * num_readers,
                        min_after_dequeue=self.batch_size * num_readers // 2,
                        shapes=([], [], [], self.pshape, []),
                        num_threads=num_readers
                    )

    def read_h5_data(self, key_idx, is_training):
        if is_training:
            key = self.train_keys[key_idx]
        else:
            key = self.val_keys[key_idx]

        subject = np.int32(self.h5file[key+'/Subject']) - 1  # Small hack to reindex the classes from 0
        action = np.int32(self.h5file[key+'/Action']) - 1  # Small hack to reindex the classes from 0
        pose = np.array(self.h5file[key+'/Pose'], dtype=np.float32)

        pose, plen = self.process_pose(pose)

        return key_idx, subject, action, pose, plen

    def process_pose(self, pose, plen=None):
        plen = np.int32(np.size(pose, 2)) if plen is None else plen
        if self.data_set == 'NTURGBD':
            if self.only_3dpos:
                pose = pose[:, :3, :]
        elif self.data_set == 'SBU_inter':
            m_fact = np.reshape(np.array([1280, 960, 0]), [1, 3, 1])
            p_fact = np.reshape(np.array([2560, 1920, 1280]), [1, 3, 1])
            pose = m_fact - (pose * p_fact)
            pose /= 1000
        elif self.data_set == 'UWA3DII':
            pose[np.isnan(pose)] = 0
            pose /= 1000
        elif self.data_set == 'NUCLA':
            pose[np.isnan(pose)] = 0
        elif self.data_set == 'MSRC12':
            pose = pose[:, :3, :]
            pose[np.isnan(pose)] = 0

        def pad_pose():
            pad_len = self.crop_len if self.random_crop else self.pick_num
            padpose = np.zeros((np.size(pose, 0), np.size(pose, 1), pad_len), dtype=np.float32)
            padpose[:, :, :plen] = pose
            return padpose

        if self.random_crop:
            if self.crop_len > plen:
                pose = pad_pose()
            elif self.crop_len < plen:
                indx = np.random.randint(0, plen - self.crop_len)
                pose = pose[:, :, indx:indx + self.crop_len]
            plen = np.int32(self.crop_len)
        if self.random_pick:
            if self.pick_num > plen:
                pose = pad_pose()
            elif self.pick_num < plen:
                subplen = plen / self.pick_num
                picks = np.random.randint(0, subplen, size=(self.pick_num)) + \
                        np.arange(0, plen, subplen, dtype=np.int32)
                pose = pose[:, :, picks]
            plen = np.int32(self.pick_num)

        return pose, plen

    def process_pose_tf(self, pose, plen):
        tf_crop_len = tf.constant(self.crop_len, dtype=self._int_type)
        tf_pick_num = tf.constant(self.pick_num, dtype=self._int_type)
        if self.data_set == 'NTURGBD':
            if self.only_3dpos:
                pose = pose[:, :3, :]
        elif self.data_set == 'SBU_inter':
            pass
        elif self.data_set == 'UWA3DII':
            pass
        elif self.data_set == 'NUCLA':
            pass
        elif self.data_set == 'MSRC12':
            pass

        def pad_pose():
            pad_len = self.crop_len if self.random_crop else self.pick_num
            tf_pad_len = tf_crop_len if self.random_crop else tf_pick_num

            def fill_pad():
                return tf.pad(pose, [[0, 0], [0, 0], [0, tf_pad_len - plen]]), plen

            def crop_pad():
                return pose[:, :, :pad_len], plen

            return tf.cond(tf.greater(tf.cast(tf.shape(pose)[2], self._int_type), plen), crop_pad, fill_pad)

        if self.random_crop:
            def crop_pose():
                indx = tf.random_uniform([], minval=0, maxval=plen - tf_crop_len, dtype=self._int_type)
                return tf.slice(pose, [0, 0, indx], [-1, -1, tf_crop_len]), tf_crop_len

            pose, plen = tf.cond(tf.greater_equal(tf_crop_len, plen), pad_pose, crop_pose)
        if self.random_pick:
            def pick_pose():
                indcs = tf.cast((tf.range(0, tf_pick_num) * plen / tf_pick_num), self._int_type) + \
                        tf.random_uniform([self.pick_num], minval=0, maxval=(plen // tf_pick_num), dtype=self._int_type)
                indcs = tf.expand_dims(indcs, 1)
                return tf.transpose(tf.gather_nd(tf.transpose(pose, [2, 1, 0]), indcs), [2, 1, 0]), tf_pick_num

            pose, plen = tf.cond(tf.greater_equal(tf_pick_num, plen), pad_pose, pick_pose)

        return pose, plen

    def get_ram_batch(self, slice_idx, is_training):
        batches = self.train_batches if is_training else self.val_batches
        return batches[slice_idx]

    def update_diff(self, accuracy, batch_idxs, batch_losses, batch_plens, loss_w=0.5, smooth_w=0.5):
        with tf.control_dependencies(
                [tf.assign(self.acc_coef, accuracy)]
        ):
            current_entropy = tf.gather(self.seq_entropy, batch_idxs)
            loss_coef = batch_losses / (tf.reduce_max(batch_losses) + 1e-8)
            new_entropy = (loss_coef * loss_w) + (batch_plens / self.max_plen * (1 - loss_w))
            updated_entropy = (current_entropy * smooth_w) + (new_entropy * (1 - smooth_w))
            update_op = tf.scatter_update(self.seq_entropy, batch_idxs, updated_entropy)

        return update_op

    def weight_losses(self, batch_idxs, batch_losses):
        current_entropy = tf.gather(self.seq_entropy, batch_idxs)
        # weighted_losses = (1 - tf.abs(current_entropy-self.acc_coef)) * batch_losses
        weighted_losses = (self.acc_coef * batch_losses) + ((1 - self.acc_coef) * batch_losses / current_entropy)

        return weighted_losses

