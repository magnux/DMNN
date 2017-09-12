from __future__ import absolute_import, division, print_function

import os

import numpy as np
import tensorflow as tf

from layers.pre_process_poses import *
from layers.learn_comb import *
from layers.dm_embedding import *
from layers.lstm import create_cell
from layers.base import linear_layer
import layers.resnet as resnet
from cells.conv_lstm import ConvLSTMCell

COMB_MATRIX_COLLECTION = 'COMB_MATRIX'


class DmnnModel(object):
    """The PoseSeq model."""

    def __init__(self, is_training, config, input_):
        self._int_type = tf.int16 if config.use_type16 else tf.int32
        self._float_type = tf.float16 if config.use_type16 else tf.float32

        self._is_training = is_training
        splitname = 'train' if self._is_training else 'val'

        hidden_size = config.hidden_size

        if config.inference_model[:4] == 'conv':
            assert config.cell_model == None, print("The convolutional (resnet-based) models don't have cells")

        self._inputs = input_.generate_batch(self._is_training if not config.only_val else False)
        idxs, subjects, actions, poses, plens = self._inputs
        actions = actions
        subjects = subjects

        consider_conf = False

        poses, dm_shape, dm_size, body_splits, joints_perms, num_perms, batch_size, max_length, fixed_length, n_dims =\
            pre_process_poses(is_training, config, poses, actions)

        #TODO: change the learn_comb switches to str instead of bool
        if config.learn_comb or config.learn_comb_sm or config.learn_comb_orth or \
            config.learn_comb_orth or config.learn_comb_orth_rmsprop or \
            config.learn_comb_unc or config.learn_comb_centered:
            poses, self._comb_matrix_image, self._comb_matrix_update = \
                learn_comb_matrix(is_training, config, poses, dm_shape, batch_size, max_length, n_dims, self._float_type)

        if config.no_dm:
            embedding = tf.reshape(poses, [batch_size, max_length, dm_shape[0] * n_dims])
        else:
            embedding = dm_embedding(is_training, config, poses, dm_shape, dm_size,
                                    batch_size, max_length, hidden_size, consider_conf, self._float_type)

        def conv_to_linear(outputs, max_length, dm_size, scope):
            if (config.keep_prob < 1.0) and self._is_training:
                outputs = tf.nn.dropout(outputs, config.keep_prob)
            r_outputs = tf.reshape(outputs, [batch_size * max_length, dm_size])
            l_outputs = tf.layers.dense(r_outputs, hidden_size,
                kernel_regularizer=tf.contrib.layers.l2_regularizer(5.0e-4),
                kernel_initializer=tf.contrib.layers.xavier_initializer(False))
            return tf.reshape(tf.nn.elu(l_outputs), [batch_size, max_length, hidden_size])

        outputs = state = None
        cell_params = {'is_training': self._is_training, 'config': config,
                       'dm_shape': dm_shape, 'dm_size': dm_size, 'batch_size': batch_size,
                       'max_plen': input_.max_plen, 'fixed_length': fixed_length, 'hidden_size': hidden_size,
                       'num_perms': num_perms, '_float_type': self._float_type}
        if config.inference_model == 'standard':
            with tf.variable_scope('standard_rnn') as scope:
                cell, self._initial_state, dm_size = create_cell(**cell_params)
                outputs, state = tf.nn.dynamic_rnn(
                    cell,
                    embedding,
                    sequence_length=plens,
                    initial_state=self._initial_state,
                    parallel_iterations=4
                )
                self._rnn_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)
        elif config.inference_model == 'bidirectional':
            with tf.variable_scope('bidirectional_rnn') as scope:
                cell_fw, initial_state_fw, dm_size = create_cell(**cell_params)
                cell_bw, initial_state_bw, dm_size = create_cell(**cell_params)
                outputs, state = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw,
                    embedding,
                    sequence_length=plens,
                    initial_state_fw=initial_state_fw, initial_state_bw=initial_state_bw,
                    parallel_iterations=4
                )
                outputs = tf.reduce_mean(tf.concat([tf.expand_dims(outputs[0], 3), tf.expand_dims(outputs[1], 3)], axis=3), axis=3)
                self._rnn_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)
        elif config.inference_model == 'siamese':
            with tf.variable_scope('siamese_rnn') as scope:
                cell, self._initial_state, dm_size = create_cell(**cell_params)
                outputs_s = []
                state_s = []
                for emb in embedding:
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
        elif config.inference_model == 'staged':
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
                cell, self._initial_state, dm_size = create_cell(**cell_params)
                outputs, state = tf.nn.dynamic_rnn(
                    cell,
                    inputs,
                    sequence_length=plens,
                    initial_state=self._initial_state,
                    parallel_iterations=4
                )
                self._rnn_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)
        elif config.inference_model[:4] == 'conv':
            with tf.variable_scope(config.inference_model) as scope:
                if config.inference_model[7:] == 'resnet':
                    resnet_model = resnet.resnet
                elif config.inference_model[7:] == 'resnext':
                    resnet_model = resnet.resnext
                outputs = resnet_model(
                    embedding, config.num_layers, config.resnet_blocks,
                    self._is_training, config.inference_model[4:6])
                max_length = 1
                self._rnn_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)

        if config.inference_model[0:4] != 'conv':
            if config.cell_model == 'phlstm':
                outputs = outputs[:, :, :hidden_size]
            elif config.cell_model[0:4] == 'conv':
                outputs = conv_to_linear(outputs, max_length, dm_size, "")

        self._outputs = outputs
        self._final_state = state
        key_logits_act = "KEY_LOGITS_ACT"
        key_logits_sub = "KEY_LOGITS_SUB"
        plens = tf.cast(plens, self._float_type)
        if config.loss_model == 'mean_pool':
            mean_pool = None
            if config.inference_model[:4] == 'conv':
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
            logits = tf.layers.dense(tf.reshape(outputs, [-1, hidden_size]), config.num_actions,
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(5.0e-4),
                                     kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
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
