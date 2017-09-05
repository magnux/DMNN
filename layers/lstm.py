from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

from cells.dense_lstm import *
from cells.conv_lstm import *


def create_cell(is_training, config, dm_shape, dm_size,
                batch_size, max_plen, fixed_length, hidden_size, _float_type=tf.float32):
    lstm_units = hidden_size
    cell = None
    if config.cell_model == 'lstm':
        lstm_cell = tf.contrib.rnn.LSTMCell(lstm_units)
    elif config.cell_model == 'bnlstm':
        lstm_cell = BNLSTMCell(lstm_units, fixed_length, is_training)
    elif config.cell_model == 'phlstm':
        lstm_cell = PhasedLSTMCell(lstm_units, is_training)
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
                is_training=is_training,
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
                is_training=is_training,
                activation=tf.nn.tanh,
                tau_init=np.log(max_plen / (2 * config.pick_num)),
                r_on_init=1e-1
            )
            num_in = dm_shape[2]
            cell.append(conv_cell)
    elif config.cell_model == 'conv_variant':
        cell = []
        conv_specs = config.conv_specs
        conv_shape = (dm_shape[0] ,dm_shape[1])
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
                is_training=is_training,
                max_length=fixed_length,
                # keep_prob=config.keep_prob if self._is_training else 1.0
                new_pool=config.new_pool
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
    initial_state = cell.zero_state(batch_size, _float_type)
    return cell, initial_state, dm_size