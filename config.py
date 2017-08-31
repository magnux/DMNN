from __future__ import absolute_import, division, print_function

import os
import pickle
import tensorflow as tf


class Config(object):
    pass


def get_config(flags):
    config = Config()

    filename = os.path.join('configs', 'base_config.py')
    with open(filename, 'r') as f:
        dict_file = eval(f.read())
        config.__dict__ = dict_file

    if flags.save_path is not None and tf.gfile.Exists(flags.save_path):
        filename = os.path.join(flags.save_path, "config.pickle")
        if tf.gfile.Exists(filename):
            with open(filename, 'rb') as f:
                pickle_file = pickle.load(f)
                config.__dict__.update(pickle_file.__dict__)
    elif flags.config_file is not None:
        filename = os.path.join('configs', flags.config_file + '_config.py')
        with open(filename, 'r') as f:
            dict_file = eval(f.read())
            config.__dict__.update(dict_file)

    if config.data_set == 'NTURGBD':
        config.num_actions = 60
        config.num_subjects = 40
    elif config.data_set == 'SBU_inter':
        config.num_actions = 8
        config.num_subjects = 7
    elif config.data_set == 'UWA3DII':
        config.num_actions = 30
        config.num_subjects = 10
    elif config.data_set == 'NUCLA':
        config.num_actions = 12
        config.num_subjects = 10
    elif config.data_set == 'MSRC12':
        config.num_actions = 12
        config.num_subjects = 30

    return config
