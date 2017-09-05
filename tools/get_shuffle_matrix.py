from __future__ import absolute_import, division, print_function

import argparse
import tensorflow as tf
from numpy import savetxt

parser = argparse.ArgumentParser(
    description='Extract shuffle matrix from network snapshots')
parser.add_argument('--name', '-n', type=str, default='Model/learn_comb/matrix',
    help='Name of the variable containing the shuffle matrix')
parser.add_argument('--size', '-s', type=int, default=25,
    help='Size of the shuffle matrix')
parser.add_argument('in_dir', type=str,
    help='Input directory containing the network snapshots')
parser.add_argument('out_file', type=str, help='Output CSV file')

if __name__ == '__main__':
    args = parser.parse_args()
    
    # Find checkpoint
    chkpt = tf.train.latest_checkpoint(args.in_dir)
    print('Loading data from', chkpt)
    
    with tf.device('/cpu:0'):
        M_var = tf.Variable(tf.zeros((args.size, args.size), dtype=tf.float32))
        saver = tf.train.Saver({args.name: M_var})
    
    with tf.Session() as sess:
        saver.restore(sess, chkpt)
        M = sess.run(M_var)
    
    savetxt(args.out_file, M)
