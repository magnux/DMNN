from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops

def relu1(x):
    return tf.minimum(tf.maximum(x, 0), 1)

def leaky_relu(x, alpha):
    return tf.maximum(tf.minimum(tf.maximum(x * alpha, x), 1), -1)

def tf_mod(x, y, name=None):
    """Differentiable mod based in numpy
    Args
        x: first argument
        y: second argument
    Returns
        mod between x and y
    """

    def np_mod(x, y):
        return np.mod(x, y, dtype=np.float32)

    def modgrad(op, grad):
        x = op.inputs[0] # the first argument (normally you need those to calculate the gradient, like the gradient of x^2 is 2x. )
        y = op.inputs[1] # the second argument

        return grad * 1, grad * 0 #the propagated gradient with respect to the first and second argument respectively

    def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
        # Need to generate a unique name to avoid duplicates:
        rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

        tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
        g = tf.get_default_graph()
        with g.gradient_override_map({"PyFunc": rnd_name}):
            return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

    with ops.name_scope(name, "mod", [x,y]) as name:
        z = py_func(np_mod,
                    [x,y],
                    [tf.float32],
                    name=name,
                    grad=modgrad)  # <-- here's the call to the gradient
        return tf.reshape(z[0], tf.shape(x))

def dk_mod(x, y):
    """Differentiable mod, Donald Knuth style
    Args
        x: first argument
        y: second argument
    Returns
        mod between x and y
    """
    return x - y * tf.floor(x / y)

# Register the gradient for the mod operation. tf.mod() does not have a gradient implemented.
@ops.RegisterGradient('Mod')
def _mod_grad(op, grad):
    x, y = op.inputs
    gz = grad
    x_grad = gz
    # y_grad = tf.reduce_mean(-(x // y) * gz, reduction_indices=[0], keep_dims=True)#[0]
    y_grad = -(x // y) * gz
    return x_grad, y_grad

def triu(x):
    """Upper triangular matrix
    Args
        x: matrix to be analized
    Returns
        Upper triangular portion of the matrix as a tensor
    """
    triu_indcs = np.triu_indices(x.get_shape().as_list()[0], x.get_shape().as_list()[1])
    triu_indcs = np.reshape(np.transpose(np.concatenate(np.expand_dims(triu_indcs,0), axis=1)), [-1, 2])
    tf_triu_indcs = tf.constant(triu_indcs)
    return tf.gather_nd(x, tf_triu_indcs)
