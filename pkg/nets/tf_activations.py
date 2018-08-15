import numpy as np
import os
import scipy
from scipy.stats import norm as gaussian
import tensorflow as tf


def elu_np(x):
    return x + np.heaviside(-x, 0) * (np.exp(-np.abs(x)) - 1 - x)


def leaky_relu_np(alpha=0.2):
    def f(x):
        return np.maximum(x, x * alpha)
    return f


def leaky_relu_tf(alpha=0.2):
    def f(x, name='leaky relu'):
        with tf.name_scope(name):
            return tf.maximum(x, x * alpha)
    return f


def relu_np(x):
    return np.maximum(x, 0)


def relu6_np(x):
    return np.minimum(np.maximum(x, 0), 6)


def selu_np(x):
    return 1.7580993408473768599402175208123 \
        * (np.exp(np.minimum(x, 0)) - 1) \
        + 1.0507009873554804934193349852946 * relu_np(x)


def selu_tf(x, name='selu'):
    with tf.name_scope(name):
        return 1.7580993408473768599402175208123 \
            * (tf.exp(tf.minimum(x, tf.zeros_like(x))) - tf.ones_like(x)) \
            + 1.0507009873554804934193349852946 * tf.nn.relu(x)


def sigmoid_num_np(x):
    return 4 * scipy.special.expit(x)


def sigmoid_num_tf(x, name):
    return 4 * tf.nn.sigmoid(x, name)


def softplus_np(x):
    return np.log(1 + np.exp(-np.abs(x))) + np.maximum(x, 0)


def softplus_p_np(alpha=1.0):
    def f(x):
        return np.log(1 + np.exp(-np.abs(x) * alpha)) / alpha + np.maximum(x, 0)
    assert alpha > 0
    return f


def softplus_p_tf(alpha=1.0):
    def softplus_p_tf(x, name='softplus parametrized'):
        with tf.name_scope(name):
            return tf.log(1 + tf.exp(-tf.abs(x) * alpha)) / alpha + \
                tf.maximum(x, 0)
    assert alpha > 0
    return softplus_p_tf


def softsign_np(x):
    return x / (1 + np.abs(x))


def softsign_num_np(x):
    return 2 * softsign_np(x)


def softsign_num_tf(x, name):
    return 2 * tf.nn.softsign(x, name)


def tanh_num_np(x):
    return 2 * np.tanh(x)


def tanh_num_tf(x, name):
    return 2 * tf.nn.tanh(x, name)


implemented_activations = {
    'elu': {'np': lambda dummy: elu_np, 'tf': lambda dummy: tf.nn.elu},
    'identity': {'np': lambda dummy: lambda x: x,
                 'tf': lambda dummy: lambda x: x},
    'leaky_relu': {'np': leaky_relu_np, 'tf': leaky_relu_tf},
    'relu': {'np': lambda dummy: relu_np, 'tf': lambda dummy: tf.nn.relu},
    'relu6': {'np': lambda dummy: relu6_np, 'tf': lambda dummy: tf.nn.relu6},
    'selu': {'np': lambda dummy: selu_np, 'tf': lambda dummy: selu_tf},
    'sigmoid': {'np': lambda dummy: scipy.special.expit,
                'tf': lambda dummy: tf.nn.sigmoid},
    'sigmoid_num': {'np': lambda dummy: sigmoid_num_np,
                    'tf': lambda dummy: sigmoid_num_tf},
    'softplus': {'np': lambda dummy: softplus_np,
                 'tf': lambda dummy: tf.nn.softplus},
    'softplus_p': {'np': softplus_p_np, 'tf': softplus_p_tf},
    'softsign': {'np': lambda dummy: softsign_np,
                 'tf': lambda dummy: tf.nn.softsign},
    'softsign_num': {'np': lambda dummy: softsign_num_np,
                     'tf': lambda dummy: softsign_num_tf},
    'tanh': {'np': lambda dummy: np.tanh, 'tf': lambda dummy: tf.nn.tanh},
    'tanh_num': {'np': lambda dummy: tanh_num_np,
                 'tf': lambda dummy: tanh_num_tf},
}


def get_activation(name, params=(None,)):
    '''
    Should return in this order:
    act: activation function
    pre_std: expecting gaussian with mean 0 and standard deviation pre_std
             to guarantee that the output has a standard deviation of 1
    post_mean: the mean after the activation function assuming the above
               described input
    '''
    assert name in implemented_activations
    if name not in get_activation._cached_activations:
        get_activation._cached_activations[name] = dict()
    if params not in get_activation._cached_activations[name]:
        np_act = implemented_activations[name]['np'](*params)
        tf_act = implemented_activations[name]['tf'](*params)
        pre_std, post_mean = compute_pre_std_post_mean(np_act)
        print('There was the need to recompute weight init parameters:')
        if params[0] is not None:
            act_name_add = '(%s)' % ', '.join([str(p) for p in params])
        else:
            act_name_add = ''
        cache_string = "    '%s': {%s: (%s, %.10f, %.10f)},\n" % \
            (name, str(params), tf_act.__name__ + act_name_add,
             pre_std, post_mean)
        print(cache_string)
        cache_fname = os.path.dirname(__file__) + '/cached_activations.txt'
        with open(cache_fname, 'r') as file:
            current_cache = set(file.readlines())
        current_cache.add(cache_string)
        current_cache = sorted(list(current_cache))
        with open(cache_fname, 'w') as file:
            [file.write(line) for line in current_cache]
        get_activation._cached_activations[name][params] = \
            (tf_act, pre_std, post_mean)
    return get_activation._cached_activations[name][params]


get_activation._cached_activations = {
    'elu': {(None,): (tf.nn.elu, 1.3219389963, 0.2504223680)},
    'leaky_relu': {(0.01,): (leaky_relu_tf(0.01), 1.7048316272, 0.6733281229),
                   (0.2,): (leaky_relu_tf(0.2), 1.5464600604, 0.4935586424)},
    'relu': {(None,): (tf.nn.relu, 1.7128585504, 0.6833316961)},
    'relu6': {(None,): (tf.nn.relu6, 1.7138376616, 0.6836224424)},
    'selu': {(None,): (selu_tf, 1.0000000000, -0.0000000000)},
    'sigmoid': {(None,): (tf.nn.sigmoid, 4.0000000001, 0.5000000000)},
    'sigmoid_num': {(None,): (sigmoid_num_tf, 1.3126771368, 2.0000000000)},
    'softplus': {(None,): (tf.nn.softplus, 1.8438395187, 1.0211922094)},
    'softplus_p': {(20,): (softplus_p_tf(20), 1.7139191933, 0.6847111081)},
    'softsign': {(None,): (tf.nn.softsign, 1.0000100000, 0.0000000000)},
    'softsign_num': {(None,): (softsign_num_tf, 1.4229583818, 0.0000000000)},
    'tanh': {(None,): (tf.nn.tanh, 1.0000000000, 0.0000000000)},
    'tanh_num': {(None,): (tanh_num_tf, 0.6563385684, 0.0000000000)},
    'identity': {(None,): ('identity', 1.0000000000, 0.0000000000)},
}


def compute_pre_std_post_mean(act):
    def post_std(pre_std):
        expect = scipy.integrate.quad(
            lambda x: act(x) * gaussian.pdf(x, 0, pre_std), -np.inf, np.inf)[0]
        expect_fsqr = scipy.integrate.quad(
            lambda x: act(x)**2 * gaussian.pdf(x, 0, pre_std),
            -np.inf, np.inf)[0]
        return np.sqrt(expect_fsqr - expect**2)
    maximum = np.max(act(np.linspace(-100, 100)))
    minimum = np.min(act(np.linspace(-100, 100)))
    max_std = 0.5 * (maximum - minimum)
    if max_std > 1.0:
        pre_std = scipy.optimize.fsolve(
            lambda pre_std: post_std(pre_std) - 1, 1.5)[0]
        post_mean = scipy.integrate.quad(
            lambda x: act(x) * gaussian.pdf(x, 0, pre_std), -np.inf, np.inf)[0]
        return float(pre_std), float(post_mean)
    else:
        print('Warning, could not compute precise weight initialization '
              'parameters!')
        delta = 1e-5
        gradient_at_zero = (act(delta) - act(-delta)) / (2 * delta)
        pre_std = 1. / gradient_at_zero
        post_mean = act(0)
        return float(pre_std), float(post_mean)
