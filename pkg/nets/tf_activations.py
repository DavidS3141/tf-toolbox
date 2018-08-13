import tensorflow as tf


def leaky_relu(x, alpha=0.2, name='leaky relu'):
    with tf.name_scope(name):
        return tf.maximum(x, x * alpha)


def selu(x, name='selu'):
    with tf.name_scope(name):
        return 1.7580993408473768599402175208123 \
            * (tf.exp(tf.minimum(x, tf.zeros_like(x))) - tf.ones_like(x)) \
            + 1.0507009873554804934193349852946 * tf.nn.relu(x)


def sigmoid_num(x, name='sigmoid num'):
    with tf.name_scope(name):
        return 2.8 * tf.nn.sigmoid(x)


def softsign_num(x, name='softsign num'):
    with tf.name_scope(name):
        return 2 * tf.nn.softsign(x)


def tanh_num(x, name='tanh num'):
    with tf.name_scope(name):
        return 1.5 * tf.nn.tanh(x)


act_name_to_act_tuple = {
    'elu': (tf.nn.elu, 1., 0.),
    'elu_num': (tf.nn.elu, 1.321939, 0.250422),
    'leaky_relu': (leaky_relu, 1.546460, 0.493559),
    'relu': (tf.nn.relu, 1.712859, 0.683332),
    'relu6': (tf.nn.relu6, 1.713838, 0.683622),
    'selu': (selu, 1., 0.),
    'sigmoid': (tf.nn.sigmoid, 4., 0.5),
    'sigmoid_num': (sigmoid_num, 2.751695, 1.4),
    'softplus': (tf.nn.softplus, 2.0, 0.693147),
    'softplus_num': (tf.nn.softplus, 1.843839, 1.021192),
    'softsign': (tf.nn.softsign, 1., 0.),
    'softsign_num': (softsign_num, 1.422959, 0.),
    'tanh': (tf.nn.tanh, 1., 0.),
    'tanh_num': (tanh_num, 1.147216, 0.),
    'identity': ('identity', 1., 0.), }


def get_activation(name):
    if name in act_name_to_act_tuple:
        return act_name_to_act_tuple[name]
    else:
        return None, None, None
