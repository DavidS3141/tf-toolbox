import tensorflow as tf


def weight_decay_regularizer(dependency_ops, total_number_iterations,
                             var_list=None, normalized_weight_decay=0.05):
    weights = tf.get_collection(tf.GraphKeys.WEIGHTS)
    if not var_list:
        var_list = weights
    weight_decay = normalized_weight_decay * (1. / total_number_iterations)**0.5
    decay_param = 1. - weight_decay
    with tf.control_dependencies(dependency_ops):
        return tf.group(*[tf.assign(var, var * decay_param)
                          for var in var_list if var in weights])
