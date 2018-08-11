# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf


def weight_decay_regularizer(dependency_ops, var_list, weight_decay=0.0005):
    weights = tf.get_collection(tf.GraphKeys.WEIGHTS)
    decay_param = 1. - weight_decay
    with tf.control_dependencies(dependency_ops):
        return tf.group(*[tf.assign(var, var * decay_param) for var in var_list if var in weights])
