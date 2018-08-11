# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf


def warm_restart_cosine_annealing_scheduler(
        lr_max=0.001,
        lr_min=0.0,
        initial_warm_restart_iterations=1024,
        warm_restart_iterations_relative_increment=2.0):
    iter = 0
    WR_iters = initial_warm_restart_iterations
    while True:
        iter += 1
        if iter > WR_iters:
            iter = 0
            WR_iters *= warm_restart_iterations_relative_increment
        lr = 0.5 * (lr_max + lr_min) + \
            0.5 * (lr_max - lr_min) * np.cos(np.pi * iter / WR_iters)
        yield lr


def tf_warm_restart_cosine_annealing_scheduler(
        iter_t,
        lr_max=0.001,
        lr_min=0.0,
        initial_warm_restart_iterations=1024,
        warm_restart_iterations_relative_increment=2.0):
    '''
        See above definition. Number of warm restarts (nwr) and total iterations
        depend on each other in the following way:
        total_iters >= initial_warm_restart_iterations *
            sum_k=0..nwr-1 warm_restart_iterations_relative_increment**k
        total_iters < initial_warm_restart_iterations *
            sum_k=0..nwr warm_restart_iterations_relative_increment**k

        This can be translated to:
        total_iters >= initial_warm_restart_iterations *
            (warm_restart_iterations_relative_increment**nwr - 1) /
            (warm_restart_iterations_relative_increment - 1)
        total_iters < ...

        Solving for nwr we find:
    '''
    iter_t = tf.to_float(iter_t)
    wri_relative_increment_to_power_nwr = iter_t * \
        (warm_restart_iterations_relative_increment - 1.) / \
        initial_warm_restart_iterations + \
        1
    nwr = tf.floor(tf.log(wri_relative_increment_to_power_nwr) /
        np.log(warm_restart_iterations_relative_increment))
    iters_since_wr = iter_t - initial_warm_restart_iterations * \
        (tf.pow(warm_restart_iterations_relative_increment, nwr) - 1) / \
        (warm_restart_iterations_relative_increment - 1)
    WR_iters = initial_warm_restart_iterations * \
        tf.pow(warm_restart_iterations_relative_increment, nwr)
    lr = 0.5 * (lr_max + lr_min) + \
        0.5 * (lr_max - lr_min) * tf.cos(np.pi * iters_since_wr / WR_iters)
    return lr
