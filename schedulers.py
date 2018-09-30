import numpy as np
import tensorflow as tf


def warm_restart_cosine_annealing_scheduler(
        eta_min=0.0,
        initial_warm_restart_iterations=1024,
        warm_restart_iterations_relative_increment=2.0):
    assert eta_min <= 1.
    iter = 0
    wr_iters = initial_warm_restart_iterations
    while True:
        iter += 1
        if iter > wr_iters:
            iter = 0
            wr_iters *= warm_restart_iterations_relative_increment
        eta = 0.5 * (1. + eta_min) + \
            0.5 * (1. - eta_min) * np.cos(np.pi * iter / wr_iters)
        yield eta


def tf_warm_restart_cosine_annealing_scheduler(
        iter_t,
        eta_min=0.0,
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
    assert eta_min <= 1.
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
    wr_iters = initial_warm_restart_iterations * \
        tf.pow(warm_restart_iterations_relative_increment, nwr)
    eta = 0.5 * (1. + eta_min) + \
        0.5 * (1. - eta_min) * tf.cos(np.pi * iters_since_wr / wr_iters)
    return eta


def warm_restart_exponential_scheduler(
        eta_min=0.001,
        initial_warm_restart_iterations=1024,
        warm_restart_iterations_relative_increment=2.0):
    assert eta_min <= 1.
    assert eta_min > 0
    iter = 0
    wr_iters = initial_warm_restart_iterations
    decay_param = eta_min**(1. / wr_iters)
    while True:
        iter += 1
        if iter > wr_iters:
            iter = 0
            wr_iters *= warm_restart_iterations_relative_increment
            decay_param = eta_min**(1. / wr_iters)
        eta = decay_param**iter
        yield eta


def tf_warm_restart_exponential_scheduler(
        iter_t,
        eta_min=0.001,
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
    assert eta_min <= 1.
    assert eta_min > 0
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
    wr_iters = initial_warm_restart_iterations * \
        tf.pow(warm_restart_iterations_relative_increment, nwr)
    decay_param = eta_min**(1. / wr_iters)
    eta = decay_param**iters_since_wr
    return eta, wr_iters


def exponential_scheduler(decay_rate=0.0):
    assert decay_rate < 1.0
    assert decay_rate >= 0.0
    iter = 0
    while True:
        iter += 1
        eta = (1.0 - decay_rate)**iter
        yield eta


def tf_exponential_scheduler(iter_t, decay_rate=0.0):
    assert decay_rate < 1.0
    assert decay_rate >= 0.0
    iter_t = tf.to_float(iter_t)
    eta = (1.0 - decay_rate)**iter_t
    return eta
