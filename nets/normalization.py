import numpy as np
import tensorflow as tf


def normalization(input_t, data_or_datalist, name='normalization'):
    if isinstance(data_or_datalist, np.ndarray):
        datalist = [data_or_datalist]
    else:
        datalist = data_or_datalist
    for data in datalist:
        assert data.shape[1:] == input_t.get_shape()[1:]
    data = np.concatenate(datalist, axis=0)
    mu = np.mean(data, axis=0, keepdims=True)
    std = np.std(data, axis=0, keepdims=True)
    with tf.name_scope(name):
        normalized_t = input_t - tf.constant(mu, dtype=tf.float32, name='mu')
        normalized_t /= tf.constant(std, dtype=tf.float32, name='std')
        mean_t = tf.reduce_mean(normalized_t)
        tf.summary.scalar('mean', mean_t, collections=['d0'])
        tf.summary.scalar('std', tf.sqrt(tf.reduce_mean(tf.square(
            normalized_t - mean_t))), collections=['d0'])
        return normalized_t
