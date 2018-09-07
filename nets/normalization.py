import numpy as np
import tensorflow as tf


def normalization(input_t, data_or_datalist, name='normalization'):
    if isinstance(data_or_datalist, np.ndarray):
        datalist = [data_or_datalist]
    else:
        datalist = data_or_datalist
    for data in datalist:
        assert data.shape[1:] == input_t.shape[1:]
    assert len(input_t.shape) <= 2
    data = np.concatenate(datalist, axis=0)
    mu = np.mean(data, axis=0, keepdims=True)
    std = np.std(data, axis=0, keepdims=True)
    invariant_features = (std == 0)
    with tf.name_scope(name):
        if np.any(invariant_features):
            assert len(input_t.shape) == 2
            print(
                'There are invariant features %s!' %
                str(np.where(invariant_features)))
            input_t = input_t[:, ~invariant_features[0]]
            mu = mu[:, ~invariant_features[0]]
            std = std[:, ~invariant_features[0]]
        normalized_t = input_t - tf.constant(mu, dtype=tf.float32, name='mu')
        assert np.all(std > 0)
        normalized_t /= tf.constant(std, dtype=tf.float32, name='std')
        mean_t = tf.reduce_mean(normalized_t)
        tf.summary.scalar('mean', mean_t, collections=['d0'])
        tf.summary.scalar('std', tf.sqrt(tf.reduce_mean(tf.square(
            normalized_t - mean_t))), collections=['d0'])
        return normalized_t


def normalization_nd(input_t, data_or_datalist, name='normalization'):
    if isinstance(data_or_datalist, np.ndarray):
        datalist = [data_or_datalist]
    else:
        datalist = data_or_datalist
    for data in datalist:
        assert data.shape[1:] == input_t.shape[1:]
    assert len(input_t.shape) > 2
    data = np.concatenate(datalist, axis=0)
    mu = np.mean(data, axis=0, keepdims=True)
    std = np.std(data, axis=0, keepdims=True)
    invariant_features = (std == 0)
    std[invariant_features] = 1
    with tf.name_scope(name):
        if np.any(invariant_features):
            print(
                'There are invariant features %s!' %
                str(np.where(invariant_features)))
        normalized_t = input_t - tf.constant(mu, dtype=tf.float32, name='mu')
        assert np.all(std > 0)
        normalized_t /= tf.constant(std, dtype=tf.float32, name='std')
        mean_t = tf.reduce_mean(normalized_t)
        tf.summary.scalar('mean', mean_t, collections=['d0'])
        tf.summary.scalar('std', tf.sqrt(tf.reduce_mean(tf.square(
            normalized_t - mean_t))), collections=['d0'])
        return normalized_t
