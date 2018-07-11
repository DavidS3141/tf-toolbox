import numpy as np
import tensorflow as tf


def normalization(input_t, data_or_datalist, name='normalization'):
    assert len(input_t.get_shape()) == 2
    if type(data_or_datalist) is not list:
        datalist = [data_or_datalist]
    else:
        datalist = data_or_datalist
    for data in datalist:
        assert len(data.shape) == 2
        assert data.shape[1] == input_t.get_shape()[1]
    data = np.concatenate(datalist, axis=0)
    mu = np.mean(data, axis=0, keepdims=True)
    std = np.std(data, axis=0, keepdims=True)
    with tf.name_scope(name):
        return (input_t - tf.constant(mu, dtype=tf.float32, name='mu')) \
            / tf.constant(std, dtype=tf.float32, name='std')
