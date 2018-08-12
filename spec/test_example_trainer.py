from pkg.nets.fc import fc_network
from pkg.nets.normalization import normalization
from pkg.schedulers import tf_warm_restart_cosine_annealing_scheduler
from pkg.trainer import TF_Trainer as Trainer
from pkg.util import lazy_property, define_scope

import numpy as np
import tensorflow as tf


def generate_toy_data(size):
    np.random.seed(42)
    A = np.random.uniform(size=(size, 1))**2
    A_w = 0.5 + A
    A_l = np.zeros_like(A)
    B = 1 - np.random.uniform(size=(2*size, 1))**2
    B_w = (1 - 2*B)**2 + 0.5
    B_l = np.ones_like(B)
    data = np.concatenate([A, B])
    weights = np.concatenate([A_w, B_w])
    assert np.all(weights > 0)
    labels = np.concatenate([A_l, B_l])
    return [(data, weights, labels)]


class ExampleTrainer(Trainer):
    def __init__(self, list_data, *args, **kwargs):
        self.average_weight = np.average(list_data[0][1])
        super(ExampleTrainer, self).__init__(list_data, *args, **kwargs)

    def get_feed_dict(self, batch):
        nbr_vals = np.sum([np.sum([np.prod(e.shape) for e in tpl])
                           for tpl in batch])
        approx_memory_bit = nbr_vals * 32
        approx_memory_mb = approx_memory_bit / 8 / 1e6
        assert approx_memory_mb < 10
        return {
            self.input_t: batch[0][0],
            self.weights_t: batch[0][1],
            self.labels_t: batch[0][2],
        }

    @lazy_property
    def input_t(self):
        return tf.placeholder(shape=[None, 1], dtype=tf.float32,
                              name='input_t')

    @lazy_property
    def weights_t(self):
        return tf.placeholder(shape=[None, 1], dtype=tf.float32,
                              name='weights_t')

    @lazy_property
    def labels_t(self):
        return tf.placeholder(shape=[None, 1], dtype=tf.float32,
                              name='labels_t')

    @define_scope
    def normalization_t(self):
        return normalization(self.input_t, self.list_feeding_data[0][0])

    @define_scope
    def network(self):
        layer_sizes = [1, 128, 128, 128, 128, 1]
        network_func, layer_variables = fc_network(layer_sizes, 'network')
        return network_func

    @define_scope
    def logits_t(self):
        return self.network(self.normalization_t)

    @define_scope
    def prediction_t(self):
        return tf.nn.sigmoid(self.logits_t)

    @define_scope
    def loss_t(self):
        xentropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.labels_t,
            logits=self.logits_t,
            name='xentropy')
        weighted_xentropy = tf.reduce_mean(
            xentropy * self.weights_t / self.average_weight,
            name='weighted_xentropy')
        return weighted_xentropy

    @define_scope
    def optimize_t(self):
        step_t = tf.Variable(0, dtype=tf.int32, name='step_t')
        lr_t = tf_warm_restart_cosine_annealing_scheduler(
            step_t, lr_min=0.000001, lr_max=0.000001)
        tf.summary.scalar('step', step_t, collections=['v0'])
        tf.summary.scalar('lr', lr_t, collections=['v0'])
        opt = tf.train.AdamOptimizer(learning_rate=lr_t)
        theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'network')
        optimizer_opt = opt.minimize(self.loss_t, var_list=theta,
                                     global_step=step_t)
        return optimizer_opt
        # return weight_decay_regularizer([optimizer_opt], theta)


def test_example_trainer():
    list_data = generate_toy_data(10000)
    trainer = ExampleTrainer(list_data, seed=42, max_epochs=32)
    trainer.train('example_trainer')
