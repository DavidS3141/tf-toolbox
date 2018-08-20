from ..nets.fc import fc_network
from ..nets.normalization import normalization
from ..regularizers import weight_decay_regularizer
from ..schedulers import tf_warm_restart_exponential_scheduler
from ..trainer import Trainer
from ..util import lazy_property, define_scope, AttrDict, \
    get_time_stamp, makedirs

import numpy as np
import skopt
from skopt.plots import plot_objective, plot_evaluations
from skopt.space import Real
from skopt.utils import use_named_args
import tensorflow as tf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


def generate_toy_data(size):
    np.random.seed(42)
    a = np.random.uniform(size=(size, 1))**2
    a_w = 0.5 + a
    a_l = np.zeros_like(a)
    b = 1 - np.random.uniform(size=(2*size, 1))**2
    b_w = (1 - 2*b)**2 + 0.5
    b_l = np.ones_like(b)
    data = np.concatenate([a, b])
    weights = np.concatenate([a_w, b_w])
    assert np.all(weights > 0)
    labels = np.concatenate([a_l, b_l])
    return [(data, weights, labels)]


class ExampleTrainer(Trainer):
    def __init__(self, list_data, lr=0.01, normalized_weight_decay=0.05, *args,
                 **kwargs):
        self.average_weight = np.average(list_data[0][1])
        super(ExampleTrainer, self).__init__(
            list_data, lr=lr, normalized_weight_decay=normalized_weight_decay,
            *args, **kwargs)

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
        act_name = self.cfg.get('act_name', 'relu')
        act_params = self.cfg.get('act_params', (None,))
        network_func, layer_variables = fc_network(layer_sizes, 'network',
                                                   act_name=act_name,
                                                   act_params=act_params)
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

    @lazy_property
    def step_t(self):
        return tf.Variable(0, dtype=tf.int32, name='step_t')

    @lazy_property
    def lr_total_its_t(self):
        return tf_warm_restart_exponential_scheduler(
            self.step_t, lr_min=0.0000001, lr_max=self.cfg.lr)

    @lazy_property
    def lr_t(self):
        return self.lr_total_its_t[0]

    @define_scope
    def optimize_t(self):
        tf.summary.scalar('step', self.step_t, collections=['v0'])
        tf.summary.scalar('lr', self.lr_t, collections=['v0'])
        opt = tf.train.AdamOptimizer(learning_rate=self.lr_t)
        theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'network')
        optimizer_opt = opt.minimize(self.loss_t, var_list=theta,
                                     global_step=self.step_t)
        # return optimizer_opt
        return weight_decay_regularizer(
            [optimizer_opt], self.lr_total_its_t[1], theta,
            normalized_weight_decay=self.cfg.normalized_weight_decay)

    def graph(self):
        self.input_t
        self.weights_t
        self.labels_t
        self.network
        self.logits_t
        self.prediction_t
        self.loss_t
        self.optimize_t

    def create_plots(self, plot_dir, **kwargs):
        data = self.list_feeding_data[0][0]
        weights = self.list_feeding_data[0][1]
        labels = self.list_feeding_data[0][2]
        x = np.linspace(0, 1, num=1000)[..., np.newaxis]
        pred = self.sess.run(self.prediction_t, feed_dict={self.input_t: x})
        a_idxs = labels == 0
        b_idxs = labels == 1
        a = data[a_idxs]
        a_w = weights[a_idxs]
        b = data[b_idxs]
        b_w = weights[b_idxs]
        bins = np.linspace(0, 1, num=20)
        bin_mids = 0.5 * (bins[:-1] + bins[1:])
        plt.close()
        a_h = plt.hist(a, bins=bins, weights=a_w, histtype='step', label='A')
        b_h = plt.hist(b, bins=bins, weights=b_w, histtype='step', label='B')
        plt.twinx()
        plt.plot(bin_mids, b_h[0] / (a_h[0] + b_h[0]), '-g', label='B/(A+B)')
        plt.plot(x, pred, '-r', label='pred B/(A+B)')
        plt.gcf().legend(loc='upper left', bbox_to_anchor=(0, 1),
                         bbox_transform=plt.gca().transAxes)
        plt.savefig(plot_dir + 'data.png')


space = [
    # Real(8, 256, 'log-uniform', name='max_epochs'),
    Real(0.0001, 0.01, 'log-uniform', name='lr'),
    Real(0.005, 0.5, 'log-uniform', name='normalized_weight_decay'),
    Real(8, 1024, 'log-uniform', name='batch_size'),
    # Categorical(['relu', 'leaky_relu', 'softplus', 'softplus_p'],
    #             name='act_name'),
    Real(0.0001, 0.2, 'log-uniform', name='leaky_param'),
    # Real(1, 20, 'log-uniform', name='softplus_param'),
    Real(0.5, 16, 'log-uniform', name='succ_validations'),
]


def test_hyper_search_example_trainer():
    list_data = generate_toy_data(100000)
    data_split_hash = ExampleTrainer(list_data, seed=42).data_split_hash

    @use_named_args(space)
    def run_example_trainer(**kwargs):
        cfg = AttrDict(kwargs)
        cfg.batch_size = int(round(cfg.batch_size))
        cfg.act_name = 'leaky_relu'
        cfg.act_params = (cfg.leaky_param,)
        del cfg['leaky_param']
        cfg.max_epochs = 1024

        trainer = ExampleTrainer(list_data, seed=42, nbr_readouts=0, **cfg)
        assert data_split_hash == trainer.data_split_hash
        makedirs('data/hyper', exist_ok=True)
        path = cfg.get_hashed_path('data/hyper')
        try:
            trainer.restore_best_state(path)
        except EnvironmentError:
            trainer.train(path)
        return trainer.best_validation_score

    res_gp = skopt.gp_minimize(run_example_trainer, space, n_calls=1,
                               random_state=0, n_random_starts=1)
    print([d.name for d in res_gp.space.dimensions])
    ts = get_time_stamp()
    plt.close()
    plot_objective(res_gp)
    plt.savefig('data/hyper/%s-objective.png' % ts)
    plt.close()
    plot_evaluations(res_gp)
    plt.savefig('data/hyper/%s-evaluations.png' % ts)
    skopt.dump(res_gp, 'data/hyper/%s-result.gz' % ts, store_objective=False)


def test_deterministic_example_trainer():
    list_data = generate_toy_data(100000)
    trainer = ExampleTrainer(list_data, seed=42, max_epochs=32,
                             nbr_readouts=0, debug_verbosity=0, verbosity=0,
                             succ_validations=0.2)
    trainer.train('data/example_trainer')
    last_loss = trainer.sess.run(
        trainer.loss_t,
        feed_dict=trainer.get_feed_dict(trainer.list_valid_data))
    trainer.restore_best_state()
    best_loss = trainer.sess.run(
        trainer.loss_t,
        feed_dict=trainer.get_feed_dict(trainer.list_valid_data))
    trainer = ExampleTrainer(list_data, seed=42, max_epochs=32,
                             nbr_readouts=0, debug_verbosity=0, verbosity=0,
                             succ_validations=0.2)
    trainer.train('data/example_trainer2')
    last_loss2 = trainer.sess.run(
        trainer.loss_t,
        feed_dict=trainer.get_feed_dict(trainer.list_valid_data))
    trainer.restore_best_state()
    best_loss2 = trainer.sess.run(
        trainer.loss_t,
        feed_dict=trainer.get_feed_dict(trainer.list_valid_data))
    assert last_loss == last_loss2
    assert best_loss == best_loss2
    assert last_loss >= best_loss
