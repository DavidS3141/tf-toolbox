# -*- coding: utf-8 -*-

import functools
import numpy as np
import os
import re
import shutil
import tensorflow as tf
import time
from tqdm import tqdm

from . import batch_generator
from . import Convergence_Checker
from . import create_summaries_on_graph
from . import write_tb_summary
from .util import AttrDict, lazy_property, munge_filename, askYN


class TF_Trainer(object):
    def __init__(self, list_feeding_data, train_cfg=dict(), max_epochs=32,
                 nbr_readouts=32, seed=None, succ_validations=64,
                 train_portion=0.8, batch_size=128):
        # #region train config
        self.train_cfg = AttrDict({
            'batch_size': batch_size,
            'max_epochs': max_epochs,
            'nbr_readouts': nbr_readouts,
            'seed': seed,
            'succ_validations': succ_validations,
            'train_portion': train_portion,
        })
        if isinstance(train_cfg, AttrDict) or isinstance(train_cfg, dict):
            self.train_cfg.update(train_cfg)
        else:
            raise ValueError('train_cfg not valid config type')
        # #endregion train config
        assert isinstance(list_feeding_data, list)
        for tpl in list_feeding_data:
            assert isinstance(tpl, tuple)
            assert isinstance(tpl[0], np.ndarray)
            n = len(tpl[0])
            for elem in tpl:
                assert isinstance(elem, np.ndarray)
                assert n == len(elem)
        self.list_feeding_data = list_feeding_data
        # build graph
        self.graph
        create_summaries_on_graph()

    @lazy_property
    def graph(self):
        tf.reset_default_graph()
        input_t = self.input_t
        normalization_t = self.normalization_t
        prediction_t = self.prediction_t
        loss_t = self.loss_t
        optimize_t = self.optimize_t
        return AttrDict(locals())

    def setup_datasets(self):
        self.list_train_data = []
        self.list_valid_data = []
        for tpl in self.list_feeding_data:
            if self.train_cfg.train_portion <= 1.0:
                nbr_train_elements = int(round(
                    self.train_cfg.train_portion * len(tpl[0])))
            else:
                nbr_train_elements = self.train_cfg_train_portion
            assert isinstance(nbr_train_elements, int)
            perm = np.random.permutation(len(tpl[0]))
            train_idxs = perm[:nbr_train_elements]
            valid_idxs = perm[nbr_train_elements:]
            local_train_list = []
            local_valid_list = []
            for elem in tpl:
                local_train_list.append(elem[train_idxs, ...])
                local_valid_list.append(elem[valid_idxs, ...])
            self.list_train_data.append(tuple(local_train_list))
            self.list_valid_data.append(tuple(local_valid_list))

    def setup_train_queues(self):
        self.train_queue = batch_generator(
            self.train_cfg.batch_size, self.list_train_data)
        self.valid_queue = batch_generator(
            self.train_cfg.batch_size, self.list_valid_data)

    def setup_logging_paths(self):
        self.variables_dir = os.path.join(self.output_dir, 'variables')
        self.best_dir = os.path.join(self.output_dir, 'best')
        self.plot_dir = os.path.join(self.output_dir, 'plot')
        self.tb_dir = os.path.join(self.output_dir, 'tb')
        for d in [self.variables_dir, self.best_dir,
                  self.plot_dir, self.tb_dir]:
            if os.path.exists(d):
                if askYN('Remove %s (necessary for Trainer to run)?' % d,
                         default=0, timeout=10):
                    shutil.rmtree(d)
                else:
                    raise EnvironmentError('Folder %s already exists!' % d)

    def setup_logging_ops(self):
        with tf.name_scope('variables_saver'):
            self.variables_saver = tf.train.Saver(max_to_keep=10)
        with tf.name_scope('best_saver'):
            self.best_saver = tf.train.Saver(max_to_keep=10)
        self.tb_saver = tf.summary.FileWriter(self.tb_dir)

    def setup_session(self):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess.run(tf.global_variables_initializer())
        self.tb_saver.add_graph(self.sess.graph)

    def setup_infrastructure_training(self):
        # #region set random seeds
        if self.train_cfg.seed is not None:
            np.random.seed(self.train_cfg.seed)
            tf.set_random_seed(self.train_cfg.seed + 1)
        # #endregion set random seeds
        self.setup_datasets()
        self.setup_train_queues()
        self.setup_logging_paths()
        self.setup_logging_ops()
        self.setup_session()

    def train(self, output_dir):
        self.output_dir = output_dir
        self.setup_infrastructure_training()
        self.train_loop()

    def train_loop(self):
        raise NotImplementedError  # TODO implement default classifier


class Adversaries_Trainer(TF_Trainer):
    def __init__(self, adversary_converge=1024, adversary_succ_validations=512,
                 train_cfg=dict(), *args, **kwargs):
        comb_train_cfg = AttrDict({
            'adversary_converge': adversary_converge,
            'adversary_succ_validations': adversary_succ_validations,
        })
        comb_train_cfg.update(train_cfg)
        super(Adversaries_Trainer, self).__init__(train_cfg=comb_train_cfg,
                                                  *args, **kwargs)

    def get_feed_dict(self, batch):
        raise NotImplementedError

    @lazy_property
    def performer_loss_t(self):
        raise NotImplementedError

    @lazy_property
    def adversary_loss_t(self):
        raise NotImplementedError

    @lazy_property
    def performer_optimize_t(self):
        raise NotImplementedError

    @lazy_property
    def adversary_optimize_t(self):
        raise NotImplementedError

    def create_plots(self, plot_dir, **kwargs):
        print('\t\tNothing to do here! Overwrite "create_plots" for action!')

    def train_loop(self):
        # #region loop variables
        epoch = 0.
        global_step = 0
        turn = 'adversary'
        performer_steps = 0
        nbr_readouts = 0
        # #endregion loop variables

        # #region initialize convergence checkers
        # initialize convergence checker for adversary
        if self.train_cfg.adversary_succ_validations > 0:
            adversary_conv_checker = Convergence_Checker(
                min_iters=0, max_iters=self.train_cfg.adversary_converge,
                min_confirmations=self.train_cfg.adversary_succ_validations)
        else:
            adversary_conv_checker = Convergence_Checker(
                min_iters=self.train_cfg.adversary_converge,
                max_iters=self.train_cfg.adversary_converge)
        # initialize validation checker
        validation_checker = Convergence_Checker(min_iters=1, max_iters=np.inf,
            min_confirmations=self.train_cfg.succ_validations)
        # #endregion initialize convergence checkers

        # #region main loop
        with tqdm(total=self.train_cfg.max_epochs, unit='epoch',
                  dynamic_ncols=True) as pbar:
            while epoch < self.train_cfg.max_epochs:
                pbar.update(int(epoch) - pbar.n)
                # create batch
                batch, epoch = next(self.train_queue)
                global_step += 1
                # #region training step
                fd = self.get_feed_dict(batch)
                if turn == 'adversary':
                    # #region train step for adversary
                    (
                        _,
                        smaller_used_summary,
                        bigger_used_summary,
                    ) = self.sess.run(
                        [
                            self.adversary_optimize_t,
                            'smaller_used_summaries_t:0',
                            'bigger_used_summaries_t:0',
                        ],
                        feed_dict=fd
                    )
                    # #endregion train step for adversary
                elif turn == 'performer':
                    # #region train step for performer
                    performer_steps += 1
                    (
                        _,
                        smaller_used_summary,
                        bigger_used_summary,
                    ) = self.sess.run(
                        [
                            self.performer_optimize_t,
                            'smaller_used_summaries_t:0',
                            'bigger_used_summaries_t:0',
                        ],
                        feed_dict=fd
                    )
                    # #endregion train step for performer
                else:
                    raise ValueError(
                        'turn(%s) is neither performer nor adversary' % turn)
                # #endregion training step
                # #region determine whos turn to train is next
                if turn == 'performer':
                    turn = 'adversary'
                    adversary_conv_checker.reset()
                elif turn == 'adversary':
                    # do small validation to check if adversary is converged
                    valid_batch, epoch_valid = next(self.valid_queue)
                    fd = self.get_feed_dict(valid_batch)
                    local_validation_value = self.sess.run(
                        self.adversary_loss_t, feed_dict=fd)
                    if adversary_conv_checker.check(local_validation_value):
                        turn = 'performer'
                else:
                    raise ValueError(
                        'turn(%s) is neither performer nor adversary' % turn)
                # #endregion determine whos turn to train is next
                # Validation values and logging makes sense when the
                # adversary is well-defined! This is only the case after
                # the adversary converged, meaning the next turn is
                # performed by the performer.
                if turn == 'performer':
                    # #region do validation
                    fd = self.get_feed_dict(self.list_valid_data)
                    validation_value = self.sess.run(
                        self.performer_loss_t, feed_dict=fd)
                    # #endregion do validation
                    name = munge_filename(
                        '%06d__%08.4f' % (performer_steps, 100.0 * epoch /
                                          self.train_cfg.max_epochs))
                    # #region readout
                    if ((epoch / self.train_cfg.max_epochs >
                         float(nbr_readouts) / self.train_cfg.nbr_readouts) or
                            (epoch >= self.train_cfg.max_epochs)):
                        nbr_readouts += 1
                        print('\nThis is the %d readout ...' % nbr_readouts)
                        print('\n\tCreate plots ...')
                        plot_dir = os.path.join(self.plot_dir, name) + '/'
                        os.makedirs(plot_dir)
                        self.create_plots(plot_dir,
                            adversary_conv_checker=adversary_conv_checker)
                        print('\tPlots saved to %s!' % plot_dir)
                        print('\n\tSave model ...')
                        save_path = os.path.join(self.variables_dir, name)
                        self.variables_saver.save(self.sess, save_path,
                            global_step=global_step)
                        print('\tModel saved in file: %s' % save_path)
                        print('\n\tWrite tensorboard summary ...')
                        write_tb_summary(self.tb_saver,
                            bigger_used_summary, global_step,
                            {
                                'adversary/convergence-steps':
                                    len(adversary_conv_checker),
                                'epoch': epoch,
                                'validation': validation_value,
                            }
                        )
                        print('\tFinished!')
                        print('Finished readout!')
                    else:
                        write_tb_summary(self.tb_saver,
                            smaller_used_summary, global_step,
                            {
                                'adversary/convergence-steps':
                                    len(adversary_conv_checker),
                                'epoch': epoch,
                                'validation': validation_value,
                            }
                        )
                    # #endregion readout
                    # #region check validation for convergence and save best
                    if validation_checker.is_best(validation_value):
                        save_path = os.path.join(self.best_dir, name)
                        self.best_saver.save(self.sess, save_path,
                                             global_step=global_step)
                    if validation_checker.check(validation_value):
                        return
                    # #endregion check validation for convergence and save best
        # #endregion main loop
