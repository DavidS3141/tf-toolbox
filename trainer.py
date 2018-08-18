from .batch_generator import batch_generator
from .convergence_checker import ConvergenceChecker
from .create_summaries_on_graph import create_summaries_on_graph
from .timer import Timer
from .util import AttrDict, ask_yn, print_graph_statistics, lazy_property, \
    hash_string, average_tf_output, makedirs
from .write_tb_summary import write_tb_summary

from abc import ABCMeta, abstractmethod
import math
import numpy as np
import os
import shutil
import tensorflow as tf
from tqdm import tqdm


class Trainer(object):
    __metaclass__ = ABCMeta

    def __init__(self, list_feeding_data, max_epochs=128,
                 nbr_readouts=256, seed=None, succ_validations=8,
                 train_portion=0.8, batch_size=64, verbosity=1,
                 debug_verbosity=0, train_valid_time_ratio=4, **kwargs):
        tf.reset_default_graph()
        # #region config
        self.cfg = AttrDict({
            'batch_size': batch_size,
            'debug_verbosity': debug_verbosity,
            'max_epochs': float(max_epochs),
            'nbr_readouts': nbr_readouts,
            'seed': seed,
            'succ_validations': succ_validations,
            'train_portion': train_portion,
            'train_valid_time_ratio': train_valid_time_ratio,
            'verbosity': verbosity,
        })
        self.cfg.update(kwargs)
        # #endregion config
        # #region set random seeds
        if self.cfg.seed is not None:
            np.random.seed(self.cfg.seed)
            tf.set_random_seed(self.cfg.seed + 1)
        # #endregion set random seeds
        assert isinstance(list_feeding_data, list)
        for tpl in list_feeding_data:
            assert isinstance(tpl, tuple)
            assert isinstance(tpl[0], np.ndarray)
            n = len(tpl[0])
            for elem in tpl:
                assert isinstance(elem, np.ndarray)
                assert n == len(elem)
        self.list_feeding_data = list_feeding_data
        self.setup_trainer_object()

    @abstractmethod
    def get_feed_dict(self, batch):
        raise NotImplementedError

    @abstractmethod
    def create_plots(self, plot_dir, **kwargs):
        print('\t\tNothing to do here! Overwrite "create_plots" for action!')

    @abstractmethod
    def graph(self):
        raise NotImplementedError

    # #region needed tensors
    @abstractmethod
    def loss_t(self):
        raise NotImplementedError

    @abstractmethod
    def lr_t(self):
        raise NotImplementedError

    @abstractmethod
    def optimize_t(self):
        raise NotImplementedError
    # #endregion needed tensors

    # #region setup
    def setup_graph(self):
        self.graph()
        create_summaries_on_graph(
            verbosity=self.cfg.verbosity,
            debug_verbosity=self.cfg.debug_verbosity)
        print_graph_statistics()

    def setup_datasets(self):
        self.list_train_data = []
        self.list_valid_data = []
        self.iterations_per_epoch = 1
        self.iterations_per_validation = 1
        for tpl in self.list_feeding_data:
            if self.cfg.train_portion <= 1.0:
                nbr_train_elements = int(round(
                    self.cfg.train_portion * len(tpl[0])))
            else:
                nbr_train_elements = self.cfg_train_portion
            assert isinstance(nbr_train_elements, int)
            nbr_valid_elements = len(tpl[0]) - nbr_train_elements
            self.iterations_per_epoch = max(
                self.iterations_per_epoch,
                round(float(nbr_train_elements) / self.cfg.batch_size))
            self.iterations_per_validation = max(
                self.iterations_per_validation,
                round(self.cfg.train_valid_time_ratio *
                      float(nbr_valid_elements) / self.cfg.batch_size))
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
            self.cfg.batch_size, self.list_train_data)
        self.valid_queue = batch_generator(
            self.cfg.batch_size, self.list_valid_data)

    def setup_logging_paths(self):
        self.variables_dir = os.path.join(self.output_dir, 'variables')
        self.best_dir = os.path.join(self.output_dir, 'best')
        self.plot_dir = os.path.join(self.output_dir, 'plot')
        self.tb_dir = os.path.join(self.output_dir, 'tb')

    def setup_logging_ops(self):
        with tf.name_scope('variables_saver'):
            self.variables_saver = tf.train.Saver(max_to_keep=10)
        with tf.name_scope('best_saver'):
            self.best_saver = tf.train.Saver(max_to_keep=10)

    def setup_session(self):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess.run(tf.global_variables_initializer())

    def clear_logging_paths(self):
        for d in [self.variables_dir, self.best_dir,
                  self.plot_dir, self.tb_dir]:
            if os.path.exists(d):
                if ask_yn('Remove %s (necessary for Trainer to run)?' % d,
                          default=0, timeout=60):
                    shutil.rmtree(d)
                else:
                    raise EnvironmentError('Folder %s already exists!' % d)

    def setup_trainer_object(self):
        self.setup_datasets()
        self.setup_train_queues()
        self.setup_graph()

    def setup_infrastructure_training(self):
        self.setup_logging_paths()
        self.setup_logging_ops()
        self.setup_session()
        self.timer = Timer()
        self.best_validation_score = np.inf
        self.clear_logging_paths()
    # #endregion setup

    @lazy_property
    def data_split_hash(self):
        hash_value = ''
        for tpl in self.list_train_data:
            for elem in tpl:
                hash_value += hash_string(str(elem.tostring()))
        for tpl in self.list_valid_data:
            for elem in tpl:
                hash_value += hash_string(str(elem.tostring()))
        return hash_value

    def restore_best_state(self, output_dir=None):
        if output_dir:
            assert not hasattr(self, 'output_dir')
            latest_best_checkpoint = tf.train.latest_checkpoint(
                os.path.join(output_dir, 'best'))
            latest_vars_checkpoint = tf.train.latest_checkpoint(
                os.path.join(output_dir, 'variables'))
        else:
            latest_best_checkpoint = tf.train.latest_checkpoint(self.best_dir)
            latest_vars_checkpoint = tf.train.latest_checkpoint(
                self.variables_dir)
        if not (latest_best_checkpoint or latest_vars_checkpoint):
            raise EnvironmentError("Why aren't there any saved variables?")
        if output_dir:
            self.output_dir = output_dir
            self.setup_logging_paths()
            self.setup_logging_ops()
            self.setup_session()
            self.best_validation_score = None
        if latest_best_checkpoint:
            self.best_saver.restore(self.sess, latest_best_checkpoint)
        elif latest_vars_checkpoint:
            self.variables_saver.restore(self.sess, latest_vars_checkpoint)
        else:
            raise Exception('Logial error in this function!')
        if self.best_validation_score is None:
            self.best_validation_score = self.safe_sess_run(
                self.loss_t, self.list_valid_data)

    def do_readout(self, readout_title, name=None, global_step=None,
                   bigger_used_summary=None, **kwargs):
        is_final = readout_title == 'final'
        is_initial = readout_title == 'initial'
        if is_final:
            assert name is None
            assert global_step is None
            assert bigger_used_summary is None
            name = 'final'
        elif is_initial:
            assert name is None
            assert global_step == 0
            assert bigger_used_summary
            name = 'initial'
        else:
            assert global_step > 0
            assert bigger_used_summary

        self.timer.start('plot')
        print('\nThis is the %s readout ...' % readout_title)
        print('\tCreate plots ...')
        plot_dir = os.path.join(self.plot_dir, name) + '/'
        makedirs(plot_dir)
        self.create_plots(plot_dir)
        print('\tPlots saved to %s!' % plot_dir)
        self.timer.stop('plot')
        if not is_final:
            print('\n\tSave model ...')
            save_path = os.path.join(self.variables_dir, name)
            self.variables_saver.save(
                self.sess, save_path, global_step=global_step)
            print('\tModel saved in file: %s' % save_path)
            self.timer.start('TB')
            print('\n\tWrite tensorboard summary ...')
            write_tb_summary(
                self.tb_saver, bigger_used_summary, global_step, kwargs)
            print('\tFinished!')
            self.timer.stop('TB')
        print('Finished readout!')

    def safe_sess_run(self, tensors, batch):
        if self.cfg.get('max_batch_size', None) and max([len(t[0]) for t in batch]) > self.cfg.max_batch_size:
            assert self.cfg.batch_size <= self.cfg.max_batch_size
            bg = batch_generator(self.cfg.max_batch_size, batch, shuffle=False,
                                 single_epoch=True)
            results = []
            for b, epoch in bg:
                results.append(
                    self.sess.run(tensors, feed_dict=self.get_feed_dict(b)))
            return average_tf_output(results)
        else:
            return self.sess.run(tensors, feed_dict=self.get_feed_dict(batch))

    def train(self, output_dir):
        self.output_dir = output_dir
        self.setup_infrastructure_training()
        self.tb_saver = tf.summary.FileWriter(self.tb_dir)
        self.tb_saver.add_graph(self.sess.graph)
        self.train_loop()
        self.timer.start('readout')
        self.restore_best_state()
        self.do_readout('final')
        self.timer.stop('readout')
        self.timer.create_plot(self.plot_dir + '/timer')

    def train_loop(self):
        self.timer.start('readout')
        # #region initial readout
        validation_value, bigger_used_summary = self.safe_sess_run(
            [self.loss_t, 'bigger_used_summaries_t:0'],
            self.list_valid_data)
        self.do_readout(
            'initial', global_step=0,
            bigger_used_summary=bigger_used_summary,
            epoch=0., validation=validation_value)
        # #endregion initial readout
        self.timer.stop('readout')

        # #region loop variables
        epoch = 0.
        global_step = 0
        nbr_readouts = 1  # counting the initial readout
        # #endregion loop variables

        # #region initialize validation checker
        validation_checker = ConvergenceChecker(
            min_iters=1,
            max_iters=int(math.ceil(self.cfg.max_epochs + 1) *
                          (round(self.iterations_per_epoch /
                                 self.iterations_per_validation) + 1)),
            min_confirmations=int(round(
                self.cfg.succ_validations *
                self.iterations_per_epoch / self.iterations_per_validation))
        )
        # #endregion initialize validation checker

        self.timer.start('loop')
        # #region main loop
        with tqdm(total=int(math.ceil(self.cfg.max_epochs)), unit='epoch',
                  dynamic_ncols=True) as pbar:
            while epoch < self.cfg.max_epochs:
                self.timer.start('progress bar')
                # #region update progress bar
                pbar.set_description(
                    'confirmations: %d/%d' %
                    (validation_checker.get_nbr_confirmations(),
                     validation_checker.min_confirmations))
                if int(epoch) > pbar.n:
                    pbar.update(int(epoch) - pbar.n)
                # #endregion update progress bar
                self.timer.stop('progress bar')
                self.timer.start('training')
                # #region training step
                batch, epoch = next(self.train_queue)
                global_step += 1
                fd = self.get_feed_dict(batch)
                (
                    _,
                    smaller_used_summary,
                    bigger_used_summary,
                ) = self.sess.run(
                    [
                        self.optimize_t,
                        'smaller_used_summaries_t:0',
                        'bigger_used_summaries_t:0',
                    ],
                    feed_dict=fd
                )
                # #endregion training step
                self.timer.stop('training')
                self.timer.start('validation')
                # #region do validation
                if global_step % self.iterations_per_validation == 0:
                    validation_value, lr = self.safe_sess_run(
                        [self.loss_t, self.lr_t], self.list_valid_data)
                    validation_checker.check(validation_value, lr)
                else:
                    validation_value = None
                # #endregion do validation
                self.timer.stop('validation')
                self.timer.start('readout')
                # #region readout
                self.timer.start('name_if')
                name = '%06d__%08.4f' % (global_step, 100.0 * epoch /
                                         self.cfg.max_epochs)
                if ((epoch * self.cfg.nbr_readouts >
                     float(nbr_readouts) * self.cfg.max_epochs) or
                        (epoch >= self.cfg.max_epochs) or
                        validation_checker.is_converged()):
                    nbr_readouts += 1
                    self.timer.stop('name_if')
                    self.do_readout(
                        str(nbr_readouts), name, global_step=global_step,
                        bigger_used_summary=bigger_used_summary,
                        validation=validation_value, epoch=epoch,
                        nbr_confirmations=validation_checker.
                        get_nbr_confirmations())
                else:
                    self.timer.stop('name_if')
                    self.timer.start('TB')
                    write_tb_summary(
                        self.tb_saver, smaller_used_summary, global_step,
                        {
                            'epoch': epoch,
                            'validation': validation_value,
                            'nbr_confirmations':
                                validation_checker.get_nbr_confirmations(),
                        }
                    )
                    self.timer.stop('TB')
                # #endregion readout
                self.timer.stop('readout')
                self.timer.start('check')
                # #region check validation for convergence and save best
                self.best_validation_score = min(
                    self.best_validation_score, validation_value or np.inf)
                if validation_value and \
                        validation_checker.is_best(validation_value):
                    save_path = os.path.join(self.best_dir, name)
                    self.best_saver.save(self.sess, save_path,
                                         global_step=global_step)
                if validation_checker.is_converged():
                    self.timer.stop('check')
                    break
                # #endregion check validation for convergence and save best
                self.timer.stop('check')
        # #endregion main loop
        self.timer.stop('loop')


class AdversariesTrainer(Trainer):
    def __init__(self, adversary_converge=64, adversary_succ_validations=4,
                 just_train_adversary=False, *args, **kwargs):
        cfg = AttrDict({
            'adversary_converge': adversary_converge,
            'adversary_succ_validations': adversary_succ_validations,
            'just_train_adversary': just_train_adversary,
        })
        kwargs.update(cfg)
        super(AdversariesTrainer, self).__init__(*args, **kwargs)

    # #region needed tensors
    def lr_t(self):
        raise AttributeError("Don't use this function name for adversaries!")

    def optimize_t(self):
        raise AttributeError("Don't use this function name for adversaries!")

    def loss_t(self):
        return self.performer_loss_t

    @abstractmethod
    def performer_loss_t(self):
        raise NotImplementedError

    @abstractmethod
    def adversary_loss_t(self):
        raise NotImplementedError

    @abstractmethod
    def performer_lr_t(self):
        raise NotImplementedError

    @abstractmethod
    def adversary_lr_t(self):
        raise NotImplementedError

    @abstractmethod
    def performer_optimize_t(self):
        raise NotImplementedError

    @abstractmethod
    def adversary_optimize_t(self):
        raise NotImplementedError
    # #endregion needed tensors

    def train_loop(self):
        self.timer.start('readout')
        # #region initial readout
        validation_value, bigger_used_summary = self.safe_sess_run(
            [self.performer_loss_t, 'bigger_used_summaries_t:0'],
            self.list_valid_data)
        self.do_readout(
            'initial', global_step=0,
            bigger_used_summary=bigger_used_summary,
            epoch=0., validation=validation_value)
        # #endregion initial readout
        self.timer.stop('readout')

        # #region loop variables
        epoch = 0.
        global_step = 0
        turn = 'adversary'
        performer_step = 0
        nbr_readouts = 1  # counting the initial readout
        # #endregion loop variables

        # #region initialize convergence checkers
        # initialize convergence checker for adversary
        if self.cfg.adversary_succ_validations > 0:
            adversary_conv_checker = ConvergenceChecker(
                min_iters=0, max_iters=self.cfg.adversary_converge,
                min_confirmations=self.cfg.adversary_succ_validations)
        else:
            adversary_conv_checker = ConvergenceChecker(
                min_iters=self.cfg.adversary_converge,
                max_iters=self.cfg.adversary_converge)
        # initialize validation checker
        validation_checker = ConvergenceChecker(
            min_iters=1,
            max_iters=int(math.ceil(self.cfg.max_epochs + 1) *
            (round(self.iterations_per_epoch / self.iterations_per_validation)
             + 1)),
            min_confirmations=int(math.ceil(
                self.cfg.succ_validations *
                self.iterations_per_epoch / self.iterations_per_validation))
        )
        # #endregion initialize convergence checkers

        self.timer.start('loop')
        # #region main loop
        with tqdm(total=int(math.ceil(self.cfg.max_epochs)), unit='epoch',
                  dynamic_ncols=True) as pbar:
            while epoch < self.cfg.max_epochs:
                self.timer.start('progress bar')
                # #region update progress pbar
                pbar.set_description(
                    'confirmations: %d/%d' %
                    (validation_checker.get_nbr_confirmations(),
                     validation_checker.min_confirmations))
                if int(epoch) > pbar.n:
                    pbar.update(int(epoch) - pbar.n)
                # #endregion update progress bar
                self.timer.stop('progress bar')
                self.timer.start('training')
                # #region training step
                # create batch
                batch, epoch = next(self.train_queue)
                global_step += 1
                fd = self.get_feed_dict(batch)
                if turn == 'adversary':
                    self.timer.start('adversary')
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
                    self.timer.stop('adversary')
                elif turn == 'performer':
                    self.timer.start('performer')
                    # #region train step for performer
                    performer_step += 1
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
                    self.timer.stop('performer')
                else:
                    raise ValueError(
                        'turn(%s) is neither performer nor adversary' % turn)
                # #endregion training step
                self.timer.stop('training')
                self.timer.start('turn')
                # #region determine whos turn to train is next
                if turn == 'performer':
                    turn = 'adversary'
                    adversary_conv_checker.reset()
                elif turn == 'adversary' and \
                        not self.cfg.just_train_adversary:
                    # do small validation to check if adversary is converged
                    valid_batch, epoch_valid = next(self.valid_queue)
                    fd = self.get_feed_dict(valid_batch)
                    local_validation_value, local_lr = self.sess.run(
                        [self.adversary_loss_t, self.adversary_lr_t],
                        feed_dict=fd)
                    if adversary_conv_checker.check(local_validation_value,
                                                    local_lr):
                        turn = 'performer'
                elif not self.cfg.just_train_adversary:
                    raise ValueError(
                        'turn(%s) is neither performer nor adversary' % turn)
                # #endregion determine whos turn to train is next
                self.timer.stop('turn')

                # Validation values and logging makes sense when the
                # adversary is well-defined! This is only the case after
                # the adversary converged, meaning the next turn is
                # performed by the performer.
                if turn == 'performer' or self.cfg.just_train_adversary:
                    self.timer.start('validation')
                    # #region do validation
                    if global_step % self.iterations_per_validation == 0:
                        print('doing validation: %d' % global_step)
                        if self.cfg.just_train_adversary:
                            validation_value, lr = self.safe_sess_run(
                                [self.adversary_loss_t,
                                 self.adversary_lr_t],
                                self.list_valid_data)
                        else:
                            validation_value, lr = self.safe_sess_run(
                                [self.performer_loss_t,
                                 self.performer_lr_t],
                                self.list_valid_data)
                        validation_checker.check(validation_value, lr)
                    else:
                        validation_value = None
                    # #endregion do validation
                    self.timer.stop('validation')
                    self.timer.start('readout')
                    # #region readout
                    self.timer.start('name_if')
                    if self.cfg.just_train_adversary:
                        name = '%06d__%08.4f' % (global_step, 100.0 * epoch /
                                                 self.cfg.max_epochs)
                    else:
                        name = '%06d__%08.4f' % (performer_step, 100.0 * epoch /
                                                 self.cfg.max_epochs)
                    if ((epoch * self.cfg.nbr_readouts >
                         float(nbr_readouts) * self.cfg.max_epochs) or
                            (epoch >= self.cfg.max_epochs) or
                            validation_checker.is_converged()):
                        nbr_readouts += 1
                        self.timer.stop('name_if')
                        self.do_readout(
                            str(nbr_readouts), name, global_step=global_step,
                            bigger_used_summary=bigger_used_summary,
                            validation=validation_value, epoch=epoch,
                            nbr_confirmations=validation_checker.
                            get_nbr_confirmations(),
                            adversary_convergence_steps=len(
                                adversary_conv_checker)
                        )
                    else:
                        self.timer.stop('name_if')
                        self.timer.start('TB')
                        write_tb_summary(
                            self.tb_saver, smaller_used_summary, global_step,
                            {
                                'adversary_convergence_steps':
                                    len(adversary_conv_checker),
                                'epoch': epoch,
                                'validation': validation_value,
                                'nbr_confirmations': validation_checker.
                                get_nbr_confirmations(),
                            }
                        )
                        self.timer.stop('TB')
                    # #endregion readout
                    self.timer.stop('readout')
                    self.timer.start('check')
                    # #region check validation for convergence and save best
                    if validation_value and \
                            validation_checker.is_best(validation_value):
                        save_path = os.path.join(self.best_dir, name)
                        self.best_saver.save(self.sess, save_path,
                                             global_step=global_step)
                    if validation_checker.is_converged():
                        self.timer.stop('check')
                        break
                    # #endregion check validation for convergence and save best
                    self.timer.stop('check')
        # #endregion main loop
        self.timer.stop('loop')
