from .batch_generator import batch_generator
from .convergence_checker import ConvergenceChecker
from .create_summaries_on_graph import create_summaries_on_graph
from .timer import Timer
from .util import AttrDict, lazy_property, munge_filename, ask_yn, \
                  print_graph_statistics
from .write_tb_summary import write_tb_summary

from abc import ABC, abstractmethod
import numpy as np
import os
import shutil
import tensorflow as tf
from tqdm import tqdm


class Trainer(ABC):
    def __init__(self, list_feeding_data, train_cfg=dict(), max_epochs=128,
                 nbr_readouts=256, seed=None, succ_validations=8,
                 train_portion=0.8, batch_size=64, verbosity=1,
                 debug_verbosity=0, **kwargs):
        tf.reset_default_graph()
        # #region train config
        self.train_cfg = AttrDict({
            'batch_size': batch_size,
            'debug_verbosity': debug_verbosity,
            'max_epochs': max_epochs,
            'nbr_readouts': nbr_readouts,
            'seed': seed,
            'succ_validations': succ_validations,
            'train_portion': train_portion,
            'verbosity': verbosity,
        })
        if isinstance(train_cfg, AttrDict) or isinstance(train_cfg, dict):
            self.train_cfg.update(train_cfg)
            self.train_cfg.update(kwargs)
        else:
            raise TypeError('train_cfg not valid config type!')
        # #endregion train config
        # #region set random seeds
        if self.train_cfg.seed is not None:
            np.random.seed(self.train_cfg.seed)
            tf.set_random_seed(self.train_cfg.seed + 1)
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
        # build graph
        self.graph
        create_summaries_on_graph(
            verbosity=self.train_cfg.verbosity,
            debug_verbosity=self.train_cfg.debug_verbosity)

    @abstractmethod
    def get_feed_dict(self, batch):
        pass

    @lazy_property
    def graph(self):
        input_t = self.input_t
        weights_t = self.weights_t
        labels_t = self.labels_t
        normalization_t = self.normalization_t
        self.network
        logits_t = self.logits_t
        prediction_t = self.prediction_t
        loss_t = self.loss_t
        optimize_t = self.optimize_t
        print_graph_statistics()
        return AttrDict(locals())

    def setup_datasets(self):
        self.list_train_data = []
        self.list_valid_data = []
        self.iterations_per_epoch = 1
        self.iterations_per_validation = 1
        for tpl in self.list_feeding_data:
            if self.train_cfg.train_portion <= 1.0:
                nbr_train_elements = round(
                    self.train_cfg.train_portion * len(tpl[0]))
            else:
                nbr_train_elements = self.train_cfg_train_portion
            assert isinstance(nbr_train_elements, int)
            nbr_valid_elements = len(tpl[0]) - nbr_train_elements
            self.iterations_per_epoch = max(
                self.iterations_per_epoch,
                round(float(nbr_train_elements) / self.train_cfg.batch_size))
            self.iterations_per_validation = max(
                self.iterations_per_validation,
                round(float(nbr_valid_elements) / self.train_cfg.batch_size))
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
                if ask_yn('Remove %s (necessary for Trainer to run)?' % d,
                          default=0, timeout=60):
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
        self.setup_datasets()
        self.setup_train_queues()
        self.setup_logging_paths()
        self.setup_logging_ops()
        self.setup_session()
        self.timer = Timer()

    def train(self, output_dir):
        self.output_dir = output_dir
        self.setup_infrastructure_training()
        # self.do_readout(
        #     'initial', global_step=0, epoch=0.,
        #     bigger_used_summary=bigger_used_summary,
        #     validation=validation_value)
        self.train_loop()
        self.restore_best_state()
        self.do_readout('final')
        self.timer.create_plot(self.plot_dir + '/timer')

    def restore_best_state(self):
        self.best_saver.restore(self.sess,
                                tf.train.latest_checkpoint(self.best_dir))

    @abstractmethod
    def create_plots(self, plot_dir, **kwargs):
        print('\t\tNothing to do here! Overwrite "create_plots" for action!')

    def do_readout(self, readout_title, global_step=None, epoch=None,
                   bigger_used_summary=None, **kwargs):
        is_final = readout_title == 'final'
        is_initial = readout_title == 'initial'
        if is_final:
            assert global_step is None
            assert epoch is None
            assert bigger_used_summary is None
            name = 'final'
        elif is_initial:
            assert global_step == 0
            assert epoch == 0.
            assert bigger_used_summary
            name = 'initial'
        else:
            assert global_step > 0
            assert epoch > 0.
            assert bigger_used_summary
            name = munge_filename(
                '%06d__%08.4f' % (global_step, 100.0 * epoch /
                                  self.train_cfg.max_epochs))

        print('\nThis is the %s readout ...' % readout_title)
        print('\tCreate plots ...')
        self.timer.start('plot')
        plot_dir = os.path.join(self.plot_dir, name) + '/'
        os.makedirs(plot_dir)
        self.create_plots(plot_dir)
        self.timer.stop('plot')
        print('\tPlots saved to %s!' % plot_dir)
        if not is_final:
            print('\n\tSave model ...')
            self.timer.start('variables')
            save_path = os.path.join(self.variables_dir, name)
            self.variables_saver.save(
                self.sess, save_path, global_step=global_step)
            self.timer.stop('variables')
            print('\tModel saved in file: %s' % save_path)
            print('\n\tWrite tensorboard summary ...')
            self.timer.start('TB')
            value_dict = {'epoch': epoch}
            value_dict.update(kwargs)
            write_tb_summary(
                self.tb_saver, bigger_used_summary, global_step, value_dict)
            self.timer.stop('TB')
            print('\tFinished!')
        print('Finished readout!')

    def train_loop(self):
        # #region loop variables
        epoch = 0.
        global_step = 0
        nbr_readouts = 0
        # #endregion loop variables

        # #region initialize validation checker
        validation_checker = ConvergenceChecker(
            min_iters=1,
            max_iters=(self.train_cfg.max_epochs + 1) *
            (round(self.iterations_per_epoch / self.iterations_per_validation)
             + 1),
            min_confirmations=round(
                self.train_cfg.succ_validations *
                self.iterations_per_epoch / self.iterations_per_validation)
        )
        # #endregion initialize validation checker

        self.timer.start('loop')
        # #region main loop
        with tqdm(total=self.train_cfg.max_epochs, unit='epoch',
                  dynamic_ncols=True) as pbar:
            while epoch < self.train_cfg.max_epochs:
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
                    # valid_batch, epoch_valid = next(self.valid_queue)
                    # fd = self.get_feed_dict(valid_batch)
                    fd = self.get_feed_dict(self.list_valid_data)
                    validation_value = self.sess.run(self.loss_t, feed_dict=fd)
                    validation_checker.check(validation_value)
                else:
                    validation_value = None
                # #endregion do validation
                self.timer.stop('validation')
                self.timer.start('readout')
                # #region readout
                name = munge_filename(
                    '%06d__%08.4f' % (global_step, 100.0 * epoch /
                                      self.train_cfg.max_epochs))
                if ((epoch * self.train_cfg.nbr_readouts >
                     float(nbr_readouts) * self.train_cfg.max_epochs) or
                        (epoch >= self.train_cfg.max_epochs) or
                        validation_checker.is_converged()):
                    nbr_readouts += 1
                    self.do_readout(
                        str(nbr_readouts), global_step=global_step, epoch=epoch,
                        bigger_used_summary=bigger_used_summary,
                        validation=validation_value)
                else:
                    self.timer.start('TB')
                    write_tb_summary(
                        self.tb_saver, smaller_used_summary, global_step,
                        {
                            'epoch': epoch,
                            'validation': validation_value,
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


class AdversariesTrainer(Trainer):
    def __init__(self, adversary_converge=1024, adversary_succ_validations=128,
                 just_train_adversary=False, train_cfg=dict(), *args, **kwargs):
        comb_train_cfg = AttrDict({
            'adversary_converge': adversary_converge,
            'adversary_succ_validations': adversary_succ_validations,
            'just_train_adversary': just_train_adversary,
        })
        comb_train_cfg.update(train_cfg)
        super(AdversariesTrainer, self).__init__(train_cfg=comb_train_cfg,
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

    def train_loop(self):
        # #region loop variables
        epoch = 0.
        global_step = 0
        turn = 'adversary'
        performer_step = 0
        nbr_readouts = 0
        # #endregion loop variables

        # #region initialize convergence checkers
        # initialize convergence checker for adversary
        if self.train_cfg.adversary_succ_validations > 0:
            adversary_conv_checker = ConvergenceChecker(
                min_iters=0, max_iters=self.train_cfg.adversary_converge,
                min_confirmations=self.train_cfg.adversary_succ_validations)
        else:
            adversary_conv_checker = ConvergenceChecker(
                min_iters=self.train_cfg.adversary_converge,
                max_iters=self.train_cfg.adversary_converge)
        # initialize validation checker
        validation_checker = ConvergenceChecker(
            min_iters=1, max_iters=np.inf,
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
                else:
                    raise ValueError(
                        'turn(%s) is neither performer nor adversary' % turn)
                # #endregion training step
                # #region determine whos turn to train is next
                if turn == 'performer':
                    turn = 'adversary'
                    adversary_conv_checker.reset()
                elif turn == 'adversary' and \
                        not self.train_cfg.just_train_adversary:
                    # do small validation to check if adversary is converged
                    valid_batch, epoch_valid = next(self.valid_queue)
                    fd = self.get_feed_dict(valid_batch)
                    local_validation_value = self.sess.run(
                        self.adversary_loss_t, feed_dict=fd)
                    if adversary_conv_checker.check(local_validation_value):
                        turn = 'performer'
                elif not self.train_cfg.just_train_adversary:
                    raise ValueError(
                        'turn(%s) is neither performer nor adversary' % turn)
                # #endregion determine whos turn to train is next
                # Validation values and logging makes sense when the
                # adversary is well-defined! This is only the case after
                # the adversary converged, meaning the next turn is
                # performed by the performer.
                if turn == 'performer' or self.train_cfg.just_train_adversary:
                    # #region do validation
                    valid_batch, epoch_valid = next(self.valid_queue)
                    fd = self.get_feed_dict(valid_batch)
                    if self.train_cfg.just_train_adversary:
                        validation_value = self.sess.run(
                            self.adversary_loss_t, feed_dict=fd)
                    else:
                        validation_value = self.sess.run(
                            self.performer_loss_t, feed_dict=fd)
                    validation_checker.check(validation_value)
                    # #endregion do validation
                    name = munge_filename(
                        '%06d__%08.4f' % (performer_step, 100.0 * epoch /
                                          self.train_cfg.max_epochs))
                    # #region readout
                    if ((epoch / self.train_cfg.max_epochs >
                         float(nbr_readouts) / self.train_cfg.nbr_readouts) or
                            (epoch >= self.train_cfg.max_epochs) or
                            validation_checker.is_converged()):
                        nbr_readouts += 1
                        print('\nThis is the %d readout ...' % nbr_readouts)
                        print('\n\tCreate plots ...')
                        plot_dir = os.path.join(self.plot_dir, name) + '/'
                        os.makedirs(plot_dir)
                        self.create_plots(
                            plot_dir,
                            adversary_conv_checker=adversary_conv_checker)
                        print('\tPlots saved to %s!' % plot_dir)
                        print('\n\tSave model ...')
                        save_path = os.path.join(self.variables_dir, name)
                        self.variables_saver.save(
                            self.sess, save_path, global_step=global_step)
                        print('\tModel saved in file: %s' % save_path)
                        print('\n\tWrite tensorboard summary ...')
                        write_tb_summary(
                            self.tb_saver, bigger_used_summary, global_step,
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
                        write_tb_summary(
                            self.tb_saver, smaller_used_summary, global_step,
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
                    if validation_checker.is_converged():
                        return
                    # #endregion check validation for convergence and save best
        # #endregion main loop
