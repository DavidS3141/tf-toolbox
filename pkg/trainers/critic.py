#!/usr/bin/env python
from ..batch_generator import batch_generator
from ..Convergence_Checker import Convergence_Checker
from ..write_tb_summary import write_tb_summary as write_general_tb_summary

import numpy as np
import os
import tensorflow as tf
from tqdm import tqdm


def write_tb_summary(tb_saver, summary, global_step, epoch,
                     valid_loss=None, **kwargs):
    values_dict = {'epoch': epoch, 'validation': valid_loss}
    for key in kwargs:
        values_dict[key] = kwargs[key]
    write_general_tb_summary(tb_saver, summary, global_step, values_dict)


def munge_filename(s):
    return s.replace(' ', '_')


def run(A_train, B_train, A_valid, B_valid, output_dir, tb_dir=None,
        train_seed=None, total_nbr_readouts=128, max_epochs=32, batch_size=32,
        succ_validations=8, sess=None, var_saver=None, tb_saver=None):
    assert isinstance(A_train, np.ndarray)
    assert len(A_train.shape) == 2
    nbr_feats = A_train.shape[1]
    for data in [B_train, A_valid, B_valid]:
        assert isinstance(data, np.ndarray)
        assert len(data.shape) == 2
        assert data.shape[1] == nbr_feats

    # set output paths
    if tb_dir is None:
        out_vars = os.path.join(output_dir, 'variables')
        tb_dir = os.path.join(output_dir, 'tb')
    else:
        out_vars = output_dir
    assert not os.path.exists(out_vars)
    assert not os.path.exists(tb_dir)

    # set random seed
    if train_seed is not None:
        np.random.seed(train_seed)

    # start session and load model
    if sess is None:
        if train_seed is not None:
            tf.set_random_seed(train_seed)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        sess.run(tf.global_variables_initializer())

    # utils for saving the trained model to disk
    if var_saver is None:
        with tf.name_scope('variables_saver'):
            var_saver = tf.train.Saver(max_to_keep=10)
    if tb_saver is None:
        tb_saver = tf.summary.FileWriter(tb_dir)

    # save the graph to TensorBoard
    tb_saver.add_graph(sess.graph)

    # set up convergence checker
    convergence_checker = Convergence_Checker(
        max_iters=np.inf, min_confirmations=succ_validations)

    # perform initial evaluation
    print('\nINIT:')
    print('\tValidation ... ', end='', flush=True)
    valid_loss = sess.run(
        'loss:0',
        feed_dict={'A_in:0': A_valid, 'B_in:0': B_valid})
    convergence_checker.check(valid_loss)
    print('\tDone.')
    save_path = os.path.join(
        out_vars,
        munge_filename('%15d-%6.2f' % (0, 100 * 0 / max_epochs)))
    print('\tSave model in %s ... ' % save_path, end='', flush=True)
    var_saver.save(sess, save_path)
    print('\tDone.')
    print('\tEvaluate ... ', end='', flush=True)
    bigger_used_summary = sess.run(
        'bigger_used_summaries:0',
        feed_dict={'A_in:0': A_train, 'B_in:0': B_train}
    )
    print('\tDone.')
    print('\tWrite TB ... ', end='', flush=True)
    write_tb_summary(tb_saver, bigger_used_summary, 0, 0, valid_loss)
    print('\tDone.')
    print('\tLOSS: %.2e' % valid_loss)

    # data batch queue
    train_queue_A = batch_generator(batch_size, A_train)
    train_queue_B = batch_generator(batch_size, B_train)

    # train
    _train(sess, max_epochs, total_nbr_readouts, convergence_checker,
           train_queue_A, train_queue_B, A_valid, B_valid, out_vars,
           var_saver, tb_saver)

    return sess


def _train(sess, max_epochs, total_nbr_readouts, convergence_checker,
           train_queue_A, train_queue_B, A_valid, B_valid, out_vars,
           var_saver, tb_saver):
    # tracking values
    epoch = 0.0
    nbr_readouts = 0
    global_step = 0

    for epoch_counter in tqdm(range(1, 1 + max_epochs), 'Train Critic'):
        while epoch < epoch_counter:
            A_batch, epoch = next(train_queue_A)
            B_batch, epoch_alt = next(train_queue_B)
            assert epoch == epoch_alt
            global_step += 1
            # do the training step
            (
                _,
                smaller_used_summary,
                bigger_used_summary,
            ) = sess.run(
                [
                    'optimizer',
                    'smaller_used_summaries:0',
                    'bigger_used_summaries:0',
                ],
                feed_dict={
                    'A_in:0': A_batch,
                    'B_in:0': B_batch,
                }
            )
            # do validation
            if epoch / max_epochs > nbr_readouts / total_nbr_readouts \
                    or epoch >= max_epochs:
                nbr_readouts += 1

                print('\nSTEP %15d:' % global_step)
                print('\tValidation ... ', end='', flush=True)
                valid_loss = sess.run(
                    'loss:0',
                    feed_dict={'A_in:0': A_valid, 'B_in:0': B_valid})
                print('\tDone.')
                save_path = os.path.join(
                    out_vars,
                    munge_filename('%15d-%6.2f' % (global_step,
                                                   100 * epoch / max_epochs)))
                print('\tSave model in %s ... ' % save_path, end='', flush=True)
                var_saver.save(sess, save_path)
                print('\tDone.')
                print('\tWrite TB ... ', end='', flush=True)
                write_tb_summary(tb_saver, bigger_used_summary, global_step,
                                 epoch, valid_loss)
                print('\tDone.')
                print('\tLOSS: %.2e' % valid_loss)

                if convergence_checker.check(valid_loss):
                    return
            else:
                write_tb_summary(tb_saver, smaller_used_summary,
                                 global_step, epoch)
