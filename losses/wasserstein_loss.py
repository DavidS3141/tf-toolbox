import tensorflow as tf


def wasserstein_loss(network, critic_gradient_penalty, **kwargs):
    assert critic_gradient_penalty >= 0
    graph = tf.get_default_graph()
    a_in = graph.get_tensor_by_name("a_in:0")
    b_in = graph.get_tensor_by_name("b_in:0")
    a_out = graph.get_tensor_by_name("a_out:0")
    b_out = graph.get_tensor_by_name("b_out:0")

    with tf.name_scope('loss_wasserstein'):
        neg_emd = tf.reduce_mean(a_out) - tf.reduce_mean(b_out)
        with tf.name_scope('create_mix'):
            epsilon = tf.random_uniform([tf.shape(a_in)[0], 1])
            with tf.control_dependencies([tf.assert_equal(
                    tf.shape(a_in), tf.shape(b_in))]):
                sample_mix = epsilon * a_in + (1 - epsilon) * b_in
        sample_mix_out = network(sample_mix)
        with tf.name_scope('compute_grad_norm'):
            gradients = tf.gradients(sample_mix_out, sample_mix)
            gradients_norm = tf.sqrt(tf.reduce_sum(
                tf.square(gradients)[0], axis=1))
            gradient_avg, gradient_std = tf.nn.moments(gradients_norm,
                                                       axes=[0])
            tf.summary.scalar('avg_gradient', gradient_avg,
                              collections=['v0'])
            tf.summary.scalar('std_gradient', tf.sqrt(gradient_std),
                              collections=['v0'])
        if kwargs.get('gradient_penalty_max', True):
            gradient_penalty = tf.reduce_mean(
                tf.square(tf.maximum(0., gradients_norm - 1.0))) * \
                critic_gradient_penalty
        else:
            gradient_penalty = tf.reduce_mean(
                tf.square(gradients_norm - 1.0)) * \
                critic_gradient_penalty
        tf.summary.scalar('neg_EMD', neg_emd,
                          collections=['v0'])
        tf.summary.scalar('GP', gradient_penalty,
                          collections=['v0'])
        tf.summary.scalar('total', neg_emd + gradient_penalty,
                          collections=['v0'])
        return neg_emd + gradient_penalty
