import tensorflow as tf


def create_summaries_on_graph(verbosity=1, debug_verbosity=0):
    v0_summaries = tf.summary.merge_all('v0')
    v1_summaries = tf.summary.merge_all('v1')
    d0_summaries = tf.summary.merge_all('d0')
    d1_summaries = tf.summary.merge_all('d1')

    watch_summaries = tf.summary.merge_all('watch')
    if verbosity == 0:
        smaller_used_summaries = v0_summaries
        bigger_used_summaries = v0_summaries
    elif verbosity == 1:
        smaller_used_summaries = v0_summaries
        bigger_used_summaries = tf.summary.merge([v0_summaries, v1_summaries])
    elif verbosity == 2:
        smaller_used_summaries = tf.summary.merge([v0_summaries, v1_summaries])
        bigger_used_summaries = tf.summary.merge([v0_summaries, v1_summaries])
    else:
        raise ValueError('Verbosity has to be a one of 0,1,2 instead '
                         'of %d!' % verbosity)
    if debug_verbosity == 0:
        pass
    elif debug_verbosity == 1:
        bigger_used_summaries = tf.summary.merge([bigger_used_summaries,
                                                  d0_summaries])
    elif debug_verbosity == 2:
        smaller_used_summaries = tf.summary.merge([smaller_used_summaries,
                                                   d0_summaries])
        bigger_used_summaries = tf.summary.merge([bigger_used_summaries,
                                                  d0_summaries, d1_summaries])
    elif debug_verbosity == 3:
        smaller_used_summaries = tf.summary.merge([smaller_used_summaries,
                                                   d0_summaries, d1_summaries])
        bigger_used_summaries = tf.summary.merge([bigger_used_summaries,
                                                  d0_summaries, d1_summaries])
    else:
        raise ValueError('Debug Verbosity has to be a one of 0,1,2,3 instead '
                         'of %d!' % debug_verbosity)
    if watch_summaries is not None:
        smaller_used_summaries = tf.summary.merge([smaller_used_summaries,
                                                   watch_summaries])
        bigger_used_summaries = tf.summary.merge([bigger_used_summaries,
                                                  watch_summaries])
    # check that all summaries are sorted into one of the verbosity groups
    all_summaries = tf.summary.merge_all()
    assert(all_summaries is None)
    # name the tf_summaries
    tf.identity(smaller_used_summaries, name='smaller_used_summaries_t')
    tf.identity(bigger_used_summaries, name='bigger_used_summaries_t')
