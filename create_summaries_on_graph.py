import tensorflow as tf


def create_summaries_on_graph(verbosity):
    v0_summaries = tf.summary.merge_all('v0')
    v1_summaries = tf.summary.merge_all('v1')
    v2_summaries = tf.summary.merge_all('v2')
    watch_summaries = tf.summary.merge_all('watch')
    if verbosity == 0:
        smaller_used_summaries = v0_summaries
        bigger_used_summaries = v0_summaries
    elif verbosity == 1:
        smaller_used_summaries = v0_summaries
        bigger_used_summaries = v1_summaries
    elif verbosity == 2:
        smaller_used_summaries = v1_summaries
        bigger_used_summaries = v2_summaries
    elif verbosity >= 3:
        smaller_used_summaries = v2_summaries
        bigger_used_summaries = v2_summaries
    else:
        raise ValueError('Verbosity has to be a non-negative number instead '
                         'of %d!' % verbosity)
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
