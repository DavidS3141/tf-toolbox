import tensorflow as tf


def write_tb_summary(tb_saver, summary, global_step, values_dict):
    """Write a summary of values to disk.

    Parameters
    ----------
    tb_saver : tf.train.Saver
        An instance to write the summary to.
    summary : scalar Tensor of type string
        The summary produced by an operation in the computation graph.
    global_step : int
        The step count associated with this summary write.
    values_dict : dict
        A dictionary with simple (string, int/float) item pairs that are
        appended to the summary.
    """
    s = tf.Summary()
    if isinstance(summary, str) or isinstance(summary, bytes):
        tb_saver.add_summary(summary, global_step)
    elif isinstance(summary, dict):
        values_dict.update(summary)
    for key in values_dict:
        if values_dict[key] is not None:
            s.value.add(tag=key, simple_value=float(values_dict[key]))
    tb_saver.add_summary(s, global_step)
