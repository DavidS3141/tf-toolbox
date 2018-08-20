import tensorflow as tf


def write_tb_summary(tb_saver, summary, global_step, values_dict):
    s = tf.Summary()
    if isinstance(summary, str) or isinstance(summary, bytes):
        tb_saver.add_summary(summary, global_step)
    elif isinstance(summary, dict):
        values_dict.update(summary)
    for key in values_dict:
        if values_dict[key] is not None:
            s.value.add(tag=key, simple_value=float(values_dict[key]))
    tb_saver.add_summary(s, global_step)
