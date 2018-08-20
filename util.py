import functools
import hashlib
from numbers import Number
import numpy as np
import os
import re
import struct
import tensorflow as tf
import time
import yaml


def get_time_stamp(with_date=True, with_delims=False):
    if with_date:
        if with_delims:
            return time.strftime('%Y/%m/%d-%H:%M:%S')
        else:
            return time.strftime('%Y%m%d-%H%M%S')
    else:
        if with_delims:
            return time.strftime('%H:%M:%S')
        else:
            return time.strftime('%H%M%S')


def munge_filename(name, mode='strict'):
    """Remove characters that might not be safe in a filename."""
    if mode == 'strict':
        non_alphabetic = re.compile('[^A-Za-z0-9_.]')
    else:
        non_alphabetic = re.compile('[^A-Za-z0-9_\-.=,:]')
    return non_alphabetic.sub('_', name)


def ask_yn(question, default=-1, timeout=0):
    import sys
    import select

    answers = '[y/n]'
    if default == 0:
        answers = '[N/y]'
    elif default == 1:
        answers = '[Y/n]'
    elif default != -1:
        raise Exception('Wrong default parameter (%d) to ask_yn!' % default)

    if timeout > 0:
        if default == -1:
            raise Exception('When using timeout, specify a default answer!')
        answers += ' (%ds time to answer!)' % timeout
    print(question + ' ' + answers)

    if timeout == 0:
        ans = input()
    else:
        i, o, e = select.select([sys.stdin], [], [], timeout)
        if i:
            ans = sys.stdin.readline().strip()
        else:
            ans = ''

    if ans == 'y' or ans == 'Y':
        return True
    elif ans == 'n' or ans == 'N':
        return False
    elif len(ans) == 0:
        if default == 0:
            return False
        elif default == 1:
            return True
        elif default == -1:
            raise Exception('There is no default option given to this '
                            'y/n-question!')
        else:
            raise Exception('Logical error in ask_yn function!')
    else:
        raise Exception('Wrong answer to y/n-question! Answer was %s!' % ans)
    raise Exception('Logical error in ask_yn function!')


def denumpyfy(tuple_list_dict_number):
    if isinstance(tuple_list_dict_number, tuple):
        return tuple([denumpyfy(elem) for elem in tuple_list_dict_number])
    if isinstance(tuple_list_dict_number, list):
        return [denumpyfy(elem) for elem in tuple_list_dict_number]
    if isinstance(tuple_list_dict_number, dict):
        return {denumpyfy(k): denumpyfy(tuple_list_dict_number[k])
                for k in tuple_list_dict_number}
    if isinstance(tuple_list_dict_number, float):
        return float(tuple_list_dict_number)
    if isinstance(tuple_list_dict_number, int):
        return int(tuple_list_dict_number)
    return tuple_list_dict_number


def summary_string2dict(summ_str):
    if isinstance(summ_str, str):
        newline_compare = '\n'

        def unpack_flt(arg):
            return struct.unpack('<f', arg)[0]

        def unpack_int(arg):
            return struct.unpack('B', arg)[0]
    else:
        assert isinstance(summ_str, bytes)
        newline_compare = b'\n'[0]

        def unpack_flt(arg):
            return struct.unpack('<f', arg)[0]

        def unpack_int(arg):
            return struct.unpack('B', bytes([arg]))[0]
    idx = 0
    ret_dict = {}
    while idx < len(summ_str):
        assert summ_str[idx] == newline_compare
        content_size = unpack_int(summ_str[idx + 1]) - 3
        if summ_str[idx + 2] != newline_compare:
            item_size = unpack_int(summ_str[idx + 2])
            add_data = 128 * (item_size - 1)
            head_len = 5
        else:
            item_size = 0
            add_data = 0
            head_len = 4
        assert summ_str[idx + head_len - 2] == newline_compare
        name_len = unpack_int(summ_str[idx + head_len - 1])
        name = summ_str[idx + head_len:idx + head_len + name_len]
        if isinstance(name, bytes):
            name = name.decode('utf8')
        data_start = idx + head_len + name_len + 1
        # print(name)
        # print('before data:' + repr(summ_str[data_start - 1]))
        # print('data:')
        # print(repr(summ_str[
        #     data_start:data_start + content_size - name_len + add_data]))
        # if len(summ_str) > data_start + content_size - name_len + add_data:
        #     print('after data:' + repr(summ_str[
        #         data_start + content_size - name_len + add_data]))
        if 42 == unpack_int(summ_str[idx + head_len + name_len]):
            value = []
        else:
            value = unpack_flt(summ_str[data_start:data_start + 4])
        if not isinstance(value, list):
            ret_dict[name] = value
        idx += head_len + content_size + 1 + add_data
    return ret_dict


def average_tf_output(list_of_outputs):
    assert isinstance(list_of_outputs, list)
    if isinstance(list_of_outputs[0], list):
        f = len(list_of_outputs[0])
        result = []
        for i in range(f):
            result.append(average_tf_output([v[i] for v in list_of_outputs]))
        return result
    elif isinstance(list_of_outputs[0], Number):
        return np.mean(np.array(list_of_outputs))
    elif isinstance(list_of_outputs[0], str) or \
            isinstance(list_of_outputs[0], bytes):
        return average_tf_output(
            [summary_string2dict(s) for s in list_of_outputs])
    elif isinstance(list_of_outputs[0], dict):
        keys = list(list_of_outputs[0])
        result = {}
        for k in keys:
            result[k] = average_tf_output([v[k] for v in list_of_outputs])
        return result
    elif isinstance(list_of_outputs[0], np.ndarray):
        return np.concatenate(list_of_outputs, axis=0)
    print(type(list_of_outputs[0]))
    assert False


def hash_string(string):
    return hashlib.md5(string.encode()).hexdigest()


def hash_array(array):
    return hashlib.md5(array.tostring()).hexdigest()


class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def get_hashed_path(self, base_path):
        c = denumpyfy(dict(self))
        hash_value = hash_string(yaml.dump(c))
        with open(os.path.join(base_path, hash_value + '.cfg'), 'w') as f:
            f.write(yaml.dump(c, default_flow_style=False))
        return os.path.join(base_path, hash_value)


class share_variables(object):  # noqa: N801
    def __init__(self, callable_):
        self._callable = callable_
        self._wrappers = {}
        self._wrapper = None

    def __call__(self, *args, **kwargs):
        return self._function_wrapper(*args, **kwargs)

    def __get__(self, instance, owner):
        decorator = self._method_wrapper
        decorator = functools.partial(decorator, instance)
        decorator = functools.wraps(self._callable)(decorator)
        return decorator

    def _method_wrapper(self, instance, *args, **kwargs):
        if instance not in self._wrappers:
            name = self._create_name(
                type(instance).__module__,
                type(instance).__name__,
                instance.name if hasattr(instance, 'name') else id(instance),
                self._callable.__name__)
            self._wrappers[instance] = tf.make_template(name, self._callable,
                                                        create_scope_now_=True)
        return self._wrappers[instance](instance, *args, **kwargs)

    def _function_wrapper(self, *args, **kwargs):
        if not self._wrapper:
            name = self._create_name(self._callable.__module__,
                                     self._callable.__name__)
            self._wrapper = tf.make_template(name, self._callable,
                                             create_scope_now_=True)
        return self._wrapper(*args, **kwargs)

    def _create_name(self, *words):
        words = [str(word) for word in words]
        words = [word.replace('_', '') for word in words]
        return '_'.join(words)


def lazy_property(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


def define_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator


def print_graph_statistics():
    stats_string = ''
    for key in [tf.GraphKeys.GLOBAL_VARIABLES,
                tf.GraphKeys.TRAINABLE_VARIABLES,
                tf.GraphKeys.WEIGHTS,
                tf.GraphKeys.BIASES]:
        vars = tf.get_collection(key)
        param_count = 0
        for var in vars:
            param_count += np.prod(var.shape)
        stats_string += str(key) + ' statistics:\n'
        stats_string += '\t#vars:   %d\n' % len(vars)
        stats_string += '\t#params: %d\n' % param_count
    stats_string += 'Only weights are considered by weight decay/L2 ' \
                    'regularization!'
    print(stats_string)
    return stats_string


# python 2 compatability functions

def makedirs(path, exist_ok=False):
    if exist_ok:
        if os.path.exists(path):
            return
    os.makedirs(path)


# custom tensorflow operations
def tf_sign_0(x, value_for_zero=0, name="sign_0"):
    with tf.name_scope(name):
        if value_for_zero == 0:
            return tf.sign(x)
        else:
            return tf.sign(x) + value_for_zero * (1 - tf.abs(tf.sign(x)))


def tf_safe_div(x, y, name="safe_div", epsilon=1e-7):
    with tf.name_scope(name):
        y = y + tf_sign_0(y, 1) * epsilon
        return x / y
