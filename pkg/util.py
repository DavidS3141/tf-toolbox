import functools
import hashlib
import numpy as np
import os
import re
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


def hash_string(string):
    return hashlib.md5(string.encode()).hexdigest()


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
