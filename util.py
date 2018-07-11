# -*- coding: utf-8 -*-

import functools
import re
import tensorflow as tf
import time


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
        NON_ALPHABETIC = re.compile('[^A-Za-z0-9_.]')
    else:
        NON_ALPHABETIC = re.compile('[^A-Za-z0-9_\-.=,:]')
    return NON_ALPHABETIC.sub('_', name)


class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


class share_variables(object):
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
