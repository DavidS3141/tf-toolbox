#!/usr/bin/env python
import numpy as np


class Convergence_Checker(object):
    def __init__(self, max_iters=int(1e13), min_iters=3, min_confirmations=1):
        self.min_iters = max(min_iters, 1 + 2 * min_confirmations)
        self.max_iters = max_iters
        self.min_confirmations = min_confirmations
        self.reset()

    def __len__(self):
        return len(self.values)

    def reset(self):
        self.values = []

    @property
    def values_of_interest(self):
        return self.values[self.min_confirmations:]

    def check(self, value):
        self.values.append(value)
        n = len(self.values)
        if n < self.min_iters:
            return False
        elif n >= self.max_iters:
            return True
        return (len(self.values_of_interest)
                - np.argmin(self.values_of_interest)
                >
                self.min_confirmations)

    def get_best(self):
        n = len(self.values)
        if n == 0:
            return - np.inf
        if n < self.min_confirmations + 1:
            return self.values[-1]
        return np.min(self.values_of_interest)

    def is_best(self, value):
        n = len(self.values)
        if n < self.min_confirmations:
            return False
        if n == self.min_confirmations:
            return True
        return np.min(self.values_of_interest) >= value
