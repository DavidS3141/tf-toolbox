import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


class ConvergenceChecker(object):
    def __init__(self, min_iters=3, max_iters=int(1e13), min_confirmations=1):
        self.min_iters = min(max(min_iters, 1 + 2 * min_confirmations),
                             max_iters)
        self.max_iters = max_iters
        self.min_confirmations = min_confirmations
        assert isinstance(self.min_iters, int)
        assert isinstance(self.max_iters, int)
        assert isinstance(self.min_confirmations, int)
        self.reset()

    def __len__(self):
        return len(self.values)

    def reset(self):
        self.values = []
        self._is_converged = False
        self._logged_lr = np.inf

    @property
    def values_of_interest(self):
        return self.values[self.min_confirmations:]

    def check(self, value, lr=1):
        if self.is_best(value):
            self._logged_lr = lr
        elif lr > 2 * self._logged_lr:
            return False
        self.values.append(value)
        n = len(self.values)
        if n < self.min_iters:
            return False
        elif n >= self.max_iters:
            self._is_converged = True
            return True
        self._is_converged = self._is_converged or \
            (self.get_nbr_confirmations() >= self.min_confirmations)
        return self._is_converged

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

    def is_converged(self):
        return self._is_converged

    def get_nbr_confirmations(self):
        if len(self.values) < self.min_confirmations + 1:
            return 0
        return len(self.values_of_interest) - \
            np.argmin(self.values_of_interest) - 1

    def create_plot(self, plot_name):
        plt.close()
        plt.plot(np.arange(len(self.values)), self.values)
        plt.axvline(x=self.min_confirmations - 0.5)
        plt.axvline(x=len(self.values) - self.min_confirmations - 0.5)
        plt.savefig(plot_name)
