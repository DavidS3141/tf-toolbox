from .util import makedirs

import numpy as np
from time import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


def merge_total_time_dict(a, b):
    all_keys = set(a).union(set(b))
    total = dict()
    for key in all_keys:
        t_a, d_a = a.get(key, (0, dict()))
        t_b, d_b = b.get(key, (0, dict()))
        total[key] = (t_a + t_b, merge_total_time_dict(d_a, d_b))
    return total


class Timer(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_times = dict()
        self.running_timer = None
        self.running_child_timer = None

    def start(self, name):
        if self.running_timer is None:
            assert self.running_child_timer is None
            self.running_timer = (name, time())
            self.running_child_timer = Timer()
        else:
            self.running_child_timer.start(name)

    def stop(self, name):
        if self.running_timer is None:
            raise ValueError('"%s" timer was not started!' % name)
        if self.running_timer[0] == name:
            self.total_times[name] = self.total_times.get(name, (0., dict()))
            if self.running_child_timer.running_timer is not None:
                raise ValueError('There is a running child timer "%s" of "%s"!'
                                 % (self.running_child_timer.running_timer[0],
                                    self.running_timer[0]))
            child_total_time = merge_total_time_dict(
                self.total_times[name][1], self.running_child_timer.total_times)
            self.total_times[name] = (
                self.total_times[name][0] + time() - self.running_timer[1],
                child_total_time
            )
            self.running_timer = None
            self.running_child_timer = None
        else:
            try:
                self.running_child_timer.stop(name)
            except ValueError:
                print(self.running_timer)
                raise

    def stop_all(self):
        running_timer = self.get_current_timer()
        for cur_timer in running_timer[::-1]:
            self.stop(cur_timer)

    def get_current_timer(self):
        if self.running_timer is None:
            return []
        return [self.running_timer[0]] + \
            self.running_child_timer.get_current_timer()

    def get_max_depth(self, d=None):
        if d is None:
            return self.get_max_depth(self.total_times)
        max_sub_depth = 0
        for k in d:
            max_sub_depth = max(max_sub_depth, 1 + self.get_max_depth(d[k][1]))
        return max_sub_depth

    def get_times_labels_colorids(self, depth, time_dict=None,
                                  color_range=(0, 1)):
        if time_dict is None:
            time_dict = self.total_times
        assert len(time_dict) > 0
        split_range = np.linspace(*color_range, num=len(time_dict) + 1)
        delta_for_cutoff = 0
        if len(time_dict) > 1:
            delta_for_cutoff = split_range[1] - split_range[0]
        if depth == 0:
            mids = 0.5 * (split_range[1:] + split_range[:-1])
            labels = sorted(list(time_dict))
            times = [time_dict[l][0] for l in labels]
            colorids = [c for c in mids]
            return times, labels, colorids
        labels = []
        times = []
        colorids = []
        for i, k in enumerate(sorted(list(time_dict))):
            if len(time_dict[k][1]) > 0:
                t, labs, c = self.get_times_labels_colorids(
                    depth - 1, time_dict=time_dict[k][1],
                    color_range=(split_range[i] + delta_for_cutoff/6,
                                 split_range[i+1] - delta_for_cutoff/6))
            else:
                t = []
                labs = []
                c = []
            t += [time_dict[k][0] - sum(t)]
            labs += ['']
            c += [-1]
            labels += labs
            times += t
            colorids += c
        return times, labels, colorids

    def create_plot(self, path, timer_dict=None):
        timer_dict = timer_dict or self.total_times

        plt.close()
        labels = sorted(list(timer_dict))
        times = [timer_dict[lab][0] for lab in labels]
        total_time = sum(times)
        labels = ['%s %.1fs' % lab_time for lab_time in zip(labels, times)]
        times = [t / min(times) for t in times]  # make sure sum > 1
        plt.pie(times, labels=labels, autopct='%.1f%%',
                shadow=True, startangle=90, counterclock=False)
        plt.axis('equal')
        plt.title('total time: %.1fs' % total_time)
        plt.tight_layout()
        plt.savefig(path + '.png')

        size = 0.3
        radius = 0.6
        plt.close()
        cmap = plt.get_cmap('gist_rainbow')
        total_time = sum([timer_dict[k][0] for k in timer_dict])
        for d in range(self.get_max_depth(timer_dict)):
            times, labels, colorids = self.get_times_labels_colorids(
                d, time_dict=timer_dict)
            colors = [cmap(cid) if cid != -1 else 'w' for cid in colorids]
            labels = ['%s %.1fs (%.1f%%)' %
                      (label, time, time * 100 / total_time)
                      if label != '' else ''
                      for label, time in zip(labels, times)]
            times = [t / min(times) for t in times]  # make sure sum > 1
            plt.pie(times, labels=labels, colors=colors, radius=radius,
                    wedgeprops=dict(width=size, edgecolor='w'), shadow=True,
                    startangle=90, counterclock=False, labeldistance=0.8)
            radius += size
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(path + '_hierarchy.png')

        for k in timer_dict:
            if len(timer_dict[k][1]) == 0:
                continue
            makedirs(path, exist_ok=True)
            self.create_plot(path + '/' + k, timer_dict[k][1])
