import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


def histnd(data, labels, bins=10, space=0.1, figsize=8, percentile_cut=0, **kwargs):
    f = data.shape[1]
    d_min = np.percentile(data, percentile_cut, axis=0)
    d_max = np.percentile(data, 100 - percentile_cut, axis=0)
    ranges = [(d_min[i], d_max[i]) for i in range(f)]

    plt.close()
    fig, axarr = plt.subplots(f, f, sharex='col', sharey='row',
                              figsize=(figsize, figsize))
    for i in range(f):
        for k in range(f):
            ax = axarr[i, k]
            if k < i:
                ax.hist2d(data[:, k], data[:, i], bins=bins,
                          range=[ranges[k], ranges[i]], **kwargs)
                if k == 0:
                    ax.set_ylabel(labels[i])
                if i == f - 1:
                    ax.set_xlabel(labels[k])
            elif k == i:
                if i == 0:
                    ax.xaxis.set_tick_params(labeltop=True)
                    ax.xaxis.tick_top()
                    ax.xaxis.set_label_position('top')
                    ax.yaxis.tick_right()
                    ax.yaxis.set_label_position('right')
                    ax.hist(data[:, i], bins=bins, range=ranges[i], **kwargs)
                    ax.set_xlabel(labels[0])
                    ax.set_ylabel('weighted count')
                elif i == f - 1:
                    plt.setp(ax.yaxis.get_major_ticks(), visible=False)
                    twinx = ax.twinx()
                    twiny = twinx.twiny()
                    twinx.get_shared_x_axes().join(twinx, twiny)
                    twiny.hist(data[:, i], bins=bins, range=ranges[i], **kwargs)
                    ax.set_xlabel(labels[i])
                    twiny.set_xlabel(labels[i])
                    twinx.set_ylabel('weighted count')
                else:
                    ax.xaxis.set_tick_params(labeltop=True)
                    ax.xaxis.tick_top()
                    ax.xaxis.set_label_position('top')
                    plt.setp(ax.yaxis.get_major_ticks(), visible=False)
                    twinx = ax.twinx()
                    twinx.hist(data[:, i], bins=bins, range=ranges[i], **kwargs)
                    ax.set_xlabel(labels[i])
                    twinx.set_ylabel('weighted count')
            if k < i < f - 1:
                plt.setp(ax.xaxis.get_major_ticks(), visible=False)
            if i > k > 0:
                plt.setp(ax.yaxis.get_major_ticks(), visible=False)
    for i in range(f):
        for k in range(i + 1, f):
            axarr[i, k].remove()
    fig.subplots_adjust(hspace=space, wspace=space)
