import numpy as np
import os
import matplotlib.pyplot as plt

__author__ = 'Otilia Stretcu'


def plot_points(x, y, title='', output_path=None, file_name=None, class_colors=None):
    colors = y > 0 if class_colors is None else [class_colors[c] for c in y]
    fig = plt.figure()
    plt.scatter(x[:, 0], x[:, 1], marker='o', c=colors)
    plt.title(title)
    plt.show()
    if output_path is not None and file_name is not None:
        fig.savefig(os.path.join(output_path, file_name))


def plot_svm_decision_boundary(svm, x, y, title='', output_path=None,
        file_name=None, class_colors=None):
    colors = y > 0 if class_colors is None else [class_colors[c] for c in y]
    fig = plt.figure()
    plt.scatter(x[:, 0], x[:, 1], c=colors)

    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx = np.linspace(xlim[0], xlim[1], 100)
    yy = np.linspace(ylim[0], ylim[1], 100)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    f = svm.decision_function(xy).reshape(XX.shape)

    ax.contour(XX, YY, f, colors='k', levels=[-1, 0, 1],
               alpha=0.5, linestyles=['--', '-', '--'])
    ax.scatter(svm.support_vectors[:, 0],
               svm.support_vectors[:, 1],
               s=100, linewidth=1, facecolors='none')
    plt.title(title)
    plt.show()
    if output_path is not None and file_name is not None:
        fig.savefig(os.path.join(output_path, file_name))