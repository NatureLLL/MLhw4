import numpy as np

__author__ = 'Otilia Stretcu'


def linear(x, y):
    """
    Calculates k(x_i, x_j) to be the dot product <x_i, x_j>.
    :return:
    """
    if len(x.shape) == 1:
        return y.dot(x)
    else:
        return x.dot(y)


def rbf(x, y, gamma=None):
	# TODO: implement this.
    eu_dist = np.sum((x-y)**2)
    res = np.exp(-gamma*eu_dist)
    return res
