import numpy as np

__author__ = 'Otilia Stretcu'


def accuracy(predictions, targets):
	# TODO: implement this.
    acc = (float) (np.sum(predictions==targets))
    acc = acc/targets.shape[0]
    return acc