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
#	# TODO: implement this.
#    num_samples1 = x.shape[0]
#    num_samples2 = y.shape[0]
#    K = np.zeros((num_samples1,num_samples2))
#    for i in range(num_samples1):
#        for j in range(num_samples2):
#            eu_dist = np.sum((x[i,:]-y[j,:])**2)
#            K[i,j] = np.exp(-gamma*eu_dist)
    if (len(x.shape) == 1) and (len(y.shape) == 1):
        eu_dist = np.sum((x-y)**2)
    else:
        eu_dist = np.sum((x-y)**2,axis=1)
    
    K = np.exp(-gamma*eu_dist)
    return K
