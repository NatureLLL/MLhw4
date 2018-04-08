from numpy import genfromtxt

__author__ = 'Otilia Stretcu'


def load_data_csv(filename):
    data = genfromtxt(filename, delimiter=',')
    x = data[:, :2]
    y = data[:, 2].astype(int)
    return x, y