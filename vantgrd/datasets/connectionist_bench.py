import numpy as np


def read_connectionist_bench(filename):
    X_ = []
    y_ = []

    with open(filename) as i_file:
        for line in i_file:
            tokens = line.strip().split(',')

            X_.append([float(tokens[x]) for x in xrange(0, 60)])
            y_.append(0. if tokens[60] == 'R' else 1.)

    # return np.asarray(X_, dtype=np.float64), np.asarray(y_, dtype=np.float64)
    return np.ascontiguousarray(X_, dtype=np.float64), np.ascontiguousarray(y_, dtype=np.float64)
