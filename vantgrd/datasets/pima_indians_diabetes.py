import numpy as np


def read_pima_indians_diabetes(filename):
    X_ = []
    y_ = []

    with open(filename) as i_file:
        for line in i_file:
            tokens = [float(x) for x in line.strip().split(',')]

            x_ = [tokens[0], tokens[1], tokens[2], tokens[3],
                  tokens[4], tokens[5], tokens[6], tokens[7]]

            X_.append(x_)
            y_.append(tokens[8])

    # return np.asarray(X_, dtype=np.float64), np.asarray(y_, dtype=np.float64)
    return np.ascontiguousarray(X_, dtype=np.float64), np.ascontiguousarray(y_, dtype=np.float64)
