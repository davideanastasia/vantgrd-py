import numpy as np


def min_max_scale(X):
    return (X - X.min(0)) / X.ptp(0)


def normalize_data(X):
    return (X - X.mean(axis=0, dtype=np.float64)) / X.std(axis=0, dtype=np.float64)
