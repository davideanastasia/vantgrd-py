from collections import Counter


def compute_class_weight(y):
    c = Counter(y)

    # n_samples = y.size
    # return {0.0: float(c[1.0]) / n_samples, 1.0: float(c[0.0]) / n_samples}
    return {0: float(c[1.0]) / c[0.0], 1: 1.0}
