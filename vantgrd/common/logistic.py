from math import log, exp, sqrt


def logloss(p, y):
    p = max(min(p, 1. - 10e-12), 10e-12)
    return -log(p) if y == 1. else -log(1. - p)


def sigmoid(wtx):
    return 1. / (1. + exp(-max(min(wtx, 20.), -20.)))


def rms(z):
    return sqrt(z + 1e-8)
