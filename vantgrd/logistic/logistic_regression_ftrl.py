import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.multiclass import check_classification_targets

from math import fabs, sqrt, exp, log
from random import random
from datetime import datetime


class LogisticRegressionFTRL(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=0.05, beta=0.05, l1=0.001, l2=0.001, epochs=1, rate=50000):
                 # class_weight=None):  # , subsample=1.):
        self.alpha = alpha
        self.beta = beta
        self.l1 = l1
        self.l2 = l2

        # self.subsample = subsample
        self.epochs = epochs
        self.rate = rate

        self.class_weight = None  # class_weight
        self.classes_ = None

        self.log_likelihood_ = 0
        self.loss_ = []

        self.target_ratio_ = 0.

        self.X_ = None
        self.y_ = None
        self.z_ = None
        self.n_ = None
        self.fit_flag_ = False

    def _clear_params(self):
        """
        If the fit method is called multiple times, all trained parameters
        must be cleared allowing for a fresh start. This function simply
        resets everything back to square one.
        :return: Nothing
        """

        # All models parameters are set to their original value (see __init__ description)
        self.classes_ = None
        self.class_weight_ = None
        self.log_likelihood_ = 0
        self.loss_ = []
        self.target_ratio_ = 0.
        self.X_ = None
        self.y_ = None
        self.z_ = None
        self.n_ = None
        self.fit_flag_ = False

    def _update_class_weight(self, _X, _y):
        if self.class_weight is None:
            self.class_weight_ = {0: 1.0, 1: 1.0}
        else:
            self.class_weight_ = self.class_weight

    def _update(self, y, p, x, w):
        d = (p - y)
        for idxi in xrange(len(x)):  # for idxi, xi in enumerate(x):
            g = d * x[idxi]
            s = (sqrt(self.n_[idxi] + g * g) - sqrt(self.n_[idxi])) / self.alpha

            self.z_[idxi] += self.class_weight_[y] * (g - s * w[idxi])
            self.n_[idxi] += self.class_weight_[y] * (g * g)

    def _get_w(self, idxi):
        if fabs(self.z_[idxi]) <= self.l1:
            return 0.
        else:
            sign = 1. if self.z_[idxi] >= 0 else -1.
            return - (self.z_[idxi] - sign * self.l1) / (self.l2 + (self.beta + sqrt(self.n_[idxi])) / self.alpha)

    def _logloss(self, p, y):
        p = max(min(p, 1. - 10e-12), 10e-12)
        return -log(p) if y == 1. else -log(1. - p)

    def _sigmoid(self, wtx):
        return 1. / (1. + exp(-max(min(wtx, 20.), -20.)))

    def _train(self, X, y, n_samples, n_features):
        start_time = datetime.now()
        iter_idx = np.arange(n_samples)
        np.random.shuffle(iter_idx)
        for t, data_idx in enumerate(iter_idx):
            curr_x = X[data_idx]
            curr_y = y[data_idx]

            # if curr_y < 1. and random() > self.subsample and (t + 1) % self.rate != 0:
            #     continue

            self.target_ratio_ = (1.0 * (t * self.target_ratio_ + curr_y)) / (t + 1)

            wtx = 0.
            curr_w = {}
            for idxi in xrange(n_features):
                curr_w[idxi] = self._get_w(idxi)
                wtx += (curr_w[idxi] * curr_x[idxi])

            curr_p = self._sigmoid(wtx)
            self.log_likelihood_ += self._logloss(curr_p, curr_y)

            if (self.rate > 0) and (t + 1) % self.rate == 0:
                # Append to the loss list.
                self.loss_.append(self.log_likelihood_)

                # Print all the current information
                print('Training Samples: {0:9} | '
                      'Loss: {1:11.2f} | '
                      'LossAdj: {2:8.5f} | '
                      'Time taken: {3:4} seconds'.format(t + 1,
                                                         self.log_likelihood_,
                                                         float(self.log_likelihood_) / (t + 1),
                                                         (datetime.now() - start_time).seconds))

            self._update(curr_y, curr_p, curr_x, curr_w)

    def fit(self, X, y):
        if self.fit_flag_:
            self._clear_params()

        X, y = check_X_y(X, y)
        check_classification_targets(y)

        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y

        total_time = datetime.now()

        # setup parameters
        n_samples, n_features = X.shape

        if self.z_ is None and self.n_ is None:
            self.z_ = np.zeros(n_features)
            self.n_ = np.zeros(n_features)

        self._update_class_weight(X, y)

        for epoch in range(self.epochs):
            epoch_time = datetime.now()
            if self.rate > 0:
                print('TRAINING EPOCH: {0:2}'.format(epoch + 1))
                print('-' * 18)

            self._train(X, y, n_samples, n_features)

            if self.rate > 0:
                print('EPOCH {0:2} FINISHED IN {1} seconds'.format(
                    epoch + 1, (datetime.now() - epoch_time).seconds))

        if self.rate > 0:
            print(' --- TRAINING FINISHED IN {0} SECONDS WITH LOSS {1:.2f} ---'.format(
                (datetime.now() - total_time).seconds, self.log_likelihood_))

        # --- Fit Flag
        # Set fit_flag to true. If fit is called again this is will trigger
        # the call of _clean_params. See partial_fit for different usage.
        self.fit_flag_ = True

        return self

    def predict(self, X):
        check_is_fitted(self, ['X_', 'y_'])

        X = check_array(X)

        n_samples, n_features = X.shape
        y_test_predict = np.zeros(n_samples)

        w = np.zeros(n_features)
        for idxi in xrange(n_features):
            w[idxi] = self._get_w(idxi)

        # print w

        for t in xrange(n_samples):
            x = X[t]
            wtx = np.dot(w, x)
            p = self._sigmoid(wtx)
            y_test_predict[t] = 0. if p < 0.5 else 1.

        return y_test_predict

    def raw_predict(self, X):
        check_is_fitted(self, ['X_', 'y_'])

        X = check_array(X)
        n_samples, n_features = X.shape

        w = np.zeros(n_features)
        for idxi in xrange(n_features):
            w[idxi] = self._get_w(idxi)

        y = np.dot(X, w)
        for idxi in xrange(y.size):
            y[idxi] = self._sigmoid(y[idxi])

        return y
