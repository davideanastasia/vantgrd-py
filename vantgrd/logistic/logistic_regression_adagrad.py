import random as rnd
import numpy as np

from collections import Counter
from datetime import datetime
from math import sqrt

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from vantgrd.common import logloss, sigmoid


class LogisticRegressionWithAdagrad(BaseEstimator, ClassifierMixin):
    def __init__(self, eta=0.001, regw=0.01, epochs=1, rate=50000):
        self.regw = regw
        self.eta = eta

        # self.subsample = subsample
        self.epochs = epochs
        self.rate = rate

        self.class_weight = None  # class_weight
        self.classes_ = None

        self.log_likelihood_ = 0
        self.loss_ = []

        self.target_ratio_ = 0.
        #
        self.X_ = None
        self.y_ = None
        self.E_ = None
        self.w_ = None
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
        self.E_ = None
        self.w_ = None
        self.fit_flag_ = False

    def _update_class_weight(self, _X, _y):
        if self.class_weight is None:
            c = Counter(_y)
            self.class_weight_ = {0: float(c[1.0])/c[0.0], 1: 1.0}
        else:
            self.class_weight_ = self.class_weight

    def _update(self, y, p, x):
        for idxi, xi in enumerate(x):
            if xi != 0.0:
                grad = self.class_weight_[y] * ((p - y - rnd.gauss(0, 0.1)) * xi + self.regw * self.w_[idxi])
                self.E_[idxi] += (grad * grad)
                self.w_[idxi] -= (self.eta / sqrt(self.E_[idxi] + 1e-8)) * grad

    def _train(self, X, y, n_iter, n_samples, n_features):
        start_time = datetime.now()
        iter_idx = np.arange(n_samples)
        np.random.shuffle(iter_idx)
        for t, data_idx in enumerate(iter_idx):
            curr_x = X[data_idx, :]
            curr_y = y[data_idx]

            # if curr_y < 1. and random() > self.subsample and (t + 1) % self.rate != 0:
            #     continue

            self.target_ratio_ = (1.0 * (t * self.target_ratio_ + curr_y)) / (t + 1)

            wtx = np.dot(curr_x, self.w_)

            curr_p = sigmoid(wtx)
            self.log_likelihood_ += logloss(curr_p, curr_y)

            t_adj = (n_iter * n_samples) + t + 1
            if self.rate > 0 and t_adj % self.rate == 0:
                # Append to the loss list.
                self.loss_.append(self.log_likelihood_)

                # Print all the current information
                print('Epoch: {0:3} | '
                      'Training Samples: {1:9} | '
                      'Loss: {2:11.2f} | '
                      'LossAdj: {3:8.5f} | '
                      'Time taken: {4:4} seconds'.format(n_iter, t_adj,
                                                         self.log_likelihood_,
                                                         float(self.log_likelihood_) / t_adj,
                                                         (datetime.now() - start_time).seconds))

            self._update(curr_y, curr_p, curr_x)

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

        if self.E_ is None and self.w_ is None:
            self.E_ = np.zeros(n_features)
            self.w_ = np.zeros(n_features)

        self._update_class_weight(X, y)

        for epoch in range(self.epochs):
            # epoch_time = datetime.now()
            # if self.rate > 0:
            #     print('TRAINING EPOCH: {0:2}'.format(epoch + 1))
            #     print('-' * 18)

            self._train(X, y, epoch, n_samples, n_features)

            # if self.rate > 0:
            #     print('WEIGHTS = {0}'.format(self.w_))
            #     print('EPOCH {0:2} FINISHED IN {1} seconds'.format(epoch + 1, (datetime.now() - epoch_time).seconds))

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

        n_samples = X.shape[0]
        y_test_predict = np.zeros(n_samples)

        for t in xrange(n_samples):
            wtx = np.dot(X[t,:], self.w_)
            p = sigmoid(wtx)
            y_test_predict[t] = 0. if p < 0.5 else 1.

        return y_test_predict

    def raw_predict(self, X):
        check_is_fitted(self, ['X_', 'y_'])

        X = check_array(X)

        y = np.dot(X, self.w_)
        for idxi in xrange(y.size):
            y[idxi] = sigmoid(y[idxi])

        return y
