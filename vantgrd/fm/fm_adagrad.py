import numpy as np
import random as rnd

from collections import Counter
from datetime import datetime
from math import exp, sqrt

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from vantgrd.common import logloss, sigmoid


class FMWithAdagrad(BaseEstimator, ClassifierMixin):
    def __init__(self, eta=0.001, k0=True, k1=True, reg0=.0, regw=.0, regv=.0, n_factors=2, epochs=1, rate=10000,
                 class_weight=None):
        self.eta = eta

        self.k0 = k0
        self.k1 = k1

        self.reg0 = reg0
        self.regw = regw
        self.regv = regv

        self.n_factors = n_factors

        self.epochs = epochs
        self.rate = rate

        self.class_weight = class_weight
        self.classes_ = None

        self.log_likelihood_ = 0
        self.loss_ = []

        self.X_ = None
        self.y_ = None

        self.w0_ = 0.
        self.w_ = None
        self.V_ = None

        self.Ew0_ = 0.
        self.Ew_ = None
        self.EV_ = None

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

        self.X_ = None
        self.y_ = None

        self.w0_ = 0.
        self.w_ = None
        self.V_ = None

        self.Ew0_ = 0.
        self.Ew_ = None
        self.EV_ = None

        self.fit_flag_ = False

    def _update_class_weight(self, _X, _y):
        if self.class_weight is None:
            c = Counter(_y)
            n_samples = _y.size
            self.class_weight_ = {0: float(c[1.0])/c[0.0], 1: 1.0}
            # self.class_weight_ = {0.0: float(c[1.0]) / n_samples, 1.0: float(c[0.0]) / n_samples}

            print self.class_weight_
        else:
            self.class_weight_ = self.class_weight

    def _update(self, curr_x, g_sum, multiplier):
        if self.k0:
            grad = multiplier
            self.Ew0_ += (grad * grad)
            self.w0_ -= (self.eta / sqrt(self.Ew0_ + 1e-8)) * (multiplier + 2. * self.reg0 * self.w0_)

        if self.k1:
            for idx in xrange(curr_x.size):
                if curr_x[idx] != 0.0:
                    grad = multiplier * curr_x[idx]
                    self.Ew_[idx] += (grad * grad)
                    self.w_[idx] -= (self.eta / sqrt(self.Ew_[idx] + 1e-8)) * (grad + 2. * self.regw * self.w_[idx])

        for f in xrange(self.n_factors):
            for idx in xrange(curr_x.size):
                if curr_x[idx] != 0.0:
                    grad = multiplier * curr_x[idx] * (g_sum[f] - self.V_[f, idx] * curr_x[idx])
                    self.EV_[f, idx] += (grad * grad)
                    self.V_[f, idx] -= (self.eta / sqrt(self.EV_[f, idx] + 1e-8)) * (grad + 2. * self.regv * self.V_[f, idx])

    def _predict_with_feedback(self, curr_x, g_sum, g_sum_sqr):
        result = 0.
        if self.k0:
            result += self.w0_

        if self.k1:
            result += np.dot(self.w_, curr_x)

        for f in xrange(self.n_factors):
            # v = self.V_[f, :]
            # g_sum[f] = float(0.)
            # g_sum_sqr[f] = float(0.)
            #
            # for idx in xrange(curr_x.size):
            #     d = v[idx] * curr_x[idx]
            #     g_sum[f] += d
            #     g_sum_sqr[f] += (d * d)
            #
            d = self.V_[f, :] * curr_x
            g_sum[f] = np.sum(d)
            g_sum_sqr[f] = np.dot(d, d)

            result += 0.5 * (g_sum[f] * g_sum[f] - g_sum_sqr[f])

        return result

    def _predict(self, curr_x):
        result = 0.
        if self.k0:
            result += self.w0_

        if self.k1:
            result += np.dot(self.w_, curr_x)

        for f in xrange(self.n_factors):
            d = self.V_[f, :] * curr_x
            g_sum = np.sum(d)
            g_sum_sqr = np.dot(d, d)

            result += 0.5 * (g_sum * g_sum - g_sum_sqr)

        return result

    def _train(self, X, y, n_iter, n_samples, n_features):
        start_time = datetime.now()
        iter_idx = np.arange(n_samples)
        np.random.shuffle(iter_idx)

        g_sum = np.zeros(self.n_factors)
        g_sum_sqr = np.zeros(self.n_factors)
        for t, data_idx in enumerate(iter_idx):
            curr_x = X[data_idx, :]
            curr_y = y[data_idx]
            curr_y_adj = -1. if curr_y == 0. else 1.

            p = self._predict_with_feedback(curr_x, g_sum, g_sum_sqr)

            # TODO: multiplier can go out of control if the learning rate is too big
            multiplier = -curr_y_adj * (1. - 1./(1. + exp(-curr_y_adj*(p + rnd.gauss(0, 0.05))))) * self.class_weight_[curr_y]

            self.log_likelihood_ += logloss(p, curr_y)

            t_adj = (n_iter * n_samples) + t + 1
            if self.rate > 0 and t_adj % self.rate == 0:
                # Append to the loss list.
                self.loss_.append(self.log_likelihood_)

                # Print all the current information
                print('Epoch: {0:3} | '
                      'Training Samples: {1:9} | '
                      'Loss: {2:11.2f} | '
                      'LossAdj: {3:8.5f} | '
                      'Time taken: {4:4} seconds'.format(n_iter, t_adj, self.log_likelihood_,
                                                         float(self.log_likelihood_) / t_adj,
                                                         (datetime.now() - start_time).seconds))

            self._update(curr_x, g_sum, multiplier)

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

        if self.w_ is None:
            self.Ew0_ = 0
            self.w0_ = 0

            self.Ew_ = np.zeros(n_features)
            self.w_ = np.zeros(n_features)

            self.EV_ = np.zeros((self.n_factors, n_features))
            self.V_ = np.random.normal(0, 0.1, (self.n_factors, n_features))

        self._update_class_weight(X, y)

        for n_iter in range(self.epochs):
            epoch_time = datetime.now()
            # if self.rate > 0:
            #     print('TRAINING EPOCH: {0:2}'.format(n_iter + 1))
            #     print('-' * 18)

            self._train(X, y, n_iter, n_samples, n_features)

            # if self.rate > 0:
            #     print('EPOCH {0:2} FINISHED IN {1} seconds'.format(
            #         n_iter + 1, (datetime.now() - epoch_time).seconds))

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

        g_sum = np.zeros(self.n_factors)
        g_sum_sqr = np.zeros(self.n_factors)

        for t in xrange(n_samples):
            p = sigmoid(self._predict_with_feedback(X[t, :], g_sum, g_sum_sqr))
            y_test_predict[t] = 0. if p < 0.5 else 1.

        return y_test_predict

    def raw_predict(self, X):
        check_is_fitted(self, ['X_', 'y_'])

        X = check_array(X)

        n_samples = X.shape[0]
        y_test_predict = np.zeros(n_samples)

        g_sum = np.zeros(self.n_factors)
        g_sum_sqr = np.zeros(self.n_factors)

        for t in xrange(n_samples):
            y_test_predict[t] = sigmoid(self._predict_with_feedback(X[t, :], g_sum, g_sum_sqr))

        return y_test_predict
