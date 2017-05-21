# Copyright 2017 Davide Anastasia
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from math import exp, sqrt

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from vantgrd.common import logloss, sigmoid, compute_class_weight
from vantgrd.common import ClassificationTrainTracker


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

        self.class_weight = class_weight
        self.classes_ = None

        self.X_ = None
        self.y_ = None

        self.w0_ = 0.
        self.w_ = None
        self.V_ = None

        self.Ew0_ = 0.
        self.Ew_ = None
        self.EV_ = None

        self.fit_flag_ = False
        self.train_tracker_ = ClassificationTrainTracker(rate)

    def _clear_params(self):
        # All models parameters are set to their original value (see __init__ description)
        self.classes_ = None
        self.class_weight_ = None

        self.X_ = None
        self.y_ = None

        self.w0_ = 0.
        self.w_ = None
        self.V_ = None

        self.Ew0_ = 0.
        self.Ew_ = None
        self.EV_ = None

        self.fit_flag_ = False
        self.train_tracker_.clear()

    def _init_model(self):
        pass

    def _update_class_weight(self, _X, _y):
        if self.class_weight is None:
            self.class_weight_ = compute_class_weight(_y)
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

    def _train(self, X, y, n_samples, _):
        iter_idx = np.arange(n_samples)
        np.random.shuffle(iter_idx)

        g_sum = np.zeros(self.n_factors)
        g_sum_sqr = np.zeros(self.n_factors)
        for t, data_idx in enumerate(iter_idx):
            curr_x = X[data_idx, :]
            curr_y = y[data_idx]
            curr_y_adj = -1. if curr_y == 0. else 1.

            p = self._predict_with_feedback(curr_x, g_sum, g_sum_sqr)

            # TODO: multiplier can go out of control if the learning rate is too big, why?
            multiplier = -curr_y_adj * (1. - 1./(1. + exp(-curr_y_adj*p))) * self.class_weight_[curr_y]
            log_likelihood = logloss(p, curr_y)

            self.train_tracker_.track(log_likelihood)
            self._update(curr_x, g_sum, multiplier)

    def fit(self, X, y):
        if self.fit_flag_:
            self._clear_params()

        X, y = check_X_y(X, y)
        check_classification_targets(y)

        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y

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
        self.train_tracker_.start_train()
        for n_epoch in range(self.epochs):
            self.train_tracker_.start_epoch(n_epoch)
            self._train(X, y, n_samples, n_features)
            self.train_tracker_.end_epoch()

        self.train_tracker_.end_train()
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
