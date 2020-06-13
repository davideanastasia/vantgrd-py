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

from math import fabs, sqrt

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.multiclass import check_classification_targets

from vantgrd.common import logloss, sigmoid, compute_class_weight
from vantgrd.common import ClassificationTrainTracker


class LogisticRegressionFTRL(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=0.05, beta=0.05, l1=.01, l2=.01, epochs=1, rate=50000, class_weight=None):
        self.alpha = alpha
        self.beta = beta
        self.l1 = l1
        self.l2 = l2

        self.epochs = epochs

        self.class_weight = class_weight
        self.classes_ = None

        self.X_ = None
        self.y_ = None
        self.z_ = None
        self.n_ = None

        self.fit_flag_ = False
        self.train_tracker_ = ClassificationTrainTracker(rate)

    def _clear_params(self):
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
            self.class_weight_ = compute_class_weight(_y)
        else:
            self.class_weight_ = self.class_weight

    def _update(self, y, p, x, w):
        d = (p - y)
        for idxi in range(len(x)):  # for idxi, xi in enumerate(x):
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

    def _train(self, X, y, n_samples, n_features):
        iter_idx = np.arange(n_samples)
        np.random.shuffle(iter_idx)
        for t, data_idx in enumerate(iter_idx):
            curr_x = X[data_idx, :]
            curr_y = y[data_idx]

            wtx = 0.
            curr_w = {}
            for idxi in range(n_features):
                curr_w[idxi] = self._get_w(idxi)
                wtx += (curr_w[idxi] * curr_x[idxi])

            curr_p = sigmoid(wtx)
            log_likelihood = logloss(curr_p, curr_y)

            self.train_tracker_.track(log_likelihood)
            self._update(curr_y, curr_p, curr_x, curr_w)

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

        if self.z_ is None and self.n_ is None:
            self.z_ = np.zeros(n_features)
            self.n_ = np.zeros(n_features)

        self._update_class_weight(X, y)
        self.train_tracker_.start_train()
        for epoch in range(self.epochs):
            self.train_tracker_.start_epoch(epoch)
            self._train(X, y, n_samples, n_features)
            self.train_tracker_.end_epoch()

        self.train_tracker_.end_train()
        self.fit_flag_ = True

        return self

    def predict(self, X):
        check_is_fitted(self, ['X_', 'y_'])

        X = check_array(X)

        n_samples, n_features = X.shape
        y_test_predict = np.zeros(n_samples)

        w = np.zeros(n_features)
        for idxi in range(n_features):
            w[idxi] = self._get_w(idxi)

        # print w

        for t in range(n_samples):
            x = X[t, :]
            wtx = np.dot(w, x)
            p = sigmoid(wtx)
            y_test_predict[t] = 0. if p < 0.5 else 1.

        return y_test_predict

    def raw_predict(self, X):
        check_is_fitted(self, ['X_', 'y_'])

        X = check_array(X)
        n_samples, n_features = X.shape

        w = np.zeros(n_features)
        for idxi in range(n_features):
            w[idxi] = self._get_w(idxi)

        y = np.dot(X, w)
        for idxi in range(y.size):
            y[idxi] = sigmoid(y[idxi])

        return y
