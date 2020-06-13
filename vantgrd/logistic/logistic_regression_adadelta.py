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

from math import sqrt

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from vantgrd.common import logloss, sigmoid, compute_class_weight
from vantgrd.common import ClassificationTrainTracker


class LogisticRegressionWithAdadelta(BaseEstimator, ClassifierMixin):
    def __init__(self, rho=0.9, regw=.01, epochs=1, rate=1000, class_weight=None):
        self.rho = rho
        self.regw = regw

        self.epochs = epochs

        self.class_weight = class_weight
        self.classes_ = None

        self.X_ = None
        self.y_ = None

        self.E_ = None
        self.Edx_ = None
        self.w_ = None

        self.fit_flag_ = False
        self.train_tracker_ = ClassificationTrainTracker(rate)

    def _clear_params(self):
        # All models parameters are set to their original value (see __init__ description)
        self.classes_ = None
        self.class_weight_ = None

        self.X_ = None
        self.y_ = None

        self.E_ = None
        self.Edx_ = None
        self.w_ = None

        self.fit_flag_ = False
        self.train_tracker_.clear()

    def _update_class_weight(self, _X, _y):
        if self.class_weight is None:
            self.class_weight_ = compute_class_weight(_y)
        else:
            self.class_weight_ = self.class_weight

    def _update(self, y, p, x):
        for idxi, xi in enumerate(x):
            if xi != 0.0:
                grad = self.class_weight_[y] * ((p - y) * xi + self.regw * self.w_[idxi])
                self.E_[idxi] = self.rho * self.E_[idxi] + (1.0 - self.rho) * grad * grad
                deltax = - (sqrt(self.Edx_[idxi] + 1e-8) / sqrt(self.E_[idxi] + 1e-8)) * grad
                self.Edx_[idxi] = self.rho * self.Edx_[idxi] + (1.0 - self.rho) * deltax * deltax
                self.w_[idxi] += deltax

    def _train(self, X, y, n_samples, _):
        iter_idx = np.arange(n_samples)
        np.random.shuffle(iter_idx)

        for t, data_idx in enumerate(iter_idx):
            curr_x = X[data_idx, :]
            curr_y = y[data_idx]

            wtx = np.dot(curr_x, self.w_)

            curr_p = sigmoid(wtx)
            log_likelihood = logloss(curr_p, curr_y)

            self.train_tracker_.track(log_likelihood)
            self._update(curr_y, curr_p, curr_x)

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

        if self.E_ is None and self.w_ is None:
            self.E_ = np.zeros(n_features)
            self.Edx_ = np.zeros(n_features)
            self.w_ = np.zeros(n_features)

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

        n_samples = X.shape[0]
        y_test_predict = np.zeros(n_samples)

        for t in range(n_samples):
            wtx = np.dot(X[t, :], self.w_)
            p = sigmoid(wtx)
            y_test_predict[t] = 0. if p < 0.5 else 1.

        return y_test_predict

    def raw_predict(self, X):
        check_is_fitted(self, ['X_', 'y_'])

        X = check_array(X)

        y = np.dot(X, self.w_)
        for idxi in range(y.size):
            y[idxi] = sigmoid(y[idxi])

        return y
