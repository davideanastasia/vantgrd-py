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
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split

from vantgrd.logistic import LogisticRegressionFTRL, \
    LogisticRegressionWithAdagrad, LogisticRegressionWithAdadelta
from vantgrd.fm import FMWithAdagrad, FMWithSGD

epochs = 1
mean_fpr = np.linspace(0, 1, 200)

X, y = datasets.make_classification(n_samples=200000, n_features=25,
                                    n_informative=7, n_redundant=5, n_repeated=3,
                                    random_state=42, weights=[0.77, 0.23])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

models = [
    LogisticRegressionWithAdagrad(eta=.01, epochs=epochs, rate=1000),
    LogisticRegressionWithAdadelta(epochs=epochs, rate=1000),
    LogisticRegressionFTRL(epochs=epochs, rate=1000),
    FMWithAdagrad(eta=0.01, reg0=.01, regw=.01, regv=.01, rate=1000, epochs=epochs, n_factors=5),
    FMWithSGD(eta=0.01, reg0=.01, regw=.01, regv=.01, rate=1000, epochs=epochs, n_factors=5)
]

colors = {0: 'blue', 1: 'red', 2: 'black', 3: 'green', 4: 'yellow'}
labels = {0: 'lr-adagrad', 1: 'lr-adadelta', 2: 'lr-ftrl', 3: 'fm-adagrad', 4: 'fm-sgd'}

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
ax.grid()
ax.set_ylabel("Logloss")

bx = fig.add_subplot(1, 2, 2)
bx.grid()
bx.set_xlabel("False positive rate")
bx.set_ylabel("True positive rate")

for idx, lr in enumerate(models):
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_test)
    print classification_report(y_test, y_pred)

    ax.plot(lr.train_tracker_.loss_, color=colors[idx], label=labels[idx])

    y_test_prob = lr.raw_predict(X_test)

    fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)
    roc_auc = auc(fpr, tpr)

    print "AUC = %f" % roc_auc

    bx.plot(fpr, tpr, color=colors[idx], label='{0} ROC area = {1:.2f}'.format(labels[idx], roc_auc))

plt.show()
