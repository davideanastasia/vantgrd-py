#
# This file is part of vantgrd-py. vantgrd-py is free software: you can
# redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 2.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program; if not, write to the Free Software Foundation, Inc., 51
# Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# Copyright 2017 Davide Anastasia
#
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
labels = {0: 'lr-adagrad', 1: 'lr-adadelta', 2: 'lr-ftrl', 3: 'lr-fm-adagrad', 4: 'fm-sgd'}

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
