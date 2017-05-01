import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split

from vantgrd.logistic import LogisticRegressionFTRL, \
    LogisticRegressionWithAdagrad, LogisticRegressionWithAdadelta


mean_fpr = np.linspace(0, 1, 200)

X, y = datasets.make_classification(n_samples=200000, n_features=25,
                                    n_informative=7, n_redundant=5, n_repeated=3,
                                    random_state=42, weights=[0.77, 0.23])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

models = [
    LogisticRegressionWithAdagrad(eta=.01, epochs=2, rate=1000),
    LogisticRegressionWithAdadelta(epochs=2, rate=1000),
    LogisticRegressionFTRL(epochs=2, rate=1000)
]

colors = {0: 'blue', 1: 'red', 2: 'black'}
labels = {0: 'adagrad', 1: 'adadelta', 2: 'ftrl'}

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
