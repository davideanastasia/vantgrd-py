import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split

from vantgrd.fm import FMWithAdagrad


mean_fpr = np.linspace(0, 1, 200)

X, y = datasets.make_classification(n_samples=100000, n_features=25,
                                    n_informative=7, n_redundant=5, n_repeated=3,
                                    random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

lr = FMWithAdagrad(eta=0.15, n_factors=5, epochs=1, rate=15000)
# lr = FMWithSGD(n_factors=5, epochs=1, rate=15000)
# lr = LogisticRegressionWithAdagrad(epochs=1, rate=15000)
# lr = LogisticRegressionWithAdadelta(epochs=1, rate=15000)
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
print classification_report(y_test, y_pred)

y_test_prob = lr.raw_predict(X_test)

fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)
roc_auc = auc(fpr, tpr)

# print "Weights = {}".format(lr.w_)
print "AUC = %f" % roc_auc

# for i in zip(y_test, y_test_prob):
#     p = 1. if i[1] >= 0.5 else 0.
#     print ('+' if i[0] == p else '-', i[0], p, i[1], )

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(fpr, tpr, color='blue', label='ROC area = %0.2f' % roc_auc)
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.grid()

# bx = fig.add_subplot(1, 2, 2)
# bx.scatter(y_test_prob, y_test, s=5, alpha=0.10, color='blue')
# bx.set_xlabel("Output Probability")
# bx.set_ylabel("Target Variable")
# bx.grid()

plt.show()
