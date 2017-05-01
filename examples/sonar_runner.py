import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix

from vantgrd.datasets import read_connectionist_bench, normalize_data
from vantgrd.logistic import LogisticRegressionWithAdadelta, LogisticRegressionWithAdagrad, LogisticRegressionFTRL
from vantgrd.fm import FMWithAdagrad, FMWithSGD

mean_fpr = np.linspace(0, 1, 200)
rate = 500
epochs = 50
cw = {0: 1., 1: 1.}

X, y = read_connectionist_bench('data/connectionist-bench/sonar.all-data')
X = normalize_data(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# lr = FMWithSGD(eta=.01, k0=True, k1=True, regw=.01, regv=.01, n_factors=4, epochs=epochs, rate=rate, class_weight=cw))
lr = FMWithAdagrad(eta=.025, k0=True, k1=True, regw=.01, regv=.01, n_factors=4, epochs=epochs, rate=rate, class_weight=cw)
# lr = LogisticRegressionWithAdagrad(eta=0.25, epochs=epochs, rate=rate, class_weight=cw))
# lr = LogisticRegressionWithAdadelta(rho=0.8, regw=.001, epochs=epochs, rate=rate, class_weight=cw))
# lr = LogisticRegressionFTRL(epochs=epochs, rate=rate, class_weight=cw))
lr.fit(X_train, y_train)

print lr.class_weight_
print lr.class_weight

y_pred = lr.predict(X_test)
print classification_report(y_test, y_pred)

print confusion_matrix(y_test, y_pred)

y_test_prob = lr.raw_predict(X_test)

fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)

roc_auc = auc(fpr, tpr)

print "AUC = %f" % roc_auc
print "Score = {}".format(lr.score(X, y))

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
ax.plot(fpr, tpr, color='blue', label='ROC area = %0.2f' % roc_auc)
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.grid()

bx = fig.add_subplot(1, 2, 2)
bx.scatter(y_test_prob, y_test, s=5, alpha=0.10, color='blue')
bx.set_xlabel("Output Probability")
bx.set_ylabel("Target Variable")
bx.grid()

plt.show()
