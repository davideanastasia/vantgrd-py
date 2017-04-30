import numpy as np

from sklearn import datasets
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from vantgrd.logistic import LogisticRegressionWithAdadelta

mean_fpr = np.linspace(0, 1, 200)

X, y = datasets.make_classification(n_samples=100000, n_features=25,
                                    n_informative=7, n_redundant=5, n_repeated=3,
                                    random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
#
# parameters = {
#     'alpha': (0.1, 0.05),
#     'beta': (0.05, 0.01),
#     'l1': (0.1, 1.0, 5.0),
#     'l2': (0.01, 0.1, 1.0)
# }
#
# lr = LogisticRegressionFTRL(epochs=1, rate=0)


parameters = {
    'rho': (0.9, 0.75, 0.6),
    'regw': (.0, 0.001, 0.01, 0.1)
}

lr = LogisticRegressionWithAdadelta(epochs=1, rate=0)


clf = GridSearchCV(lr, param_grid=parameters, verbose=1, n_jobs=4)
clf.fit(X_train, y_train)

# print sorted(clf.cv_results_.keys())

print clf.best_params_

y_pred = clf.predict(X_test)

print classification_report(y_test, y_pred)
# print confusion_matrix(y_test, y_pred)