__author__ = 'thibautxiong'

import sklearn.linear_model.LinearRegression as LR


class LinearRegression():
    def __init__(self, fit_intercept=True, normalize=False, copy_X=True, n_jobs=1):
        self.LR = LR(fit_intercept, normalize, copy_X, n_jobs)

    def decision_function(self, x):
        return self.LR.decision_function(x)

    def fit(self, x, y):
        return self.LR.fit(x, y)

    def get_params(self):
        return self.LR.get_params()

    def predict(self, x):
        return self.LR.predict(x)

    def set_params(self, **params):
        self.LR.set_params(params)