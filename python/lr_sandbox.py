from __future__ import division
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import os
from pykalman import KalmanFilter
import statsmodels.api as sm
from backtest.backtest import backtest, Order
from datautils.data_utils import get_data, get_more_data
import datautils.features as features


from sklearn import cross_validation, svm


def cross_validation(data, clf, feature_columns, ycol='log_returns_100+', label='label', K=5,
                     fit_method=lambda cl, X, y: cl.fit(X, y),
                     predict_method=lambda cl, X: cl.predict(X)):
    """

    :param data:
    :param clf:
    :param feature_columns:
    :param label:
    :param K:
    :param fit_method:
    :param predict_method:
    :return:

    define false positive as predicting incorrect 1 or -1 value
    examples:
    true    predicted
    0,-1       1
    0,1       -1

    counter-examples:
    0,1,-1      0
    1           1
    -1          -1
    """
    #data.apply(np.random.shuffle, axis=0)
    partitions = np.array_split(data, K)

    results = {
        'run': range(1, K+1) + ['Total'],
        'acc': [None]*(K+1),
        'fpr': [None]*(K+1),
        'fnr': [None]*(K+1),
        'bpr': [None]*(K+1),
        'gpr': [None]*(K+1)
    }

    for k in xrange(K):
        training_data = pd.concat(partitions[:k] + partitions[(k+1):])
        testing_data = partitions[k]
        train_x, train_y = training_data[feature_columns], training_data[ycol]
        test_x, test_y = testing_data[feature_columns], testing_data[label]
        fit_method(clf, train_x, train_y)
        pred_y = predict_method(clf, test_x)
        n = np.size(test_y)
        results['acc'][k] = np.sum(pred_y == test_y) / n
        results['fpr'][k] = np.sum((pred_y != test_y) & (pred_y != 0)) / np.sum(pred_y != 0)
        results['fnr'][k] = np.sum((pred_y != test_y) & (pred_y == 0)) / np.sum(pred_y == 0)
        results['bpr'][k] = np.sum((pred_y * test_y) == -1) / np.sum(test_y != 0)
        results['gpr'][k] = np.sum((pred_y == test_y) & (test_y != 0)) / np.sum(test_y != 0)

    for col in results:
        if col != 'run':
            results[col][K] = np.mean(results[col][:K])

    return pd.DataFrame(results).set_index('run')


def clf_output(cv_results, y, K, feature_names, w=None, summary=None):
    print """
                                 Results
==============================================================================
    """
    print pd.DataFrame({'%': np.array([len(y[y == -1]),
                                len(y[y == 0]),
                                len(y[y == 1])])/len(y)},
                       index=[-1, 0, 1])
    print pd.DataFrame({'values': [(len(y)/K)*(K-1)]}, index=['Training Size / Fold'])
    print cv_results
    print "=============================================================================="

# time period cutoffs in minutes from start
time_period_cutoffs = (30, 180, 420, 450)

#quotes, trades = get_data('XLE', 2012, 1, 5, bar_width='second')
data = get_more_data('XLE', 2012, 2, 1, days=1, bar_width='second')

hls = [10, 40, 100]

for quotes, trades in data:
    features.label_data(quotes, label_hls=hls)
    features.add_ema(quotes, halflives=hls)
    features.add_dema(quotes, halflives=hls)
    features.add_momentum(quotes, halflives=hls)
    features.add_log_return_ema(quotes, halflives=hls)
    features.add_trade_momentum(quotes, trades, bar_width='second')
    features.add_price_diff(quotes)
    features.add_size_diff(quotes)
    features.add_trade_momentum_dema(quotes, trades, halflife=40)

quotes_list, trades_list = zip(*data)

quotes = pd.concat(quotes_list)
trades = pd.concat(trades_list)

feature_names = ['momentum', 'dEMA_10', 'dEMA_40', 'dEMA_100', 'trade_momentum',
                 'log_returns_10-', 'log_returns_40-', 'log_returns_100-',
                 'log_returns_std_10-', 'log_returns_std_40-', 'size_diff',
                 'price_diff', 'trade_momentum_dema_40']

quotes = quotes.fillna(0)

# normalize features
quotes[feature_names] = (quotes[feature_names] - quotes[feature_names].mean()) / quotes[feature_names].std()


"""
Linear regression [features] -> log_returns_100+
Entry threshold is ~ 0.000005 for log_returns_100+
"""
class LRclf(object):
    def __init__(self, thresh):
        self.thresh = thresh

    def fit(self, X, y):
        X = sm.add_constant(X)
        self.mod = sm.OLS(y, X).fit()

    def predict(self, X):
        X = sm.add_constant(X)
        pred = self.mod.predict(X)
        results = np.zeros(len(pred))
        results[pred >= thresh] = 1
        results[pred <= -thresh] = -1
        return results

print """
Linear Regression Classifier, 3-class 5-fold CV, no-class weights
"""
thresh = 0.000005
print "thresh = {}".format(thresh)
hl = 100
K = 5

quotes['label'] = 0
quotes.ix[quotes['log_returns_100+'] > thresh, 'label'] = 1
quotes.ix[quotes['log_returns_100+'] < -thresh, 'label'] = -1

clf = LRclf(thresh)
cv_results = cross_validation(quotes, clf, feature_names, label='label', K=5)
y = quotes['log_returns_{}+'.format(hl)].values
reg = sm.OLS(y, quotes[feature_names]).fit()
print reg.summary()
clf_output(cv_results, y, K, feature_names)


print """
Linear Regression Classifier, 3-class 5-fold CV, no-class weights
"""
thresh = 0.000005*100
print "thresh = {}".format(thresh)
hl = 100
K = 5

quotes['label'] = 0
quotes.ix[quotes['log_returns_100+'] > thresh, 'label'] = 1
quotes.ix[quotes['log_returns_100+'] < -thresh, 'label'] = -1

clf = LRclf(thresh)
cv_results = cross_validation(quotes, clf, feature_names, label='label', K=5)
clf_output(cv_results, y, K, feature_names)


print """
Linear Regression Classifier, 3-class 5-fold CV, no-class weights
"""
thresh = 0.00001
print "thresh = {}".format(thresh)
hl = 100
K = 5

quotes['label'] = 0
quotes.ix[quotes['log_returns_100+'] > thresh, 'label'] = 1
quotes.ix[quotes['log_returns_100+'] < -thresh, 'label'] = -1

clf = LRclf(thresh)
cv_results = cross_validation(quotes, clf, feature_names, label='label', K=5)
clf_output(cv_results, y, K, feature_names)
