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

# time period cutoffs in minutes from start
time_period_cutoffs = (30, 180, 420, 450)

#quotes, trades = get_data('XLE', 2012, 1, 5, bar_width='second')
# data = get_more_data('XLE', 2012, 2, 1, days=29, bar_width='second')
data = get_more_data('XLE', 2012, 1, 5, days=1, bar_width='second')

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


def svm_output(scores, w, y, K):
    w = w / np.linalg.norm(w)
    print """
                                SVM Results
==============================================================================
    """
    print pd.DataFrame({'weight{}'.format(i): w[i] for i in xrange(0, len(w))}, index=feature_names)
    print pd.DataFrame({'%': np.array([len(y[y == -1]),
                                len(y[y == 0]),
                                len(y[y == 1])])/len(y)},
                       index=[-1, 0, 1])
    print pd.DataFrame({'values': [(len(y)/K)*(K-1), np.mean(scores), np.std(scores)]},
                 index=['Training Size / Fold', 'Mean', 'STD'])
    print pd.DataFrame({'Score': scores}, index=range(1, K+1))
    print "=============================================================================="


"""
Linear regression [features] -> log_returns_100+
Entry threshold is ~ 0.000005 for log_returns_100+
"""
hl = 100
reg = sm.OLS(quotes['log_returns_{}+'.format(hl)],
             quotes[feature_names]).fit()
print reg.summary()


print """
SVM, 3-class 5-fold CV, custom-class weights, weighted precision score
"""
class_weights = {
    -1: 1,
    0: 2,
    1: 1
}
print "class_weights = " + str(class_weights)
thresh = 0.000005
hl = 100
K = 5
quotes['label'] = 0
quotes.ix[quotes['log_returns_100+'] > thresh, 'label'] = 1
quotes.ix[quotes['log_returns_100+'] < -thresh, 'label'] = -1

X = quotes[feature_names]
y = quotes['label']

clf = svm.LinearSVC(C=1, class_weight=class_weights)
scores = cross_validation.cross_val_score(clf, X, y, cv=K)
#scores = cross_validation.cross_val_score(clf, X, y, cv=K, scoring='precision_weighted')

clf = svm.LinearSVC(C=1, class_weight=class_weights)
clf.fit(X, y)
w = clf.coef_
svm_output(scores, w, y, K)


print """
SVM, 3-class 5-fold CV, custom-class weights, weighted precision score
"""
class_weights = {
    -1: 1,
    0: 1,
    1: 1
}
print "class_weights = " + str(class_weights)
thresh = 0.000005
hl = 100
K = 5
quotes['label'] = 0
quotes.ix[quotes['log_returns_100+'] > thresh, 'label'] = 1
quotes.ix[quotes['log_returns_100+'] < -thresh, 'label'] = -1

X = quotes[feature_names]
y = quotes['label']

clf = svm.LinearSVC(C=1, class_weight=class_weights)
scores = cross_validation.cross_val_score(clf, X, y, cv=K)
#scores = cross_validation.cross_val_score(clf, X, y, cv=K, scoring='precision_weighted')

clf = svm.LinearSVC(C=1, class_weight=class_weights)
clf.fit(X, y)
w = clf.coef_
svm_output(scores, w, y, K)


print """
SVM, 3-class 5-fold CV, auto-class weights, weighted precision score
"""
thresh = 0.000005
hl = 100
K = 5
quotes['label'] = 0
quotes.ix[quotes['log_returns_100+'] > thresh, 'label'] = 1
quotes.ix[quotes['log_returns_100+'] < -thresh, 'label'] = -1

X = quotes[feature_names]
y = quotes['label']

clf = svm.LinearSVC(C=1, class_weight='auto')
scores = cross_validation.cross_val_score(clf, X, y, cv=K)
#scores = cross_validation.cross_val_score(clf, X, y, cv=K, scoring='precision_weighted')

clf = svm.LinearSVC(C=1, class_weight='auto')
clf.fit(X, y)
w = clf.coef_
svm_output(scores, w, y, K)
