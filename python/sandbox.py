import sys
import os
os.chdir('C:\\Users\\Greg\\Documents\\CS 598PS\\CS598PS-Project\\python')
sys.path.append(os.getcwd())

from __future__ import division
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
import statsmodels.api as sm
from backtest.backtest import backtest, Order
from datautils.data_utils import *
import datautils.features as features


from sklearn import cross_validation, svm



DO_UNFILTERED = False


def cross_validation(data, clf, feature_columns, ycol='label', label='label', K=5,
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

    weights = []

    results = {
        'run': range(1, K+1) + ['Total'],
        'acc': [None]*(K+1),
        'acc*': [None]*(K+1),
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
        results['acc*'][k] = np.mean([np.sum((pred_y == test_y) & (test_y == L)) / np.sum(test_y == L) for L in [-1, 0, 1]])
        results['fpr'][k] = np.sum((pred_y != test_y) & (pred_y != 0)) / np.sum(pred_y != 0)
        results['fnr'][k] = np.sum((pred_y != test_y) & (pred_y == 0)) / np.sum(pred_y == 0)
        results['bpr'][k] = np.sum((pred_y * test_y) == -1) / np.sum(test_y != 0)
        results['gpr'][k] = np.sum((pred_y == test_y) & (test_y != 0)) / np.sum(test_y != 0)
        #results['-1'][k] = np.sum(pred_y == -1)
        #results['0'][k] = np.sum(pred_y == 0)
        #results['1'][k] = np.sum(pred_y == 1)

        if hasattr(clf, 'coef_'):
            W = clf.coef_
            weights.append(pd.DataFrame({'-1': W[0], '0': W[1], '1': W[2]}, index=feature_columns))

    for col in results:
        if col != 'run':
            results[col][K] = np.mean(results[col][:K])

    if hasattr(clf, 'coef_'):
        mean_weights = sum(weights) / K
    else:
        mean_weights = []

    return pd.DataFrame(results).set_index('run'), mean_weights


def clf_output(cv_results, y, K):
    print """
                                 Results
==============================================================================
    """
    print pd.DataFrame({'%': np.array([len(y[y == -1]),
                                len(y[y == 0]),
                                len(y[y == 1])])/len(y)},
                       index=[-1, 0, 1])
    print pd.DataFrame({'values': [(len(y)/K)*(K-1)]}, index=['Training Size / Fold'])
    print cv_results[0]
    print cv_results[1]
    print "=============================================================================="

# time period cutoffs in minutes from start
time_period_cutoffs = (30, 180, 420, 450)

#pred_col = 'log_returns_100+'
#pred_col = 'log_returns_200+'

#quotes, trades = get_data('XLE', 2012, 1, 5, bar_width='second')
datas = get_more_data('XLE', 2012, 2, 1, days=29, bar_width='second', start_hour=10)
#datas = get_trades_and_quotes('XLE', 2012, 2, 1, days=1, bar_width='second', start_hour=10)
#datas = get_trades_and_quotes('XLE', 2012, 2, 1, days=29, bar_width='second', start_hour=10)

hls = [10, 40, 100]
crossover_hls = (10, 40)

feature_names = []

for data, trades in datas:
    features.add_future_log_returns(data, label_hls=hls)
    features.add_future_log_returns_rolling(data, windows=(10, 30, 60))
    feature_names += features.add_price_dema(data, halflives=hls)
    feature_names += features.add_momentum(data, halflives=hls)
    feature_names += features.add_log_return_ema(data, halflives=hls)
    feature_names += features.add_price_diff(data)
    feature_names += features.add_size_diff(data)
    #feature_names += features.add_vpin_time(data, window=dt.timedelta(seconds=20))
    #feature_names += features.add_dema_sum(data, halflives=hls)
    feature_names += add_dema_sum(data, halflives=hls)
    feature_names += features.add_dema(data, features=['momentum', 'dEMA_10', 'dEMA_40', 'dEMA_100', 'dEMA_sum'])
    features.add_crossover(data, halflives=crossover_hls)


feature_names = list(set(feature_names))

data = pd.concat(zip(*datas)[0])

#quotes_list, trades_list = zip(*data)
#quotes = pd.concat(quotes_list)
#trades = pd.concat(trades_list)

data = data.fillna(0)

# normalize features
data[feature_names] = (data[feature_names] - data[feature_names].mean()) / data[feature_names].std()


pred_col = 'log_returns_10+'
#pred_col = 'log_returns_40+'
#pred_col = 'log_returns_100+'


if DO_UNFILTERED:
    print """
    Linear SVM, 3-class 5-fold CV, auto class weights
    """
    print "crossover_hls = {}".format(crossover_hls)
    print "Pred_col = {}".format(pred_col)

    thresh = 0.000005/2
    hl = 100
    K = 5

    data['label'] = 0
    data.ix[data[pred_col] > thresh, 'label'] = 1
    data.ix[data[pred_col] < -thresh, 'label'] = -1

    #clf = LRclf(thresh)

    clf = svm.LinearSVC(C=1.0, class_weight='auto')
    cv_results = cross_validation(data, clf, feature_names, label='label', K=5)
    y = data['label'].values
    #reg = sm.OLS(y, quotes[feature_names]).fit()
    #print reg.summary()
    print "thresh = {}".format(thresh)
    clf_output(cv_results, y, K)



filtered_data = data[data['crossover?'] == True]

# TODO - expand this filter to include n seconds after the crossover
# TODO - try to get points before the crossover
# TODO - experiment with different crossovers, include all crossovers from a set of EMAs
# TODO - look at second derivatives

#run 1
thresh = 0.000005/2
hl = 100
K = 5

filtered_data['label'] = 0
filtered_data.ix[filtered_data[pred_col] > thresh, 'label'] = 1
filtered_data.ix[filtered_data[pred_col] < -thresh, 'label'] = -1

#clf = LRclf(thresh)

clf = svm.LinearSVC(C=1.0, class_weight='auto')
cv_results = cross_validation(filtered_data, clf, feature_names, label='label', K=5)
y = filtered_data['label'].values
#reg = sm.OLS(y, quotes[feature_names]).fit()
#print reg.summary()

print """
Filtered-data Linear SVM, 3-class 5-fold CV, auto class weights
"""
print "crossover_hls = {}".format(crossover_hls)
print "Pred_col = {}".format(pred_col)
print "thresh = {}".format(thresh)
clf_output(cv_results, y, K)

#run 2
thresh = 0.000005
hl = 100
K = 5

filtered_data['label'] = 0
filtered_data.ix[filtered_data[pred_col] > thresh, 'label'] = 1
filtered_data.ix[filtered_data[pred_col] < -thresh, 'label'] = -1

#clf = LRclf(thresh)

clf = svm.LinearSVC(C=1.0, class_weight='auto')
cv_results = cross_validation(filtered_data, clf, feature_names, label='label', K=5)
y = filtered_data['label'].values
#reg = sm.OLS(y, quotes[feature_names]).fit()
#print reg.summary()

print """
Filtered-data Linear SVM, 3-class 5-fold CV, auto class weights
"""
print "crossover_hls = {}".format(crossover_hls)
print "Pred_col = {}".format(pred_col)
print "thresh = {}".format(thresh)
clf_output(cv_results, y, K)


#run 3
thresh = 0.000005*2
hl = 100
K = 5

filtered_data['label'] = 0
filtered_data.ix[filtered_data[pred_col] > thresh, 'label'] = 1
filtered_data.ix[filtered_data[pred_col] < -thresh, 'label'] = -1

#clf = LRclf(thresh)

clf = svm.LinearSVC(C=1.0, class_weight='auto')
cv_results = cross_validation(filtered_data, clf, feature_names, label='label', K=5)
y = filtered_data['label'].values
#reg = sm.OLS(y, quotes[feature_names]).fit()
#print reg.summary()

print """
Filtered-data Linear SVM, 3-class 5-fold CV, auto class weights
"""
print "crossover_hls = {}".format(crossover_hls)
print "Pred_col = {}".format(pred_col)
print "thresh = {}".format(thresh)
clf_output(cv_results, y, K)



#run 1
thresh = 0.000005/2
hl = 100
K = 5

filtered_data['label'] = 0
filtered_data.ix[filtered_data[pred_col] > thresh, 'label'] = 1
filtered_data.ix[filtered_data[pred_col] < -thresh, 'label'] = -1

clf = svm.SVC(C=1.0, kernel='rbf', class_weight='auto')
cv_results = cross_validation(filtered_data, clf, feature_names, label='label', K=5)
y = filtered_data['label'].values
#reg = sm.OLS(y, quotes[feature_names]).fit()
#print reg.summary()

print """
Filtered-data Rbf SVM, 3-class 5-fold CV, auto class weights
"""
print "crossover_hls = {}".format(crossover_hls)
print "Pred_col = {}".format(pred_col)
print "thresh = {}".format(thresh)
clf_output(cv_results, y, K)

#run 2
thresh = 0.000005
hl = 100
K = 5

filtered_data['label'] = 0
filtered_data.ix[filtered_data[pred_col] > thresh, 'label'] = 1
filtered_data.ix[filtered_data[pred_col] < -thresh, 'label'] = -1

clf = svm.SVC(C=1.0, kernel='rbf', class_weight='auto')
cv_results = cross_validation(filtered_data, clf, feature_names, label='label', K=5)
y = filtered_data['label'].values
#reg = sm.OLS(y, quotes[feature_names]).fit()
#print reg.summary()

print """
Filtered-data Rbf SVM, 3-class 5-fold CV, auto class weights
"""
print "crossover_hls = {}".format(crossover_hls)
print "Pred_col = {}".format(pred_col)
print "thresh = {}".format(thresh)
clf_output(cv_results, y, K)


#run 3
thresh = 0.000005*2
hl = 100
K = 5

filtered_data['label'] = 0
filtered_data.ix[filtered_data[pred_col] > thresh, 'label'] = 1
filtered_data.ix[filtered_data[pred_col] < -thresh, 'label'] = -1

#clf = LRclf(thresh)

clf = svm.SVC(C=1.0, kernel='rbf', class_weight='auto')
cv_results = cross_validation(filtered_data, clf, feature_names, label='label', K=5)
y = filtered_data['label'].values
#reg = sm.OLS(y, quotes[feature_names]).fit()
#print reg.summary()

print """
Filtered-data Rbf SVM, 3-class 5-fold CV, auto class weights
"""
print "crossover_hls = {}".format(crossover_hls)
print "Pred_col = {}".format(pred_col)
print "thresh = {}".format(thresh)
clf_output(cv_results, y, K)