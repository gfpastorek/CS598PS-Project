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

#quotes, trades = get_data('XLE', 2012, 1, 5, bar_width='second')
data = get_more_data('XLE', 2012, 1, 5, days=3, bar_width='second')

hls = [10, 40, 100]

for quotes, trades in data:
    features.label_data(quotes, label_hls=hls)
    features.add_ema(quotes, halflives=hls)
    features.add_dema(quotes, halflives=hls)
    features.add_momentum(quotes, halflives=hls)
    features.add_log_return_ema(quotes, halflives=hls)
    features.add_size_diff(quotes)

quotes_list, trades_list = zip(*data)

quotes = pd.concat(quotes_list)
trades = pd.concat(trades_list)

feature_names = ['momentum', 'dEMA_10', 'dEMA_40', 'dEMA_100', 'size_diff',
                 'log_returns_10-', 'log_returns_40-', 'log_returns_100-',
                 'log_returns_std_10-', 'log_returns_std_40-', 'log_returns_std_100-']

quotes = quotes.fillna(0)


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
"""
                            OLS Regression Results
==============================================================================
Dep. Variable:       log_returns_100+   R-squared:                       0.052
Model:                            OLS   Adj. R-squared:                  0.052
Method:                 Least Squares   F-statistic:                     193.4
Date:                Fri, 20 Nov 2015   Prob (F-statistic):               0.00
Time:                        14:58:25   Log-Likelihood:             4.0805e+05
No. Observations:               38750   AIC:                        -8.161e+05
Df Residuals:                   38739   BIC:                        -8.160e+05
Df Model:                          11
Covariance Type:            nonrobust
========================================================================================
                           coef    std err          t      P>|t|      [95.0% Conf. Int.]
----------------------------------------------------------------------------------------
momentum              2.056e-05   9.82e-07     20.938      0.000      1.86e-05  2.25e-05
dEMA_10                  0.0026      0.000     12.160      0.000         0.002     0.003
dEMA_40                  0.0015      0.000      4.539      0.000         0.001     0.002
dEMA_100                 0.0006      0.000      2.101      0.036      4.08e-05     0.001
size_diff             3.259e-09   7.25e-10      4.494      0.000      1.84e-09  4.68e-09
log_returns_10-          0.0859      0.007     11.746      0.000         0.072     0.100
log_returns_40-         -0.6591      0.051    -13.023      0.000        -0.758    -0.560
log_returns_100-         0.5532      0.049     11.206      0.000         0.456     0.650
log_returns_std_10-      0.0029      0.001      2.419      0.016         0.001     0.005
log_returns_std_40-     -0.0292      0.002    -15.348      0.000        -0.033    -0.025
log_returns_std_100-     0.0260      0.002     16.192      0.000         0.023     0.029
==============================================================================
Omnibus:                     7244.132   Durbin-Watson:                   0.036
Prob(Omnibus):                  0.000   Jarque-Bera (JB):            67349.649
Skew:                          -0.633   Prob(JB):                         0.00
Kurtosis:                       9.333   Cond. No.                     1.00e+08
==============================================================================
"""

"""
Notes:
-regression on 2012/01/05 - 2012/01/09
-is there a non-linear relationship with momentum and the dEMA_* features?
-negative betas for dEMA_100 and log_returns_40-, implying mean reversion instead of momentum
"""




"""
SVM, 3-class 5-fold CV, auto-class weights
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

clf = svm.LinearSVC(C=1, class_weight='auto')
clf.fit(X, y)
w = clf.coef_
svm_output(scores, w, y, K)


"""

                                SVM Results
==============================================================================

                       weight0   weight1   weight2
momentum              0.021698 -0.206676  0.221851
dEMA_10              -0.076281  0.270374 -0.477989
dEMA_40              -0.388825  0.136267  0.167016
dEMA_100             -0.556140  0.250879  0.153617
size_diff             0.000683 -0.001006  0.000412
log_returns_10-      -0.001432  0.006081 -0.010255
log_returns_40-      -0.003314  0.004019 -0.004411
log_returns_100-     -0.006647  0.003676  0.000179
log_returns_std_10-   0.036988 -0.034754  0.023471
log_returns_std_40-   0.027537 -0.030182  0.024559
log_returns_std_100- -0.006924 -0.012224  0.029614
           %
-1  0.193910
 0  0.628852
 1  0.177239
                            values
Training Size / Fold  31000.000000
Mean                      0.536935
STD                       0.055608
      Score
1  0.468585
2  0.564185
3  0.513353
4  0.508453
5  0.630098
==============================================================================
"""


"""
SVM, 2-class 5-fold CV, filtered
"""
thresh = 0.000005/10
hl = 100
K = 5
quotes['label'] = 0
quotes.ix[quotes['log_returns_100+'] > thresh, 'label'] = 1
quotes.ix[quotes['log_returns_100+'] < -thresh, 'label'] = -1
filtered_quotes = quotes[quotes['label'] != 0]

X = filtered_quotes[feature_names]
y = filtered_quotes['label']

clf = svm.LinearSVC(C=1, class_weight='auto')
scores = cross_validation.cross_val_score(clf, X, y, cv=K)

clf = svm.LinearSVC(C=1, class_weight='auto')
clf.fit(X, y)
w = clf.coef_

svm_output(scores, w, y, K)

"""
                                SVM Results
==============================================================================

                       weight0
momentum              0.337731
dEMA_10              -0.322543
dEMA_40               0.603920
dEMA_100              0.644097
size_diff             0.000409
log_returns_10-      -0.010908
log_returns_40-      -0.000358
log_returns_100-      0.007543
log_returns_std_10-  -0.020170
log_returns_std_40-   0.007243
log_returns_std_100-  0.041078
          %
-1  0.48797
 0  0.00000
 1  0.51203
                            values
Training Size / Fold  28695.200000
Mean                      0.517272
STD                       0.033043
      Score
1  0.506899
2  0.478394
3  0.541957
4  0.491287
5  0.567824
==============================================================================
"""