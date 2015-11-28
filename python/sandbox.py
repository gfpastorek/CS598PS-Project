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
    features.add_trade_momentum(quotes, trades, bar_width='second')

quotes_list, trades_list = zip(*data)

quotes = pd.concat(quotes_list)
trades = pd.concat(trades_list)

feature_names = ['momentum', 'dEMA_10', 'dEMA_40', 'dEMA_100', 'trade_momentum',
                 'log_returns_10-', 'log_returns_40-', 'log_returns_100-',
                 'log_returns_std_10-', 'log_returns_std_40-']

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
"""
                            OLS Regression Results
==============================================================================
Dep. Variable:       log_returns_100+   R-squared:                       0.046
Model:                            OLS   Adj. R-squared:                  0.046
Method:                 Least Squares   F-statistic:                     186.7
Date:                Fri, 27 Nov 2015   Prob (F-statistic):               0.00
Time:                        17:42:55   Log-Likelihood:             4.0792e+05
No. Observations:               38750   AIC:                        -8.158e+05
Df Residuals:                   38740   BIC:                        -8.157e+05
Df Model:                          10
Covariance Type:            nonrobust
=======================================================================================
                          coef    std err          t      P>|t|      [95.0% Conf. Int.]
---------------------------------------------------------------------------------------
momentum             2.115e-06   8.87e-08     23.837      0.000      1.94e-06  2.29e-06
dEMA_10              2.444e-06   2.01e-07     12.171      0.000      2.05e-06  2.84e-06
dEMA_40              8.757e-07   1.51e-07      5.810      0.000       5.8e-07  1.17e-06
dEMA_100             2.293e-07   9.34e-08      2.454      0.014      4.62e-08  4.12e-07
trade_momentum      -1.896e-07   3.33e-08     -5.697      0.000     -2.55e-07 -1.24e-07
log_returns_10-      1.779e-06   1.59e-07     11.165      0.000      1.47e-06  2.09e-06
log_returns_40-     -7.388e-06   5.69e-07    -12.995      0.000      -8.5e-06 -6.27e-06
log_returns_100-     4.326e-06   3.76e-07     11.507      0.000      3.59e-06  5.06e-06
log_returns_std_10-  1.135e-07   4.08e-08      2.781      0.005      3.35e-08  1.94e-07
log_returns_std_40- -2.903e-07   4.19e-08     -6.929      0.000     -3.72e-07 -2.08e-07
==============================================================================
Omnibus:                     7133.827   Durbin-Watson:                   0.033
Prob(Omnibus):                  0.000   Jarque-Bera (JB):            64620.219
Skew:                          -0.627   Prob(JB):                         0.00
Kurtosis:                       9.201   Cond. No.                         47.2
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
momentum            -0.216936  0.111939  0.209902
dEMA_10             -0.140160 -0.042351  0.197784
dEMA_40             -0.137513  0.051638  0.157879
dEMA_100            -0.082296  0.131149 -0.033857
trade_momentum       0.009951 -0.005242 -0.004739
log_returns_10-     -0.057481 -0.095249  0.150720
log_returns_40-      0.239331  0.338320 -0.608134
log_returns_100-    -0.103740 -0.199465  0.346236
log_returns_std_10- -0.001515 -0.007009  0.002253
log_returns_std_40-  0.060197 -0.090672  0.070939
           %
-1  0.193910
 0  0.628852
 1  0.177239
                            values
Training Size / Fold  31000.000000
Mean                      0.626425
STD                       0.023286
      Score
1  0.638369
2  0.611663
3  0.657335
4  0.590012
5  0.634744
==============================================================================

"""


"""
SVM, 2-class 5-fold CV, filtered, + vs -
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
momentum             0.257510
dEMA_10              0.245852
dEMA_40              0.127780
dEMA_100            -0.033758
trade_momentum      -0.009124
log_returns_10-      0.167149
log_returns_40-     -0.750050
log_returns_100-     0.513951
log_returns_std_10- -0.012832
log_returns_std_40-  0.029482
          %
-1  0.48797
 0  0.00000
 1  0.51203
                            values
Training Size / Fold  28695.200000
Mean                      0.556441
STD                       0.004622
      Score
1  0.561533
2  0.558545
3  0.555478
4  0.548027
5  0.558623
==============================================================================
"""


"""
SVM, 2-class 5-fold CV, filtered
"""
thresh = 0.000005/10
hl = 100
K = 5
quotes['label'] = 0
quotes.ix[abs(quotes['log_returns_100+']) > thresh, 'label'] = 1
quotes.ix[abs(quotes['log_returns_100+']) < -thresh, 'label'] = -1

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

                      weight0
momentum            -0.647538
dEMA_10             -0.172094
dEMA_40             -0.478397
dEMA_100             0.067800
trade_momentum      -0.021980
log_returns_10-      0.090825
log_returns_40-      0.129755
log_returns_100-    -0.385438
log_returns_std_10- -0.067429
log_returns_std_40-  0.372747
           %
-1  0.000000
 0  0.074348
 1  0.925652
                            values
Training Size / Fold  31000.000000
Mean                      0.485953
STD                       0.115670
      Score
1  0.566637
2  0.542581
3  0.521032
4  0.543097
5  0.256420
==============================================================================
"""