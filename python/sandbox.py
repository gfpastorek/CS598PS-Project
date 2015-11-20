from __future__ import division
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import os
from pykalman import KalmanFilter
import statsmodels.api as sm
from backtest.backtest import backtest, Order
from datautils.data_utils import get_data
import datautils.features as features

from sklearn import cross_validation, svm


quotes, trades = get_data('XLE', 2012, 1, 5, bar_width='second')

hls = [10, 40, 100]

features.label_data(quotes, label_hls=hls)

features.add_ema(quotes, halflives=hls)
features.add_dema(quotes, halflives=hls)
features.add_momentum(quotes, halflives=hls)
features.add_log_return_ema(quotes, halflives=hls)
features.add_size_diff(quotes)

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
Dep. Variable:       log_returns_100+   R-squared:                       0.057
Model:                            OLS   Adj. R-squared:                  0.057
Method:                 Least Squares   F-statistic:                     73.55
Date:                Thu, 19 Nov 2015   Prob (F-statistic):          2.27e-161
Time:                        14:57:19   Log-Likelihood:             1.3985e+05
No. Observations:               13301   AIC:                        -2.797e+05
Df Residuals:                   13290   BIC:                        -2.796e+05
Df Model:                          11
Covariance Type:            nonrobust
========================================================================================
                           coef    std err          t      P>|t|      [95.0% Conf. Int.]
----------------------------------------------------------------------------------------
momentum              2.025e-05   1.64e-06     12.374      0.000       1.7e-05  2.35e-05
dEMA_10                  0.0037      0.000     10.046      0.000         0.003     0.004
dEMA_40                  0.0031      0.001      5.505      0.000         0.002     0.004
dEMA_100                -0.0020      0.000     -4.100      0.000        -0.003    -0.001
size_diff             2.333e-09   1.23e-09      1.898      0.058      -7.6e-11  4.74e-09
log_returns_10-          0.1471      0.012     11.928      0.000         0.123     0.171
log_returns_40-         -1.0289      0.085    -12.105      0.000        -1.195    -0.862
log_returns_100-         0.8560      0.083     10.293      0.000         0.693     1.019
log_returns_std_10-      0.0068      0.002      3.329      0.001         0.003     0.011
log_returns_std_40-     -0.0275      0.003     -8.123      0.000        -0.034    -0.021
log_returns_std_100-     0.0226      0.003      7.855      0.000         0.017     0.028
==============================================================================
Omnibus:                      755.460   Durbin-Watson:                   0.030
Prob(Omnibus):                  0.000   Jarque-Bera (JB):             3175.099
Skew:                          -0.051   Prob(JB):                         0.00
Kurtosis:                       5.391   Cond. No.                     9.93e+07
==============================================================================
"""

"""
Notes:
-is there a non-linear relationship with momentum and the dEMA_* features?
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

clf = svm.SVC(kernel='linear', C=1, class_weight='auto')
scores = cross_validation.cross_val_score(clf, X, y, cv=K)

clf.fit(X, y)
w = clf.coef_
svm_output(scores, w, y, K)


"""

                                SVM Results
==============================================================================

                       weight1   weight2   weight3
momentum              0.538066 -0.484641 -0.677849
dEMA_10              -0.007214  0.069543  0.074101
dEMA_40              -0.028418 -0.034758 -0.007410
dEMA_100             -0.043508 -0.041361  0.002492
size_diff             0.000348 -0.000595 -0.000426
log_returns_10-      -0.000266  0.000995  0.001279
log_returns_40-      -0.000252  0.000535  0.000760
log_returns_100-     -0.000490 -0.000220  0.000254
log_returns_std_10-   0.003873  0.000813 -0.003216
log_returns_std_40-   0.003125 -0.000521 -0.003750
log_returns_std_100-  0.000381 -0.002426 -0.002934
           %
-1  0.202992
 0  0.606421
 1  0.190587
                            values
Training Size / Fold  10640.800000
Mean                      0.432914
STD                       0.103482
      Score
1  0.248403
2  0.420301
3  0.518045
4  0.542857
5  0.434962
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

clf = svm.SVC(kernel='linear', C=1)
scores = cross_validation.cross_val_score(clf, X, y, cv=K)

clf = svm.SVC(kernel='linear', C=1)
clf.fit(X, y)
w = clf.coef_

svm_output(scores, w, y, K)

"""
                                SVM Results
==============================================================================
          %
-1  0.47138
 0  0.00000
 1  0.52862
                           values
Training Size / Fold  9867.200000
Mean                     0.528620
STD                      0.000086


      Score
1  0.528577
2  0.528577
3  0.528577
4  0.528577
5  0.528792
==============================================================================
"""