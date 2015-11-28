from __future__ import division
import os
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn import svm

from backtest import backtest, Order
from datautils.data_utils import get_data, get_more_data
import datautils.features as features

from pykalman import KalmanFilter


"""
os.chdir(os.path.join(os.getcwd(), 'python'))
sys.path.append(os.getcwd())
"""
# XLE XOM CVX SLB KMI EOG COP OXY PXD VLO USO


prev_momentum = {}

count = 0


def test_strategy(quotes, trades, positions):
    orders = []
    momentum_threshold = 0.0001
    entry_threshold = 0.000035
    exit_threshold = 0.00002
    global count
    count += 1
    if count < 30:
        return[]
    for sym in quotes:
        pos = positions.get(sym, 0)
        qty = 10*int(quotes[sym]['dEMA_10']/entry_threshold) - pos
        if quotes[sym]['momentum'] >= momentum_threshold and \
                        quotes[sym]['dEMA_10'] >= entry_threshold and qty > 0:
            orders.append(Order(sym, qty, type='market'))
        elif quotes[sym]['momentum'] <= -momentum_threshold and \
                        quotes[sym]['dEMA_10'] <= -entry_threshold and qty < 0:
            orders.append(Order(sym, qty, type='market'))
        elif quotes[sym]['dEMA_10'] >= exit_threshold and (pos < 0):
            orders.append(Order(sym, min(-pos, qty*5), type='market'))
        elif quotes[sym]['dEMA_10'] <= -exit_threshold and (pos > 0):
            orders.append(Order(sym, max(-pos, qty*5), type='market'))
        prev_momentum[sym] = quotes[sym]['momentum']
    return orders


def svm_strategy(quotes, trades, positions, svm_clf):
    orders = []
    indicator = 'log_returns_100+'

    feature_names = ['momentum', 'dEMA_10', 'dEMA_40', 'dEMA_100', 'trade_momentum',
                 'log_returns_10-', 'log_returns_40-', 'log_returns_100-',
                 'log_returns_std_10-', 'log_returns_std_40-']

    global count
    count += 1
    if count < 30:
        return[]

    base_qty = 10
    max_pos = 100

    for sym in quotes:
        pos = positions.get(sym, 0)
        qty = base_qty - pos
        feats = quotes[sym][feature_names].values
        pred = svm_clf.predict(feats)
        if abs(qty) > 0 and np.sign(qty) == np.sign(pred) and abs(pos+qty) <= max_pos:
            orders.append(Order(sym, qty, type='market'))
        #elif pred == 0:
        #    orders.append(Order(sym, -pos, type='market'))

    return orders


def magic_strategy(quotes, trades, positions):
    orders = []
    indicator = 'log_returns_100+'
    entry_threshold = 0.000005
    exit_threshold = 0.000001

    global count
    count += 1
    if count < 30:
        return[]

    base_qty = 10
    max_pos = 100

    for sym in quotes:
        pos = positions.get(sym, 0)
        qty = base_qty*int(quotes[sym][indicator]/entry_threshold) - pos
        if abs(qty) > 0 and np.sign(qty) == np.sign(quotes[sym][indicator]) and abs(pos+qty) <= max_pos:
            orders.append(Order(sym, qty, type='market'))
        elif abs(quotes[sym][indicator]) < exit_threshold:
            orders.append(Order(sym, -pos, type='market'))

    return orders


data = get_more_data('XLE', 2012, 1, 5, days=3, bar_width='second')

hls = [10, 40, 100]

for quotes, trades in data:
    features.label_data(quotes, label_hls=hls)
    features.add_ema(quotes, halflives=hls)
    features.add_dema(quotes, halflives=hls)
    features.add_momentum(quotes, halflives=hls)
    features.add_log_return_ema(quotes, halflives=hls)
    features.add_trade_momentum(quotes, trades, bar_width='second')

quotes_list, trades_list = zip(*data[:2])
training_quotes = pd.concat(quotes_list)
training_trades = pd.concat(trades_list)

quotes_list, trades_list = zip(*data[2:])
testing_quotes = pd.concat(quotes_list)
testing_trades = pd.concat(trades_list)

feature_names = ['momentum', 'dEMA_10', 'dEMA_40', 'dEMA_100', 'trade_momentum',
                 'log_returns_10-', 'log_returns_40-', 'log_returns_100-',
                 'log_returns_std_10-', 'log_returns_std_40-']

training_quotes = training_quotes.fillna(0)
testing_quotes = testing_quotes.fillna(0)

# normalize features
# quotes[feature_names] = (quotes[feature_names] - quotes[feature_names].mean()) / quotes[feature_names].std()

thresh = 0.000005
label_hl = 100
training_quotes['label'] = 0
training_quotes.ix[training_quotes['log_returns_{}+'.format(label_hl)] > thresh, 'label'] = 1
training_quotes.ix[training_quotes['log_returns_{}+'.format(label_hl)] < -thresh, 'label'] = -1

X = training_quotes[feature_names]
y = training_quotes['label']

svm_clf = svm.LinearSVC(C=1, class_weight='auto')
svm_clf.fit(X, y)

#testing_quotes['label'] = svm_clf.predict(testing_quotes[feature_names].values)

pnl_history, order_history = backtest(testing_quotes, testing_trades, svm_strategy,
                                      transaction_costs=0.005, slippage_rate=0.25, delay_fill=True, svm_clf=svm_clf)

fig, axes = plt.subplots(nrows=4)

axes[0].plot(quotes['DATE_TIME'].values, quotes['price'].values)

for hl in hls:
    axes[0].plot(quotes['DATE_TIME'].values, quotes['EMA_{}'.format(hl)].values)

long_orders = filter(lambda x: x[2] > 0, order_history)
short_orders = filter(lambda x: x[2] < 0, order_history)
long_order_times = map(lambda x: x[0], long_orders)
short_order_times = map(lambda x: x[0], short_orders)
long_order_prices = map(lambda x: x[3], long_orders)
short_order_prices = map(lambda x: x[3], short_orders)

axes[0].plot(long_order_times, long_order_prices, '^', ms=8, color='g')
axes[0].plot(short_order_times, short_order_prices, 'v', ms=8, color='r')

ax2 = axes[0].twinx()
ax2.plot(quotes['DATE_TIME'].values, (quotes['ASK_PRICE']-quotes['BID_PRICE']).values)

axes[1].plot(quotes['DATE_TIME'].values, quotes['momentum'].values, label='momentum')

axes[2].plot(quotes['DATE_TIME'].values, quotes['log_returns_10+'].values, label='lr_10+')
axes[2].plot(quotes['DATE_TIME'].values, quotes['log_returns_40+'].values, label='lr_40+')
axes[2].plot(quotes['DATE_TIME'].values, quotes['log_returns_100+'].values, label='lr_100+')
plt.legend()

axes[3].plot(quotes['DATE_TIME'].values, pnl_history, label='pnl')

plt.show()

