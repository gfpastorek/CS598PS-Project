from __future__ import division
import os
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn import svm

from backtest import backtest, Order
from datautils.data_utils import *
import datautils.features as features


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


def svm_strategy(quotes, trades, positions, svm_clf, feature_names):
    orders = []

    global count
    count += 1
    if count < 30:
        return[]

    base_qty = 10
    max_pos = 100

    for sym in quotes:
        if not quotes[sym]['crossover?']:
            continue
        feats = quotes[sym][feature_names].values
        pred = svm_clf.predict(feats)[0]
        pos = positions.get(sym, 0)
        qty = base_qty*pred - pos
        if abs(qty) > 0 and np.sign(qty) == np.sign(pred) and abs(pos+qty) <= max_pos:
            orders.append(Order(sym, qty, type='market'))
        elif (pred == 0) and (pos != 0):
            orders.append(Order(sym, -pos, type='market'))

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


datas = get_more_data('XLE', 2012, 2, 1, days=10, bar_width='second')

#datas, trades, quotes = get_trades_and_quotes('XLE', 2012, 2, 1, days=10, bar_width='second', return_unmerged=True)
#datas = get_trades_and_quotes('XLE', 2012, 2, 1, days=10, bar_width='second')

hls = [10, 40, 100]

feature_names = []

for data, trades in datas:
    features.add_future_log_returns(data, label_hls=hls)
    features.add_future_log_returns_rolling(data, windows=(10, 30, 60))
    feature_names += features.add_price_dema(data, halflives=hls)
    feature_names += features.add_momentum(data, halflives=hls)
    feature_names += features.add_log_return_ema(data, halflives=hls)
    feature_names += features.add_price_diff(data)
    feature_names += features.add_size_diff(data)
    feature_names += features.add_dema(data, features=['momentum', 'dEMA_10', 'dEMA_40', 'dEMA_100'])
    features.add_crossover(data, halflives=(10, 100))

datas = zip(*datas)[0]

feature_names = list(set(feature_names))

test_ind = -1
thresh = 0.000005
pred_col = 'log_returns_10+'

training_data = pd.concat(datas[:test_ind] + datas[test_ind+1:]).fillna(0)
testing_data = datas[test_ind].fillna(0)

training_data['label'] = 0
training_data.ix[training_data[pred_col] >= thresh, 'label'] = 1
training_data.ix[training_data[pred_col] < -thresh, 'label'] = -1

filtered_training_data = training_data[training_data['crossover?']]

X = filtered_training_data[feature_names]
y = filtered_training_data['label']

svm_clf = svm.LinearSVC(C=1, class_weight='auto')
svm_clf.fit(X, y)

#testing_quotes['label'] = svm_clf.predict(testing_data[feature_names].values)

pnl_history, order_history = backtest(svm_strategy, testing_data,
                                      transaction_costs=0.005, slippage_rate=0.25, delay_fill=True,
                                      svm_clf=svm_clf, feature_names=feature_names)

fig, axes = plt.subplots(nrows=4)

axes[0].plot(testing_data['DATE_TIME'].values, testing_data['price'].values)

for hl in hls:
    axes[0].plot(testing_data['DATE_TIME'].values, testing_data['EMA_{}'.format(hl)].values)

long_orders = filter(lambda x: x[2] > 0, order_history)
short_orders = filter(lambda x: x[2] < 0, order_history)
long_order_times = map(lambda x: x[0], long_orders)
short_order_times = map(lambda x: x[0], short_orders)
long_order_prices = map(lambda x: x[3], long_orders)
short_order_prices = map(lambda x: x[3], short_orders)

axes[0].plot(long_order_times, long_order_prices, '^', ms=8, color='g')
axes[0].plot(short_order_times, short_order_prices, 'v', ms=8, color='r')

ax2 = axes[0].twinx()
ax2.plot(testing_data['DATE_TIME'].values, (testing_data['ASK_PRICE']-testing_data['BID_PRICE']).values)

axes[1].plot(testing_data['DATE_TIME'].values, testing_data['momentum'].values, label='momentum')

axes[2].plot(testing_data['DATE_TIME'].values, testing_data['log_returns_10+'].values, label='lr_10+')
axes[2].plot(testing_data['DATE_TIME'].values, testing_data['log_returns_40+'].values, label='lr_40+')
axes[2].plot(testing_data['DATE_TIME'].values, testing_data['log_returns_100+'].values, label='lr_100+')
plt.legend()

axes[3].plot(testing_data['DATE_TIME'].values, pnl_history, label='pnl')

plt.show()

