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



def test_strategy(quotes, trades, positions):
    orders = []
    momentum_threshold = 0.0001
    entry_threshold = 0.000035
    exit_threshold = 0.00002

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


p = {
    0: 0,
    1: 0
}

hold_time = np.inf
count = 0


def momentum_strategy(quotes, trades, positions, threshold=0.0001, threshold2=0.001, threshold3=0.0002, min_hold_time=30):
    global hold_time, count
    orders = []

    count += 1

    if count < 60:
        return []

    for sym in quotes:
        pos = positions.get(sym, 0)

        dEMA_10 = quotes[sym]['dEMA_40']
        dEMA_10_sign_change = quotes[sym]['dEMA_10_sign_change?']
        dEMA_40 = quotes[sym]['dEMA_40']
        dEMA_100 = quotes[sym]['dEMA_100']
        momentum = quotes[sym]['momentum']

        if abs(momentum) >= threshold and abs(dEMA_40) >= threshold2 and np.sign(dEMA_40) == -np.sign(momentum) and hold_time > min_hold_time:
            qty = 10*np.sign(dEMA_40) - pos
            orders.append(Order(sym, qty, type='market'))
            hold_time = 0
        #elif hold_time > min_hold_time and np.sign(dEMA_100) == np.sign(-pos) and abs(dEMA_40) >= threshold3:
        #    orders.append(Order(sym, -pos, type='market'))
        #    hold_time = 0
        elif pos != 0:
            hold_time += 1

        #elif quotes[sym]['dEMA_100_sign_change?']:
        #    orders.append(Order(sym, -pos, type='market'))

    return orders


def momentum_strategy2(quotes, trades, positions, min_hold_time=30):
    global hold_time, count
    orders = []

    count += 1

    if count < 60:
        return []

    for sym in quotes:
        pos = positions.get(sym, 0)

        EMA_10 = quotes[sym]['EMA_40']
        EMA_300 = quotes[sym]['EMA_300']
        diff = EMA_10 - EMA_300

        if hold_time > min_hold_time:
            qty = 10*np.sign(diff) - pos
            orders.append(Order(sym, qty, type='market'))
            hold_time = 0
        #elif hold_time > min_hold_time and np.sign(dEMA_100) == np.sign(-pos) and abs(dEMA_40) >= threshold3:
        #    orders.append(Order(sym, -pos, type='market'))
        #    hold_time = 0
        elif pos != 0:
            hold_time += 1

        #elif quotes[sym]['dEMA_100_sign_change?']:
        #    orders.append(Order(sym, -pos, type='market'))

    return orders


def svm_strategy(quotes, trades, positions, svm_clf, feature_names, filter_cols):
    orders = []

    base_qty = 10
    max_pos = 100

    for sym in quotes:
        if not quotes[sym]['yes?']:
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


def svm_strategy2(quotes, trades, positions, svm_clf, feature_names, filter_cols):
    orders = []

    base_qty = 10
    max_pos = 100

    for sym in quotes:
        if not quotes[sym]['yes?']:
            continue
        feats = quotes[sym][feature_names].values
        pred = svm_clf.predict(feats)[0]
        pos = positions.get(sym, 0)
        qty = base_qty*pred - pos
        if abs(qty) > 0 and np.sign(qty) == np.sign(pred) and abs(pos+qty) <= max_pos:
            orders.append(Order(sym, qty, type='market'))
        #elif (pred == 0) and (pos != 0):
        #    orders.append(Order(sym, -pos, type='market'))

    return orders


def magic_strategy(quotes, trades, positions):
    orders = []
    indicator = 'log_returns_10+'
    entry_threshold = 0.000005*2
    exit_threshold = 0.000001

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


def train_and_backtest(datas, strategy, i, feature_names, filter_cols, pred_col='log_returns_10+', thresh=0.000005*2):
    test_ind = -i

    training_data = pd.concat(datas[:test_ind] + datas[test_ind+1:]).fillna(0).reset_index()
    testing_data = datas[test_ind].fillna(0).reset_index()

    training_data['label'] = 0
    training_data.ix[training_data[pred_col] >= thresh, 'label'] = 1
    training_data.ix[training_data[pred_col] < -thresh, 'label'] = -1

    filtered_training_data = training_data[reduce(lambda x,y: x | training_data[y], filter_cols, training_data[filter_cols[0]])]
    testing_data['yes?'] = False
    testing_data['yes?'] = reduce(lambda x,y: x | testing_data[y], filter_cols, testing_data[filter_cols[0]])

    X = filtered_training_data[feature_names]
    y = filtered_training_data['label']

    svm_clf = svm.LinearSVC(C=1, class_weight='auto')
    svm_clf.fit(X, y)

    pnl_history, order_history = backtest(strategy, testing_data,
                                          transaction_costs=0.005, slippage_rate=0.25, delay_fill=True,
                                          svm_clf=svm_clf, feature_names=feature_names, filter_cols=filter_cols)

    fig, axes = plt.subplots(nrows=2)

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
    axes[0].set_title("Price and Orders")

    axes[1].plot(testing_data['DATE_TIME'].values, pnl_history, label='pnl')
    axes[1].set_title("PnL")

    #axes[2].plot(testing_data['DATE_TIME'].values, testing_data['momentum'].values, label='momentum')

    plt.show()


def backtest_and_plot(datas, strategy, i, **kwargs):
    global hold_time, count
    count = 0
    hold_time = np.inf
    test_ind = -i

    testing_data = datas[test_ind].fillna(0).reset_index()

    pnl_history, order_history = backtest(strategy, testing_data,
                                          transaction_costs=0.005, slippage_rate=0.25, delay_fill=True, **kwargs)

    fig, axes = plt.subplots(nrows=4)

    axes[0].plot(testing_data['DATE_TIME'].values, testing_data['price'].values)

    for hl in [10, 40, 100, 150, 200, 300]:
        axes[0].plot(testing_data['DATE_TIME'].values, testing_data['EMA_{}'.format(hl)].values)

    long_orders = filter(lambda x: x[2] > 0, order_history)
    short_orders = filter(lambda x: x[2] < 0, order_history)
    long_order_times = map(lambda x: x[0], long_orders)
    short_order_times = map(lambda x: x[0], short_orders)
    long_order_prices = map(lambda x: x[3], long_orders)
    short_order_prices = map(lambda x: x[3], short_orders)

    axes[0].plot(long_order_times, long_order_prices, '^', ms=8, color='g')
    axes[0].plot(short_order_times, short_order_prices, 'v', ms=8, color='r')
    axes[0].set_title("Price and Orders")

    axes[1].plot(testing_data['DATE_TIME'].values, pnl_history, label='pnl')
    axes[1].set_title("PnL")

    axes[2].plot(testing_data['DATE_TIME'].values, testing_data['dmtm_100'].values, label='dmtm_100')
    axes[2].plot(testing_data['DATE_TIME'].values, testing_data['dmtm_200'].values, label='dmtm_200')
    axes[2].plot(testing_data['DATE_TIME'].values, [-0.2]*len(testing_data['DATE_TIME'].values), label='-0.2')
    axes[2].plot(testing_data['DATE_TIME'].values, [0]*len(testing_data['DATE_TIME'].values), label='0')
    axes[2].plot(testing_data['DATE_TIME'].values, [0.2]*len(testing_data['DATE_TIME'].values), label='0.2')
    axes[3].plot(testing_data['DATE_TIME'].values, testing_data['EMA_40'] - testing_data['EMA_300'].values, label='momentum')
    axes[3].plot(testing_data['DATE_TIME'].values, [0]*len(testing_data['DATE_TIME'].values), label='0')

    plt.show()


datas = get_more_data('XLE', 2012, 2, 1, days=10, bar_width='second')

#datas, trades, quotes = get_trades_and_quotes('XLE', 2012, 2, 1, days=10, bar_width='second', return_unmerged=True)
#datas = get_trades_and_quotes('XLE', 2012, 2, 1, days=10, bar_width='second')

hls = [10, 40, 100]

feature_names = []
filter_feature_names = []

for data, trades in datas:
    features.add_future_log_returns(data, label_hls=hls)
    features.add_future_log_returns_rolling(data, windows=(10, 30, 60))
    feature_names += features.add_price_dema(data, halflives=hls)
    feature_names += features.add_momentum(data, halflives=hls)
    feature_names += features.add_log_return_ema(data, halflives=hls)
    feature_names += features.add_price_diff(data)
    feature_names += features.add_size_diff(data)
    feature_names += features.add_dema(data, features=['momentum', 'dEMA_10', 'dEMA_40', 'dEMA_100'])
    filter_feature_names += features.add_crossover_filter(data, halflives=(10, 100))
    filter_feature_names += features.add_high_momentum_filter(data, cutoff=0.1, halflives=(10, 40, 100))
    filter_feature_names += features.add_dema_sign_change_filter(data, halflife=10)
    filter_feature_names += features.add_dema_sign_change_filter(data, halflife=40)
    filter_feature_names += features.add_dema_sign_change_filter(data, halflife=100)

quotes = zip(*datas)[0]

feature_names += filter_feature_names

feature_names = list(set(feature_names))

train_and_backtest(quotes, svm_strategy, 2, feature_names, filter_feature_names, thresh=0.000005*2)

