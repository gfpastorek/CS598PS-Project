from __future__ import division
import os
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import statsmodels.api as sm

from backtest import backtest, Order
from datautils.data_utils import get_data
import datautils.features as features

from pykalman import KalmanFilter



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


quotes, trades = get_data('XLE', 2012, 1, 5, bar_width='second')

hls = [10, 40, 100]

features.label_data(quotes, label_hls=hls)

features.add_ema(quotes, halflives=hls)
features.add_dema(quotes, halflives=hls)
features.add_momentum(quotes, halflives=hls)

kf = KalmanFilter(transition_matrices=[1],
                  observation_matrices = [1],
                  initial_state_mean = quotes['price'].iloc[0],
                  initial_state_covariance = 1,
                  observation_covariance=1,
                  transition_covariance=.01)

quotes['kf'], _ = kf.filter(quotes['price'].values)

quotes = quotes.fillna(0)

#pnl_history, order_history = backtest(data, test_strategy, transaction_costs=0.005)
pnl_history, order_history = backtest(quotes, trades, magic_strategy, transaction_costs=0.005, slippage_rate=0.25, delay_fill=True)

fig, axes = plt.subplots(nrows=3)

axes[0].plot(quotes['DATE_TIME'].values, quotes['price'].values)

for hl in hls:
    axes[0].plot(quotes['DATE_TIME'].values, quotes['EMA_{}'.format(hl)].values)

axes[0].plot(quotes['DATE_TIME'].values, quotes['kf'].values)

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

#axes[1].plot(data['DATE_TIME'].values, data['log_returns_10+'].values, label='lr_10+')
#axes[1].plot(data['DATE_TIME'].values, data['log_returns_40+'].values, label='lr_40+')
#axes[1].plot(data['DATE_TIME'].values, data['log_returns_100+'].values, label='lr_100+')
#plt.legend()

axes[2].plot(quotes['DATE_TIME'].values, pnl_history, label='pnl')

plt.show()

