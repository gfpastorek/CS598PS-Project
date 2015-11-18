from __future__ import division
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import os


from backtest.backtest import backtest, Order


# XLE XOM CVX SLB KMI EOG COP OXY PXD VLO USO

# DATE,TIME_M,SYM_ROOT,SYM_SUFFIX,BID,BIDSIZ,ASK,ASKSIZ,BIDEX,ASKEX,NATBBO_IND

def convert_time(d):
    h, m, s = d.split(':')
    s, ms = map(int, s.split('.'))
    return dt.datetime(2012, 1, 3, int(h), int(m), int(s), int(ms)*1000)


# FEATURES/TRANSFORMATIONS

# data = get_data('AAPL', start, end)
#
#
# def make_ohlc(df, params):
# 	price_header = params['price_header']
# 	df.resample('1s', how={''})

def plot_series(data, sec1n, start_hour=4, start_min=0, end_hour=20, end_min=0):
    data = data[data['TIME_M'] >= dt.datetime(2012,1,3,start_hour,start_min,0,0)]
    data = data[data['TIME_M'] <= dt.datetime(2012,1,3,end_hour,end_min,0,0)]

    sec1 = data[data['SYM_ROOT'] == sec1n]

    sec1_seconds = \
        sec1.set_index('TIME_M')\
        .groupby(lambda x: dt.datetime(x.year, x.month, x.day, x.hour, x.minute, x.second, 0))\
        .mean()

    sec1_seconds['BID'].plot()
    sec1_seconds['ASK'].plot()

    plt.show()


def clean_quotes(data, sec1n, start_hour=9, start_min=30, end_hour=15, end_min=30, bar_width='second'):

    data = data.reset_index()

    data = data[data['BID'] != 0]
    data = data[data['ASK'] != 0]
    data = data[data['NATBBO_IND'] == 1]

    minute_bars = (bar_width == 'minute')

    data = data[data['TIME_M'] >= dt.datetime(2012,1,3,start_hour,start_min,0,0)]
    data = data[data['TIME_M'] <= dt.datetime(2012,1,3,end_hour,end_min,0,0)]

    sec1 = data[data['SYM_ROOT'] == sec1n]

    sec1 = sec1.reset_index().drop(['DATE', 'SYM_SUFFIX', 'index'], 1)

    if bar_width is None:
        return sec1
    else:
        sec1_seconds = \
            sec1.set_index('TIME_M')\
            .groupby(['SYM_ROOT', lambda x: dt.datetime(x.year, x.month, x.day, x.hour, x.minute, 0 if minute_bars else x.second, 0)])\
            .agg({
                'ASK': 'mean',
                'BID': 'mean',
                'ASKSIZ': 'max',
                'BIDSIZ': 'max'
            })
        return sec1_seconds.reset_index().rename(columns={'level_1': 'TIME_M'})

prev_momentum = {}

count = 0

def test_strategy(data, positions):
    orders = []
    #momentum threshold = 0.001
    entry_threshold = 0.000035
    exit_threshold = 0.00002
    global count
    count += 1
    if count < 30:
        return[]
    for sym in data:
        pos = positions.get(sym, 0)
        #qty = 10
        qty = 10*int(data[sym]['dEMA_10']/entry_threshold) - pos
        if data[sym]['momentum'] >= 0 and data[sym]['dEMA_10'] >= entry_threshold and qty > 0:# and (pos <= 0):
            orders.append(Order(sym, qty, type='market'))
        elif data[sym]['momentum'] <= 0 and data[sym]['dEMA_10'] <= -entry_threshold and qty < 0:# and (pos >= 0):
            orders.append(Order(sym, qty, type='market'))
        elif data[sym]['dEMA_10'] >= exit_threshold and (pos < 0):
            orders.append(Order(sym, min(-pos, qty*5), type='market'))
        elif data[sym]['dEMA_10'] <= -exit_threshold and (pos > 0):
            orders.append(Order(sym, max(-pos, qty*5), type='market'))
        prev_momentum[sym] = data[sym]['momentum']
    return orders


def magic_strategy(data, positions):
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

    for sym in data:
        pos = positions.get(sym, 0)
        qty = base_qty*int(data[sym][indicator]/entry_threshold) - pos
        if abs(qty) > 0 and np.sign(qty) == np.sign(data[sym][indicator]) and abs(pos+qty) <= max_pos:
            orders.append(Order(sym, qty, type='market'))
        elif abs(data[sym][indicator]) < exit_threshold:
            orders.append(Order(sym, -pos, type='market'))

    return orders


#os.chdir(os.path.join(os.getcwd(), 'python'))
root_dir = os.path.realpath(os.path.dirname(os.getcwd()))
#fname = 'xle_jan_3_12'
#fname = 'xle_only_jan_5_12'
fname = 'xle_only_jan_6_12'
#fname = 'xle_only_jan_7_12'
fpath = os.path.join(root_dir, 'data', fname, '{}.csv'.format(fname))
data = pd.read_csv(fpath, parse_dates=['TIME_M'], date_parser=convert_time)

data = clean_quotes(data, 'XLE', 9, 30, 15, 30, bar_width='second')
#data = clean_quotes(data, 'XLE', 9, 30, 15, 30, bar_width=None)

data['price'] = (data['BID'] + data['ASK'])/2
data['log_price'] = np.log((data['BID'] + data['ASK'])/2)

#hls = [10, 20, 30, 40, 50, 60, 70, 80, 90]

#hls = [3, 5, 10, 20, 30]

#hls = [10, 20, 30, 40]

hls = [10, 40, 100]

for hl in hls:
    data.ix[:, 'EMA_{}'.format(hl)] = pd.ewma(data['price'], halflife=hl)
    data['dEMA_{}'.format(hl)] = 0
    #data.ix[1:, 'dEMA_{}'.format(hl)] = np.diff(pd.ewma(data['price'], halflife=hl))
    data.ix[1:, 'dEMA_{}'.format(hl)] = pd.ewma(np.diff(data['log_price']), halflife=hl)
    data['dEMA_std_{}'.format(hl)] = pd.rolling_std(data['dEMA_{}'.format(hl)], 1000)
data['dEMA'] = np.mean([data['dEMA_{}'.format(hl)] for hl in hls], axis=0)

data['log_returns'] = np.concatenate([[0], np.diff(data['log_price'])])

return_windows = [2, 3, 4, 5, 10, 20, 50, 100]

for hl in hls:
    data['log_returns_{}+'.format(hl)] = np.concatenate([(pd.ewma(data['log_returns'].values[::-1], hl))[:0:-1], [None]])
    data['log_returns_{}-'.format(hl)] = np.concatenate([[0], pd.ewma(data['log_returns'], hl).values[:-1]])
    data['log_returns_std_{}+'.format(hl)] = np.concatenate([(pd.rolling_std(data['log_returns'].values[::-1], 2*hl))[:0:-1], [None]])
    data['log_returns_std_{}-'.format(hl)] = np.concatenate([[0], pd.rolling_std(data['log_returns'], 2*hl).values[:-1]])
    data['nlog_returns_{}+'.format(hl)] = data['log_returns_{}+'.format(hl)] / (1+data['log_returns_std_{}+'.format(hl)])
    data['nlog_returns_{}-'.format(hl)] = data['log_returns_{}-'.format(hl)] / (1+data['log_returns_std_{}-'.format(hl)])

data = data.fillna(0)

data['std'] = pd.rolling_std(data['log_returns'], 1000)

f = lambda window: plt.scatter(data['log_returns'], data['log_returns_{}+'.format(window)])
g = lambda w1, w2: plt.scatter(data['log_returns_{}-'.format(w1)], data['log_returns_{}+'.format(w2)])
g2 = lambda w1, w2, min_w1: plt.scatter(
    data[np.abs(data['log_returns_{}-'.format(w1)].fillna(0)) >= min_w1]['log_returns_{}-'.format(w1)],
    data[np.abs(data['log_returns_{}-'.format(w1)].fillna(0)) >= min_w1]['log_returns_{}+'.format(w2)])

g3 = lambda w1, w2, min_w1: plt.scatter(
    np.abs(data[np.abs(data['log_returns_{}-'.format(w1)].fillna(0)) >= min_w1]['log_returns_{}-'.format(w1)]),
    np.sign(data[np.abs(data['log_returns_{}-'.format(w1)].fillna(0)) >= min_w1]['log_returns_{}-'.format(w1)])*data[np.abs(data['log_returns_{}-'.format(w1)].fillna(0)) >= min_w1]['log_returns_{}+'.format(w2)])


#data['momentum'] = data['dEMA']
data['momentum'] = data['EMA_10'] - data['EMA_40']

#pnl_history, order_history = backtest(data, test_strategy, transaction_costs=0.005)
pnl_history, order_history = backtest(data, magic_strategy, transaction_costs=0.005)

fig, axes = plt.subplots(nrows=3)

axes[0].plot(data['TIME_M'].values, data['price'].values)

for hl in hls:
    axes[0].plot(data['TIME_M'].values, data['EMA_{}'.format(hl)].values)

long_orders = filter(lambda x: x[2] > 0, order_history)
short_orders = filter(lambda x: x[2] < 0, order_history)
long_order_times = map(lambda x: x[0], long_orders)
short_order_times = map(lambda x: x[0], short_orders)
long_order_prices = map(lambda x: x[3], long_orders)
short_order_prices = map(lambda x: x[3], short_orders)

axes[0].plot(long_order_times, long_order_prices, '^', ms=8, color='g')
axes[0].plot(short_order_times, short_order_prices, 'v', ms=8, color='r')

ax2 = axes[0].twinx()
ax2.plot(data['TIME_M'].values, (data['ASK']-data['BID']).values)
#ax2.plot(data['TIME_M'].values, data['ASKSIZ'].values)
#ax2.plot(data['TIME_M'].values, data['BIDSIZ'].values)

axes[1].plot(data['TIME_M'].values, data['log_returns_10+'].values, label='lr_10+')
axes[1].plot(data['TIME_M'].values, data['log_returns_40+'].values, label='lr_40+')
axes[1].plot(data['TIME_M'].values, data['log_returns_100+'].values, label='lr_100+')

plt.legend()

axes[2].plot(pnl_history, label='pnl')

plt.legend()

plt.show()

