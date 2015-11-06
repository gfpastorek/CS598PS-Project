from __future__ import division
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import os


from backtest import backtest, Order


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


def make_bars(data, sec1n, start_hour=9, start_min=30, end_hour=15, end_min=30, bar_width='second'):

    data = data.reset_index()

    data = data[data['BID'] != 0]
    data = data[data['ASK'] != 0]
    data = data[data['NATBBO_IND'] == 1]

    minute_bars = (bar_width == 'minute')

    data = data[data['TIME_M'] >= dt.datetime(2012,1,3,start_hour,start_min,0,0)]
    data = data[data['TIME_M'] <= dt.datetime(2012,1,3,end_hour,end_min,0,0)]

    sec1 = data[data['SYM_ROOT'] == sec1n]

    sec1_seconds = \
        sec1.set_index('TIME_M')\
        .groupby(['SYM_ROOT', lambda x: dt.datetime(x.year, x.month, x.day, x.hour, x.minute, 0 if minute_bars else x.second, 0)])\
        .mean()

    return sec1_seconds.drop(['DATE', 'SYM_SUFFIX', 'index'], 1).reset_index().rename(columns={'level_1': 'TIME_M'})


def test_strategy(data, positions, pnl):
    orders = []
    for sym in data:
        pos = positions.get(sym, 0)
        ema1 = data[sym]['dEMA_0.5']
        ema2 = data[sym]['dEMA_1']
        ema3 = data[sym]['dEMA_2']
        if (ema1 > 0.01) and (ema2 > 0.01) and (ema3 > 0.01):
            orders.append(Order(sym, 10-pos, type='market'))
        elif (ema1 < -0.01) and (ema2 < -0.01) and (ema3 < -0.01) and (pos >= 0):
            orders.append(Order(sym, -10-pos, type='market'))
    print positions
    print pnl
    return orders


root_dir = os.path.realpath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
#fname = 'xle_jan_3_12'
fname = 'xle_only_jan_5_12'
fpath = os.path.join(root_dir, 'data', fname, '{}.csv'.format(fname))
data = pd.read_csv(fpath, parse_dates=['TIME_M'], date_parser=convert_time)

data = make_bars(data, 'XLE', 9, 30, 15, 30, bar_width='second')

data['price'] = (data['BID'] + data['ASK'])/2

for hl in [0.5, 1, 2]:
    data['dEMA_{}'.format(hl)] = 0
    data.ix[1:, 'dEMA_{}'.format(hl)] = np.diff(pd.ewma(data['price'], halflife=hl))

pnl_history, order_history = backtest(data, test_strategy, transaction_costs=0.05)

pd.DataFrame(pnl_history).plot()

plt.show()