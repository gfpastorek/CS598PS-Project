from __future__ import division
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import os


# XLE XOM CVX SLB KMI EOG COP OXY PXD VLO USO

# DATE,TIME_M,SYM_ROOT,SYM_SUFFIX,BID,BIDSIZ,ASK,ASKSIZ,BIDEX,ASKEX,NATBBO_IND

def convert_time(d):
    h, m, s = d.split(':')
    s, ms = map(int, s.split('.'))
    return dt.datetime(2012, 1, 3, int(h), int(m), int(s), int(ms)*1000)


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


def make_bars(data, sec1n, start_hour=9, start_min=30, end_hour=15, end_min=30, bar_width='second'):

    minute_bars = (bar_width == 'minute')

    data = data[data['TIME_M'] >= dt.datetime(2012,1,3,start_hour,start_min,0,0)]
    data = data[data['TIME_M'] <= dt.datetime(2012,1,3,end_hour,end_min,0,0)]

    sec1 = data[data['SYM_ROOT'] == sec1n]

    sec1_seconds = \
        sec1.set_index('TIME_M')\
        .groupby(['SYM_ROOT', lambda x: dt.datetime(x.year, x.month, x.day, x.hour, x.minute, 0 if minute_bars else x.second, 0)])\
        .mean()

    return sec1_seconds.drop(['DATE', 'SYM_SUFFIX', 'index'], 1).reset_index().rename(columns={'level_1': 'TIME_M'})


def plot_spread(data, sec1n, sec2n, B, start_hour=9, start_min=30, end_hour=15, end_min=30):
    data = data[data['TIME_M'] >= dt.datetime(2012,1,3,start_hour,start_min,0,0)]
    data = data[data['TIME_M'] <= dt.datetime(2012,1,3,end_hour,end_min,0,0)]
    sec1 = data[data['SYM_ROOT'] == sec1n]
    sec2 = data[data['SYM_ROOT'] == sec2n]

    idx = np.searchsorted(sec1['TIME_M'], sec2['TIME_M'])
    mask = idx >= 0

    df = pd.DataFrame().reset_index()
    df['TIME'] = sec2['TIME_M'].values
    df["{}_BID".format(sec1n)] = sec1['BID'].iloc[idx-1][mask].values
    df["{}_ASK".format(sec1n)] = sec1['ASK'].iloc[idx-1][mask].values
    df["{}_BID".format(sec2n)] = sec2['BID'].values
    df["{}_ASK".format(sec2n)] = sec2['ASK'].values
    df = df.drop('index', 1)

    df['{}-{}'.format(sec1n, sec2n)] = df['{}_ASK'.format(sec1n)] - B*df['{}_BID'.format(sec2n)]
    df['{}-{}'.format(sec1n, sec2n)].plot()
    df['{}-{}'.format(sec2n, sec1n)] = df['{}_BID'.format(sec1n)] - B*df['{}_ASK'.format(sec2n)]
    df['{}-{}'.format(sec2n, sec1n)].plot()
    plt.show()


next_order_id = 1

class Order(object):
    def __init__(self, sym, qty, type="market"):
        self.sym = sym
        self.qty = qty
        self.type = type
        global next_order_id
        self.order_id = next_order_id
        next_order_id += 1


def backtest(data, strategy):

    cash = [0.0]
    pnl_history = []
    order_history = []
    positions = {}
    security_data = {}

    def check_signals(quotes):
        for i in xrange(0, len(quotes)):
            security_data[quotes.iloc[i]['SYM_ROOT']] = quotes.iloc[i]
        orders = strategy(security_data)
        for order in orders:
            sdata = security_data[order.sym]
            positions[order.sym] = positions.get(order.sym, 0) + order.qty
            price = sdata['BID'] if order.qty > 0 else sdata['ASK']  # TODO - what if qty is larger than BID/ASK QTY?
            order_history.append((quotes.iloc[0]['TIME_M'], order.sym, order.qty, price))
            cash[0] += -1 * order.qty * price
        portfolio_value = np.sum(
            [positions[sym] * (security_data[sym]['BID'] if positions[sym] > 0 else security_data[sym]['ASK']) for sym in
             positions])
        pnl_history.append(cash[0] + portfolio_value)

    data.groupby('TIME_M').apply(check_signals)

    return pnl_history, order_history





def test_strategy(data):
    print data
    return []

root_dir = os.path.realpath(os.path.dirname(os.path.dirname(__file__)))
#fname = 'xle_jan_3_12'
fname = 'xle_only_jan_5_12'
fpath = os.path.join(root_dir, 'data', fname, '{}.csv'.format(fname))
data = pd.read_csv(fpath, parse_dates=['TIME_M'], date_parser=convert_time)

data = data.reset_index()

data = data[data['BID'] != 0]
data = data[data['ASK'] != 0]
data = data[data['NATBBO_IND'] == 1]

#plot_series(data, 'XLE', 9, 30, 15, 30)

data = make_bars(data, 'XLE', 9, 30, 15, 30, bar_width='second')
backtest(data, test_strategy)

"""
plot_spread(data, 'IAU', 'SGOL', 0.1, 8, 30, 16, 30)
plot_spread(data, 'IAU', 'UGLD', 1, 8, 30, 16, 30)
plot_spread(data, 'IAU', 'DZZ', 1, 8, 30, 16, 30)
"""



