from __future__ import division
import pandas as pd
import datetime as dt
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


def test_strategy(data):
    print data
    return []


root_dir = os.path.realpath(os.path.dirname(os.path.dirname(__file__)))
#fname = 'xle_jan_3_12'
fname = 'xle_only_jan_5_12'
fpath = os.path.join(root_dir, 'data', fname, '{}.csv'.format(fname))
data = pd.read_csv(fpath, parse_dates=['TIME_M'], date_parser=convert_time)

data = make_bars(data, 'XLE', 9, 30, 15, 30, bar_width='second')

backtest(data, test_strategy)
