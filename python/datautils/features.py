import unittest
import pandas as pd
import numpy as np
import datetime as dt
from scipy import stats, spatial


def generate_features():
    raise NotImplemented("TODO")


def label_data(data, label_hls=(10, 40, 100)):

    data['price'] = (data['BID_PRICE']*data['BID_SIZE'] + data['ASK_PRICE']*data['ASK_SIZE']) / (data['BID_SIZE'] + data['ASK_SIZE'])
    data['log_returns'] = data['log_returns'] = np.concatenate([[0], np.diff(np.log(data['price']))])

    # TODO - which halflife to use? Kalman filter?
    for hl in label_hls:
        data['log_returns_{}+'.format(hl)] = \
            np.concatenate([(pd.ewma(data['log_returns'].values[::-1], hl))[:0:-1], [0]])
        # TODO - how to get the EWMA decay to match the rolling_std window?
        data['log_returns_std_{}+'.format(hl)] = \
            np.concatenate([(pd.rolling_std(data['log_returns'].values[::-1], 2*hl))[:0:-1], [0]])


def add_dema(data, halflives=[10, 40, 100], colname='dEMA'):
    for hl in halflives:
        ema = pd.ewma(data['price'], halflife=hl)
        data['{}_{}'.format(colname, hl)] = 0
        data.ix[1:, '{}_{}'.format(colname, hl)] = pd.ewma(np.diff(ema), halflife=hl)
        data['{}_std_{}'.format(colname, hl)] = \
            pd.rolling_std(data['{}_{}'.format(colname, hl)], 2*hl)   # TODO, what window to use?


def add_ema(data, halflives=[10, 40, 100], colname='EMA'):
    for hl in halflives:
        data['{}_{}'.format(colname, hl)] = pd.ewma(data['price'], halflife=hl)


def add_momentum(data, halflives=[10, 40, 100], colname='momentum'):
    add_ema(data, halflives=halflives)
    data[colname] = 0
    for hl1 in halflives:
        for hl2 in halflives:
            if hl2 < hl1:
                data[colname] += data['EMA_{}'.format(hl1)] - data['EMA_{}'.format(hl2)]


def add_log_return_ema(data, halflives=(10, 40, 100)):

    data['price'] = (data['BID_PRICE']*data['BID_SIZE'] + data['ASK_PRICE']*data['ASK_SIZE']) / (data['BID_SIZE'] + data['ASK_SIZE'])
    data['log_returns'] = data['log_returns'] = np.concatenate([[0], np.diff(np.log(data['price']))])

    for hl in halflives:
        data['log_returns_{}-'.format(hl)] = \
            np.concatenate([[0], (pd.ewma(data['log_returns'].values[:-1], hl))])
        # TODO - how to get the EWMA decay to match the rolling_std window?
        data['log_returns_std_{}-'.format(hl)] = \
            np.concatenate([[0], (pd.rolling_std(data['log_returns'].values[:-1], 2*hl))])


def add_size_diff(data):
    data['size_diff'] = data['BID_SIZE'] - data['ASK_SIZE']


def add_trade_momentum(data, trades, bar_width='second'):
    minute_bars = (bar_width == 'second')
    trades = trades.set_index('DATE_TIME')
    trades['PRICExSIZE'] = trades['PRICE'] * trades['SIZE']
    trades = \
        trades.groupby(['SYM', lambda x: dt.datetime(x.year, x.month, x.day, x.hour, x.minute,
                                                       0 if minute_bars else x.second, 0)])\
        .agg({
                 'PRICE': 'mean',
                 'SIZE': 'sum',
                 'PRICExSIZE': 'mean'
             })
    trades['PRICE'] = trades['PRICExSIZE'] / trades['SIZE']
    trades = trades.reset_index().rename(columns={'level_1': 'DATE_TIME'})
    trades = trades.drop('PRICExSIZE', 1)
    data = pd.join(data, trades, index=['SYM', 'DATE_TIME'])
    return data.reset_index()
