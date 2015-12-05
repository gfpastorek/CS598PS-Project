import unittest
import pandas as pd
import numpy as np
import datetime as dt
from scipy import stats, spatial
import unittest


def generate_features():
    raise NotImplemented("TODO")


def label_data(data, label_hls=(10, 40, 100)):

    data['price'] = (data['BID_PRICE']*data['BID_SIZE'] + data['ASK_PRICE']*data['ASK_SIZE']) / (data['BID_SIZE'] + data['ASK_SIZE'])
    data['log_returns'] = data['log_returns'] = np.concatenate([[0], np.diff(np.log(data['price']))])

    # TODO - which halflife to use? Kalman filter?
    for hl in label_hls:
        data['log_returns_{}+'.format(hl)] = \
            np.concatenate([(pd.ewma(data['log_returns'].values[::-1], halflife=hl))[:0:-1], [0]])
        # TODO - how to get the EWMA decay to match the rolling_std window?
        data['log_returns_std_{}+'.format(hl)] = \
            np.concatenate([(pd.rolling_std(data['log_returns'].values[::-1], 2*hl))[:0:-1], [0]])


def add_dema(data, halflives=[10, 40, 100], colname='dEMA'):
    for hl in halflives:
        ema = pd.ewma(data['price'], halflife=hl)
        data['{}_{}'.format(colname, hl)] = 0
        data.ix[1:, '{}_{}'.format(colname, hl)] = np.diff(ema)
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
            np.concatenate([[0], (pd.ewma(data['log_returns'].values[:-1], halflife=hl))])
        # TODO - how to get the EWMA decay to match the rolling_std window?
        data['log_returns_std_{}-'.format(hl)] = \
            np.concatenate([[0], (pd.rolling_std(data['log_returns'].values[:-1], 2*hl))])


def add_size_diff(data):
    data['size_diff'] = data['BID_SIZE'] - data['ASK_SIZE']


def add_price_diff(data):
    data['price_diff'] = data['ASK_PRICE'] - data['BID_PRICE']


def add_trade_momentum(data, trades, bar_width='second', colname='trade_momentum'):
    minute_bars = (bar_width == 'minute')
    trades = trades.set_index('DATE_TIME')
    trades['PRICExSIZE'] = trades['PRICE'] * trades['SIZE']
    trades = \
        trades.groupby(['SYM', lambda x: dt.datetime(x.year, x.month, x.day, x.hour, x.minute,
                                                       0 if minute_bars else x.second, 0)])\
        .agg({
                 'PRICE': 'mean',
                 'SIZE': 'sum',
                 'PRICExSIZE': 'sum'
             })
    trades['MEAN_TRADE_PRICE'] = trades['PRICExSIZE'] / trades['SIZE']
    trades = trades.reset_index().rename(columns={'level_1': 'DATE_TIME'})
    data[['MEAN_TRADE_PRICE', 'SIZE']] = data.merge(trades, on=['SYM', 'DATE_TIME'], how='left')[['MEAN_TRADE_PRICE', 'SIZE']]
    data[colname] = data['MEAN_TRADE_PRICE'] - data['price']   # TODO - normalize
    data.drop('MEAN_TRADE_PRICE', axis=1, inplace=True)
    data.reset_index(inplace=True)


def add_trade_momentum_dema(data, trades, halflife=10, bar_width='second', colname='trade_momentum_dema'):
    if 'trade_momentum' not in data.columns:
        add_trade_momentum(data, trades, bar_width=bar_width)
    feat_col_name = '{}_{}'.format(colname, halflife)
    data[feat_col_name] = 0
    ema = pd.ewma(data['trade_momentum'], halflife=halflife)
    data.ix[1:, feat_col_name] = np.diff(ema)
    print "Added {}".format(feat_col_name)


def add_mid_price(data):
    data['MID_PRICE'] = data['ASK_PRICE']-data['BID_PRICE']

def add_cum_trade_volume(data):
    pass



class TestFeatures(unittest.TestCase):
    def test_add_mid_price(self):
        pass
