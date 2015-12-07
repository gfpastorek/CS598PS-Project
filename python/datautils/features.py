import unittest
import pandas as pd
import numpy as np
import datetime as dt
from scipy import stats, spatial
import unittest
import sklearn as sk


def generate_features():
    raise NotImplemented("TODO")


def standardize_features(data, feature_names):

    data[feature_names] = (data[feature_names] - data[feature_names].mean()) / data[feature_names].std()


def add_future_log_returns(data, label_hls=(10, 40, 100)):
    data['price'] = (data['BID_PRICE']*data['BID_SIZE'] + data['ASK_PRICE']*data['ASK_SIZE']) / (data['BID_SIZE'] + data['ASK_SIZE'])
    data['log_returns'] = data['log_returns'] = np.concatenate([[0], np.diff(np.log(data['price']))])

    for hl in label_hls:
        data['log_returns_{}+'.format(hl)] = \
            np.concatenate([(pd.ewma(data['log_returns'].values[::-1], halflife=hl))[:0:-1], [0]])
        data['log_returns_std_{}+'.format(hl)] = \
            np.concatenate([(pd.rolling_std(data['log_returns'].values[::-1], 2*hl))[:0:-1], [0]])


def add_future_log_returns_rolling(data, windows=(10, 40, 100)):

    data['price'] = (data['BID_PRICE']*data['BID_SIZE'] + data['ASK_PRICE']*data['ASK_SIZE']) / (data['BID_SIZE'] + data['ASK_SIZE'])
    data['log_returns'] = data['log_returns'] = np.concatenate([[0], np.diff(np.log(data['price']))])

    for w in windows:
        data['log_returns_w{}+'.format(w)] = \
            np.concatenate([(pd.rolling_mean(data['log_returns'].values[::-1], w))[:0:-1], [0]])

def add_price_dema(data, halflives=[10, 40, 100], colname='dEMA'):
    for hl in halflives:
        ema = pd.ewma(data['price'], halflife=hl)
        data['{}_{}'.format(colname, hl)] = 0
        data.ix[1:, '{}_{}'.format(colname, hl)] = np.diff(ema)
    return ['{}_{}'.format(colname, hl) for hl in halflives]


def add_ema(data, halflives=[10, 40, 100], colname='EMA'):
    for hl in halflives:
        data['{}_{}'.format(colname, hl)] = pd.ewma(data['price'], halflife=hl)
    return ['{}_{}'.format(colname, hl) for hl in halflives]


def add_momentum(data, halflives=[10, 40, 100], colname='momentum'):
    add_ema(data, halflives=halflives)
    data[colname] = 0
    for hl1 in halflives:
        for hl2 in halflives:
            if hl2 < hl1:
                data[colname] += data['EMA_{}'.format(hl1)] - data['EMA_{}'.format(hl2)]
    return [colname]


def add_dema_sum(data, halflives=[10, 40, 100], colname='dEMA_sum'):
    add_price_dema(data, halflives=halflives)
    data[colname] = 0
    for hl in halflives:
        data[colname] += data['dEMA_{}'.format(hl)]
    return [colname]


def add_log_return_ema(data, halflives=(10, 40, 100)):

    data['price'] = (data['BID_PRICE']*data['BID_SIZE'] + data['ASK_PRICE']*data['ASK_SIZE']) / (data['BID_SIZE'] + data['ASK_SIZE'])
    data['log_returns'] = data['log_returns'] = np.concatenate([[0], np.diff(np.log(data['price']))])

    for hl in halflives:
        data['log_returns_{}-'.format(hl)] = \
            np.concatenate([[0], (pd.ewma(data['log_returns'].values[:-1], halflife=hl))])
        # TODO - how to get the EWMA decay to match the rolling_std window?
        data['log_returns_std_{}-'.format(hl)] = \
            np.concatenate([[0], (pd.rolling_std(data['log_returns'].values[:-1], 2*hl))])
    return ['log_returns_{}-'.format(hl) for hl in halflives] + ['log_returns_std_{}-'.format(hl) for hl in halflives]


def add_trade_momentum(data, trades, bar_width='second', colname='trade_momentum'):
    minute_bars = (bar_width == 'minute')
    data['price'] = (data['BID_PRICE']*data['BID_SIZE'] + data['ASK_PRICE']*data['ASK_SIZE']) / (data['BID_SIZE'] + data['ASK_SIZE'])
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
    return [colname]


def add_trade_momentum_dema(data, trades, halflife=10, bar_width='second', colname='trade_momentum_dema'):
    if 'trade_momentum' not in data.columns:
        add_trade_momentum(data, trades, bar_width=bar_width)
    feat_col_name = '{}_{}'.format(colname, halflife)
    data[feat_col_name] = 0
    ema = pd.ewma(data['trade_momentum'], halflife=halflife)
    data.ix[1:, feat_col_name] = np.diff(ema)
    print "Added {}".format(feat_col_name)


def add_dema(data, features, halflife=10):
    if type(features) == 'str':
        features = [features]
    feat_cols = []
    for feature in features:
        feat_col_name = '{}_dema_{}'.format(feature, halflife)
        data[feat_col_name] = 0
        ema = pd.ewma(data[feature], halflife=halflife)
        data.ix[1:, feat_col_name] = np.diff(ema)
        feat_cols.append(feat_col_name)
    return feat_cols


def add_size_diff(data):
    data['size_diff'] = data['BID_SIZE'] - data['ASK_SIZE']
    return ['size_diff']


def add_price_diff(data):
    data['price_diff'] = data['ASK_PRICE'] - data['BID_PRICE']
    return ['price_diff']


def add_mid_price(data):
    data['MID_PRICE'] = data['ASK_PRICE']-data['BID_PRICE']


def add_rolling_trade_sum(data, window):
    start_dates = data['DATE_TIME'] - window
    data['start_index'] = data['DATE_TIME'].values.searchsorted(start_dates, side='right')
    data['end_index'] = np.arange(len(data))
    def sum_window(row):
        return data['SIZE'].iloc[row['start_index']:row['end_index']+1].sum()
    data['TRADE_SUM'] = data.apply(sum_window, axis=1)
    data = data.drop(['start_index', 'end_index'], axis=1, inplace=True)




def add_vpin_time(data, window):

    data['lift'] = ((2*data['PRICE'] - data['BID_PRICE'] - data['ASK_PRICE']) > 0).apply(int) * data['SIZE']

    start_dates = data['DATE_TIME'] - window
    data['start_index'] = data['DATE_TIME'].values.searchsorted(start_dates, side='right')
    data['end_index'] = np.arange(len(data))

    def sum_window(row):
        return data['SIZE'].iloc[row['start_index']:row['end_index']+1].sum()

    def sum_lifts(row):
        return data['lift'].iloc[row['start_index']:row['end_index']+1].sum()

    data['trade_sum'] = data.apply(sum_window, axis=1)
    data['lift_sum'] = data.apply(sum_lifts, axis=1)
    data['VPIN_TIME'] = data['lift_sum']/data['trade_sum']
    data.drop(['start_index', 'end_index', 'lift', 'trade_sum', 'lift_sum'], axis=1, inplace=True)
    return ['VPIN_TIME']


def add_crossover(data, halflives):
    add_ema(data, halflives)
    crossover_list = []
    for hl1 in halflives:
        for hl2 in halflives:
            if hl2 < hl1:
                diffs = ((data['EMA_{}'.format(hl1)] - data['EMA_{}'.format(hl2)]) >= 0).values
                crossover_list.append(diffs[:-1] ^ diffs[1:])
    data['crossover?'] = np.concatenate([[False], np.any(crossover_list, 0)])


class TestFeatures(unittest.TestCase):
    def test_add_mid_price(self):
        pass
