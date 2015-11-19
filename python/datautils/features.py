import unittest
import pandas as pd
import numpy as np
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