import unittest
import pandas as pd
import numpy as np
from scipy import stats, spatial

def generate_features():
    raise NotImplemented("TODO")


def label_data(data, label_hls=(10, 40, 100)):

    data['price'] = (data['BID']*data['BIDSIZ'] + data['ASK']*data['ASKSIZ']) / (data['BIDSIZ'] + data['ASKSIZ'])
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
    for i in xrange(0, len(data)):
        ranking = stats.rankdata(data[['EMA_{}'.format(hl) for hl in halflives]].iloc[i])
        data.ix[i, colname] = spatial.distance.hamming(ranking, range(1, len(halflives)+1))