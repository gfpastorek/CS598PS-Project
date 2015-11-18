import unittest
import pandas as pd
import numpy as np
import scipy as sp

def generate_features():
    raise NotImplemented("TODO")


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
        ranking = sp.stats.rankdata(data[['EMA_{}'.format(hl) for hl in halflives]].iloc[i])
        data[colname].iloc[i] = sp.spatial.distance.hamming(ranking, range(1, len(halflives)+1))