import unittest
import pandas as pd
import numpy as np

def generate_features():
    raise NotImplemented("TODO")


def add_dema(data, halflives=[10, 40, 100]):

    for hl in halflives:
        ema = pd.ewma(data['price'], halflife=hl)
        data['dEMA_{}'.format(hl)] = 0
        data.ix[1:, 'dEMA_{}'.format(hl)] = pd.ewma(np.diff(ema), halflife=hl)
        data['dEMA_std_{}'.format(hl)] = pd.rolling_std(data['dEMA_{}'.format(hl)], 2*hl)   # TODO, what window to use?


def add_ema(data, halflives=[10, 40, 100]):

    for hl in halflives:
        data['EMA_{}'.format(hl)] = pd.ewma(data['price'], halflife=hl)