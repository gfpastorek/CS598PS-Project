__author__ = 'thibautxiong'

import unittest
import pandas as pd
import datetime as dt


def apply_transformations():
    raise NotImplemented("TODO")


def ohlc(data, params):
    """
    Adds open/high/low/close/volume columns to the data
    :param data:
    :param params:
    :return:
    """
    freq = params['freq']
    mid_prices = (data['ASK_PRICE']+data['BID_PRICE'])/2
    ohlc = mid_prices.resample(freq, how='ohlc', fill_method='ffill')
    bid_size_sum = data['BID_SIZE'].resample(freq, how='sum', fill_method='ffill')
    ask_size_sum = data['ASK_SIZE'].resample(freq, how='sum', fill_method='ffill')

    open_resampled = data['DATE_TIME'].apply(lambda x: ohlc['open'].asof(x))
    high_resampled = data['DATE_TIME'].apply(lambda x: ohlc['high'].asof(x))
    low_resampled = data['DATE_TIME'].apply(lambda x: ohlc['low'].asof(x))
    close_resampled = data['DATE_TIME'].apply(lambda x: ohlc['close'].asof(x))
    bid_size_resampled = data['DATE_TIME'].apply(lambda x: bid_size_sum.asof(x))
    ask_size_resampled = data['DATE_TIME'].apply(lambda x: ask_size_sum.asof(x))
    data['BAR_OPEN'] = open_resampled
    data['BAR_HIGH'] = high_resampled
    data['BAR_LOW'] = low_resampled
    data['BAR_CLOSE'] = close_resampled
    data['BAR_BID_SIZE'] = bid_size_resampled
    data['BAR_ASK_SIZE'] = ask_size_resampled


def get_ohlc(data, params):
    """
    Return ohlc data (instead of adding columns to original data)
    :param data:
    :param params:
    :return:
    """
    freq = params['freq']
    mid_prices = (data['ASK_PRICE']+data['BID_PRICE'])/2
    ohlc = mid_prices.resample(freq, how='ohlc', fill_method='ffill')
    bid_size_sum = data['BID_SIZE'].resample(freq, how='sum', fill_method='ffill')
    ask_size_sum = data['ASK_SIZE'].resample(freq, how='sum', fill_method='ffill')
    ohlc['BID_SIZE'] = bid_size_sum
    ohlc['ASK_SIZE'] = ask_size_sum

    return ohlc


from data_utils import get_test_data


class TestTransformations(unittest.TestCase):
    def test_ohlc(self):
        data = get_test_data()
        ohlc_params = {'freq': '5min'}
        ohlc(data, ohlc_params)
        print data.head(5)
        # print data
        # print ohlc_result
        # print data.head(1)

