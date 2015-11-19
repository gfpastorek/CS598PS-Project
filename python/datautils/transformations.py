__author__ = 'thibautxiong'

import unittest
import pandas as pd
import datetime as dt


def parse_params(params, key, default):
    try:
        return params[key]
    except KeyError:
        return default
  

def apply_transformations():
    raise NotImplemented("TODO")


def add_ohlc(data, params):
    """
    Adds open/high/low/close/volume columns to the datautils
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
    Return ohlc datautils (instead of adding columns to original datautils)
    :param data:
    :param params:
    :return:
    """
    freq = params['freq']
    mid_prices = (data['ASK_PRICE']+data['BID_PRICE'])/2
    ohlc = mid_prices.resample(freq, how='ohlc', fill_method='ffill')
    bid_size_sum = data['BID_SIZE'].resample(freq, how='sum', fill_method='ffill')
    ask_size_sum = data['ASK_SIZE'].resample(freq, how='sum', fill_method='ffill')
    ohlc.columns = ['OPEN', 'HIGH', 'LOW', 'CLOSE']
    ohlc['BID_SIZE'] = bid_size_sum
    ohlc['ASK_SIZE'] = ask_size_sum

    return ohlc


def add_percent_change(data, params):
    input_column = parse_params(params, 'input_column', 'HIGH')
    output_column = parse_params(params, 'output_column', 'PCT_CHANGE')
    periods = parse_params(params, 'periods', 1)
    freq = parse_params(params, 'freq', None)
    pct = data[input_column].pct_change(periods=periods, freq=freq)
    data[output_column] = pct


# Tests
# from data_utils import get_test_data
#
#
# class TestTransformations(unittest.TestCase):
#
#     def test_ohlc(self):
#         datautils = get_test_data()
#         ohlc_params = {'freq': '5min'}
#         add_ohlc(datautils, ohlc_params)
#
#     def test_percent_return(self):
#         ohlc_params = {'freq': '5min'}
#         datautils = get_ohlc(get_test_data(), ohlc_params)
#         pct_return_params = {'input_column': 'HIGH',
#                              'output_column': 'HIGH_PCT_CHANGE'}
#         add_percent_change(datautils, pct_return_params)
#         # print datautils.head(5)
