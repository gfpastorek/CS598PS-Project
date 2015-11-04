__author__ = 'thibautxiong'


import unittest
import pandas as pd


def generate_features():
    raise NotImplemented("TODO")


def parse_params(params, key, default):
    try:
        return params[key]
    except KeyError:
        return default


def percent_change(data, params):
    """

    :param data:
    :param params:
    :return:
    """
    input_column = parse_params(params, 'input_column', 'HIGH')
    output_column = parse_params(params, 'output_column', 'PCT_CHANGE')
    periods = parse_params(params, 'periods', 1)
    freq = parse_params(params, 'freq', None)
    pct = data[input_column].pct_change(periods=periods, freq=freq)
    data[output_column] = pct



from data_utils import get_test_data
from transformations import get_ohlc


class TestFeatures(unittest.TestCase):
    def test_percent_return(self):
        ohlc_params = {'freq': '5min'}
        data = get_ohlc(get_test_data(), ohlc_params)
        pct_return_params = {'input_column': 'HIGH',
                             'output_column': 'HIGH_PCT_CHANGE'}
        percent_change(data, pct_return_params)
        # print data.head(5)

