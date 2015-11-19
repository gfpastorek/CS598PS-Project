import datetime as dt
import numpy as np
import pandas as pd
import os
from transformations import get_ohlc

# TODO: filter by time (maybe in get_data? --> pass in start_time and end_time for intraday limits)

# DATE,TIME_M,SYM_ROOT,SYM_SUFFIX,BID,BIDSIZ,ASK,ASKSIZ,BIDEX,ASKEX,NATBBO_IND


def _convert_time(date, time):
    """
    :param date: string representation of a date, format: yyyymmdd
    :param time: string representation of a time, format: hh:mm:ss.ssss
    """
    year = date[:4]
    month = date[4:6]
    day = date[-2:]
    h, m, s = time.split(':')
    s, ms = map(int, s.split('.'))
    return dt.datetime(int(year), int(month), int(day), int(h), int(m), int(s), int(ms)*1000)


def filter_data_by_time(data, start_time, end_time):
    """
    Removes data before start_time and end_time
    """
    # data = data[data['DATE_TIME_M'] >= start_time]
    # data = data[data['DATE_TIME_M'] <= end_time]
    data = data.between_time(start_time, end_time, include_start=True, include_end=True)
    return data


def filter_invalid_quotes(data):
    """
    Filter out invalid quotes
    """
    data = data[data['BID'] != 0]
    data = data[data['ASK'] != 0]
    data = data[data['NATBBO_IND'] == 1]
    return data


def clean_quotes(data, sec=None, start_hour=9, start_min=30, end_hour=15, end_min=30, bar_width='second'):

    """

    :param data:
    :param sec:
    :param start_hour:
    :param start_min:
    :param end_hour:
    :param end_min:
    :param bar_width:
    :return:
    """

    data = filter_invalid_quotes(data)

    data = data.set_index('DATE_TIME_M', drop=True)
    data.index.names = ['DATE_TIME']
    data.columns = ['SYM', 'SYM_SUFFIX', 'BID_PRICE', 'BID_SIZE', 'ASK_PRICE', 'ASK_SIZE', 'NATBBO_IND']
    data['DATE_TIME'] = data.index.values

    start_time = dt.time(start_hour, start_min, 0, 0)
    end_time = dt.time(end_hour, end_min, 0, 0)
    data = filter_data_by_time(data, start_time, end_time)

    return data

    # minute_bars = (bar_width == 'minute')
    #
    # datautils = datautils[datautils['TIME_M'] >= dt.datetime(2012, 1, 3, start_hour, start_min, 0, 0)]
    # datautils = datautils[datautils['TIME_M'] <= dt.datetime(2012, 1, 3, end_hour, end_min, 0, 0)]
    #
    # if sec is not None:
    #     datautils = datautils[datautils['SYM_ROOT'] == sec]
    #
    # datautils = datautils.reset_index().drop(['DATE', 'SYM_SUFFIX', 'index'], 1)
    #
    # if bar_width is None:
    #     return datautils
    # else:
    #     datautils = \
    #         datautils.set_index('TIME_M')\
    #         .groupby(['SYM_ROOT', lambda x: dt.datetime(x.year, x.month, x.day, x.hour, x.minute, 0 if minute_bars else x.second, 0)])\
    #         .agg({
    #             'ASK': 'mean',
    #             'BID': 'mean',
    #             'ASKSIZ': 'max',
    #             'BIDSIZ': 'max'
    #         })
    #     return datautils.reset_index().rename(columns={'level_1': 'TIME_M'})


def get_data(ticker, year, month, day, bar_width='second', label_halflives=[10, 40, 100]):
    # filename = "{}_{}".format(ticker, dt.datetime(year, month, day).strftime("%m_%d_%y"))
    # root_dir = os.path.realpath(os.path.dirname(os.getcwd()))
    # fpath = os.path.join(root_dir, 'datautils', filename, '{}.csv'.format(filename))
    data = pd.read_csv("/Users/thibautxiong/Documents/Development/CS598PS-Project/xle_test.csv",
                       parse_dates=[['DATE', 'TIME_M']], date_parser=_convert_time)
    data = clean_quotes(data, bar_width=bar_width)
    return data


import unittest


class TestDataUtils(unittest.TestCase):
    def test_get_test_data(self):
        # get_data(None, None, None, None)
        pass

    def test_convert_time(self):
        date = "20120105"
        time = "09:30:00.291"
        dt = _convert_time(date, time)
        assert dt.year == 2012
        assert dt.month == 1
        assert dt.day == 5
        assert dt.hour == 9
        assert dt.minute == 30
        assert dt.second == 0

    def test_filter_data_by_time(self):
        pass



