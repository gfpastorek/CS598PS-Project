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
    return data.between_time(start_time, end_time, include_start=True, include_end=True)


def filter_invalid_quotes(data):
    """
    Filter out invalid quotes
    """
    data = data[data['BID'] != 0]
    data = data[data['ASK'] != 0]
    data = data[data['NATBBO_IND'] == 1]
    return data


def _clean_quotes(data, start_hour=9, start_min=30, end_hour=15, end_min=30, bar_width='second'):

    data = filter_invalid_quotes(data)
    data = data.drop(['SYM_SUFFIX', 'NATBBO_IND'], 1)

    data.columns = ['DATE_TIME', 'SYM', 'BID_PRICE', 'BID_SIZE', 'ASK_PRICE', 'ASK_SIZE']

    data = data.set_index('DATE_TIME')

    start_time = dt.time(start_hour, start_min, 0, 0)
    end_time = dt.time(end_hour, end_min, 0, 0)
    data = filter_data_by_time(data, start_time, end_time)

    if bar_width is None:
        return data
    else:
        minute_bars = (bar_width == 'minute')
        data = \
            data.groupby(['SYM', lambda x: dt.datetime(x.year, x.month, x.day, x.hour, x.minute,
                                                       0 if minute_bars else x.second, 0)])\
            .agg({
                 'ASK_PRICE': 'mean',
                 'BID_PRICE': 'mean',
                 'ASK_SIZE': 'max',
                 'BID_SIZE': 'max'
             })
    return data.reset_index().rename(columns={'level_1': 'DATE_TIME'})


def _clean_trades(data, start_hour=9, start_min=30, end_hour=15, end_min=30):

    data = data.drop('SYM_SUFFIX', 1)
    data.columns = ['DATE_TIME', 'SYM', 'SIZE', 'PRICE']

    data = data.set_index('DATE_TIME')

    start_time = dt.time(start_hour, start_min, 0, 0)
    end_time = dt.time(end_hour, end_min, 0, 0)
    data = filter_data_by_time(data, start_time, end_time)

    return data.reset_index().rename(columns={'level_1': 'DATE_TIME'})


def get_quotes(ticker, year, month, day, bar_width='second'):
    filename = "{}_{}".format(ticker.lower(), dt.datetime(year, month, day).strftime("%m_%d_%y"))
    try:
        root_dir = os.path.realpath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    except NameError:
        root_dir = os.path.realpath(os.path.dirname(os.getcwd()))
    fpath = os.path.join(root_dir, 'data', filename, '{}_quotes.csv'.format(filename))
    data = pd.read_csv(fpath, parse_dates=[['DATE', 'TIME_M']], date_parser=_convert_time)
    data = _clean_quotes(data, bar_width=bar_width)
    return data


def get_trades(ticker, year, month, day):
    filename = "{}_{}".format(ticker.lower(), dt.datetime(year, month, day).strftime("%m_%d_%y"))
    try:
        root_dir = os.path.realpath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    except NameError:
        root_dir = os.path.realpath(os.path.dirname(os.getcwd()))
    fpath = os.path.join(root_dir, 'data', filename, '{}_trades.csv'.format(filename))
    data = pd.read_csv(fpath, parse_dates=[['DATE', 'TIME_M']], date_parser=_convert_time)
    data = _clean_trades(data)
    return data


def date_iter(year, month, day, days=1):
    start_date = dt.datetime(year, month, day)
    for cur_date in (start_date + dt.timedelta(n) for n in range(days)):
        yield cur_date.year, cur_date.month, cur_date.day


def get_data(ticker, year, month, day, bar_width='second'):
    quotes = get_quotes(ticker, year, month, day, bar_width=bar_width)
    trades = get_trades(ticker, year, month, day)
    return quotes, trades


# TODO - days counts empty days as days (i.e weekends), fix?
def get_more_data(tickers, year, month, day, days=1, bar_width='second'):
    data = []
    if type(tickers) == str:
        tickers = [tickers]
    for y, m, d in date_iter(year, month, day, days=days):
        for ticker in tickers:
            try:
                quotes, trades = get_data(ticker, y, m, d, bar_width=bar_width)
                data.append((quotes, trades))
            except (IOError, ValueError):
                continue
    return data


def get_dev_data():
    f_quotes = "/Users/thibautxiong/Documents/Development/CS598PS-Project/dev_data/xle_02_01_12_quotes_dev.csv"
    print "reading dev quotes"
    quotes = pd.read_csv(f_quotes, parse_dates=[['DATE', 'TIME_M']], date_parser=_convert_time)
    print "cleaning dev quotes"
    quotes = _clean_quotes(quotes, bar_width='second')
    f_trades = "/Users/thibautxiong/Documents/Development/CS598PS-Project/dev_data/xle_02_01_12_trades_dev.csv"
    print "reading dev trades"
    trades = pd.read_csv(f_trades, parse_dates=[['DATE', 'TIME_M']], date_parser=_convert_time)
    print "cleaning dev trades"
    trades = _clean_trades(trades)
    data = [(quotes, trades)]
    return data


def make_dev_data():
    """
    make some smaller data sets for dev purposes
    """
    quotes = open("/Users/thibautxiong/Documents/Development/CS598PS-Project/data/xle_02_01_12/xle_02_01_12_quotes.csv",
                  'r')
    quotes_out = open("/Users/thibautxiong/Documents/Development/CS598PS-Project/data/xle_02_01_12/xle_02_01_12_quotes_dev.csv", 'w')
    n_quotes = 0
    for line in quotes:
        quotes_out.write(line)
        if n_quotes > 1000:
            break
    trades = open("/Users/thibautxiong/Documents/Development/CS598PS-Project/data/xle_02_01_12/xle_02_01_12_trades.csv", 'r')
    trades_out = open("/Users/thibautxiong/Documents/Development/CS598PS-Project/data/xle_02_01_12/xle_02_01_12_trades_dev.csv", 'w')
    n_trades = 0
    for line in trades:
        trades_out.write(line)
        if n_trades > 4000:
            break


def merge_trades_and_quotes(data):
    for i in range(len(data)):
        trades_and_quotes = data[i]
        quotes = trades_and_quotes[0]
        trades = trades_and_quotes[1]
        merged = pd.ordered_merge(quotes, trades, fill_method='ffill')
        data[i] = merged
    return data



def get_test_data():
    # TODO
    pass

if __name__ == "__main__":
    pass
    # data = get_more_data('XLE', 2012, 2, 1, days=1, bar_width='second')
    # print data[0][0]
    # make_dev_data()
    # merge_trades_and_quotes()

# import unittest
#
# class TestDataUtils(unittest.TestCase):
#     def test_make_dev_data(self):
#         make_dev_data()
#     def test_get_test_data(self):
#         # get_data(None, None, None, None)
#         pass
#
#     def test_convert_time(self):
#         pass
#         date = "20120105"
#         time = "09:30:00.291"
#         dt = _convert_time(date, time)
#         assert dt.year == 2012
#         assert dt.month == 1
#         assert dt.day == 5
#         assert dt.hour == 9
#         assert dt.minute == 30
#         assert dt.second == 0
#
#     def test_filter_data_by_time(self):
#         pass
#         data = get_more_data('XLE', 2012, 2, 1, days=1, bar_width='second')
#         print len(data)
#         print data
#         merged = merge_trades_and_quotes(data[0], data[1])
#
#
