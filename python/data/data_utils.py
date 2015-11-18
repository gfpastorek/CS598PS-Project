import datetime as dt
import numpy as np
import pandas as pd
import os

# TODO: filter by time (maybe in get_data? --> pass in start_time and end_time for intraday limits)


def convert_time(d):
    h, m, s = d.split(':')
    s, ms = map(int, s.split('.'))
    return dt.datetime(2012, 1, 3, int(h), int(m), int(s), int(ms)*1000)


def get_test_data():
    df = pd.read_csv('/Users/thibautxiong/Documents/Development/CS598PS-Project/xle_test.csv',
                     parse_dates=[['DATE', 'TIME_M']], date_parser=convert_time)
    df = df[df.NATBBO_IND != 0]
    df = df.set_index('DATE_TIME_M', drop=True)
    df.index.names = ['DATE_TIME']
    df.columns = ['SYM', 'SYM_SUFFIX', 'BID_PRICE', 'BID_SIZE', 'ASK_PRICE', 'ASK_SIZE', 'NATBBO_IND']
    df['DATE_TIME'] = df.index.values
    return df


def _clean_quotes(data, sec=None, start_hour=9, start_min=30, end_hour=15, end_min=30, bar_width='second'):

    data = data.reset_index()

    data = data[data['BID'] != 0]
    data = data[data['ASK'] != 0]
    data = data[data['NATBBO_IND'] == 1]

    minute_bars = (bar_width == 'minute')

    data = data[data['TIME_M'] >= dt.datetime(2012, 1, 3, start_hour, start_min, 0, 0)]
    data = data[data['TIME_M'] <= dt.datetime(2012, 1, 3, end_hour, end_min, 0, 0)]

    if sec is not None:
        data = data[data['SYM_ROOT'] == sec]

    data = data.reset_index().drop(['DATE', 'SYM_SUFFIX', 'index'], 1)

    if bar_width is None:
        return data
    else:
        data = \
            data.set_index('TIME_M')\
            .groupby(['SYM_ROOT', lambda x: dt.datetime(x.year, x.month, x.day, x.hour, x.minute, 0 if minute_bars else x.second, 0)])\
            .agg({
                'ASK': 'mean',
                'BID': 'mean',
                'ASKSIZ': 'max',
                'BIDSIZ': 'max'
            })
        return data.reset_index().rename(columns={'level_1': 'TIME_M'})


# TODO - come up with a cooler way to label data, optimally adjust the half-life
def _label_data(data, label_hls=(10, 40, 100)):

    data['price'] = (data['BID']*data['BIDSIZ'] + data['ASK']*data['ASKSIZ']) / (data['BIDSIZ'] + data['ASKSIZ'])
    data['log_returns'] = data['log_returns'] = np.concatenate([[0], np.diff(np.log(data['price']))])

    # TODO - which halflife to use? Kalman filter?
    for hl in label_hls:
        data['log_returns_{}+'.format(hl)] = \
            np.concatenate([(pd.ewma(data['log_returns'].values[::-1], hl))[:0:-1], [0]])
        # TODO - how to get the EWMA decay to match the rolling_std window?
        data['log_returns_std_{}+'.format(hl)] = \
            np.concatenate([(pd.rolling_std(data['log_returns'].values[::-1], 2*hl))[:0:-1], [0]])

    return data


def get_data(ticker, year, month, day, bar_width='second', label_halflives=[10, 40, 100]):
    filename = "{}_{}".format(ticker, dt.datetime(year, month, day).strftime("%m_%d_%y"))
    root_dir = os.path.realpath(os.path.dirname(os.getcwd()))
    fpath = os.path.join(root_dir, 'data', filename, '{}.csv'.format(filename))
    data = pd.read_csv(fpath, parse_dates=['TIME_M'], date_parser=convert_time)
    data = _clean_quotes(data, bar_width=bar_width)
    data = _label_data(data, label_hls=label_halflives)
    return data



