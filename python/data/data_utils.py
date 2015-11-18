__author__ = 'thibautxiong'

import datetime as dt
import pandas as pd

# TODO: filter by time (maybe in get_data? --> pass in start_time and end_time for intraday limits)


def convert_time(d):
    """
    :param d: string "YYYYMMDD HH:MM:SSSS"
    :return: datetime object
    """
    date = d.split(' ')[0]
    year = date[:4]
    month = date[4:6]
    day = date[6:]
    time = d.split(' ')[1]
    h, m, s = time.split(':')
    s, ms = map(int, s.split('.'))
    return dt.datetime(int(year), int(month), int(day), int(h), int(m), int(s), int(ms)*1000)


def get_test_data():
    df = pd.read_csv('/Users/thibautxiong/Documents/Development/CS598PS-Project/xle_test.csv',
                     parse_dates=[['DATE', 'TIME_M']], date_parser=convert_time)
    df = df[df.NATBBO_IND != 0]
    df = df.set_index('DATE_TIME_M', drop=True)
    df.index.names = ['DATE_TIME']
    df.columns = ['SYM', 'SYM_SUFFIX', 'BID_PRICE', 'BID_SIZE', 'ASK_PRICE', 'ASK_SIZE', 'NATBBO_IND']
    df['DATE_TIME'] = df.index.values
    return df


def get_data(filename):

    

    for hl in hls:
        data['log_returns_{}+'.format(hl)] = np.concatenate([(pd.ewma(data['log_returns'].values[::-1], hl))[:0:-1], [None]])
        data['log_returns_{}-'.format(hl)] = np.concatenate([[0], pd.ewma(data['log_returns'], hl).values[:-1]])
        data['log_returns_std_{}+'.format(hl)] = np.concatenate([(pd.rolling_std(data['log_returns'].values[::-1], 2*hl))[:0:-1], [None]])
        data['log_returns_std_{}-'.format(hl)] = np.concatenate([[0], pd.rolling_std(data['log_returns'], 2*hl).values[:-1]])


def get_data(ticker, start, end):
    """

    :param ticker:
    :param start:
    :param end:
    :return:
    """
    raise NotImplemented("TODO")

