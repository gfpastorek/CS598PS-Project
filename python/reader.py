from __future__ import division
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt


# XLE XOM CVX SLB KMI EOG COP OXY PXD VLO USO

# DATE,TIME_M,SYM_ROOT,SYM_SUFFIX,BID,BIDSIZ,ASK,ASKSIZ,BIDEX,ASKEX,NATBBO_IND

def convert_time(d):
    h, m, s = d.split(':')
    s, ms = map(int, s.split('.'))
    return dt.datetime(2012, 1, 3, int(h), int(m), int(s), int(ms)*1000)


def plot_series(data, sec1n, start_hour=4, start_min=0, end_hour=20, end_min=0):
    data = data[data['TIME_M'] >= dt.datetime(2012,1,3,start_hour,start_min,0,0)]
    data = data[data['TIME_M'] <= dt.datetime(2012,1,3,end_hour,end_min,0,0)]

    sec1 = data[data['SYM_ROOT'] == sec1n]

    sec1_seconds = \
        sec1.set_index('TIME_M')\
        .groupby(lambda x: dt.datetime(x.year, x.month, x.day, x.hour, x.minute, x.second, 0))\
        .mean()

    sec1_seconds['BID'].plot()
    sec1_seconds['ASK'].plot()

    plt.show()


def plot_spread(data, sec1n, sec2n, B, start_hour=4, start_min=0, end_hour=20, end_min=0):
    data = data[data['TIME_M'] >= dt.datetime(2012,1,3,start_hour,start_min,0,0)]
    data = data[data['TIME_M'] <= dt.datetime(2012,1,3,end_hour,end_min,0,0)]
    sec1 = data[data['SYM_ROOT'] == sec1n]
    sec2 = data[data['SYM_ROOT'] == sec2n]

    idx = np.searchsorted(sec1['TIME_M'], sec2['TIME_M'])
    mask = idx >= 0

    df = pd.DataFrame().reset_index()
    df['TIME'] = sec2['TIME_M'].values
    df["{}_BID".format(sec1n)] = sec1['BID'].iloc[idx-1][mask].values
    df["{}_ASK".format(sec1n)] = sec1['ASK'].iloc[idx-1][mask].values
    df["{}_BID".format(sec2n)] = sec2['BID'].values
    df["{}_ASK".format(sec2n)] = sec2['ASK'].values
    df = df.drop('index', 1)

    df['{}-{}'.format(sec1n, sec2n)] = df['{}_ASK'.format(sec1n)] - B*df['{}_BID'.format(sec2n)]
    df['{}-{}'.format(sec1n, sec2n)].plot()
    df['{}-{}'.format(sec2n, sec1n)] = df['{}_BID'.format(sec1n)] - B*df['{}_ASK'.format(sec2n)]
    df['{}-{}'.format(sec2n, sec1n)].plot()
    plt.show()


fname = 'xle_jan_3_12'
fpath = "C:\Users\Greg\Documents\CS 598PS\project\data\\{}\{}.csv".format(fname, fname)
data = pd.read_csv(fpath, parse_dates=['TIME_M'], date_parser=convert_time)

data = data.reset_index()

data = data[data['BID'] != 0]
data = data[data['ASK'] != 0]
data = data[data['NATBBO_IND'] == 1]

plot_series(data, 'XLE', 8, 30, 16, 30)

"""
plot_spread(data, 'IAU', 'SGOL', 0.1, 8, 30, 16, 30)
plot_spread(data, 'IAU', 'UGLD', 1, 8, 30, 16, 30)
plot_spread(data, 'IAU', 'DZZ', 1, 8, 30, 16, 30)
"""



