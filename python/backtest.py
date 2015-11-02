from __future__ import division
import numpy as np
import pandas as pd
import os


next_order_id = 1


class Order(object):
    def __init__(self, sym, qty, type="market"):
        if type != 'market':
            raise NotImplementedError("Only market orders implemented")
        self.sym = sym
        self.qty = qty
        self.type = type
        global next_order_id
        self.order_id = next_order_id
        next_order_id += 1


def backtest(data, strategy):
    """
    description: backtest a trading strategy
    inputs:
        @data - market data as a Pandas DataFrame, columns - [TIME_M,SYM_ROOT,BID,BIDSIZ,ASK,ASKSIZ]
        @strategy - callable function taking one parameter 'data': dictionary of symbol(string) -> security_data(pd.Series)
                    function is called at each minimum time increment, and should return a list of Order objects
    output:
        pnl history: list of numbers, your pnl at each time increment
        order history: list of order objects, at each time increment
    """

    cash = [0.0]
    pnl_history = []
    order_history = []
    positions = {}
    security_data = {}

    def check_signals(quotes):
        for i in xrange(0, len(quotes)):
            security_data[quotes.iloc[i]['SYM_ROOT']] = quotes.iloc[i]
        orders = strategy(security_data)
        for order in orders:
            sdata = security_data[order.sym]
            positions[order.sym] = positions.get(order.sym, 0) + order.qty
            price = sdata['BID'] if order.qty > 0 else sdata['ASK']  # TODO - what if qty is larger than BID/ASK QTY?
            order_history.append((quotes.iloc[0]['TIME_M'], order.sym, order.qty, price))
            cash[0] += -1 * order.qty * price
        portfolio_value = np.sum(
            [positions[sym] * (security_data[sym]['BID'] if positions[sym] > 0 else security_data[sym]['ASK']) for sym in
             positions])
        pnl_history.append(cash[0] + portfolio_value)

    data.groupby('TIME_M').apply(check_signals)

    return pnl_history, order_history




