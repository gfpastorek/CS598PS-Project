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


def backtest(data, strategy, transaction_costs=0.005, delay_fill=False):
    """
    description: backtest a trading strategy
    inputs:
        @datautils - market datautils as a Pandas DataFrame, columns - [DATE_TIME, SYM, BID_PRICE, BID_SIZE, ASK_PRICE, ASK_SIZE]
        @strategy - callable function taking two parameters
                        'datautils': dictionary of symbol(string) -> security_data(pd.Series)
                        'positions': dictionary of symbol(string) -> position(int)
                    function is called at each minimum time increment, and should return a list of Order objects
        @transaction_costs - transaction cost per contract per trade
    output:
        pnl history: list of numbers, your pnl at each time increment
        order history: list of order objects, at each time increment
    """

    cash = [0.0]
    pnl_history = [0]
    order_history = []
    positions = {}
    security_data = {}

    slippage = lambda price, qty, _book_qty: price + 0.01/4.0 * (qty/_book_qty + 1)

    def check_signals(quotes):
        # run strategy
        if len(security_data) > 0 and delay_fill:
            orders = strategy(security_data, positions)
        for i in xrange(0, len(quotes)):
            security_data[quotes.iloc[i]['SYM']] = quotes.iloc[i]
        if not delay_fill:
            orders = strategy(security_data, positions)
        # process orders
        for order in orders:
            sdata = security_data[order.sym]
            positions[order.sym] = positions.get(order.sym, 0) + order.qty
            price = sdata['BID_PRICE'] if order.qty > 0 else sdata['ASK_PRICE']
            book_qty = sdata['BID_SIZE'] if order.qty > 0 else sdata['ASK_SIZE']
            price = slippage(price, order.qty, book_qty)
            order_history.append((quotes.iloc[0]['DATE_TIME'], order.sym, order.qty, price))
            cash[0] -= order.qty * price
            cash[0] -= transaction_costs * abs(order.qty)
        portfolio_value = np.sum(
            [positions[sym] * (security_data[sym]['BID_PRICE'] if positions[sym] > 0 else security_data[sym]['ASK_PRICE']) for sym in
             positions])
        pnl_history.append(cash[0] + portfolio_value)

    data.groupby('DATE_TIME').apply(check_signals)

    return pnl_history[1:], order_history
