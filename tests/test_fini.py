
import numpy as np
from alpaca_examples.singleton import Singleton
from datetime import timedelta
import matplotlib.pyplot as plt
from stockstats import StockDataFrame as sdf
import pandas as pd
import sys
sys.path.append('../../')
# from market_app.overview.refreshFinancials import refreshFinancials

import pytz
utc = pytz.UTC
# from alpaca_examples.buy_sell import BuySell
# from sklearn import linear_model, preprocessing
from pytz import timezone
localtz = timezone('Europe/Prague')
from alpaca_examples.plot_i import PlotI
from alpaca_examples.fin_i import FinI 
from alpaca_examples.utils import Utils
from alpaca_examples.check_indicators import CheckIndicators as chi
from alpaca_examples.market_db import Database, TableName

import pytest

db = Database()
sym="INTC"

def test_get_nearest_values():
    # price_levels = np.array([10,8,5,3,-1,-5,-10,-15])
    df = db.load_data(table_name=TableName.DAY, symbols=[sym])
    df = FinI.add_levels(df)
    price = df.iloc[-1].close
    
    low, high = FinI.get_nearest_values(
        price_levels=df.price_level.dropna().tolist(), price=price)
    print("low" + str(low))
    print("high" + str(high))
    
    assert low is not None and low[0] <  price,"low  array failed"
    assert high is not None and high[0] > price, "high  array failed"


def test_get_green_red_in_row():
    # price_levels = np.array([10,8,5,3,-1,-5,-10,-15])
    df = db.load_data(table_name=TableName.DAY, time_from = "-200d",symbols=[sym])
    price = df.iloc[-1].close

    df = FinI.get_green_red_in_row(df)
        
    assert False, " Not finished test"


def test_get_green_red_sum_in_row():
    # price_levels = np.array([10,8,5,3,-1,-5,-10,-15])
    df = db.load_data(table_name=TableName.DAY,
                      time_from="-200d", symbols=[sym])
    price = df.iloc[-1].close

    df = FinI.get_green_red_sum_in_row(df)

    assert False, " Not finished test"

def test_get_up_down_sum_in_row():
    # price_levels = np.array([10,8,5,3,-1,-5,-10,-15])
    df = db.load_data(table_name=TableName.DAY,
                      time_from="-200d", symbols=[sym])
    price = df.iloc[-1].close

    df = FinI.get_up_down_sum_in_row(df)

    assert False, " Not finished test"
