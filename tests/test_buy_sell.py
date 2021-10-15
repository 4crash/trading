import sys
sys.path.append('../../')
# from singleton import Singleton
# from datetime import timedelta
# import matplotlib.pyplot as plt
# from stockstats import StockDataFrame as sdf
import pandas as pd

# from market_app.overview.refreshFinancials import refreshFinancials
import pytz
utc = pytz.UTC
# from buy_sell import BuySell
# from sklearn import linear_model, preprocessing
from pytz import timezone
localtz = timezone('Europe/Prague')
# from plot_i import PlotI
# from fin_i import FinI 
# from utils import Utils
# from check_indicators import CheckIndicators as chi
from alpaca_examples.market_db import Database, TableName
from buy_sell import BuySell
import logging
# import pytest

db = Database()
bs = BuySell()
data_intc = db.load_data(table_name=TableName.DAY,
                    time_from="-10d", symbols=["INTC"])
data_amd = db.load_data(table_name=TableName.DAY,
                         time_from="-10d", symbols=["AMD"])



def test_buy_sell_stock_t():
    bs.buy_sell_open = pd.DataFrame()
    bs.money = 10000

    bs.buy_stock_t(data_intc.iloc[0], data_intc.iloc[0].close, qty=1, profit_loss={
                   "profit": data_intc.iloc[0].close+1, "loss": data_intc.iloc[0].close-1})

    bs.buy_stock_t(data_amd.iloc[0], data_amd.iloc[0].close, qty=1, profit_loss={
        "profit": data_amd.iloc[0].close+1, "loss": data_amd.iloc[0].close-1})
    
    logging.warning(bs.buy_sell_open)
    assert bs.buy_sell_open is not None , str(bs.buy_sell_open) + " -- fill buy sell failed"
    assert len(bs.buy_sell_open) > 1, str(
        bs.buy_sell_open) + " -- fill buy sell failed"
    

    b_stock = bs.sell_stock_t(sym="INTC", price=(data_intc.iloc[-1].close), qty=1, sell_date =data_intc.index[0])
    

    assert b_stock is not None
    assert b_stock.sell is not None
    assert b_stock.sell > 0
    assert len(bs.buy_sell_open) == 0, str(b_stock) + \
        " closed trade wasnt removed from array"
    assert len(bs.buy_sell_closed) == 1, str(b_stock) + " closed trade wasnt add to array"
    

def test_buy_sell_stats():
    
    results = bs.trading_results()
    logging.warning(bs.buy_sell_closed)
    logging.warning(results)

    assert results is not None , " -- stats failed"
    assert len(results) > 0, " -- s_profit array doesnt exists"
