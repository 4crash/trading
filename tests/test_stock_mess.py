
from stockstats import StockDataFrame as sdf
import pandas as pd

import pytz
utc = pytz.UTC
# from ..buy_sell import BuySell
# from sklearn import linear_model, preprocessing
from pytz import timezone
localtz = timezone('Europe/Prague')
from ..stock_mess import StockMess
from ..market_db import Database, TableName
from ..plot_i import PlotI
from ..fin_i import FinI 
from ..check_indicators import CheckIndicators as chi
import asyncio
import io
import pytest

sm = StockMess()
sm.set_fundamentals("PLUG")
# st.write(index)
mess, curr_price, days_to_earnings = sm.get_subj_mess("Test", "PLUG")

def test_subject_mess():
    assert mess != "", "Subj fundamental message fail!"
   

def test_subject_earnings_day():
    assert days_to_earnings != "", "Days to earnings is none or doesnt exists"
    

def test_get_common_mess():
    db = Database()
    data = db.load_data(time_from="-10d")
    data = FinI.add_indicators(sdf.retype(data))
    sector_mess, spy_mess, vol_mess = sm.get_common_mess(data)
    assert sector_mess is not None and spy_mess is not None and vol_mess is not None, " get_common_mess failed"
    

def test_create_sector_trend_mess():
    mess = sm.create_sectors_trend_mess("industrials")
    assert mess is not None, mess + " create_sectors_trend_mess failed"

def test_get_fund_mess():
    # sm.set_prices()
    sm.set_fundamentals("INTC")
    mess = sm.get_fund_mess(sm.financials, 65.12, sm.earnings,
                     sm.sentiment, 2,  sm.stocks)
    assert mess is not None, " test_get_fund_mess failed"
