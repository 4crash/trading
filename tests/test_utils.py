
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


def test_add_first_last_perc_diff():
    
    data = db.load_data(time_from = "-10d")
    out = Utils.add_first_last_perc_diff(data)
    
    assert out is not None and "close" in out and out.iloc[0].flpd > 0 and out.iloc[0].flpd is not None, str(
        out) + "flpd hasnot been calculated"
   


    
