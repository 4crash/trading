
from ..singleton import Singleton
from ..market_db import Database, TableName
from ..sector_stats import SectorStats
from datetime import timedelta
import matplotlib.pyplot as plt
from stockstats import StockDataFrame as sdf
import pandas as pd

# from market_app.overview.refreshFinancials import refreshFinancials

import pytz
utc = pytz.UTC
# from ..buy_sell import BuySell
# from sklearn import linear_model, preprocessing
from pytz import timezone
localtz = timezone('Europe/Prague')


import pytest

ss = SectorStats()

def test_sectors_day_stats():
    out = ss.sectors_day_stats(table_name = TableName.DAY, time_from = "-180d", time_to = "0d")
    assert out is not None and len(out) > 1, str(out) + "Failed"
   

def test_set_spy_sectors_trend():
    out = ss.set_spy_sectors_trend()
    assert ss.spy_trend is not None and len(ss.spy_trend) > 0 and ss.sectors_trend is not None and len(
        ss.sectors_trend) > 1, "Failed"


def test_sectors_uptrend_by_month():
    out = ss.sectors_uptrend_by_month(yf=2017, yt=None, show_chart=False)
    assert out is not None and len(out) > 1, str(out) + "Failed"


def test_classify_sectors_uptrend():
    out = ss.classify_sectors_uptrend(
        table_name=TableName.DAY, time_from="-180d", time_to="0d")
    assert out is not None and len(out) > 1, str(out) + "Failed"


def test_get_trend_slice():
    out = ss.get_trend_slice(
        table_name=TableName.DAY, time_from="-180d", time_to="0d")
    assert out is not None and len(out) > 1, str(out) + "Failed"

    
