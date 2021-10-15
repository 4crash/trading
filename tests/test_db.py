
import pytest
import pandas as pd
from pytz import HOUR
from stockstats import StockDataFrame as sdf
from datetime import datetime, timedelta
from ..market_db import Database, TableName
import warnings


db:Database = Database()
sym = "INTC"

def test_get_last_n_days():
    out = db.get_last_n_days(sym, n_days_back=1, table=TableName.MIN15)
    assert out is not None
    assert len(out) > 1, str(out) + "Failed"


def test_last_date():
    last_wednesday = datetime.today().weekday() + 5
    db.last_date = datetime.today().replace(hour=23, minute=59) - \
        timedelta(days=(last_wednesday))
    df = db.load_data(table_name=TableName.DAY, symbols=[sym])
    print(db)
    # print(df.index[-1])
    # print(df.index[-1].weekday())
    earnings, sentiment, financials = db.get_fundamentals([sym])
    print(earnings)
    print(sentiment)
    print(financials)

    assert df is not None, "load data with last date Failed"
    assert db.last_date.day == df.index[-1].day, "last date in data is different"
    
    assert earnings is not None, "Earnings failed"
    if len(earnings)>0:
        assert  db.last_date <= earnings.iloc[-1].startdatetime + timedelta(
        days=30), "last date in data is different"
    else:
        warnings.warn("earnings missing data ")
        
    assert financials is not None,"Financials failed"
    assert len(financials) > 0 and db.last_date >= financials.iloc[-1].date.replace(
        tzinfo=None), "last date in data is different"
    
    if sentiment is not None and len(sentiment) > 0:
        assert db.last_date >= sentiment.iloc[-1].check_day, "last date in data is different"
    else:
        warnings.warn("sentiment missing data ")

