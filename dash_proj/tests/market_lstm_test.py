from alpaca_examples.market_db import Database, TableName
from datetime import timedelta
from stockstats import StockDataFrame as sdf
import pandas as pd
import pytest
from dash_proj.market_lstm import MarketLSTM
import sys
sys.path.append('../../')


db = Database()


def test_market_lstm():
    
    m_df = db.load_data(
        table_name=TableName.DAY,  time_from="-180d", symbols=["TSLA"])

    m_df_spy = db.load_data(
        table_name=TableName.DAY,  time_from="-180d", symbols=["SPY"])

    m_df_spy["oc_mean"] = ((m_df_spy.close + m_df_spy.open)/2)
    m_df = sdf.retype(m_df)
    m_df.get("boll")
    # m_df = FinI.add_sma(9, m_df)
    # m_df = FinI.add_weekday(m_df)
    # m_df = FinI.add_week_of_month(m_df)
    # m_df = FinI.add_levels(m_df)
    
    lstm = MarketLSTM(m_df[["close"]])
    assert lstm is not None, str("LSTM") + "Failed"
