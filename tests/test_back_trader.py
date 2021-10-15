from datetime import datetime
from alpaca_examples.stock_whisperer import StockWhisperer
from ..back_trader import BackTrader
from ..singleton import Singleton
from datetime import timedelta
import matplotlib.pyplot as plt
from stockstats import StockDataFrame as sdf
import pandas as pd
from ..stock_whisperer import StockWhisperer

# from market_app.overview.refreshFinancials import refreshFinancials
import pytz
utc = pytz.UTC
# from buy_sell import BuySell
# from sklearn import linear_model, preprocessing
from pytz import timezone
localtz = timezone('Europe/Prague')
from ..plot_i import PlotI
from ..fin_i import FinI
from ..utils import Utils
from ..check_indicators import CheckIndicators as chi
from ..market_db import Database, TableName
from ..buy_sell import BuySell
import pytest
import warnings


def test_find_stocks_to_buy():
    sw = StockWhisperer()
    output = sw.find_stocks(TableName.DAY, False)
    assert output is not None, "find stocks failed"
    assert len(output) > 1, "no stocks to buy or something go wrong"




# @pytest.mark.skip
def test_back_trader():
    btr = BackTrader()
    btr.symbols = ["INTC"]
    # bt.stock_stats.KDJ_WINDOW = 14
    # if not btr.db.time_from:
    #     btr.db.set_time_from("-365d")

    btr.trading_alg(table_name=TableName.DAY, buy_now=True, strategy_name="sma9_keep")
    res = btr.bs.trading_results()
    print("Prof. price" + str(res.s_profit.sum()))
    print("Prof. perc." + str(res.s_profit_p.mean()))
    assert res is not None, "Trading results there is nothing"
    assert len(res)>0, str(res)

def test_profit_loss():
    back_days = 30
    profit_level = 1
    loss_level = 2
    
    sw = StockWhisperer()
    stats = sw.profit_loss(back_days=back_days,
                           profit_level=profit_level, loss_level=loss_level)


    print(stats)
    if len(stats)> 0:
        print("profit l. set." + str(profit_level))
        print("loss l. set." + str(loss_level))
        # print("stocks: " + str(len(buyed_stocks.groupby(by="sym"))))
        print("trades: " + str(len(stats)))
        print("sum price: " + str(stats.buy_price.sum()))
        print("profit: " + str(stats.price_diff.sum()))
        print("mean days¨: " + str(stats.days.mean()))
    
        


    # earnings, sentiment, financials = db.get_fundamentals([sym])
    warnings.warn("Finish this code")
    assert True, "Finish code in this test"


def test_bt_top_stocks():
    back_days = 50
    profit_level = 2
    loss_level = 1

    sw = StockWhisperer()
    stats = sw.bt_top_stocks(back_days=back_days,
                           profit_level=profit_level, loss_level=loss_level)

    print(sw.bs.buy_sell_closed)
    
    assert sw.bs.buy_sell_closed is not None and len(
        sw.bs.buy_sell_closed) > 0, "No trades has been executed... sssstrange"

