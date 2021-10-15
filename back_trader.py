from typing import List, Literal
from alpaca_examples.indicators import Indicators
import logging
import sys
sys.path.append('../')
from _datetime import timedelta
from alpaca_examples.back_tester import TableName
from alpaca_examples.fin_i import FinI
from alpaca_examples.utils import Utils
import pandas as pd
from stockstats import StockDataFrame as sdf
from alpaca_examples.market_db import Database
from alpaca_examples.buy_sell import BuySell
import numpy
from datetime import datetime
from alpaca_examples.check_indicators import CheckIndicators
class BackTrader(object):
    """
    docstring
    """
    def __init__(self):
        """
        docstring
        """
        # start tradining stats
        # self.stock_stats = sdf()
       
        # self.bs.money = 0
        # self.bs.profit_perc = 0
        # self.bs.profit_perc_all = 0
        # self.bs.transactions = 0
        # self.buy_sell_hist = sdf()
        # self.bs.shares_amount = 1
        # self.bs.buyed_stocks = 0
        self.bt_stocks = pd.DataFrame()
        # self.buy_marks = pd.DataFrame()
        # self.sell_marks = pd.DataFrame()
        self.buy_sell = sdf()
        self.rsi_params = { "buyBand": { "optimal":0 ,"max":90, "min":60, "step":10}, 
                            "sellBand": {"optimal":0, "max":60, "min":10, "step":10 }, 
                            "buySignalGap":{"optimal":0 ,"max":1, "min":0, "step":0.2},
                            "sellSignalGap":{"optimal":0 ,"max":1, "min":0, "step":0.2} 
                        }
        self.bollinger_params = { 
                            "buySignalGap":{"optimal":0 ,"max":1, "min":0, "step":0.2, "mid":0},
                            "sellSignalGap":{"optimal":0 ,"max":1, "min":0, "step":0.2, "mid":0} 
                        }
        self.best_settings = {}
        self.best_at_settings = {}
        self.best_rsi_params = 0
        self.db = Database()
        self.symbols = None
        self.bs = BuySell()
        
        
    def load_data(self, table_name = None, symbols = None, sectors = None, limit = None, time_from = None, time_to=None):
        symbols = symbols if symbols is not None else self.symbols
        return sdf.retype(self.db.load_data(table_name, symbols=symbols, sectors=sectors, 
                                            limit=limit, time_from=time_from, time_to=time_to))
        
    def trading_alg(self,table_name = None, buy_now = False, strategy_name = "sma9"):
        """back test with specific strategy

        Args:
            table_name ([type], optional): [description]. Defaults to None.
            buy_now (bool, optional): [description]. Defaults to False.
            strategy_name (str, optional): [description]. Defaults to "sma9".
        """
                
        self.bs.buyed_stocks = 0
        self.bs.money = self.bs.startCredit
        spy_stocks = self.load_data(table_name = table_name, symbols = ["SPY"])
        spy_stocks = FinI.add_indicators(spy_stocks)
        
        if self.symbols:
            symbols = self.symbols
        else:
            symbols = self.db.get_symbols()

        # symbols = ["INTC","BYND","ZM","NKE","HIMX","JKS","ENPH","DUK","GE","DIS","LEVI","NVAX","SLCA","GPS"]
        
        for symbol in symbols:
            print("symbol: " + str(symbol))
            
            sub_data = self.load_data(table_name = table_name, symbols = symbol)
            if len(sub_data) < 1:
                break

            self.bt_stocks = FinI.add_indicators(sub_data)
            self.bt_stocks = FinI.add_fib(self.bt_stocks)
            # print(self.bt_stocks)
            print(self.bt_stocks["sma30"])
            print("calculating percent change:" + str(symbol))
            # sub_data = self.stocks.loc[self.stocks.sym ==symbol[0]].sort_values(by='index')
            
            self.symbols = symbol[0]
           
            # self.prev_stock = sub_data.iloc[0]
            # self.bt_stocks.iloc[0] = sub_data.iloc[0]

            # self.sell_marks = self.sell_marks.iloc[0:0]
            # self.buy_marks = self.buy_marks.iloc[0:0]
            self.bs.transactions = 0
            self.bs.profit_perc = 0
           
            # trend_indicator = 
            # TODO mechanism for select strategies
            # self.ts_boll(buy_now = buy_now, at_settings = None, symbol = symbol, spy_stocks = spy_stocks)
            self.ts_eval(buy_now = buy_now, at_settings = None, symbol = symbol, spy_stocks = spy_stocks, strategy_logic = strategy_name)

            # call the method with passed and assembled name
            # method = getattr(self, 'ts_' + strategy_name)
            # method(buy_now = buy_now, at_settings = None, symbol = symbol, spy_stocks = spy_stocks, strategy_name = strategy_name)

    def pars_input_for_eval(self, input_params:str)->List[str]:
        """ pars input params to check logic code

        Args:
            input_params (str): example -  [sma9-and-macd-or-boll,sma9-boll?2-candles?4-boll?_up"]

        Returns:
            List[str]: buy and sell eval string with check funciotns calling
        """
       # buy sell composed eval code
        output: List[str] = ["", ""]
        bs_params: List[str] = ["", ""]

        # split buy sell logic 
        buy_sell_params = input_params.split(",")

        # set code parts
        method_preposition = "CheckIndicators.check_"
        
        bs_params[0] = "(stocks=self.bt_stocks.iloc[0:val],buy=True{params})"
        bs_params[1] = bs_params[0].replace("True", "False")

        # split commands like OR AND and checker names
        valid_eval_buy:List[str] = buy_sell_params[0].split("-")
        valid_eval_sell: List[str] = buy_sell_params[1].split("-") if len(buy_sell_params)>1 else []

        # spliter for each function parameters
        param_spliter = "?"
        # buy and sell aplittedd args  iteration
        for i in range(len(buy_sell_params)):
            valid_eval_bs:List[str] = buy_sell_params[i].split("-")
            # iterate over buy and sell variants
            for item in valid_eval_bs:
                if item.lower() == "or" or item.lower() == "and":
                    output[i] = f"{output[i]} {item} "
                else:
                    fce_fields = item.split(param_spliter)
                    # check if input params are numeric or not
                    if len(fce_fields) > 1:
                        fce_fields[1] = fce_fields[1] if  fce_fields[1].isnumeric() else f"'{fce_fields[1]}'"
                    params = f", params={fce_fields[1]}" if len(fce_fields)>1 else ""
                    bs_params[i] = bs_params[i].format(params=params)
                    # check if its number or string
                    output[i] = f"{output[i]}{method_preposition}{fce_fields[0]}{bs_params[i]}"
        
        # print(output)          
        if len(buy_sell_params) < 2:
            output[1] = output[0].replace("True", "False")

        return output
        
    def ts_eval(self, buy_now=False, at_settings=None, symbol=None, spy_stocks=None, strategy_logic:str=""):

        buy_now_process = buy_now
        buy_indic = 0
        macd_mem = {"up": 100, "down": 100, "range": None}
        self.bs.profit_perc = 0
        eval_cmd = self.pars_input_for_eval(strategy_logic)
        print(eval_cmd)
        

        for val in range(1, len(self.bt_stocks)):
            #conditions for buy sell are composed to string from input params and than eval
            if self.bs.buyed_stocks == 0 and  \
               eval(eval_cmd[0]):

                self.bs.buy_stock_t(
                    self.bt_stocks.iloc[val], self.bt_stocks.iloc[val].close)
                buy_now_process = False 

            #comment this block for selling at the end of the date
            elif self.bs.buyed_stocks != 0 and \
                  eval(eval_cmd[1]):
                        
                # self.bs.buyed_stocks = 0
                #  (self.bt_stocks.iloc[val]['boll_mid_lb'] <= self.bt_stocks.iloc[val-1]['close'] and self.bt_stocks.iloc[val]['boll_mid_lb'] > self.bt_stocks.iloc[val]['close'])):
                # self.bs.sell_stock(
                #     self.bt_stocks.iloc[val], self.bt_stocks.iloc[val].sma9)
                self.bs.sell_stock_t(
                    sym=self.bt_stocks.iloc[val].sym,
                    price=self.bt_stocks.iloc[val].close,
                    sell_date=self.bt_stocks.iloc[val].date)

            # ----------------------------------------------------------------------------------------------------------

        if self.bs.transactions > 0:
            print("--------------TRADING RESULTSSS------------------------")
            self.bs.trading_results()
            # self.show_settings_stats()
            self.bs.show_sym_bs_stats(symbol)
            # self.plot_stats(sub_data, spy_stocks)

        else:
            print("Theres no transactions please change BUY/SELL params")

        
    def ts_hammer(self, buy_now=False, at_settings=None, symbol=None, spy_stocks=None, strategy_name:str = None):

        buy_now_process = buy_now
        buy_indic = 0
        macd_mem = {"up": 100, "down": 100, "range": None}
        self.bs.profit_perc = 0
        method = getattr(CheckIndicators, 'check_' + strategy_name)

        for val in range(1, len(self.bt_stocks)):

            if self.bs.buyed_stocks == 0 and  \
               method(stocks=self.bt_stocks.iloc[0:val], buy=True):

                self.bs.buy_stock_t(
                    self.bt_stocks.iloc[val], self.bt_stocks.iloc[val].sma9)
                buy_now_process = False

            #comment this block for selling at the end of the date
            elif self.bs.buyed_stocks != 0 and \
                  method(stocks=self.bt_stocks.iloc[0:val], buy=False):
                        
                # self.bs.buyed_stocks = 0
                #  (self.bt_stocks.iloc[val]['boll_mid_lb'] <= self.bt_stocks.iloc[val-1]['close'] and self.bt_stocks.iloc[val]['boll_mid_lb'] > self.bt_stocks.iloc[val]['close'])):
                # self.bs.sell_stock(
                #     self.bt_stocks.iloc[val], self.bt_stocks.iloc[val].sma9)
                self.bs.sell_stock_t(
                    sym=self.bt_stocks.iloc[val].sym,
                    price=self.bt_stocks.iloc[val].close,
                    sell_date=self.bt_stocks.iloc[val].date)

            # ----------------------------------------------------------------------------------------------------------

        if self.bs.transactions > 0:
            print("--------------TRADING RESULTSSS------------------------")
            self.bs.trading_results()
            # self.show_settings_stats()
            self.bs.show_sym_bs_stats(symbol)
            # self.plot_stats(sub_data, spy_stocks)

        else:
            print("Theres no transactions please change BUY/SELL params")

    def ts_macd_rsi_i(self, buy_now=False, at_settings=None, symbol=None, spy_stocks=None):

        buy_now_process = buy_now
        buy_indic = 0
        macd_mem = {"up": 100, "down": 100, "range": None}
        self.bs.profit_perc = 0
        for val in range(1, len(self.bt_stocks)):

            if self.bs.buyed_stocks == 0 and  \
                CheckIndicators.check_macd(self.bt_stocks.iloc[0:val]) and \
                CheckIndicators.check_rsi(self.bt_stocks.iloc[0:val],buy=True):

                self.bs.buy_stock_t(
                    self.bt_stocks.iloc[val], self.bt_stocks.iloc[val].sma9)
                buy_now_process = False

            #comment this block for selling at the end of the date
            elif self.bs.buyed_stocks != 0 and \
                CheckIndicators.check_macd(self.bt_stocks.iloc[0:val],buy=False) and \
                CheckIndicators.check_rsi(self.bt_stocks.iloc[0:val],buy=False):
                        
                # self.bs.buyed_stocks = 0
                #  (self.bt_stocks.iloc[val]['boll_mid_lb'] <= self.bt_stocks.iloc[val-1]['close'] and self.bt_stocks.iloc[val]['boll_mid_lb'] > self.bt_stocks.iloc[val]['close'])):
                # self.bs.sell_stock(
                #     self.bt_stocks.iloc[val], self.bt_stocks.iloc[val].sma9)
                self.bs.sell_stock_t(
                    sym=self.bt_stocks.iloc[val].sym,
                    price=self.bt_stocks.iloc[val].close,
                    sell_date=self.bt_stocks.iloc[val].date)

            # ----------------------------------------------------------------------------------------------------------

        if self.bs.transactions > 0:
            print("--------------TRADING RESULTSSS------------------------")
            self.bs.trading_results()
            # self.show_settings_stats()
            self.bs.show_sym_bs_stats(symbol)
            # self.plot_stats(sub_data, spy_stocks)

        else:
            print("Theres no transactions please change BUY/SELL params")
    def ts_sma9_i(self, buy_now=False, at_settings=None, symbol=None, spy_stocks=None):

        buy_now_process = buy_now
        buy_indic = 0
        macd_mem = {"up": 100, "down": 100, "range": None}
        self.bs.profit_perc = 0
        for val in range(1, len(self.bt_stocks)):

            if self.bs.buyed_stocks == 0 and  \
                   CheckIndicators.check_sma9(self.bt_stocks.iloc[0:val]):

                self.bs.buy_stock_t(
                    self.bt_stocks.iloc[val], self.bt_stocks.iloc[val].sma9)
                buy_now_process = False

            #comment this block for selling at the end of the date
            elif self.bs.buyed_stocks != 0 and \
                    CheckIndicators.check_sma9(self.bt_stocks.iloc[0:val],buy=False):
                # self.bs.buyed_stocks = 0
                #  (self.bt_stocks.iloc[val]['boll_mid_lb'] <= self.bt_stocks.iloc[val-1]['close'] and self.bt_stocks.iloc[val]['boll_mid_lb'] > self.bt_stocks.iloc[val]['close'])):
                # self.bs.sell_stock(
                #     self.bt_stocks.iloc[val], self.bt_stocks.iloc[val].sma9)
                self.bs.sell_stock_t(
                    sym=self.bt_stocks.iloc[val].sym,
                    price=self.bt_stocks.iloc[val].close,
                    sell_date=self.bt_stocks.iloc[val].date)

            # ----------------------------------------------------------------------------------------------------------

        if self.bs.transactions > 0:
            print("--------------TRADING RESULTSSS------------------------")
            self.bs.trading_results()
            # self.show_settings_stats()
            self.bs.show_sym_bs_stats(symbol)
            # self.plot_stats(sub_data, spy_stocks)

        else:
            print("Theres no transactions please change BUY/SELL params")
            
    def ts_sma9_keep(self, buy_now=False, at_settings=None, symbol=None, spy_stocks=None):

        buy_now_process = buy_now
        buy_indic = 0
        macd_mem = {"up": 100, "down": 100, "range": None}
        self.bs.profit_perc = 0
        for val in range(1, len(self.bt_stocks)):

            if self.bs.buyed_stocks == 0 and  \
                    self.bt_stocks.iloc[val-1].sma9 > self.bt_stocks.iloc[val-1].close and \
                    self.bt_stocks.iloc[val].sma9 < self.bt_stocks.iloc[val].close:

                self.bs.buy_stock_t(
                    self.bt_stocks.iloc[val], self.bt_stocks.iloc[val].sma9)
                buy_now_process = False

            #comment this block for selling at the end of the date
            elif self.bs.buyed_stocks != 0 and \
                    self.bt_stocks.iloc[val-1].sma9 < self.bt_stocks.iloc[val-1].close and \
                    self.bt_stocks.iloc[val].sma9 > self.bt_stocks.iloc[val].close:
                # self.bs.buyed_stocks = 0
                #  (self.bt_stocks.iloc[val]['boll_mid_lb'] <= self.bt_stocks.iloc[val-1]['close'] and self.bt_stocks.iloc[val]['boll_mid_lb'] > self.bt_stocks.iloc[val]['close'])):
                # self.bs.sell_stock(
                #     self.bt_stocks.iloc[val], self.bt_stocks.iloc[val].sma9)
                self.bs.sell_stock_t(
                    sym=self.bt_stocks.iloc[val].sym,
                    price=self.bt_stocks.iloc[val].close,
                    sell_date=self.bt_stocks.iloc[val].date)

            # ----------------------------------------------------------------------------------------------------------

        if self.bs.transactions > 0:
            print("--------------TRADING RESULTSSS------------------------")
            self.bs.trading_results()
            # self.show_settings_stats()
            self.bs.show_sym_bs_stats(symbol)
            # self.plot_stats(sub_data, spy_stocks)

        else:
            print("Theres no transactions please change BUY/SELL params")

    def ts_macd_boll(self, buy_now=False, at_settings=None, symbol=None, spy_stocks=None):

        buy_now_process = buy_now
        buy_indic = 0
        macd_mem = {"up": 100, "down": 100, "range": None}
        self.bs.profit_perc = 0
        for val in range(1, len(self.bt_stocks)):

            # or \
            #     (self.bt_stocks.iloc[val-1].boll_ub_macd > self.bt_stocks.iloc[val-1].macd and \
            #     self.bt_stocks.iloc[val].boll_ub_macd < self.bt_stocks.iloc[val].macd):

            if self.bs.buyed_stocks == 0 and  \
                (self.bt_stocks.iloc[val-1].boll_lb_macd > self.bt_stocks.iloc[val-1].macd and
                 self.bt_stocks.iloc[val].boll_lb_macd < self.bt_stocks.iloc[val].macd):

                self.bs.buy_stock_t(
                    self.bt_stocks.iloc[val], self.bt_stocks.iloc[val].sma9)
                buy_now_process = False

            #comment this block for selling at the end of the date
            elif self.bs.buyed_stocks != 0 and \
                    self.bt_stocks.iloc[val-1].boll_ub_macd < self.bt_stocks.iloc[val-1].macd and \
                    self.bt_stocks.iloc[val].boll_ub_macd > self.bt_stocks.iloc[val].macd:

                # self.bs.buyed_stocks = 0
                #  (self.bt_stocks.iloc[val]['boll_mid_lb'] <= self.bt_stocks.iloc[val-1]['close'] and self.bt_stocks.iloc[val]['boll_mid_lb'] > self.bt_stocks.iloc[val]['close'])):
                self.bs.sell_stock_t(
                    sym=self.bt_stocks.iloc[val].sym,
                    price=self.bt_stocks.iloc[val].close,
                    sell_date=self.bt_stocks.iloc[val].date)

            # ----------------------------------------------------------------------------------------------------------

        if self.bs.transactions > 0:

            self.bs.trading_results()
            self.bs.show_sym_bs_stats(symbol)
            # # self.show_settings_stats()
            
            # self.plot_stats(sub_data, spy_stocks)

        else:
            print("Theres no transactions please change BUY/SELL params")

    def ts_sma9(self, buy_now=False, at_settings=None, symbol=None, spy_stocks=None):

        buy_now_process = buy_now
        buy_indic = 0
        macd_mem = {"up": 100, "down": 100, "range": None}
        self.bs.profit_perc = 0
        for val in range(1, len(self.bt_stocks)):

            if self.bs.buyed_stocks == 0 and  \
                    self.bt_stocks.iloc[val].sma9 < self.bt_stocks.iloc[val].open:

                self.bs.buy_stock_t(
                    self.bt_stocks.iloc[val], self.bt_stocks.iloc[val].open)
                buy_now_process = False

                print("buying")

            #comment this block for selling at the end of the date
            elif self.bs.buyed_stocks != 0 and \
                (spy_stocks.iloc[val-1].open < spy_stocks.iloc[val].close or
                 spy_stocks.iloc[val].close < spy_stocks.iloc[val].sma9):

                # self.bs.buyed_stocks = 0
                #  (self.bt_stocks.iloc[val]['boll_mid_lb'] <= self.bt_stocks.iloc[val-1]['close'] and self.bt_stocks.iloc[val]['boll_mid_lb'] > self.bt_stocks.iloc[val]['close'])):
                # self.bs.sell_stock_t(
                #     self.bt_stocks.iloc[val], self.bt_stocks.iloc[val].close)
                self.bs.sell_stock_t(
                   sym =  self.bt_stocks.iloc[val].sym, 
                   price = self.bt_stocks.iloc[val].close, 
                   sell_date=self.bt_stocks.iloc[val].date)
                print("selling")

            # ----------------------------------------------------------------------------------------------------------

        if self.bs.transactions > 0:
            print("--------------TRADING RESULTSSS------------------------")
            self.bs.trading_results()
            # self.show_settings_stats()
            self.bs.show_sym_bs_stats(symbol)
            # self.plot_stats(sub_data, spy_stocks)

        else:
            print("Theres no transactions please change BUY/SELL params")

    def ts_sma9_fake(self, buy_now=False, at_settings=None, symbol=None, spy_stocks=None):

        buy_now_process = buy_now
        buy_indic = 0
        macd_mem = {"up": 100, "down": 100, "range": None}
        self.bs.profit_perc = 0
        for val in range(1, len(self.bt_stocks)):

            if self.bs.buyed_stocks == 0 and  \
                    self.bt_stocks.iloc[val].sma9 < self.bt_stocks.iloc[val].close:
                self.bs.buy_stock(
                    self.bt_stocks.iloc[val], self.bt_stocks.iloc[val].sma9)
                buy_now_process = False

            #comment this block for selling at the end of the date
            if self.bs.buyed_stocks != 0 and \
                (spy_stocks.iloc[val-1].open < spy_stocks.iloc[val].close or
                 spy_stocks.iloc[val].close < spy_stocks.iloc[val].sma9):
                # self.bs.buyed_stocks = 0
                #  (self.bt_stocks.iloc[val]['boll_mid_lb'] <= self.bt_stocks.iloc[val-1]['close'] and self.bt_stocks.iloc[val]['boll_mid_lb'] > self.bt_stocks.iloc[val]['close'])):
                self.bs.sell_stock(
                    self.bt_stocks.iloc[val], self.bt_stocks.iloc[val].sma9)

            # ----------------------------------------------------------------------------------------------------------

        if self.bs.transactions > 0:

            self.bs.trading_stats(symbol, self.bt_stocks)
            # self.plot_stats(sub_data, spy_stocks)

        else:
            print("Theres no transactions please change BUY/SELL params")

    def get_day_stocks(self, tf, sym):
        # store origin dates
        time_hist = {"from": self.db.time_from, "to": self.db.time_to}

        tf = tf.replace(minute=0, second=0, hour=0)
        tt = tf.replace(minute=0, second=0, hour=0)
        tt = tt + timedelta(days=1)
        self.db.time_from = tf
        self.db.time_to = tt
        out = self.load_data(TableName.MIN15, sym)

        # set back origin dates
        self.db.time_from = time_hist["from"]
        self.db.time_to = time_hist["to"]
        return out

    def ts_macd(self, buy_now=False, at_settings=None, symbol=None, spy_stocks=None):

        buy_now_process = buy_now
        buy_indic = 0
        macd_mem = {"up": 100, "down": 100, "range": None}
        self.bs.profit_perc = 0
        spy = self.load_data(TableName.DAY,symbols=["SPY"])
        spy = FinI.add_indicators(spy)

        for val in range(1, len(self.bt_stocks)):
            if self.bt_stocks.iloc[val].name in spy:
                pass

            if self.bs.buyed_stocks == 0 and  \
                    self.bt_stocks.iloc[val-1].macd - self.bt_stocks.iloc[val].macd <= 0 and \
                    self.bt_stocks.iloc[val].macd < self.bt_stocks.iloc[val].macdh:
                # self.bt_stocks.iloc[val-1].close >= self.bt_stocks.iloc[val].open:

                day_stocks = self.get_day_stocks(
                    self.bt_stocks.iloc[val].name, self.bt_stocks.iloc[val].sym)
                print("DAY STOCKS" + str(day_stocks))
                print("SPY Val" +
                      str(spy.loc[self.bt_stocks.iloc[val].name].close))
                self.bs.buy_stock(
                    self.bt_stocks.iloc[val], buy_price=self.bt_stocks.iloc[val].open)

            elif self.bs.buyed_stocks != 0 and   \
                    self.bt_stocks.iloc[val-1].macd - self.bt_stocks.iloc[val].macd >= 0 and \
                    self.bt_stocks.iloc[val].macd > self.bt_stocks.iloc[val].macdh and \
                    self.bt_stocks.iloc[val].sma9 > self.bt_stocks.iloc[val].open:
                # self.bt_stocks.iloc[val-1].close <= self.bt_stocks.iloc[val].open:
                day_stocks = self.get_day_stocks(
                    self.bt_stocks.iloc[val].name, self.bt_stocks.iloc[val].sym)
                self.bs.sell_stock(
                    self.bt_stocks.iloc[val], sell_price=self.bt_stocks.iloc[val].open)

        if self.bs.transactions > 0:
            self.bs.trading_stats(symbol, self.bt_stocks)
            # self.plot_stats(sub_data, spy_stocks)

        else:
            print("Theres no transactions please change BUY/SELL params")

    def ts_macd_fake(self, buy_now=False, at_settings=None, symbol=None, spy_stocks=None):

        buy_now_process = buy_now
        buy_indic = 0
        macd_mem = {"up": 100, "down": 100, "range": None}
        self.bs.profit_perc = 0

        for val in range(1, len(self.bt_stocks)):
            if self.bs.buyed_stocks == 0 and  \
                    self.bt_stocks.iloc[val-1].macd - self.bt_stocks.iloc[val].macd <= 0:
                # self.bt_stocks.iloc[val-1].close >= self.bt_stocks.iloc[val].open:

                self.bs.buy_stock(
                    self.bt_stocks.iloc[val], buy_price=self.bt_stocks.iloc[val].open)

            elif self.bs.buyed_stocks != 0 and   \
                    self.bt_stocks.iloc[val-1].macd - self.bt_stocks.iloc[val].macd >= 0:
                # self.bt_stocks.iloc[val-1].close <= self.bt_stocks.iloc[val].open:

                self.bs.sell_stock(
                    self.bt_stocks.iloc[val], sell_price=self.bt_stocks.iloc[val].open)

        if self.bs.transactions > 0:
            self.bs.trading_stats(symbol, self.bt_stocks)
            # self.plot_stats(sub_data, spy_stocks)

        else:
            print("Theres no transactions please change BUY/SELL params")

    def ts_boll2(self, buy_now=False, at_settings=None, symbol=None, spy_stocks=None):

        buy_now_process = buy_now
        buy_indic = 0
        macd_mem = {"up": 100, "down": 100, "range": None}
        self.bs.profit_perc = 0

        for val in range(1, len(self.bt_stocks)):

            # --------------------------------detect peaks high or low sofifticated 50% -------------------*----------------

            macd_mem["range"] = self.bt_stocks.iloc[1:val].macd.max(
            ) - self.bt_stocks.iloc[1:val].macd.min()
            # ---------------------------------------- change indicators by trend ----
            if at_settings is None:
                at_settings = {"macd_b": 0, "macd_s": 0,
                               "boll_b": "boll_6", "boll_s": "boll_10",
                               "profit": -10000,
                                            "sym": symbol,
                                            "transactions": 0,
                                            "date": datetime.now(),
                                            "date_from": self.db.time_from,
                                            "price_change": 0}

                # macd_mem["range"] = macd_mem["up"] -  macd_mem["down"]
                # [0.4,0.65] []
                boll_range = ["boll", "boll_2", "boll_3",
                              "boll_5", "boll_6", "boll_10"]
                macd_range = [0.2, 0, 1]
                # trend = round(
                #     self.bt_stocks.iloc[val]["boll"] - self.bt_stocks.iloc[val]["sma50"], 2)

                trend = round(
                    self.bt_stocks.iloc[val]["close"] - self.bt_stocks.iloc[val]["boll"], 2)

                # if trend > 0:
                #     at_settings["macd_b"] = 0.7
                #     at_settings["macd_s"] = 0.7
                #     at_settings["boll_b"] = "boll_5"
                #     at_settings["boll_s"] = "boll_5"
                # elif trend < 0:
                #     at_settings["macd_b"] = 0.7
                #     at_settings["macd_s"] = 0.7
                #     at_settings["boll_b"] = "boll_5"
                #     at_settings["boll_s"] = "boll_5"

                # ------------------------------------------------------------------------
            # print(self.bt_stocks.iloc[1:val].macd.max())+
            # print("macd_b" + str(at_settings))
            if self.bs.buyed_stocks == 0 and  \
                (self.bt_stocks.iloc[1:val].macd.max() - (macd_mem["range"]*at_settings["macd_b"])) > self.bt_stocks.iloc[val].macd and \
                    self.bt_stocks.iloc[val-2].macd - self.bt_stocks.iloc[val].macd <= 0:

                macd_mem["down"] = self.bt_stocks.iloc[val].macd

            if self.bs.buyed_stocks != 0 and   \
                    macd_mem["range"]*at_settings["macd_s"] + self.bt_stocks.iloc[1:val].macd.min() < self.bt_stocks.iloc[val].macd and \
                    self.bt_stocks.iloc[val-1].macd - self.bt_stocks.iloc[val].macd >= 0:

                # print("sell macd limit: " + str((macd_mem["range"]) +
                #             self.bt_stocks.iloc[1:val].macd.min()))
                # print("macd: " + str(self.bt_stocks.iloc[val].macd))

                macd_mem["up"] = self.bt_stocks.iloc[val].macd
                # Utils.countdown(2)

            # SELL or BUY rules:
            # after peaks,
            # below or above boll_fib band,
            # macds > macd or Vise Versa
            # macd trend range > half previous range

            if macd_mem["down"] is not None and \
                    self.bt_stocks.iloc[val].macds > self.bt_stocks.iloc[val].macd and \
                    (self.bt_stocks.iloc[val].boll - self.bt_stocks.iloc[val][at_settings["boll_b"]]) > self.bt_stocks.iloc[val].close:

                macd_mem["down"] = None
                buy_indic = 1
                # print("macd buy: " + str(buy_macd))
                # print("macd : > " + str(self.bt_stocks.iloc[val].macd))

                print("go up " + str(self.bt_stocks.iloc[val].name) + " - boll:" + str(self.bt_stocks.iloc[val-1]['boll']) + " -prev: " + str(
                    self.bt_stocks.iloc[val-1]['close']) + " - curr:" + str(self.bt_stocks.iloc[val]['close']))

            if macd_mem["up"] is not None and \
                    self.bt_stocks.iloc[val].macds < self.bt_stocks.iloc[val].macd and \
                    (self.bt_stocks.iloc[val].boll + self.bt_stocks.iloc[val][at_settings["boll_s"]]) < self.bt_stocks.iloc[val].close:
                macd_mem["up"] = None or \
                    spy_stocks.iloc[val].close < spy_stocks.iloc[val].boll

                buy_indic = -1
                print("go down " + str(self.bt_stocks.iloc[val].name) + " - boll:" + str(self.bt_stocks.iloc[val-1]['boll']) + " -prev: " + str(
                    self.bt_stocks.iloc[val-1]['close']) + " - curr:" + str(self.bt_stocks.iloc[val]['close']))

            #---------------------------------------------------------------------------------------------------------------------------------------

            if self.bs.buyed_stocks == 0 and \
                (buy_indic == 1) or \
                    buy_now_process:

                self.bs.buy_stock(self.bt_stocks.iloc[val])
                buy_now_process = False

            #comment this block for selling at the end of the date
            if self.bs.buyed_stocks != 0 and \
                    buy_indic == -1:
                #  (self.bt_stocks.iloc[val]['boll_mid_lb'] <= self.bt_stocks.iloc[val-1]['close'] and self.bt_stocks.iloc[val]['boll_mid_lb'] > self.bt_stocks.iloc[val]['close'])):
                self.bs.sell_stock(self.bt_stocks.iloc[val])

            # ----------------------------------------------------------------------------------------------------------

        # self.bt_stocks.iloc[-1] = self.bt_stocks.iloc[val]
        # check_boll()
        # print(self.bs.profit_perc)
        if self.bs.transactions > 0:
            if self.best_at_settings is None:
                self.best_at_settings = at_settings

            if self.best_at_settings['profit'] < self.bs.profit_perc:
                at_settings["sym"] = symbol
                at_settings["profit"] = self.bs.profit_perc
                at_settings["transactions"] = self.bs.transactions
                at_settings["price_change"] = Utils.calc_perc(
                    self.bt_stocks.iloc[0].close, self.bt_stocks.iloc[-1].close)
                self.best_at_settings = at_settings
                print("at settings" + str(at_settings))
            self.bs.trading_stats(symbol, self.bt_stocks)
            # self.plot_stats(sub_data, spy_stocks)

        else:
            print("Theres no transactions please change BUY/SELL params")

    def ts_boll(self, buy_now=False, at_settings=None, symbol=None, spy_stocks=None):

        buy_now_process = buy_now
        buy_indic = 0
        macd_mem = {"up": 100, "down": 100, "range": None}
        self.bs.profit_perc = 0
        for val in range(1, len(self.bt_stocks)):

            # def check_boll():
            """
            docstring
            """
            # -------------------MACD, RSI, VOLUME 30% profit--------------------------------------------------
            # BUY
            # if self.bt_stocks.iloc[val].kdjk <= 30 and \
            #     self.bt_stocks.iloc[val].macd <= -0.5 and \
            #     self.bt_stocks.iloc[val-2].macd - self.bt_stocks.iloc[val].macd <= 0 and \
            #     self.bt_stocks.iloc[val]['boll_mid_lb'] >= self.bt_stocks.iloc[val-1]['close'] :
            #     # self.bt_stocks.iloc[val]['sma30'] < self.bt_stocks.iloc[val]['sma100'] :
            #     print("go up " + str(self.bt_stocks.iloc[val].name) +  " - boll:" + str(self.bt_stocks.iloc[val-1]['boll']) + " -prev: " + str(self.bt_stocks.iloc[val-1]['close']) + " - curr:" + str(self.bt_stocks.iloc[val]['close']))
            #     buy_indic = 1
            # # SELL
            # if self.bt_stocks.iloc[val].kdjk >= 60 and \
            #     self.bt_stocks.iloc[val].macd >= 2 and \
            #     self.bt_stocks.iloc[val-2].macd - self.bt_stocks.iloc[val].macd >= 0 and \
            #     self.bt_stocks.iloc[val]['boll_mid_ub'] <= self.bt_stocks.iloc[val-1]['close'] :
            #     # self.bt_stocks.iloc[val]['sma30'] > self.bt_stocks.iloc[val]['sma100'] :
            #     print("go down " + str(self.bt_stocks.iloc[val].name) +  " - boll:" + str(self.bt_stocks.iloc[val-1]['boll']) + " -prev: " + str(self.bt_stocks.iloc[val-1]['close']) + " - curr:" + str(self.bt_stocks.iloc[val]['close']))
            #     buy_indic = -1

            # -----------------------------------MACD + Volume change trend 50% profit------------------------------
            # BUY
            # vol_move = Utils.calc_perc(self.bt_stocks.iloc[val-1].volume, self.bt_stocks.iloc[val].volume)
            # bull = True if self.bt_stocks.iloc[val].close > self.bt_stocks.iloc[val].open else False

            # if  self.bt_stocks.iloc[val].macd <= -0.1 and \
            #     self.bt_stocks.iloc[val-2].macd - self.bt_stocks.iloc[val].macd <= 0 and \
            #     vol_move > 5  :
            #     # self.bt_stocks.iloc[val]['boll_mid_lb'] >= self.bt_stocks.iloc[val-1]['close'] :
            #     # self.bt_stocks.iloc[val]['sma30'] < self.bt_stocks.iloc[val]['sma100'] :
            #     print("go up " + str(self.bt_stocks.iloc[val].name) +  " - boll:" + str(self.bt_stocks.iloc[val-1]['boll']) + " -prev: " + str(self.bt_stocks.iloc[val-1]['close']) + " - curr:" + str(self.bt_stocks.iloc[val]['close']))
            #     buy_indic = 1
            # # SELL
            # if  self.bt_stocks.iloc[val].macd >= 1.5 and \
            #     self.bt_stocks.iloc[val-2].macd - self.bt_stocks.iloc[val].macd >= 0 :
            #     # self.bt_stocks.iloc[val]['boll_mid_ub'] <= self.bt_stocks.iloc[val-1]['close'] :
            #     # self.bt_stocks.iloc[val]['sma30'] > self.bt_stocks.iloc[val]['sma100'] :
            #     print("go down " + str(self.bt_stocks.iloc[val].name) +  " - boll:" + str(self.bt_stocks.iloc[val-1]['boll']) + " -prev: " + str(self.bt_stocks.iloc[val-1]['close']) + " - curr:" + str(self.bt_stocks.iloc[val]['close']))
            #     buy_indic = -1
            # -----------------------------------MACD change trend 80% profit------------------------------

            # --------------------------------detect peaks high or low sofifticated 50% -------------------*----------------

            macd_mem["range"] = self.bt_stocks.iloc[1:val].macd.max(
            ) - self.bt_stocks.iloc[1:val].macd.min()
            # ---------------------------------------- change indicators by trend ----
            if at_settings is None:
                at_settings = {"macd_b": 0, "macd_s": 0,
                               "boll_b": "boll_6", "boll_s": "boll_10",
                               "profit": -10000,
                                            "sym": symbol,
                                            "transactions": 0,
                                            "date": datetime.now(),
                                            "date_from": self.db.time_from,
                                            "price_change": 0}

                # macd_mem["range"] = macd_mem["up"] -  macd_mem["down"]
                # [0.4,0.65] []
                boll_range = ["boll", "boll_2", "boll_3",
                              "boll_5", "boll_6", "boll_10"]
                macd_range = [0.2, 0, 1]
                # trend = round(
                #     self.bt_stocks.iloc[val]["boll"] - self.bt_stocks.iloc[val]["sma50"], 2)

                trend = round(
                    self.bt_stocks.iloc[val]["close"] - self.bt_stocks.iloc[val]["boll"], 2)

                # if trend > 0:
                #     at_settings["macd_b"] = 0.7
                #     at_settings["macd_s"] = 0.7
                #     at_settings["boll_b"] = "boll_5"
                #     at_settings["boll_s"] = "boll_5"
                # elif trend < 0:
                #     at_settings["macd_b"] = 0.7
                #     at_settings["macd_s"] = 0.7
                #     at_settings["boll_b"] = "boll_5"
                #     at_settings["boll_s"] = "boll_5"

                # ------------------------------------------------------------------------
            # print(self.bt_stocks.iloc[1:val].macd.max())+
            # print("macd_b" + str(at_settings))
            if self.bs.buyed_stocks == 0 and  \
                (self.bt_stocks.iloc[1:val].macd.max() - (macd_mem["range"]*at_settings["macd_b"])) > self.bt_stocks.iloc[val].macd and \
                    self.bt_stocks.iloc[val-2].macd - self.bt_stocks.iloc[val].macd <= 0:

                macd_mem["down"] = self.bt_stocks.iloc[val].macd

            if self.bs.buyed_stocks != 0 and   \
                    macd_mem["range"]*at_settings["macd_s"] + self.bt_stocks.iloc[1:val].macd.min() < self.bt_stocks.iloc[val].macd and \
                    self.bt_stocks.iloc[val-1].macd - self.bt_stocks.iloc[val].macd >= 0:

                # print("sell macd limit: " + str((macd_mem["range"]) +
                #             self.bt_stocks.iloc[1:val].macd.min()))
                # print("macd: " + str(self.bt_stocks.iloc[val].macd))

                macd_mem["up"] = self.bt_stocks.iloc[val].macd
                # Utils.countdown(2)

            # SELL or BUY rules:
            # after peaks,
            # below or above boll_fib band,
            # macds > macd or Vise Versa
            # macd trend range > half previous range
            # self.bt_stocks.iloc[val].macds > self.bt_stocks.iloc[val].macd and \
            if macd_mem["down"] is not None and \
                    (self.bt_stocks.iloc[val].boll - self.bt_stocks.iloc[val][at_settings["boll_b"]]) > self.bt_stocks.iloc[val].close:

                macd_mem["down"] = None
                buy_indic = 1
                # print("macd buy: " + str(buy_macd))
                # print("macd : > " + str(self.bt_stocks.iloc[val].macd))

                print("go up " + str(self.bt_stocks.iloc[val].name) + " - boll:" + str(self.bt_stocks.iloc[val-1]['boll']) + " -prev: " + str(
                    self.bt_stocks.iloc[val-1]['close']) + " - curr:" + str(self.bt_stocks.iloc[val]['close']))

            if macd_mem["up"] is not None and \
                    self.bt_stocks.iloc[val].macds < self.bt_stocks.iloc[val].macd and \
                    (self.bt_stocks.iloc[val].boll + self.bt_stocks.iloc[val][at_settings["boll_s"]]) < self.bt_stocks.iloc[val].close:
                macd_mem["up"] = None or \
                    spy_stocks.iloc[val].close < spy_stocks.iloc[val].boll

                buy_indic = -1
                print("go down " + str(self.bt_stocks.iloc[val].name) + " - boll:" + str(self.bt_stocks.iloc[val-1]['boll']) + " -prev: " + str(
                    self.bt_stocks.iloc[val-1]['close']) + " - curr:" + str(self.bt_stocks.iloc[val]['close']))

            #---------------------------------------------------------------------------------------------------------------------------------------

            # # BUY
            # if  self.bt_stocks.iloc[val].macd <= -0.1 and \
            #     self.bt_stocks.iloc[val-2].macd - self.bt_stocks.iloc[val].macd <= 0 :

            #     macd_mem["up"] = self.bt_stocks.iloc[val].macd
            #     # self.bt_stocks.iloc[val]['boll_mid_lb'] >= self.bt_stocks.iloc[val-1]['close'] :
            #     # self.bt_stocks.iloc[val]['sma30'] < self.bt_stocks.iloc[val]['sma100'] :
            #     print("go up " + str(self.bt_stocks.iloc[val].name) +  " - boll:" + str(self.bt_stocks.iloc[val-1]['boll']) + " -prev: " + str(self.bt_stocks.iloc[val-1]['close']) + " - curr:" + str(self.bt_stocks.iloc[val]['close']))
            #     buy_indic = 1
            # # SELL
            # if  self.bt_stocks.iloc[val].macd >= 1.5 and \
            #     self.bt_stocks.iloc[val-2].macd - self.bt_stocks.iloc[val].macd >= 0 :
            #     macd_mem["down"] = self.bt_stocks.iloc[val].macd
            #     # self.bt_stocks.iloc[val]['boll_mid_ub'] <= self.bt_stocks.iloc[val-1]['close'] :
            #     # self.bt_stocks.iloc[val]['sma30'] > self.bt_stocks.iloc[val]['sma100'] :
            #     print("go down " + str(self.bt_stocks.iloc[val].name) +  " - boll:" + str(self.bt_stocks.iloc[val-1]['boll']) + " -prev: " + str(self.bt_stocks.iloc[val-1]['close']) + " - curr:" + str(self.bt_stocks.iloc[val]['close']))
            #     buy_indic = -1
            # -----------------------------------RSI change trend 20% profit------------------------------
            # if  self.bt_stocks.iloc[val].kdjk <= 50 and \
            #     self.bt_stocks.iloc[val-2].kdjk - self.bt_stocks.iloc[val].kdjk <= 0:
            #     # self.bt_stocks.iloc[val]['boll_mid_lb'] >= self.bt_stocks.iloc[val-1]['close'] :
            #     # self.bt_stocks.iloc[val]['sma30'] < self.bt_stocks.iloc[val]['sma100'] :
            #     print("go up " + str(self.bt_stocks.iloc[val].name) +  " - boll:" + str(self.bt_stocks.iloc[val-1]['boll']) + " -prev: " + str(self.bt_stocks.iloc[val-1]['close']) + " - curr:" + str(self.bt_stocks.iloc[val]['close']))
            #     buy_indic = 1
            # # SELL
            # if  self.bt_stocks.iloc[val].kdjk >= 60 and \
            #     self.bt_stocks.iloc[val-2].kdjk - self.bt_stocks.iloc[val].kdjk >= 0 :
            #     # self.bt_stocks.iloc[val]['boll_mid_ub'] <= self.bt_stocks.iloc[val-1]['close'] :
            #     # self.bt_stocks.iloc[val]['sma30'] > self.bt_stocks.iloc[val]['sma100'] :
            #     print("go down " + str(self.bt_stocks.iloc[val].name) +  " - boll:" + str(self.bt_stocks.iloc[val-1]['boll']) + " -prev: " + str(self.bt_stocks.iloc[val-1]['close']) + " - curr:" + str(self.bt_stocks.iloc[val]['close']))
            #     buy_indic = -1

            # -------------------------bollinger fibbonachi + macd test -----------------------
            # # BUY
            # if  (self.bt_stocks.iloc[val].boll -self.bt_stocks.iloc[val].boll_5) > self.bt_stocks.iloc[val].close and \
            #     self.bt_stocks.iloc[val-5].macd - self.bt_stocks.iloc[val].macd <= 0:
            #     # self.bt_stocks.iloc[val]['boll_mid_lb'] >= self.bt_stocks.iloc[val-1]['close'] :
            #     # self.bt_stocks.iloc[val]['sma30'] < self.bt_stocks.iloc[val]['sma100'] :
            #     print("go up " + str(self.bt_stocks.iloc[val].name) +  " - boll:" + str(self.bt_stocks.iloc[val-1]['boll']) + " -prev: " + str(self.bt_stocks.iloc[val-1]['close']) + " - curr:" + str(self.bt_stocks.iloc[val]['close']))
            #     buy_indic = 1
            # # SELL
            # if  (self.bt_stocks.iloc[val].boll_5 +self.bt_stocks.iloc[val].boll) < self.bt_stocks.iloc[val].close and \
            #     self.bt_stocks.iloc[val-5].macd - self.bt_stocks.iloc[val].macd >= 0 :
            #     # self.bt_stocks.iloc[val]['boll_mid_ub'] <= self.bt_stocks.iloc[val-1]['close'] :
            #     # self.bt_stocks.iloc[val]['sma30'] > self.bt_stocks.iloc[val]['sma100'] :
            #     print("go down " + str(self.bt_stocks.iloc[val].name) +  " - boll:" + str(self.bt_stocks.iloc[val-1]['boll']) + " -prev: " + str(self.bt_stocks.iloc[val-1]['close']) + " - curr:" + str(self.bt_stocks.iloc[val]['close']))
            #     buy_indic = -1

            # # ------------------------------ only bollinger band trading ------------------------------------
            # if self.bt_stocks.iloc[val]['boll'] >= self.bt_stocks.iloc[val-1]['close'] and \
            #     self.bt_stocks.iloc[val]['boll'] <= self.bt_stocks.iloc[val]['close'] :

            #     print("go up " + str(self.bt_stocks.iloc[val].name) +  " - boll:" + str(self.bt_stocks.iloc[val-1]['boll']) + " -prev: " + str(self.bt_stocks.iloc[val-1]['close']) + " - curr:" + str(self.bt_stocks.iloc[val]['close']))
            #     buy_indic = 1

            # elif self.bt_stocks.iloc[val]['boll'] <= self.bt_stocks.iloc[val-1]['close'] and \
            #     self.bt_stocks.iloc[val]['boll'] >= self.bt_stocks.iloc[val]['close']:

            #     print("go down " + str(self.bt_stocks.iloc[val].name) +  " - boll:" + str(self.bt_stocks.iloc[val-1]['boll']) + " -prev: " + str(self.bt_stocks.iloc[val-1]['close']) + " - curr:" + str(self.bt_stocks.iloc[val]['close']))
            #     buy_indic = -1

            # else:
            #     buy_indic = 0
            #--------------------------------------------------------

            if self.bs.buyed_stocks == 0 and \
                (buy_indic == 1) or \
                    buy_now_process:

                self.bs.buy_stock(self.bt_stocks.iloc[val])
                buy_now_process = False

            #comment this block for selling at the end of the date
            if self.bs.buyed_stocks != 0 and \
                    buy_indic == -1:
                #  (self.bt_stocks.iloc[val]['boll_mid_lb'] <= self.bt_stocks.iloc[val-1]['close'] and self.bt_stocks.iloc[val]['boll_mid_lb'] > self.bt_stocks.iloc[val]['close'])):
                self.bs.sell_stock(self.bt_stocks.iloc[val])

            # ----------------------------------------------------------------------------------------------------------

            # if  self.bs.buyed_stocks == 0  and \
            #     (buy_indic == 1 or \
            #     (stock['boll_mid_ub'] >= self.bt_stocks.iloc[val-1]['close'] and stock['boll_mid_ub'] < stock['close']) or \
            #     (stock['boll_mid_lb'] >= self.bt_stocks.iloc[val-1]['close'] and stock['boll_mid_lb'] < stock['close'])):
            #     self.bs.buy_stock(stock)

            # if   self.bs.buyed_stocks != 0 and \
            #      (buy_indic == -1 or \
            #      (stock['boll_mid_ub'] <= self.bt_stocks.iloc[val-1]['close'] and stock['boll_mid_ub'] > stock['close']) or \
            #      (stock['boll_mid_lb'] <= self.bt_stocks.iloc[val-1]['close'] and stock['boll_mid_lb'] > stock['close'])):
            #     self.bs.sell_stock(stock)

            # if  self.bs.buyed_stocks == 0  and \
            #     (buy_indic == 1 or \
            #     (stock['boll_mid_ub'] >= self.bt_stocks.iloc[val-1]['close'] and stock['boll_mid_ub'] < stock['close'])):
            #     self.bs.buy_stock(stock)

            # if   self.bs.buyed_stocks != 0 and \
            #      (buy_indic == -1 or \
            #      (stock['boll_mid_ub'] <= self.bt_stocks.iloc[val-1]['close'] and stock['boll_mid_ub'] > stock['close'])):
            #     self.bs.sell_stock(stock)

        # self.bt_stocks.iloc[-1] = self.bt_stocks.iloc[val]
        # check_boll()
        # print(self.bs.profit_perc)
        if self.bs.transactions > 0:
            if self.best_at_settings is None:
                self.best_at_settings = at_settings
            logging.info(self.bs.profit_perc)
            # logging.info(self.bs.best_at_settings['profit'])
            if self.best_at_settings['profit'] < self.bs.profit_perc:
                at_settings["sym"] = symbol
                at_settings["profit"] = self.bs.profit_perc
                at_settings["transactions"] = self.bs.transactions
                at_settings["price_change"] = Utils.calc_perc(
                    self.bt_stocks.iloc[0].close, self.bt_stocks.iloc[-1].close)
                self.best_at_settings = at_settings
                print("at settings" + str(at_settings))
            self.bs.trading_stats(symbol, self.bt_stocks)
            # self.plot_stats(sub_data, spy_stocks)

        else:
            print("Theres no transactions please change BUY/SELL params")

    def trading_simulation_rsi(self, buyBand, sellBand, buySignalGap, sellSignalGap):
        
        self.bs.buyed_stocks = 0
        self.bs.money = self.bs.startCredit
        self.first_stock = self.stocks.iloc[0]

        for index, stock in self.stocks.iterrows():
            
            if stock.valRSIclsf <= sellBand and stock.upRSIclsf >= sellSignalGap and self.bs.buyed_stocks == 0:
            # if stock.valRSIclsf <= sellBand and stock.upRSIclsf >= sellSignalGap and self.bs.money > stock.close:
               self.bs.buy_stock(stock)

            if stock.valRSIclsf >= buyBand and stock.upRSIclsf <= buySignalGap and self.bs.buyed_stocks != 0 and stock.close >= self.last_buyed_stock['close']:
            #if stock.valRSIclsf > 60 and stock.upRSIclsf < 0.2 and self.bs.buyed_stocks != 0:
                self.bs.sell_stock(stock)

            self.prev_stock = stock
        # print("last_buyed_stock['close']: " + str(self.last_buyed_stock['close']) + "Transactions: " + str(self.bs.transactions))
        # print("PARAMS buyBand: " + str(buyBand) + " sellBand:" + str(sellBand) + " buySignalGap:" + str(buySignalGap) + " sellSignalGap:" + str(sellSignalGap))
        
        if self.bs.transactions > 0:
            afterSellingStockMoney = round(self.money + (self.bs.buyed_stocks * self.last_buyed_stock['close']),2)
            tradingGainPercent = (afterSellingStockMoney - self.bs.startCredit) / ( self.bs.startCredit/100)/self.shares_amount
            
            if  tradingGainPercent > self.best_rsi_params:
                self.best_rsi_params = tradingGainPercent
               
                self.best_settings[self.symbols[0]] = {}
                self.best_settings[self.symbols[0]]['buyBand'] = buyBand
                self.best_settings[self.symbols[0]]['sellBand'] = sellBand
                self.best_settings[self.symbols[0]]['buySignalGap'] = buySignalGap
                self.best_settings[self.symbols[0]]['sellSignalGap'] = sellSignalGap
                print("RSI PARAMS: buyBand: " + str(buyBand) + " sellBand:" + str(sellBand) + " buySignalGap:" + str(buySignalGap) + " sellSignalGap:" + str(sellSignalGap))
                print("RSI Gain: " + str(round(tradingGainPercent,2) ) + "%" + " | Transactions: " + str(self.bs.transactions)+ " | StocksNum: " + str(self.bs.buyed_stocks)+ " | Money: " + str(afterSellingStockMoney))

        

    def test_rsi_params(self):
        
        for buyBand in range(self.rsi_params['buyBand']["min"],self.rsi_params['buyBand']["max"],self.rsi_params['buyBand']["step"]):
            for sellBand in range(self.rsi_params['sellBand']["min"],self.rsi_params['sellBand']["max"],self.rsi_params['sellBand']["step"]):
                for buySignalGap in numpy.arange(self.rsi_params['buySignalGap']["min"],self.rsi_params['buySignalGap']["max"],self.rsi_params['buySignalGap']["step"]):
                    for sellSignalGap in numpy.arange(self.rsi_params['sellSignalGap']["min"],self.rsi_params['sellSignalGap']["max"],self.rsi_params['sellSignalGap']["step"]):
                        self.bs.transactions = 0
                        # self.last_buyed_stock['close'] = 0
                        self.trading_simulation_rsi(buyBand, sellBand, buySignalGap, sellSignalGap)
     
        
    def buy_sell_japanese_food_strategy(self,time_from = None, sym = None):
        # self.buy_sell_hist
        dfp, financials, sentiment, earnings, spy  = self.db.get_all_data(time_from, sym)
        dfp = FinI.add_indicators(dfp)
       
        dfs = self.load_data(TableName.DAY,symbols=["SPY"])
        self.bt_stocks = dfp
        up_down_iters = [0,0]
        buy_indic = 0
        seq_stats = {"up_bull":0,"up_bull_3":0,"up_bull_4":0,"up_bear":0,"down_bull":0,"down_bear":0,"max_change": self.bt_stocks.ch_oc.max() }
        for val in range(2, len(self.bt_stocks)):
            # check up down for previous week, day
            
            # how long is stock under bollinger band.
            
            # get volatility
            
            # calculate speed of price action by volatility 
            # calculate or check gain sell price with bollinger band
            # calculate loss sell price from volatility
            # check bollinger band, RSI, MACD
            # Store trades in the database
            # Tune chinese food trading method
            # Check volume increasing and decreasing in different symbols at the same time, where money goes
            # check high low sectors
            # watch earning dates and check price moves
            # watch sentiment, recognize trends 
            # watch fundamentals as shortRatio, price to book, traillingEps, forwardEps should be uptrend, 
            
            if self.bt_stocks.iloc[val-1].ch_oc > 0 and \
                self.bt_stocks.iloc[val].ch_oc > 0 and \
                self.bt_stocks.iloc[val].macd > self.bt_stocks.iloc[val].macds and \
                self.bt_stocks.iloc[val].close > self.bt_stocks.iloc[val].boll:
                seq_stats["up_bull"] += 1
                
            if self.bt_stocks.iloc[val-2].ch_oc > 0 and \
                self.bt_stocks.iloc[val-1].ch_oc > 0 and \
                self.bt_stocks.iloc[val].ch_oc > 0 and \
                self.bt_stocks.iloc[val].macd > self.bt_stocks.iloc[val].macds and \
                self.bt_stocks.iloc[val].close > self.bt_stocks.iloc[val].boll:
                seq_stats["up_bull_3"] += 1
            if self.bt_stocks.iloc[val-3].ch_oc > 0 and \
               self.bt_stocks.iloc[val-2].ch_oc > 0 and \
                self.bt_stocks.iloc[val-1].ch_oc > 0 and \
                self.bt_stocks.iloc[val].ch_oc > 0 and \
                self.bt_stocks.iloc[val].macd > self.bt_stocks.iloc[val].macds and \
                self.bt_stocks.iloc[val].close > self.bt_stocks.iloc[val].boll:
                seq_stats["up_bull_4"] += 1
            
            if self.bt_stocks.iloc[val-1].ch_oc < 0 and \
                self.bt_stocks.iloc[val].ch_oc < 0 and \
                self.bt_stocks.iloc[val].macd < self.bt_stocks.iloc[val].macds and \
                self.bt_stocks.iloc[val].close > self.bt_stocks.iloc[val].boll:
                seq_stats["up_bear"] += 1
            
            if self.bt_stocks.iloc[val-1].ch_oc > 0 and \
                self.bt_stocks.iloc[val].ch_oc > 0 and \
                self.bt_stocks.iloc[val].macd > self.bt_stocks.iloc[val].macds and \
                self.bt_stocks.iloc[val].close < self.bt_stocks.iloc[val].boll:
                seq_stats["down_bull"] += 1
            
            if self.bt_stocks.iloc[val-1].ch_oc < 0 and \
                self.bt_stocks.iloc[val].ch_oc < 0 and \
                self.bt_stocks.iloc[val].macd < self.bt_stocks.iloc[val].macds and \
                self.bt_stocks.iloc[val].close < self.bt_stocks.iloc[val].boll:
                seq_stats["down_bear"] += 1

            try:
                if self.bt_stocks.iloc[val-1].close < self.bt_stocks.iloc[val-1].open and \
                    self.bt_stocks.iloc[val].close < self.bt_stocks.iloc[val].open:
                    buy_indic = 1
                
                elif self.bt_stocks.iloc[val-1].close > self.bt_stocks.iloc[val-1].open and \
                    self.bt_stocks.iloc[val-2].close > self.bt_stocks.iloc[val-2].open and \
                        self.bt_stocks.iloc[val].close > self.bt_stocks.iloc[val].open and \
                            self.bt_stocks.iloc[val].close > self.bt_stocks.iloc[val].boll:
                    buy_indic = -1
              
                
            except IndexError:
                print("Index out of range")
            
            
            
                
            if self.bs.buyed_stocks == 0 and \
                (buy_indic == 1) :
                self.bs.buy_stock(self.bt_stocks.iloc[val])
                

            #comment this block for selling at the end of the date
            if self.bs.buyed_stocks != 0 and \
                    buy_indic == -1:
                #  (self.bt_stocks.iloc[val]['boll_mid_lb'] <= self.bt_stocks.iloc[val-1]['close'] and self.bt_stocks.iloc[val]['boll_mid_lb'] > self.bt_stocks.iloc[val]['close'])):
                self.bs.sell_stock(self.bt_stocks.iloc[val])
                
        if self.bs.transactions > 0:
            self.bs.trading_stats(sym, self.bt_stocks)
            self.show_settings_stats()
        
        print(seq_stats)  
        # print(dfp[["change","close","open","boll_lb","boll","boll_ub","boll_mid_lb","kdjk","kdjd"]])
        
       
    def calculate_overnight(self, time_from = datetime.today()):
        
        if self.symbols is None:
            print("Please specify symbol")
            exit()
        if time_from is None and self.db.time_from is None:
            print("Please set time from")
            exit()

        # print(self.symbols)
        # df = self.load_data(table_name=TableName.DAY, symbols=self.symbols)
        self.buy_sell_japanese_food_strategy(
            sym=self.symbols, time_from=time_from)
        # fig, axs = plt.subplots(1, 1, figsize=(16, 8))
        # self.db.set_time_from("30d")
        
        # self.plot_overnight(axs, df)
        
        # plt.show()
    
    def show_settings_stats(self): 
        print("best settings" + str(self.best_settings))
        print("best A.T. settings" + str(self.best_at_settings))
    
    def test_strategy_1(self):
        bollingers = ["boll","boll_2","boll_3","boll_5","boll_6","boll_10"]
        step = 0.2
        spy_stocks = self.load_data(table_name=TableName.DAY, symbols="SPY")
        spy_stocks = FinI.add_indicators(spy_stocks)
        
        if self.symbols is None:
            self.symbols = self.db.get_symbols()
     
            
        for symbol in self.symbols:
            self.best_at_settings = None
            sub_data = self.load_data(table_name = TableName.DAY, symbols = symbol)
            
            if len(sub_data) < 1:
                print("No data")
                exit()
                
            self.bt_stocks = FinI.add_indicators(sub_data)
            self.bt_stocks = FinI.add_fib(self.bt_stocks)
            
            for macd_b in numpy.arange(0,1.2, step):
                for macd_s in numpy.arange(0, 1.2, step):
                    for boll_b in range(0,6, 1):
                        for boll_s in range(0,6, 1):
                            self.transactions = 0  
                            at_settings = {"macd_b": macd_b, 
                                            "macd_s": macd_s,
                                            "boll_b": bollingers[boll_b],
                                            "boll_s":  bollingers[boll_s], 
                                            "profit":-10000, 
                                            "sym":symbol,
                                            "transactions": 0,
                                            "date": datetime.now(),
                                            "date_from": self.db.time_from,
                                            "price_change": 0}
                            
                            if self.best_at_settings is None:
                                self.best_at_settings = at_settings
                                
                            
                            self.ts_boll(False, at_settings, symbol, spy_stocks = spy_stocks)
                            print(self.best_at_settings)
            
            settings = pd.DataFrame()
            settings = settings.append(self.best_at_settings, ignore_index=True)
            self.db.save_data("at_settings", settings, "append")
       
        
