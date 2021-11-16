
from martingale import ALPACA_API_KEY
from datetime import datetime
import types
from typing import Dict, Optional, Union
import pandas as pd
from traitlets.traitlets import Bool, Enum
from market_db import Database, TableName
from matplotlib import pyplot as plt
from plot_i import PlotI
from fin_i import FinI
from alpaca_buy_sell import AlpacaBuySell
from utils import Utils
from stockstats import StockDataFrame
import logging
import time
from dataclasses import dataclass

# create logger
logger = logging.getLogger("buy_sell")
logger.setLevel(logging.DEBUG)

class TradingProviders(Enum):
    ALPACA = "alpaca"


class TradingStrategies(Enum):
    OTHER = "other"
    LOGISTIC_REGRESSION = "logistic_regression"

class OpenTradesSource(Enum):
    ALPACA="alpaca"
    DB = "db"
    
@dataclass
class TradingProvider:
    name:str
    instance:AlpacaBuySell
    
class BuySell():
    trading_provider: TradingProvider
    # buyed_titles = [{"buy_price","sell_price", "sym", "shares_amount", "buy_date", "sell_date", "profit", "remain_credit"}]
    # credit = {"start_amount", "curr_amount"}
    
    def __init__(self, trading_provider: Optional[TradingProviders] = None, 
                 trading_strategy: Optional[TradingStrategies] = None, 
                 fill_open_trades=OpenTradesSource.DB, 
                 table_name="buy_sell_bt"):


        if trading_provider == TradingProviders.ALPACA:
            self.trading_provider = TradingProvider(name=trading_provider, instance = AlpacaBuySell())

        self.trading_strategy = trading_strategy
        # self.credit["start_amount"] = 1000
        # self.credit['curr_amount'] = credit['start_amount']
        self.buyed_stocks = pd.DataFrame()
        # self.stock_stats = Sdf()
        # self.startCredit = 10000
        self.credit = {'curr_amount':10000, "start_amount":10000}
        
        self.money:float = 100000
        self.prev_stock = None
        self.buyed_stocks = 0
        self.first_stock = pd.DataFrame()
        self.buy_marks = pd.DataFrame()
        self.sell_marks = pd.DataFrame()
        self.top_stocks_list = pd.DataFrame()
        self.last_buyed_stock = Optional[dict]
        self.first_buyed_price = 0.0
        self.profit_perc = 0.0
        self.profit_perc_all = 0.0
        self.transactions = 0
        self.buy_sell_closed = pd.DataFrame()
        self.share_amount = 1
        self.close_alpaca_postition = False
        self.db = Database()
        self.shares_amount = 1
        self.startCredit = 10000
        self.table_name = table_name
        # self.tv = TradingVars()
        if fill_open_trades == OpenTradesSource.DB:
            self.buy_sell_open = self.fill_open_trades_from_db()
        elif fill_open_trades == OpenTradesSource.ALPACA:
            # FIX DOESNT WORK 
            self.buy_sell_open = self.fill_trades_from_alpaca()

    def fill_trades_from_alpaca(self):
        positions = self.trading_provider.instance.get_positions()
        # logger.debug(positions['symbol'])
        # todo parse this motherfucker to buy:sell structure
        for item in positions:
            print(item)
        return positions
        
        
    def fill_open_trades_from_db(self):
        sql_string = f"select * from {self.table_name} where state='open'"
        logger.info(sql_string)
        try:
            return pd.read_sql(sql_string, con=self.db.engine)
        except:
            return pd.DataFrame()

    def fill_closed_trades_from_db(self):
        sql_string = f"select * from {self.table_name} where state='closed'"
        logger.info(sql_string)
        return pd.read_sql(sql_string, con=self.db.engine)
        
        
    # def sell_stock(self, stock:Union[pd.DataFrame,StockDataFrame],  sell_price: float= None, table_name: str= "buy_sell_bt",  shares_amount:float = 1)->None:
    #     """DEPRECATED

    #     Args:
    #         stock (Union[pd.DataFrame,StockDataFrame]): [description]
    #         sell_price (float, optional): [description]. Defaults to None.
    #         table_name (str, optional): [description]. Defaults to "buy_sell_bt".
    #         shares_amount (float, optional): [description]. Defaults to 1.
    #     """
    #     sell_price = sell_price if sell_price else stock["close"]
    #     self.profit_perc += round((sell_price - self.last_buyed_stock['buy']) / (
    #         self.last_buyed_stock['buy']/100), 2)

    #     self.buyed_stocks -= shares_amount
    #     self.money = self.money + (sell_price * shares_amount)

    #     buy_sell_stock = pd.DataFrame()
    #     buy_sell_stock.append(stock)
    #     buy_sell_stock["sell"] = sell_price
    #     buy_sell_stock["buy"] = self.last_buyed_stock['buy']
    #     buy_sell_stock["buy_date"] = self.last_buyed_stock.index
    #     buy_sell_stock["shares_amount"] = shares_amount
    #     buy_sell_stock["state"] = "closed"
      
    #     self.buy_sell_open = self.buy_sell_open.append(buy_sell_stock)

    #     logger.info("selling: " + str(stock['sym']) + " " +
    #           str(stock.name) + ' -- ' + str((sell_price * shares_amount)))

    #     if table_name:
    #         self.db.save_data(table_name=table_name,
    #                           data=self.buy_sell_open, if_exists="replace")


    # def buy_stock(self, stock: Union[pd.DataFrame, StockDataFrame], buy_price: float=None, table_name:str="buy_sell_bt", shares_amount:float=1, process_live:bool=False,  profit_loss:float=None):
    #     """DEPRECATED

    #     Args:
    #         stock (Union[pd.DataFrame, StockDataFrame]): [description]
    #         buy_price (float, optional): [description]. Defaults to None.
    #         table_name (str, optional): [description]. Defaults to "buy_sell_bt".
    #         shares_amount (float, optional): [description]. Defaults to 1.
    #         process_live (bool, optional): [description]. Defaults to False.
    #         profit_loss (float, optional): [description]. Defaults to None.
    #     """
    #     buy_price = buy_price if buy_price else stock["close"]
    #     if self.transactions == 0:
    #         self.first_buyed_price = buy_price

        
    #     self.buyed_stocks += shares_amount
    #     self.money = self.money - (buy_price * shares_amount)
    #     self.transactions += 1
    #     buy_sell_stock = pd.DataFrame()
    #     buy_sell_stock.append(stock)
    #     buy_sell_stock["buy"] = buy_price
    #     buy_sell_stock["shares_amount"] = shares_amount
    #     buy_sell_stock["state"] = "open"
    #     if profit_loss:
    #         buy_sell_stock["profit"] = profit_loss["profit"]
    #         buy_sell_stock["loss"] = profit_loss["loss"]
        
    #     logger.info("buying: " + str(stock['sym']) + " " +
    #           str(stock.name) + ' -- ' + str((buy_price * shares_amount)))
        
    #     self.buy_sell_open = self.buy_sell_open.append(buy_sell_stock)
    #     self.last_buyed_stock = buy_sell_stock
       


    def sell_stock_t(self, sym: str,  price: float, table_name: str = "buy_sell_close",  qty: Optional[float] = 1, sell_date: datetime = None, stock: pd.DataFrame = None) -> Optional[pd.DataFrame]:

        if stock is not None:
            sell_date = stock.index[0]
            price = stock.close
            
        buyed_stock = self.buy_sell_open[self.buy_sell_open["sym"] == sym]
        #sell only if some buyed stocks exist
        if buyed_stock is not None and len(buyed_stock) == 1:

            buyed_stock = buyed_stock.iloc[0]
            self.profit_perc += round((price - self.last_buyed_stock['buy']) / (
                self.last_buyed_stock['buy']/100), 2)

            if qty is None:
                qty = buyed_stock.shares_amount
            
            self.buyed_stocks = self.buyed_stocks - \
                qty if buyed_stock.shares_amount >= qty else self.buyed_stocks - buyed_stock.shares_amount
            self.money = self.money + (price * qty)

            buyed_stock.sell = price
            buyed_stock.sell_date = sell_date
            buyed_stock.shares_amount = (buyed_stock.shares_amount - qty) if buyed_stock.shares_amount - qty > 0 else 0
            buyed_stock.profit = buyed_stock.sell - buyed_stock.buy
            # print(buyed_stock.buy)
            # print(buyed_stock.sell)
            buyed_stock.perc_profit = Utils.calc_perc(
                                    buyed_stock.buy, buyed_stock.sell,2)
            if buyed_stock.shares_amount == 0:
                buyed_stock.state = "closed"
                self.buy_sell_open.drop(self.buy_sell_open[self.buy_sell_open["sym"] == sym].index, inplace = True)
               

            # self.buy_sell = self.buy_sell.append(buy_sell_stock)

            logging.info("selling: " + str(buyed_stock.sell_date) )
            
            self.buy_sell_closed = self.buy_sell_closed.append(buyed_stock)
            self.db.save_data(table_name=table_name,
                              data=self.buy_sell_closed, if_exists="replace")
        # sell stocks on provider platform
        if self.trading_provider is not None and self.trading_provider.name == TradingProviders.ALPACA:

            if self.close_alpaca_postition == False:
                self.trading_provider.instance.submitOrder(
                    qty, sym, "sell")
            else:
                self.trading_provider.instance.close_postions(sym)
            
            
            return buyed_stock
        else:
            return None

    
    def buy_stock_t(self, stock: pd.DataFrame, price:float=None, table_name:str="buy_sell_bt", qty:float=1, process_live:bool=False,  profit_loss:Dict=None, buy_already_buyed:bool = False)->pd.DataFrame:
        

        buy_sell_stock:pd.DataFrame = None
        price = price if price else stock["close"]
        if self.transactions == 0:
            self.first_buyed_price = price

        logger.warning(self.buy_sell_open)
        
        money:float = (self.money - (price * qty))
        
        already_buyed:Bool = (buy_already_buyed is True or \
            (buy_already_buyed is False and stock.sym not in self.buy_sell_open ))
        
        if money <= 0:
            logger.warning("not enough money")
        if already_buyed:
            logger.warning("symbol is already buyed")
        logger.info(money)
        logger.debug(already_buyed)
        if money > 0 and already_buyed:
            logger.debug("MOENY OK........................ ALREADY_BUUYED................OK")
            self.money = money
            self.buyed_stocks += qty
        
            self.transactions += 1
            # buy_sell_stock = pd.DataFrame()
            buy_sell_stock = stock.copy()
            
            buy_sell_stock["buy"] = price
            buy_sell_stock["shares_amount"] = qty
            buy_sell_stock["state"] = "open"
            buy_sell_stock["sell"] = None
            buy_sell_stock["sell_date"] = None
            buy_sell_stock["profit"] = None
            buy_sell_stock["perc_profit"] = None
            buy_sell_stock["strategy"] = self.trading_strategy
            if profit_loss:
                buy_sell_stock["est_profit"] = profit_loss["profit"]
                buy_sell_stock["est_loss"] = profit_loss["loss"]

            logger.info("buying: " + str(buy_sell_stock))
            self.buy_sell_open = self.buy_sell_open.append(buy_sell_stock)
            self.last_buyed_stock = buy_sell_stock
            logger.debug(self.trading_provider)
            logger.debug("TRADIGN PROVIDER")
            # buy stocks on provider platform
            if self.trading_provider is not None and self.trading_provider.name == TradingProviders.ALPACA:
                logger.debug("BUYIIIIIIIIIIIIIIIIIIIIIIIIIIIING ON ALPACA")
                self.trading_provider.instance.submitOrder(
                    qty, stock.tail(1)["sym"], "buy")
                time.sleep(5)

            try:
                self.db.save_data(table_name=table_name,
                                  data=self.buy_sell_open, if_exists="replace")
            except Exception as e: 
                logger.error("Database save failed:" + str(e))
                
        return buy_sell_stock
    
    def close_all_alpaca_postitions(self):
        self.trading_provider.instance.close_all_postions()
        
    def clean_all_db_postitions(self):
        # self.buy_sell_closed = self.fill_closed_trades_from_db()
        
        self.db.save_data(table_name=self.table_name,
                          data=pd.DataFrame(), if_exists="append")

        # self.db.save_data(table_name=self.table_name,
        #                   data=self.buy_sell_closed, if_exists="replace")
        

    def get_buyed_symbols_on_alpaca(self):
        
        buyed_stocks_table = "buyed_stocks"
        
        try:
            #get buyed stocks on alpaca
            abs = AlpacaBuySell()
            alp_pos, alp_orders = abs.get_positions()
            symbols = list()
            
            for position in alp_pos:
                symbols.append(position.symbol)

            for position in alp_orders:
                symbols.append(position.symbol)
                #------------------------------------------------
            
            df_sym = pd.DataFrame(symbols, columns =['sym']) 
            self.db.save_data(buyed_stocks_table, df_sym, "replace")
                
        except Exception as e:
            logger.error("loading buyed symbols from localDB | Err: " + str(e))
            symbols = self.db.get_data(buyed_stocks_table)
            symbols = symbols.sym.tolist()
        
        return symbols    
    
    def check_credit(self, price_sum):
        if self.credit["curr_amount"] is not None and self.credit["curr_amount"] >= price_sum:
            return True
        else:
            return False
        
    
    def get_credit(self):
        """
        return available credit
        """
        pass
        # return credit["curr_amount"]

    
    
    # def show_stats(self, symbol):
    #     print(" ---------SUMMARY--------------- ")
    #     print("STOCKS: " + str(symbol))
    #     print("Stocks in buy: " + str(self.buyed_stocks) + " | Last price: " + str(self.prev_stock.close)+ " | Last buyed price: " + str(self.last_buyed_stock['close']))

    #     afterSellingStockMoney = round(self.money + (self.buyed_stocks * self.last_buyed_stock['close']),2)
    #     # profit = afterSellingStockMoney - self.startCredit; 
    #     # tradingGainPercent = (afterSellingStockMoney - self.startCredit) / ( self.startCredit/100)/self.share_amount
    #     tradingGainPercent = (afterSellingStockMoney - self.first_buyed_price) / ( self.first_buyed_price/100)/self.share_amount
    #     tradingGainPercentCurrentPrice = (round(self.money + (self.buyed_stocks * self.prev_stock.close), 2) -
    #                                       self.credit["start_amount"]) / (self.credit["start_amount"]/100)/self.share_amount
    #     gainWithoutTradingPerc = (self.last_buyed_stock['close'] - self.first_stock.close) / (self.first_stock.close/100)
    #     afterSellingStockMoneyLastPrice = round(self.money + (self.buyed_stocks * self.last_buyed_stock['close']),2)
    #     gainWithoutTradingPercLastPrice = (self.prev_stock.close - self.first_stock.close) / (self.first_stock.close/100)


    #     if self.buyed_stocks > 0:
    #         self.sell_stock(self.prev_stock)
        
        
    #     # calculate average perc profit
    #     perc_divider = 1
    #     if self.profit_perc_all > 0:
    #         perc_divider = 2
    #     self.profit_perc_all = round((self.profit_perc_all+self.profit_perc)/perc_divider,2)

    #     # print("Money: " + str(self.money))

    #     # print("After selling remaining stocks for last buyed price: "  + str(afterSellingStockMoney) + " |  "  + str(round(afterSellingStockMoney-self.startCredit,2)) + " | " + str(round(tradingGainPercent,2)) + "%")
    #     # print("After selling remaining stocks for current price: "  + str(afterSellingStockMoney) + " |  "  + str(round(afterSellingStockMoney-self.startCredit,2)) + " | " + str(round(tradingGainPercentCurrentPrice,2)) + "%")
    #     print("First stock p.: " + str(self.first_stock.close) + " | Last stock p. : " +str(self.prev_stock.close) + " | Last buyed stock p. : " +str(self.last_buyed_stock['close']))
    #     print("Profit from " + str(symbol) +"  trade: " + str(self.profit_perc) + '%')
    #     print("Profit from all trades: " + str(self.profit_perc_all) + '%')
    #     print("Profit without trading: " + str(round(gainWithoutTradingPercLastPrice,2)) + "%")
    #     # print("Gain without trading last buyed price: " + str(round(self.last_buyed_stock['close'] - self.first_stock.close, 2)*self.share_amount) + " | " + str(round(gainWithoutTradingPerc,2)) + "%")
    #     # print("Profit without trading: " + str(round(self.prev_stock.close - self.first_stock.close, 2)*self.share_amount) + " | " + str(round(gainWithoutTradingPercLastPrice,2)) + "%")
        
    #     print("transactions: " + str(self.transactions))
    #     print("best settings" + str(self.best_settings))


    def trading_results(self, show=True) -> Union[None,pd.DataFrame]:
        res:pd.DataFrame = self.buy_sell_closed.copy()
        if len(res) > 0:
            res["profit_money"] = res.buy
            res["profit_percents"] = round((res.buy) / (res.buy/100),2)
            res["sum_profit_money"] = res["profit_money"].sum()
            res["sum_profit_percents"] = res["profit_percents"].sum()
            
            if show:
                logger.info("Profit money: ")
                logger.info(res["profit_money"])

                logger.info("Profit percents: ")
                logger.info(res["profit_percents"])

                logger.info("SUM Profit money: ")
                logger.info(res["sum_profit_money"][0])

                logger.info("SUM Profit percents: ")
                logger.info(res["sum_profit_percents"][0])
                
            
            return res
        else:
            return None

        
        
    def trading_stats(self, symbol:str, bt_stocks:pd.DataFrame)->str:
        stats_mess:str = ""
        stats_mess += ("Last Buyed Stocks: " + str(self.last_buyed_stock))
        stats_mess +=("buy sell dataframe : " + str(self.buy_sell_open))
        stats_mess +=(" ---------SUMMARY--------------- ")
        stats_mess +=("STOCKS: " + str(symbol))
        stats_mess +=("Stocks in buy: " + str(self.buyed_stocks) + " | Last price: " + str(bt_stocks.iloc[-1].close)+ " | Last buyed price: " + str(self.last_buyed_stock.buy))

        afterSellingStockMoney = round(self.money + (self.buyed_stocks * self.last_buyed_stock.buy),2)
        # profit = afterSellingStockMoney - self.startCredit; 
        # tradingGainPercent = (afterSellingStockMoney - self.startCredit) / ( self.startCredit/100)/self.share_amount
        tradingGainPercent = (afterSellingStockMoney - self.first_buyed_price) / ( self.first_buyed_price/100)/self.shares_amount
        tradingGainPercentCurrentPrice = (round(
            self.money + (self.buyed_stocks * bt_stocks.iloc[-1].close), 2) - self.startCredit) / (self.startCredit/100)/self.shares_amount
        gainWithoutTradingPerc = (self.last_buyed_stock.buy - bt_stocks.iloc[0].close) / (bt_stocks.iloc[0].close/100)
        # afterSellingStockMoneyLastPrice = round(self.money + (self.buyed_stocks * self.last_buyed_stock['close']),2)
        gainWithoutTradingPercLastPrice = (
            bt_stocks.iloc[-1].close - bt_stocks.iloc[0].close) / (bt_stocks.iloc[0].close/100)


        if self.buyed_stocks > 0:
            self.sell_stock(bt_stocks.iloc[-1])
        
        
        # calculate average perc profit
        perc_divider = 1
        if self.profit_perc_all > 0:
            perc_divider = 2
        self.profit_perc_all = round((self.profit_perc_all+self.profit_perc)/perc_divider,2)

        # stats_mess +=("Money: " + str(self.money))

        # stats_mess +=("After selling remaining stocks for last buyed price: "  + str(afterSellingStockMoney) + " |  "  + str(round(afterSellingStockMoney-self.startCredit,2)) + " | " + str(round(tradingGainPercent,2)) + "%")
        # stats_mess +=("After selling remaining stocks for current price: "  + str(afterSellingStockMoney) + " |  "  + str(round(afterSellingStockMoney-self.startCredit,2)) + " | " + str(round(tradingGainPercentCurrentPrice,2)) + "%")
        print(self.last_buyed_stock)
        stats_mess +=("First stock p.: " + str(bt_stocks.iloc[0].close) + " | Last stock p. : " +
              str(bt_stocks.iloc[-1].close) + " | Last buyed stock p. : " ) #+ str(self.last_buyed_stock['close']))
        stats_mess +=("Profit from " + str(symbol) +"  trade: " + str(self.profit_perc) + '%')
        stats_mess +=("Profit from all trades: " + str(self.profit_perc_all) + '%')
        stats_mess +=("Price change : " + str(round(gainWithoutTradingPercLastPrice,2)) + "%")
        # stats_mess +=("Gain without trading last buyed price: " + str(round(self.last_buyed_stock['close'] - bt_stocks.iloc[0].close, 2)*self.share_amount) + " | " + str(round(gainWithoutTradingPerc,2)) + "%")
        # stats_mess +=("Profit without trading: " + str(round(bt_stocks.iloc[-1].close - bt_stocks.iloc[0].close, 2)*self.share_amount) + " | " + str(round(gainWithoutTradingPercLastPrice,2)) + "%")
        
        stats_mess +=("transactions: " + str(self.transactions))
        self.show_sym_bs_stats(symbol)
        return stats_mess
    
    def show_sym_bs_stats(self, sym:str):
            
        self.db.limit = None
        dfp, financials, sentiment, earnings, spy = self.db.get_all_data(self.db.time_from, sym)
        dfp = FinI.add_indicators(dfp)
        dfp = FinI.add_fib(dfp, last_rows=10)
        # print(dfp)
        # last 5 financials
        financials = financials.tail(5)
        days_to_earnings = FinI.days_to_earnings(earnings)
        # print(earnings)
        if days_to_earnings is None:
            earnings = None
        fig, axs = plt.subplots(2, 1, figsize=(16, 16))
        PlotI.set_margins(plt)
     
        
        PlotI.plot_spy(axs[0], spy.loc[spy.index.isin(dfp.index)])
        axsp=PlotI.plot_stock_prices(axs[0].twinx(), dfp, sym, 0.5)
        axsp = PlotI.plot_sma(axsp,dfp,["sma9", "sma50"])
        axsp.xaxis.set_major_formatter(plt.NullFormatter())
        axsp = PlotI.plot_candles( dfp, axsp, body_w=0.5, shadow_w=0.1, alpha=0.5)
        axsp = PlotI.plot_fib( dfp, axsp, alpha=0.4)
        axsp = PlotI.plot_fib( dfp, axsp, alpha=0.4, fib_name = "fb_mid")
        axsp = PlotI.plot_fib( dfp, axsp, alpha=0.4, fib_name="fb_bot")
        axsp = PlotI.plot_weeks(axsp, dfp)
        
        PlotI.plot_boll(axsp, dfp, sym)
        
        
        axsp2 = PlotI.plot_volume_bars(axsp.twinx(), dfp)
        
        axsp2 = PlotI.plot_rsi(axs[1], dfp)
        axsp2 = PlotI.plot_macd_boll(axs[1].twinx(), dfp)
        # ax_macd = PlotI.plot_macd(ax_macd,dfp)
        axsp2 = PlotI.plot_volume_bars(axs[1].twinx(), dfp)
        
        if self.buy_sell_closed is not None and len(self.buy_sell_closed) > 0:
            axsp = PlotI.plot_bs(axsp, self.buy_sell_closed)
            axsp2 = PlotI.plot_bs(axsp2.twinx(), self.buy_sell_closed)
        
       
        plt.show()
        # self.plot_boll(axsp, dfp, sym)
        # self.plot_weeks(axsp, dfp)
       
        # self.plot_rsi(axs[1], dfp )
       
        # last_prices = self.load_data(TableName.MIN15, sym, limit = 26)
        # # self.plot_volume(axs[2], last_prices)
        # ax = self.plot_candlesticks2(axs[2], last_prices)
        # self.plot_boll(ax, last_prices, sym)
    


# class TradingVars():
#     classificators = {}
#     stock_stats = sdf()
#     startCredit = 10000
#     money = 0
#     prev_stock = None
#     buyed_stocks = 0
#     first_stock = pd.DataFrame()
#     buy_marks = pd.DataFrame()
#     sell_marks = pd.DataFrame()
#     top_stocks_list = pd.DataFrame()
#     last_buyed_stock = None
#     first_buyed_price = 0
#     profit_perc = 0
#     profit_perc_all = 0
#     transactions = 0
#     buy_sell_hist = pd.DataFrame()
#     share_amount = 1
