import asyncio
import logging
import twitterSentiment
import matplotlib.pyplot as plt
from stockstats import StockDataFrame
import pandas as pd
from sqlalchemy import create_engine
import sys
import numpy
sys.path.append('../')
import alpaca2Login as al
# from market_app.overview.refreshFinancials import refreshFinancials

import numpy as np
from datetime import datetime, timedelta
import pytz
utc = pytz.UTC
# from .buy_sell import BuySell
# from sklearn import linear_model, preprocessing
from pytz import timezone
localtz = timezone('Europe/Prague')
from utils import Utils
from market_db import Database
from buy_sell import BuySell
from plot_i import PlotI
from fin_i import FinI 
from alpaca_buy_sell import AlpacaBuySell
from check_indicators import CheckIndicators as chi
from market_db import TableName
from stock_mess import StockMess
from sector_stats import SectorStats
from back_trader import BackTrader

class StockWhisperer():
    def __init__(self):
        
        # self.best_at_settings = None
        self.stocks = StockDataFrame()
        self.spy = StockDataFrame()
        self.db = Database()
        self.ss = SectorStats()
        # self.processed_data = None
        self.db_name = "nyse_financials"
        self.price_table_name =TableName.DAY.to_str()
        self.engine = create_engine('postgresql://postgres:crasher@localhost:5432/'+self.db_name)
        self.df = pd.DataFrame()
        self.classificators= {}
        # start tradining stats
        # self.stock_stats = StockDataFrame()
        self.startCredit = 10000
        self.money = 0
        # self.prev_stock = None
        self.buyed_stocks = 0
        self.bt_stocks = pd.DataFrame()
       
        self.comp_sym_spy = pd.DataFrame()
       
        # comment BUY SELL on ALPACA
        self.account_api = al.alpaca2Login().getApi()
        self.time_from = None
        self.time_to = None
        self.symbols = None
        self.sectors = []
        self.sectors_trend = []
        self.spy_trend = []
        # self.buyed_stocks_list = list(pd.read_csv("buyed_stocks.csv"))
        self.bs = BuySell()
        self.alpaca_buy_sell = AlpacaBuySell()
        self.buyed_symbols = list()
        self.warning_check_list = []
        self.sm = StockMess()
        self.bt = BackTrader()

    
        
   
    def load_data(self, table_name = None, symbols = None, sectors = None, limit = None, time_from = None, time_to=None):
        symbols = symbols if symbols is not None else self.symbols
        return StockDataFrame.retype(self.db.load_data(table_name, symbols=symbols, sectors=sectors, 
                                            limit=limit, time_from=time_from, time_to=time_to))
    
        
    def load_spy(self, table_name = None, time_from = None, time_to=None):
        return StockDataFrame.retype(self.db.load_spy(table_name, time_from=time_from, time_to=time_to))
    
    def check_entered_sym(self, symbol):
        # print(symbol)
        if len(symbol) < 1:
            print("Please add stock symbol for simulation. example: py script-name.py TSLA")
            exit()

    def start_rsi_test(self, table_name, buyed_shares_amount, symbol):
        
        
        self.check_entered_sym(symbol)

        self.symbols = symbol
        self.share_amount = buyed_shares_amount
        self.stocks = self.load_data(table_name, symbols=self.symbols)
        self.stocks.get('kdjk')
        self.stocks = FinI.classify_rsi_oscilator(self.stocks)
        self.bt.test_rsi_params()

        self.trading_stats(symbol)
       

    
    def show_sec_day_stats(self, sectors):
        fig, axs = plt.subplots(1, 1, figsize=(16, 4))
        # PlotI.set_margins(plt)
        PlotI.plot_sector_stats(axs,sectors)
        # plt.legend()

                     
   
    def top_sectors(self, table_name, loosers):
        
        self.stocks = self.sm.classify_sectors_uptrend(table_name =table_name)
        self.stocks = self.stocks.sort_values(by='flpd', ascending=loosers)
        print(self.stocks.head())
        self.stocks.plot(kind="barh",use_index=True,y="flpd",legend = False)
        plt.show()


    def top_stocks(self, show_stocks_num = 20, from_top = 0, table_name = None, top_losers = True):
        
        subject = "Loosers:" if top_losers else "Gainers:"
        self.stocks =  self.load_data(table_name = table_name, sectors = self.sectors)
        self.stocks = FinI.add_change(self.stocks)
        self.stocks = Utils.add_first_last_perc_diff(self.stocks)
        # print("STOCKS: " + str(self.stocks))
        if len(self.stocks) > 0:
            stocks = self.stocks.groupby(by="sym").mean()
            stocks = stocks.sort_values(by='flpd', ascending=top_losers)
            
            top_stocks = stocks.iloc[from_top:(from_top + show_stocks_num)]
            # print(top_stocks)
           
            # self.top_stocks_list = top_stocks.index.tolist()
            self.draw_chart_values(top_stocks)
           
            asyncio.run(self.sm.mail_stats(top_stocks, subject))
        else:
            print('No stocks has been found')
    

    def draw_chart_values(self, data):
        data.plot(kind="barh", use_index=True, y="flpd", legend = False)
        for index, value in enumerate(data.flpd):
            plt.text(value, index, str(round(value,2)))
        plt.show()

    
            
    def plot_stats(self, sub_data, spy_indicator, spy_move):
        fig, axes = plt.subplots(nrows=2, ncols=1)

 

        plt.xticks(np.arange(0, len(sub_data), step=1))
        # ax = sub_data.plot(y=['boll','boll_ub','boll_lb','boll_mid_lb','boll_mid_ub'], legend = True,ax=axes[0] )
        # ax.grid('on', which='minor', axis='x',linestyle='--', linewidth=0.15, )
        
        # if len(self.buy_marks)>0:
        #     ax2 = self.buy_marks.plot(marker='o', y='close',ax=ax, color='g', linestyle="None", label="buy")
        # if len(self.sell_marks)>0:
        #     ax3 = self.sell_marks.plot(marker='o' ,y='close', ax=ax2, color='r', linestyle="None", label = "sell", grid=True)
        
        ax4 = sub_data.plot( y=['macds','macd'],ax=axes[0], grid=True)
        plt.xticks(np.arange(0, len(sub_data), step=1))
        ax5 = sub_data.plot(kind='line', linestyle=":", y=["macdh"], ax=ax4, grid=True)
        ax5.grid('on', which='minor', axis='x',linestyle=':', linewidth=0.15 )
        
        ax6 = sub_data.plot(kind='line', linestyle="None", marker='o', y=[
                            "kdjk", "kdjd", "kdjj"], ax=axes[1], grid=False)
        ax6.axhline(y=20,color="r",linestyle="-", linewidth=0.25, )
        ax6.axhline(y=80,color="r",linestyle="-", linewidth=0.25, )

        if spy_indicator:
            plt.text(0.2, 0, "Spy indicator: " + str(spy_indicator) + " (100 buy/0 sell)", size=10, ha="right", va="top", color="r")

        if spy_indicator:
            plt.text(0.2, 5.8, "Spy move: " + str(round(spy_move,2)) + "%", size=10, ha="right", va="top", color="red")
         
        # ax7 = spy_stocks.plot(y=['close'], legend=True, linestyle=":", color="r", ax=axes[0], label="SPY")
        self.plot_prices(sub_data)

        plt.show()

    def plot_prices(self, data):
        print(data)
        for key, value in data.iterrows():
            try:
                print(value)
                self.sm.show_sym_stats(value.sym)
            
            except AttributeError:
                self.sm.show_sym_stats(key)
                
        plt.show() 
        
  
    def show_fin_earn_price(self, sym):
        if sym is None:
            print("Please specify symbol")
            exit()
        self.ss.set_spy_sectors_trend()
        # if type(sym) is str:
        #     # self.stock_mess.show_sym_stats(sym)
        #     self.mail_sym_stats(sym, "Stats: ")
        # else:
        for s in sym:
            self.sm.set_fundamentals(s)
            self.sm.set_prices(sym=s)
            asyncio.run(self.sm.a_mail_sym_stats(s, "Stats: "))
        # plt.show()
        
    def prices_to_groups(self, sym):
        # fig, axes = plt.subplots(nrows=1, ncols=1)
       
        sub_data = self.load_data(table_name = self.price_table_name, symbol = sym)
        # prepare data for price profile
        prices = sub_data['close'].values
        x = list(prices)
        prices = ((x-min(x))/(max(x)-min(x))) * 100 +1
        # create price profile
        values=[]
        counters=[]

        for cl in prices:
            cl= int(cl)
            values.append(cl)
            counters.append(values.count(cl))

        profile_df = pd.DataFrame({'value': values, 'index': counters})
        return profile_df

          
    def plot_price_vol(self, sym):
        
        sub_data = self.load_data(table_name = self.price_table_name, symbols = sym)
        sub_data.get('volume_delta')
        sub_data.get('open_-2_r')
        plt.scatter(sub_data.volume, sub_data.close, alpha=0.25, marker='.',)
        plt.grid(axis="y")
        plt.yticks(np.arange(min(sub_data.close), max(sub_data.close), step=(max(sub_data.close) - min(sub_data.close))/20))
       
        plt.twiny()
        plt.plot(sub_data.index, sub_data.close, color="r")
        plt.legend([sym])
        
        plt.twinx()
        plt.plot(sub_data.index, sub_data.volume_delta, color="y", alpha=0.2) 
        plt.show()
      
        
    def iterate_by_symbol(self, table_name, mail_stats, call_back_f ):
        df_out = StockDataFrame()
        if self.symbols:
            symbols = self.symbols
        else:
            symbols = self.db.get_symbols()
        # print(str(symbols))
        for symbol in symbols:
            try:
                print("symbol: " + str(symbol))
                print(table_name)
                sub_data = self.load_data(
                    table_name, symbol, time_from=self.time_from)
                if len(sub_data) < 1:
                    break
                # print(sub_data)
                buyed_stocks = call_back_f(sub_data, mail_stats)
                df_out = df_out.append(buyed_stocks)
            except KeyboardInterrupt:
                exit()
        return df_out

    def compare_spy_sym(self, data, comp_const = - 0.2):
        
        spy_perc = Utils.calc_perc(self.spy.iloc[0].close, self.spy.iloc[-1].close)
        stock_perc = Utils.calc_perc(data.iloc[0].close, data.iloc[-1].close)
        spy_stock_comp = stock_perc - spy_perc 
       
        if spy_stock_comp >= comp_const:
            cache_comp_list = data.iloc[-1]
            cache_comp_list['perc'] = stock_perc
            cache_comp_list['spy_perc'] = spy_perc
            cache_comp_list['f_price'] = data.iloc[0].close
            cache_comp_list['spy_stock_comp'] = spy_stock_comp
            self.comp_sym_spy = self.comp_sym_spy.append(cache_comp_list)
            

    def classify_sap_index(self, stocks=None):
        buy_indicator = None
        
        stocks = FinI.add_indicators(stocks)
        if stocks is not None and len(stocks) > 0:

            if stocks.iloc[-1]["close"] < stocks.iloc[-1]['boll_lb'] or \
                stocks.iloc[-1]["close"] < stocks.iloc[-1]['boll']:
            
                buy_indicator = 20

            elif stocks.iloc[-1]["close"] > stocks.iloc[-1]['boll'] and \
                stocks.iloc[-1]["close"] > stocks.iloc[-1]['boll_mid_ub']:
                
                buy_indicator = 35

            elif  stocks.iloc[-1]["close"] > stocks.iloc[-1]['boll'] and \
                stocks.iloc[-1]["close"] < stocks.iloc[-1]['boll_mid_ub']:

                buy_indicator = 70

            elif  stocks.iloc[-1]["close"] > stocks.iloc[-1]['boll_lb'] :
                
                buy_indicator = 80

            elif  stocks.iloc[-1]["close"] > stocks.iloc[-1]['boll_mid_lb']:
                buy_indicator = 90
            

            if  stocks.iloc[-1]['open_-2_r'] < 0:
                buy_indicator -= 50
            

        return buy_indicator

    def sap_moving(self, table_name = None):
        
        stocks =  self.db.load_spy(table_name)
        stocks = FinI.add_indicators(stocks)

        if stocks.iloc[-1]["close"] < stocks.iloc[-1]['boll_lb'] or \
           stocks.iloc[-1]["close"] < stocks.iloc[-1]['boll'] and stocks.iloc[-1]['open_-2_r'] < 0:
           
           buy_indicator = 20

        elif stocks.iloc[-1]["close"] > stocks.iloc[-1]['boll'] and \
            stocks.iloc[-1]["close"] < stocks.iloc[-1]['boll_mid_ub'] and \
            stocks.iloc[-1]['open_-2_r'] > 0:
            
            buy_indicator = 70

        elif  stocks.iloc[-1]["close"] > stocks.iloc[-1]['boll_lb'] and \
            stocks.iloc[-1]['open_-2_r'] > 0:
            
            buy_indicator = 80

        elif  stocks.iloc[-1]["close"] > stocks.iloc[-1]['boll_mid_lb'] and \
            stocks.iloc[-1]['open_-2_r'] > 0:
            
            buy_indicator = 90

        return buy_indicator
    
       
    
    def prepare_buy_logic(self, infinite = False, return_stocks = True, mail_stats = True):
        
        if not self.price_table_name:
            self.price_table_name = TableName.DAY
        
        if not self.time_from:
            self.time_from = "-120d"
          
        # maybe this two rows are not necessary  
        self.spy = self.load_data(table_name=TableName.DAY, symbols=[
                                  "SPY"], time_from=self.time_from)
        self.spy = FinI.add_indicators(self.spy)
        
         # CALLBACK
        
        if infinite:
            try:
                while True:
                    # self.stock_mess.set_spy_sectors_trend()
                     #GET BUYED STOVKS
                    self.buyed_symbols = self.bs.get_buyed_symbols()
                    self.iterate_by_symbol(self.db.price_table_name, mail_stats, self.find_stocks_to_buy)
                    Utils.countdown(300)
                    
            except KeyboardInterrupt:
                print("-----------Checking stocks for Sale script: Stopped-----------------")
                exit()
        else:
            try:
                self.iterate_by_symbol(
                    self.db.price_table_name,mail_stats, self.find_stocks_to_buy)
            except KeyboardInterrupt:
                print("-----------Checking stocks for Sale script: Stopped-----------------")
                exit()
                
            
    
        # self.find_stock_to_buy()
        empty_stocks = True
        spy_change = 0
        spy_stocks = None
        if self.stocks is not None and len(self.stocks) > 0:
            self.stocks = self.stocks.groupby(by='sym').mean()
            self.stocks = self.stocks.sort_values(by='open_-2_r', ascending=False)
        

            spy_stocks = self.load_spy(self.db.price_table_name)
            spy_indicator = self.classify_sap_index(spy_stocks)
            empty_stocks = False
        else:
            print("No stocks to buy. DONE...")

        if spy_stocks is not None and len(spy_stocks) > 1:
            spy_change = Utils.calc_perc(
                spy_stocks.iloc[-3].close, spy_stocks.iloc[-1].close)
            print("Buy indicator by S&P index 0-100: " + str(spy_indicator))
            print("S&P move: " + str(spy_change) + "%")

        if return_stocks and not empty_stocks:
            return self.stocks
        
        if not empty_stocks:
            self.plot_stats(self.stocks.iloc[:30], spy_indicator, spy_change)
         
    

    def plot_sectors(self, plt):
        data = self.sectors_uptrend_by_month(yf=2021,yt=2021, show_chart = False)  
        plt = self.sector_stats_to_plt(self,plt, data)
        return plt

    def classify_sectors(self, time_from = "7d", table_name = "p_day", loosers = False):
        stocks = self.classify_sectors_uptrend(table_name)
        stocks = stocks.sort_values(by='flpd', ascending=loosers)
        return stocks
    
    def create_sectors_trend_mess(self, sector):
       
        sector0 = self.sectors_trend[0][self.sectors_trend[0].index == sector]
        sector1 = self.sectors_trend[1][self.sectors_trend[1].index == sector]
        sector_mess = ""
        if len(sector0) > 0 and len(sector1)>0:
            sector_mess = " | " + str(sector0.iloc[0].name) + ": " + str(round(sector0.iloc[0].flpd,1)) + "% -> " +  str(round(sector1.iloc[0].flpd,1))+ "%"
        
        return sector_mess          
    
    def create_spy_trend_mess(self):
        spy_mess = ""
        if len(self.spy_trend[0]) > 0 and len(self.spy_trend[1]) > 0:
            spy_mess = " | SPY: " + str(round(
            self.spy_trend[0].iloc[0].flpd, 1)) + "% -> " + str(round(self.spy_trend[1].iloc[0].flpd, 1)) + "%"
        
        return spy_mess

    def find_stocks(self, table_name: TableName, mail_stats)->StockDataFrame:
        df_out = StockDataFrame()
        symbols = self.db.get_symbols()
        
        # if self.time_from is None:
        #     self.time_from = time_from
            
        data = self.load_data(table_name, time_from=self.time_from)
        earnings, sentiment, financials = self.db.get_fundamentals()
        latest_stocks = self.load_data(
            TableName.MIN15, limit=len(symbols)*4)
        # print(str(symbols))
        try:
            for symbol in symbols:
           
                print("symbol: " + str(symbol))
                sub_data = data[data.sym == symbol]
                sub_earnings = earnings[earnings.ticker == symbol]
                sub_sentiment = sentiment[sentiment.stock == symbol]
                sub_financials = financials[financials.symbol == symbol]
                sub_latest_stocks = latest_stocks[latest_stocks.symbol == symbol]
                if len(sub_data) > 0:
                    
                    # print(sub_data)
                    buyed_stocks = self.find_stocks_to_buy(
                        sub_data, mail_stats=False, earnings=sub_earnings, sentiment=sub_sentiment, financials=sub_financials, latest_stocks=sub_latest_stocks)
                    
                    if buyed_stocks is not None and len(buyed_stocks)>0:
                        df_out =  pd.concat([df_out,buyed_stocks])
        except KeyboardInterrupt:
            exit()
            
        print(df_out)
        return df_out

    def bt_top_stocks(self,  back_days: int = 50, profit_level: int = 1, loss_level: int = 1) -> None:
       self.bs.money = 10000      

       while  back_days >= 0:
            self.db.last_date = datetime.today().replace(hour=23, minute=59) - \
                timedelta(days=(back_days))
            selected_stocks = self.find_stocks(TableName.DAY, False)
            back_days -= 1
            stats = pd.DataFrame()
            log.info("date")
            log.info(self.db.last_date)
            log.info("open trades")
            log.info(self.bs.buy_sell_open)
            log.info("closed trades")
            log.info(self.bs.buy_sell_closed)
            log.info("selected stocks ")
            log.info(selected_stocks)
            for idx, stock in selected_stocks.iterrows():

                # extract buyed stock if exists
                buyed_stock = None
                if self.bs.buy_sell_open is not None and len(self.bs.buy_sell_open) > 0:
                    buyed_stock = self.bs.buy_sell_open[self.bs.buy_sell_open.sym == stock["sym"]]

                profit  = stock["profit"][profit_level] if len(stock["profit"])> profit_level else None
                loss = stock["loss"][loss_level] if len(stock["loss"]) > loss_level else None

                self.bs.buy_stock_t(price=stock["close"],
                                    stock=stock,
                                    qty=1,
                                    profit_loss={"profit": profit, "loss": loss})
                              
                
                if buyed_stock is not None:
                
                    # sell stock in last day of test
                    # sell stock above profit level 
                    # sell stock under loss level
                    if buyed_stock.profit is not None and len(buyed_stock.profit) > profit_level and stock.close > buyed_stock.profit[profit_level] or \
                        buyed_stock.loss is not None and len(buyed_stock.loss) > loss_level and stock.close < buyed_stock.loss[loss_level] or \
                        back_days == 0:

                        self.bs.sell_stock_t(sym=stock.sym,
                                             price=stock.close,
                                             sell_date=stock.index[0],
                                             qty=1)
                      
                   
          

    def profit_loss(self, back_days:int = 50, profit_level:int=1, loss_level:int=1)->pd.DataFrame:
       
        self.db.last_date = datetime.today().replace(hour=23, minute=59) - \
            timedelta(days=(back_days))
        buyed_stocks = self.find_stocks(TableName.DAY, False)

        log.info(buyed_stocks)
        time_from = self.db.last_date
        self.db.last_date = None
        
        if len(buyed_stocks) < 1:
            return None

        stocks = self.db.load_data(symbols=buyed_stocks.sym, time_from=time_from)
        # print(stocks)

        stats = pd.DataFrame()

        for sym in buyed_stocks.sym:
            sym_s = stocks[stocks.sym == sym]
            i = 1
            loss_tolerance_perc = 1

            for idx, row in sym_s.iterrows():

                profit = buyed_stocks[buyed_stocks.sym == sym].iloc[-1].profit
                loss = buyed_stocks[buyed_stocks.sym == sym].iloc[-1].loss
                print(profit)
                print(row.close)
                b_price = buyed_stocks[buyed_stocks.sym == sym].iloc[-1].close
                price_diff = row.close - b_price
                if len(stats) == 0 or (len(stats) > 0 and stats.iloc[-1].sym != sym and stats.iloc[-1].price_diff != price_diff):
                    # sell stock above profit level
                    if profit is not None and len(profit) > profit_level and row.close > profit[profit_level]:

                        stats_l = {"sym": sym,
                                "days": i,
                                "price_diff":  price_diff,
                                "buy_price": b_price,
                                "sell_price": row.close}
                        stats = stats.append(stats_l, ignore_index=True)
                    # sell stock under loss level
                    elif loss is not None and len(loss) > loss_level and row.close < loss[loss_level]:
                        stats_l = {"sym": sym,
                                "days": i,
                                "price_diff":  price_diff,
                                "buy_price": b_price,
                                "sell_price": row.close}
                        stats = stats.append(stats_l, ignore_index=True)
                    # sell stock in last day of test
                    elif (i) == len(sym_s):
                        stats_l = {"sym": sym,
                                "days": len(sym_s),
                                "price_diff":  sym_s.iloc[-1].close - b_price,
                                "buy_price": b_price,
                                "sell_price": sym_s.iloc[-1].close}
                        stats = stats.append(stats_l, ignore_index=True)

                i += 1

        return stats

    
    def find_stocks_to_buy(self, stocks, mail_stats=True, earnings = None, sentiment= None, financials = None, latest_stocks = None):
            out = self.find_stocks_to_buy_full(stocks = stocks, 
                                               mail_stats=mail_stats, 
                                               earnings=earnings, 
                                               sentiment=sentiment, 
                                               financials=financials, 
                                               latest_stocks=latest_stocks)
            if out is not None:
                last = out.iloc[-1]
                last["loss"] ,last["profit"] = FinI.get_nearest_values(price_levels=out.price_level, price= last.close)
                return last
            else:
                return None
            
    @staticmethod           
    def find_stock_to_buy_set_1(stocks:pd.DataFrame,
                                latest_stocks:pd.DataFrame,
                                financials:pd.DataFrame, 
                                sentiment:pd.DataFrame)->bool:
        if len(stocks) > 3 and \
            stocks.iloc[-1]["flpd"] > 0 and \
            (chi.check_pre_sma9(stocks, live_stocks=latest_stocks) or
            chi.check_sma9(stocks, live_stocks = latest_stocks)  or \
            chi.check_boll_sma9_cross(stocks,buy=True)) and \
            chi.check_financials(financials, max_short_ratio=3) and \
            chi.check_sentiment(sentiment) and \
            chi.check_macd(stocks):
            return True
        else:
            return False

    @staticmethod           
    def find_stock_to_buy_set_2(stocks:pd.DataFrame,
                                latest_stocks:pd.DataFrame,
                                financials:pd.DataFrame, 
                                sentiment:pd.DataFrame)->bool:
       
        if len(stocks) > 3 and \
            stocks.iloc[-1]["flpd"] > 0 and \
            chi.check_hammer(stocks) and \
            chi.check_sentiment(sentiment):
            #chi.check_financials(financials, max_short_ratio=3) and \
            return True
        else:
            return False
              
    def find_stocks_to_buy_full(self, stocks, mail_stats=True, earnings = None, sentiment= None, financials = None, latest_stocks = None):
            
        
            if len(stocks) > 0 and stocks.iloc[-1].sym not in self.buyed_symbols:

                stocks.get('change')
                stocks['change'] =  round(stocks['change'], 2)
                stocks = FinI.add_indicators(stocks)
                # stocks.sort_index(ascending = True, inplace = True)
                stocks["flpd"] = Utils.calc_flpd(stocks)
                # print(stocks.iloc[-1]["volume"])

                if financials is None:
                    earnings, sentiment, financials = self.db.get_fundamentals()
                    
                # todo implement days_to_earnings to decision process
                days_to_earnings = FinI.days_to_earnings(earnings)
                
                if latest_stocks is None:
                    latest_stocks = self.load_data(
                        TableName.MIN15,  limit=5)
                # self.check_sma9(stocks, live_stocks = latest_stocks) and \
                    # self.spy.iloc[-1].boll < self.spy.iloc[-1].close and \
                # print(stocks)
                if StockWhisperer.find_stock_to_buy_set_2(stocks, 
                                           latest_stocks,
                                           financials,
                                           sentiment):
                    #  or \
                    # chi.check_rsi(stocks)):
                    # chi.check_candles_in_row(stocks)[0])>=2:
                   
                    # self.bs.buy_stock(stocks.iloc[-1],stocks.iloc[-1].open, table_name = "buy_sell_lt")
                    self.stocks = self.stocks.append(stocks.iloc[-1])
                    sym = stocks.iloc[-1].sym
                    # send only one buy suggestion per day
                    hash_warn = hash(
                        sym + str(datetime.today().strftime("%d/%m/%Y")))
                    
                    if hash_warn not in self.warning_check_list:

                        # sector_mess, spy_mess, vol_mess = self.stock_mess.get_common_mess(
                        # stocks)
                        if mail_stats:
                            self.sm.set_fundamentals(sym)
                            self.sm.set_prices(sym)
                            self.sm.a_mail_sym_stats(sym, "Buy Suggestion")
                            
                        # insert hash with already buyed stock at exact time point and price   
                        self.warning_check_list.append(hash_warn)
                    
                    return stocks
                
                else:
                    # print(stocks.iloc[0].sym)
                    return None
                  
            else:    
                return None     

    # def send_compl_mail(self, curr_price, financials,  earnings, sentiment, days_to_earnings, mess, day_stocks):
    #     sector_mess, spy_mess, vol_mess = self.stock_mess.get_common_mess(
    #         day_stocks)
    #     details = self.stock_mess.get_fund_mess(
    #                         financials,curr_price , earnings, sentiment, days_to_earnings,day_stocks)
    #     details += sector_mess +" \r\n" + spy_mess +" \r\n" + vol_mess
    #     plt = self.stock_mess.show_sym_stats(day_stocks.iloc[0].sym, True)
    #     Utils.send_mm_mail(mess, details, plt)
          
    
    def check_stocks_for_sale(self, symbols = None):
        print('This script goes infinitly, waiting minute to process new check, kill it by ctrl+c')
        check_list = []
        table_name = TableName.MIN15
        table_name_stats = TableName.DAY

        # stock_list = self.db.get_data("buy_sell_lt")
        # stock_list.where( cond = ["sell"] == 0 )
        # print(stock_list)
       
        try:
            while True:
                
                print('Checking ...')
                spy = self.load_data(table_name, "SPY", limit = 2)       
                spy_stats = self.load_data(table_name_stats, "SPY", time_from="120d")
                spy_stats = FinI.add_indicators(spy_stats)
                
                symbols = self.bs.get_buyed_symbols()
             
                #--------------------------spy check index drop under sma20 (boll) send warning    -----------------------
                if chi.check_boll(spy_stats,live_stocks = spy, buy = False):
                    hash_warn = hash(
                        "spy" + str(spy_stats.iloc[-1].boll))
                    
                    if hash_warn not in check_list:
                        mess = "!!! BOLL S&P Warning: boll/price" + \
                            str(spy_stats.iloc[-1].boll) + \
                            " / " + str(spy_stats.iloc[-1].close)
                        print(mess)
                        check_list.append(hash_warn)
                # SMA9 SPY check -------------------------------------------
                if chi.check_sma9(spy_stats, live_stocks=spy, buy=False):
                    hash_warn = hash(
                        "spy" + str(spy_stats.iloc[-1].sma9))
                    
                    if hash_warn not in check_list:
                        mess = "!!! SMA9 S&P Warning sma9/price: " + \
                            str(spy_stats.iloc[-1].sma9) + \
                            " / " + str(spy_stats.iloc[-1].close)
                        # print(mess)
                        Utils.send_mail(mess,mess)
                        check_list.append(hash_warn)
                # ------------------------------- SPY check end -----------------------------
                # self.set_spy_sectors_trend()
                for sym in symbols:
                    mess = ""
                    
                    stocks = self.load_data(table_name, sym, limit = 10)
                    
                    stock_stats = self.load_data(table_name_stats, sym, time_from="120d")
                    stock_stats = FinI.add_indicators(stock_stats)
                    earnings, sentiment, financials = self.db.get_fundamentals(sym)
                    # sector_mess, spy_mess, vol_mess = self.stock_mess.get_common_mess(
                    #     stock_stats)
                    days_to_earnings = FinI.days_to_earnings(earnings)
                   
                    perc_move = []
                    perc_move = round(Utils.calc_perc(stocks.iloc[-2]['open'], stocks.iloc[-1]['close']),2)
                    hash_warn = hash(sym + str(stocks.iloc[-1].close)  + str(stocks.iloc[-1].index))
                    
                    if perc_move < -3 :
                        
                        if hash_warn not in check_list:
                            mess += "PERC Step Down suggestion to sell: " + \
                                str(sym) + " | perc move: " + str(perc_move) + "%: "
                            
                            check_list.append(hash_warn)
                            

                    # if ( stock_stats.iloc[-2].boll_ub_macd < stock_stats.iloc[-2].macd and \
                    #      stock_stats.iloc[-1].boll_ub_macd > stock_stats.iloc[-1].macd):
                    if chi.check_sma9(stock_stats, live_stocks=stocks, buy=False):
                            
                            hash_warn = hash(sym + str(stocks.iloc[-1].close))
                            
                            if hash_warn not in check_list:
                                mess += "!! SMA9 Suggestion to sell: " + str(sym) + " sma9|price: " + \
                                    str(round(stock_stats.iloc[-1]['sma9'], 2)) + " | " + \
                                    str(stocks.iloc[-1])
                                check_list.append(hash_warn)
                             
                    
                    if chi.check_financials(financials, buy=False):
                            
                        hash_warn = hash(sym + str(financials.iloc[-1].shortRatio) + str(financials.iloc[-1].forwardPE))
                        financials = financials.round(2)
                        if hash_warn not in check_list:
                            mess += "FINANCIALS Suggestion to sell: "
                            check_list.append(hash_warn)
                           
                    
                    if chi.check_sentiment(sentiment, buy=False):
                                
                        hash_warn = hash(
                            sym + str(sentiment.iloc[-1].sentiment_summary_avg))
                        
                        
                        if hash_warn not in check_list:
                            mess += "SENTIMENT Suggestion to sell: " + str(sym) + " |  Sentiment" + str(sentiment.iloc[0].sentiment_summary_avg) + \
                            " -> " + str(sentiment.iloc[-1].sentiment_summary_avg) 
                            check_list.append(hash_warn)
                    
                    if chi.check_earnings(earnings, buy=False):
                                
                        hash_warn = hash(
                            sym + str(earnings.iloc[-1].epssurprisepct))
                        
                        if hash_warn not in check_list:
                            mess += "EARNINGS Suggestion to sell: " + str(sym) + \
                               " |  Earnings: " + str(earnings.iloc[-1].epssurprisepct)
                            check_list.append(hash_warn)
                    
                    if days_to_earnings is not None and days_to_earnings.days < 3:
                        hash_warn = hash(
                              sym + str(days_to_earnings.days))

                        if hash_warn not in check_list:
                            mess += "Earnings will be published in " + str(days_to_earnings.days) + " Days: "
                            check_list.append(hash_warn)
                        
                    
                    # fibonachi stepped down
                    # fib_levels = chi.check_fib(stock_stats, live_stocks = stocks)
                    
                    # if  stocks.iloc[-2].close < fib_levels["t1"] < stocks.iloc[-1].close or \
                    #     stocks.iloc[-2].close < fib_levels["t2"] < stocks.iloc[-1].close :
                        
                    #     hash_warn = hash(
                    #         sym + str(stocks.iloc[-1].close))
                        
                    #     if hash_warn not in check_list:
                    #         self.stock_mess.mail_sym_stats(
                    #             sym, "FINANCIALS Suggestion to sell: ")
                    #         mess = "FIB T1/T2 up-level  overstepped : " + str(sym) + \
                    #              " Price/T1/T2: " + \
                    #             str(stocks.iloc[-2].close) + "/" + str(fib_levels["t1"]) + "/" + str(fib_levels["t2"]) +\
                    #             str(stocks.iloc[-1].close) + " Perc: " + \
                    #             str(Utils.calc_perc(
                    #                 fib_levels["c1"], stocks.iloc[-1].close)) + sect_spy_mess
                                      
                    #         print(mess)
                    #         self.send_compl_mail(
                    #             stocks.iloc[-1].close,  financials, earnings, sentiment, days_to_earnings, mess, stock_stats)
                    #         check_list.append(hash_warn)
                    
                    # if  stocks.iloc[-2].close > fib_levels["c1"] > stocks.iloc[-1].close:
                        
                    #     hash_warn = hash(
                    #         sym + str(fib_levels["c1"]))
                        
                    #     if hash_warn not in check_list:
                    #         self.stock_mess.mail_sym_stats(
                    #             sym, "FINANCIALS Suggestion to sell: ")
                    #         mess = "FIB level overstepped warning: " + str(sym) + \
                    #              " Price/Fib/Price: " + \
                    #             str(stocks.iloc[-2].close) + "/" + str(fib_levels["c1"]) + "/" + \
                    #             str(stocks.iloc[-1].close) + " Perc: " + \
                    #             str(Utils.calc_perc(
                    #                 fib_levels["c1"], stocks.iloc[-1].close)) + sect_spy_mess
                                      
                    #         print(mess)
                    #         self.send_compl_mail(
                    #             stocks.iloc[-1].close,  financials, earnings, sentiment, days_to_earnings, mess, stock_stats)
                    #         check_list.append(hash_warn)
                    
                    if  abs(stock_stats.iloc[-1].sma9 - stock_stats.iloc[-1].boll) <= 1:
                        hash_warn = hash(
                            sym + str(stock_stats.iloc[-1].sma9))
                        
                        if hash_warn not in check_list:
                            mess += "SMA9 cross Boll warning: " + str(sym) + \
                                " sma9(-2)/sma9(-1)/boll/Price: " + \
                                str(round(stock_stats.iloc[-2].sma9, 1)) + "/" + str(round(stock_stats.iloc[-1].sma9, 1)) + \
                                "/" + str(round(stock_stats.iloc[-1].boll, 1)) + "/" + str(round(stocks.iloc[-1].close, 1))
                            check_list.append(hash_warn)
                                    
                    if chi.check_boll(stock_stats, live_stocks=stocks, buy=False):
                        
                        hash_warn = hash(
                            sym + str(stocks.iloc[-1].close))
                        
                        if hash_warn not in check_list:
                            mess += "!!! BOLL  level overstepped warning: " + str(sym) + \
                                " Price/boll/Price: " + \
                                str(stocks.iloc[-2].close) + "/" + str(round(stock_stats.iloc[-1].boll,2)) + "/" + \
                                str(stocks.iloc[-1].close) + " Perc: " + \
                                str( Utils.calc_perc(stock_stats.iloc[-1].boll, stocks.iloc[-1].close))
                            check_list.append(hash_warn)
                           
                if mess != "":
                    self.sm.set_fundamentals(sym)
                    self.sm.set_prices(sym)
                    asyncio.run(self.sm.a_mail_sym_stats(
                        sym, mess))
                    
                print('-------------------------Waiting for new data......------------------')
                if table_name == TableName.MIN15:
                    Utils.countdown(900)
                else:
                    Utils.countdown(59)

        except KeyboardInterrupt:
            print("-----------Checking stocks for Sale script: Stopped-----------------")
            
  
    
    
  
            
    # def stock_move_by_weekday(self, sym):
    #     self.price_table_name = TableName.DAY
    #     stocks = self.load_data(symbol=sym)
    #     stocks.index = pd.to_datetime(stocks.index, utc=True)

    #     stocks.get('change')
    #     stocks['change'] = round(stocks['change']*100,2)
    #     # print(stocks.head())
    #     stocks["weekday"] = stocks.index.to_series().dt.dayofweek
    #     stocks['monthday'] = stocks.index.to_series().dt.day
    #     stocks['month'] =  stocks.index.to_series().dt.month
    #     stocks['year'] =  stocks.index.to_series().dt.year
    #     # stocks_weekday = stocks.copy()
    #     stocks_weekday = stocks.groupby(by="weekday").mean()
    #     stocks_month_day = stocks.groupby(by="monthday").mean()
    #     stocks_week_in_month = FinI.add_week_of_month(stocks).groupby(by="week_in_month").mean()
        
        
    #     print(stocks_month_day.change)
    #     print(stocks_weekday.change)
    #     print(stocks_week_in_month.change)
    #     # print(stocks)
    #     self.classify_data(stocks)


                

    def move_against_spy_last_2_prices(self, data, spy_data):
        data_delta = Utils.calc_perc( data[-1].close, data[-2].close)
        spy_data_delta = Utils.calc_perc( spy_data[-1].close, spy_data[-2].close)
        return (data_delta - spy_data_delta) 
    
       
    def multiple_linear_regression(self, data, input_values, row):
        # print("iv: " + str(input_values))
        X = data[1:len(data)][['weekday', 'monthday', 'month','week_in_month']]
        # X = data[1:len(data)][['weekday']] 
        y = data[1:len(data)]['change']
    
        regr = linear_model.LinearRegression()
        regr.fit(X, y)
        predicted_change = regr.predict([[ input_values['weekday'], input_values['monthday'], input_values['month'], input_values['week_in_month'] ]])
        # predicted_change = regr.predict([[ input_values['weekday']]])
        
        
        
        # data = data[data.month.eq(input_values['monthday'])]
        # data = data[data.weekday.eq(input_values['weekday'])]
        # data = data[data.monthday.eq(input_values['month'])]
        # # data = data[data.year.eq(datetime.today().year)]

       
        # print("predicted: "  + str(predicted_change[0]))
        # print(str(row.change) + " | " + str(row.weekday))

        return round(predicted_change[0],3)

    def classify_data(self, data):
        input_values = {}
        deviation = []
        deviation_by_week = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}
        for index, row in data.iterrows():
            input_values['weekday'] = row.weekday
            input_values['monthday'] = row.monthday
            input_values['month'] = row.month
            input_values['week_in_month'] = row.week_in_month
            
            predicted_val = self.multiple_linear_regression(data, input_values, row)
            
            if ~numpy.isnan(row.change):
                deviation.append(abs(row.change - predicted_val))
                deviation_by_week[row.weekday].append(abs(row.change - predicted_val))  
                deviation_by_week[row.weekday+5].append(predicted_val)
                
        for key in deviation_by_week.keys():
            if key < 5:
                index = int(key)
                print(" L.R. Predict deviation " + str(index) + " :" + str(round(sum(deviation_by_week[index])/len(deviation_by_week[index])/100,3)) + " | L.R. pred: " + str(round(sum(deviation_by_week[index+5])/len(deviation_by_week[index+5])/100,3) ))

       
        print("\n General Predict deviation: " + str(round(sum(deviation)/len(deviation)/100,3)) )   

            # plt.scatter(x, y)
            # plt.plot(myline, mymodel(myline))
            # plt.show()

            # plt.hist(data.close, 100, orientation="horizontal")
            # plt.show()   
    # return 
    def volume_movement(self, stocks):
        # stocks = self.load_data(table_name=table_name, limit=2)
        stocks['vm'] = round(Utils.calc_perc(stocks.iloc[0].volume,stocks.iloc[1].volume),2)
        # self.stocks = self.stocks.append(stocks.iloc[0])
      
    def volume_stats(self, stocks_count = 10, mail_stats = False):
        # self.stocks = StockDataFrame()
        self.db.limit = 2
        stocks = self.iterate_by_symbol(self.db.price_table_name, mail_stats, self.volume_movement)
        
        stocks = stocks.groupby(by="sym").mean()
        stocks.sort_values(by="vm", inplace=True, ascending=False)
        print(stocks.head(stocks_count))
        
        stocks.head(stocks_count).plot(
            kind="barh", use_index=True, y="vm", legend=False)
        plt.show()
        for key, value in stocks.head(stocks_count).iterrows():
            self.sm.show_sym_stats(key)
        plt.show()
        
    def find_bottom(self, price_descents_in_row  = 5, table = TableName.MIN15):
        # 1st part -  find sap index long down trend for last 14 days
        spy_stocks = self.db.load_spy(table)
        counter  = 1
        last_value = 10000
        ignore_first_n_rows = 2
        # sequence = True
        data =  temp_data = {"start_price":None,"end_price":None,"start_date":None, "end_date":None}
        # temp_data = {"start_price":None,"end_price":None}
        
        for index, row in spy_stocks.iterrows():
            
            if row.close < last_value:
                # sequence = True
               
                if counter == 1:
                    temp_data = {key: 0 for key in temp_data}
                    temp_data["start_date"] = index
                    temp_data["start_price"] = row.close
                    
                    # print(row)
                if counter >= price_descents_in_row:
                    temp_data["end_date"] = index
                    temp_data["end_price"] = row.close
                    # print(row)
                
                counter += 1
            else:
                counter = 1
                # sequence = False
                temp_data = {key: 0 for key in temp_data}
            if temp_data["end_date"]:
                data = temp_data.copy()  
            last_value = row.close 
        print(data)           
        
        # 2nd part - load other symbols with data between SAP down trend date
        symbols = self.db.get_symbols()
        self.db.time_from = data["start_date"]
        self.db.time_to = data["end_date"]
        stocks = self.load_data(table, symbols=symbols)
        print(stocks)
        # 3rd part - find bottom of symbols, at least for 4 x 15 min candles it doesnt has to drop under first candle value
        candidates = pd.DataFrame()
        for sym in symbols:
            sym_stocks = stocks.loc[stocks['sym'] == sym]
            sym_stocks = Utils.add_first_last_perc_diff(
                sym_stocks.tail(len(sym_stocks)-ignore_first_n_rows))
            # print(sym_stocks.tail(1).loc['flpd'])
            if len(sym_stocks) > 0:
                candidates = candidates.append(sym_stocks.tail(1))
         
        candidates.sort_values(by = "flpd", inplace = True, ascending=False)
        self.plot_prices(candidates.head(10))
        # print(candidates)
     
        
    def show_earning_dates(self, date_from, date_to, symbols = None):

        if date_from is None:
            date_from = "-5d"
        
        if date_to is None:
            date_to = "5d"

        earnings = pd.DataFrame()
        if symbols == None:
            symbols = self.db.get_symbols(TableName.DAY)
        
        # earnings_dates = self.db.get_financials(symbol=None, type="earnings", 
        #                                          date_from=date_from, date_to=date_to)
        # earnings = earnings_dates
        # earnings_dates.sort_values(by = 'epsestimate', ascending = False, inplace= True)
        # print(earnings_dates)
        for sym in symbols:
            # print(sym)
            earnings_dates = self.db.get_financials(sym, type="earnings", 
                                                 date_from=date_from, date_to=date_to)
                                                 
            if len(earnings_dates)>0:
                earnings = earnings.append(earnings_dates)
        print(earnings)
        try:
            earnings.sort_values(by = 'epsestimate', ascending = False, inplace= True)
        except KeyError as e:
            print(e)
            print(earnings)
            
        for sym in earnings.ticker:
           self.sm.show_sym_stats(sym)
        plt.show()
        
        return earnings
   
        
    def get_twitter_sentiment(self, phrase):
        connection = twitterSentiment.API(
            client_key="XbKRpYxd10bJD1DCvg2Gz6WHd", client_secret="G8kBkGu2EBV5xVy9jrY101u85h9lwUD5X9liHcH6hB6EpApGYU")
        search = connection.querySearch(
            phrase, count=1, result_type='recent', lang='en')
        data = twitterSentiment.StructureStatusesData(search)
        sentiment = twitterSentiment.SentimentScore(data.getTweet())
        print(sentiment.getSentimentClassification())
        
    def find_best_short_ratio(self, date_from, date_to, limit=5, asc = True):
        symbols = self.db.get_symbols()
        # print(str(symbols))
        fin_out = pd.DataFrame()
        for sym in symbols:
            fin = self.calc_short_ratio_chng(sym,limit)
            
            if fin is not None:
                fin_out = fin_out.append(fin.iloc[-1])
                
        fin_out.sort_values(by='short_perc', inplace=True, ascending=asc)
        print(fin_out.head(20)[["symbol","short_perc","shortRatio"]])
        
        for index, row in fin_out.head(20).iterrows():
            self.sm.show_sym_stats(row.symbol)
        plt.show()
        return fin_out
    
    def calc_short_ratio_chng(self,sym, limit):
        financials = self.db.get_financials(
                sym, type="financials", limit=limit)
        financials.sort_values(by="date", inplace=True, ascending=True)
        
        if financials is not None and len(financials) > 1 and financials.iloc[0].shortRatio is not None and financials.iloc[-1].shortRatio is not None:
            financials['short_perc'] = Utils.calc_perc(
                financials.iloc[0].shortRatio, financials.iloc[-1].shortRatio)
            # print(str(sym) + " | " + str(financials.iloc[-1].short_perc) + " --- " + str(financials.iloc[0].date) + " -- "+ str(financials.iloc[-1].date))
            return financials
        else:
            return None
                    
