import twitterSentiment
from matplotlib.dates import date2num
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from stockstats import StockDataFrame as sdf
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import mplfinance as mpf
import math
import sys
import numpy
sys.path.append('../')
import market_src.alpaca2Login as al
from enum import Enum
# from market_app.overview.refreshFinancials import refreshFinancials
import asyncio

import numpy as np
from datetime import datetime, timedelta
import pytz
utc = pytz.UTC
# from alpaca_examples.buy_sell import BuySell
import time
import io
# from sklearn import linear_model, preprocessing
import statsmodels.api as sm
from pytz import timezone
localtz = timezone('Europe/Prague')
from mpl_toolkits.mplot3d import Axes3D
from alpaca_examples.get_data import getData
import ta
from alpaca_examples.utils import Utils
from alpaca_examples.market_db import Database
from alpaca_examples.buy_sell import BuySell
from alpaca_examples.plot_i import PlotI
from alpaca_examples.fin_i import FinI 
from alpaca_examples.alpaca_buy_sell import AlpacaBuySell
from alpaca_examples.check_indicators import CheckIndicators as chi
from alpaca_examples.market_db import TableName

import io


class BackTest():
    def __init__(self):
        
        # self.best_at_settings = None
        self.stocks = sdf()
        self.spy = sdf()
        self.db = Database()
        # self.processed_data = None
        self.db_name = "nyse_financials"
        self.price_table_name =TableName.DAY.to_str()
        self.engine = create_engine('postgresql://postgres:crasher@localhost:5432/'+self.db_name)
        self.df = pd.DataFrame()
        self.classificators= {}
        # start tradining stats
        # self.stock_stats = sdf()
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
        self.buyed_stocks_list = list(pd.read_csv("buyed_stocks.csv"))
        self.bs = BuySell()
        self.alpaca_buy_sell = AlpacaBuySell()
        self.buyed_symbols = list()
        self.warning_check_list = []
                     

    
        
   
    def load_data(self, table_name = None, symbols = None, sectors = None, limit = None, time_from = None, time_to=None):
        symbols = symbols if symbols is not None else self.symbols
        return sdf.retype(self.db.load_data(table_name, symbols=symbols, sectors=sectors, 
                                            limit=limit, time_from=time_from, time_to=time_to))
    
        
    def load_spy(self, table_name = None, time_from = None, time_to=None):
        return sdf.retype(self.db.load_spy(table_name, time_from=time_from, time_to=time_to))
    
    def check_entered_sym(self, symbol):
        # print(symbol)
        if len(symbol) < 1:
            print("Please add stock symbol for simulation. example: py script-name.py TSLA")
            exit()

    # def start_rsi_test(self, table_name, buyed_shares_amount, symbol):
        
        
    #     self.check_entered_sym(symbol)

    #     self.symbols = symbol
    #     self.share_amount = buyed_shares_amount
    #     self.stocks = self.load_data(table_name, symbols=self.symbols)
    #     self.stocks.get('kdjk')
    #     self.stocks = FinI.classify_rsi_oscilator(self.stocks)
    #     self.test_rsi_params()

    #     self.trading_stats(symbol)
       

    # def sectors_day_stats(self, table_name = TableName.DAY, time_from = "180d", time_to = None):
    #     # technical, healthcare, financials etc.
    #     sectors = self.load_data(
    #         table_name, time_from=time_from, time_to=time_to)
    #     sectors = sdf.retype(sectors)
    #     # normalize data
    #     sectors["date"] = sectors.index

    #     # print(sectors)
    #     # sectors.get('close_-1_r')
        
    #     # sectors = Utils.add_first_last_perc_diff(sectors)
    #     sectors = sectors.groupby(by=["sector","date"]).mean()
    #     # for item in sectors
    #     # sectors = sectors.sort_(by = ["sector","date"])
    #     # print(sectors.where(cond=sectors.index.name == "Basic Materials"))
    #     res = sdf()
    #     for index in sectors.index.unique('sector'):
    #         sec =sectors.iloc[sectors.index.get_level_values('sector') == index]
    #         sec = sdf.retype(sec)
    #         sec['close_n'] = (sec.close - sec.close.min()) / (sec.close.max() - sec.close.min())
    #         sec.get('close_-1_r')
    #         sec.get('volume_-1_r')
    #         # print(sec)
    #         res = res.append(sec)
    #     # self.show_sec_day_stats(res)
    #     # print(res)
    #     return res
    #     # print(self.stocks.head())  
    
    # def show_sec_day_stats(self, sectors):
    #     fig, axs = plt.subplots(1, 1, figsize=(16, 4))
    #     # PlotI.set_margins(plt)
    #     PlotI.plot_sector_stats(axs,sectors)
    #     # plt.legend()
    #     plt.show()


    def classify_sectors_uptrend(self, table_name, time_from = None, time_to = None):
        # technical, healthcare, financials etc.
        stocks = self.load_data(table_name, time_from=time_from, time_to = time_to)
        stocks = Utils.add_first_last_perc_diff(stocks)
        stocks = stocks.groupby(by="sector").mean()
        return stocks
        # print(self.stocks.head())
                     
    # def sectors_uptrend_by_month(self, yf=2017,yt=None, show_chart = True):
    #     # technical, healthcare, financials etc.
    #     sectors = self.db.get_sectors()
    #     sectors_month_stats = pd.DataFrame()
    #     sec_year_stats = {}
    #     # print(sectors)
    #     # self.sectors = [sec]
    #     stocks = self.load_data(TableName.DAY)
    #     stocks.index = pd.to_datetime(stocks.index, utc=True)
    #     stocks['month'] =  stocks.index.to_series().dt.month
    #     stocks['year'] =  stocks.index.to_series().dt.year
        
    #     if not yt:
    #         max_year = stocks.year.max()
    #     else:
    #         max_year = yt
    #     # data contain more sectors since 2016
    #     for year in range(yf, max_year+1):
    #         one_year_stocks = stocks.where(cond = stocks['year'] == year).dropna()
    #         sectors_month_stats = pd.DataFrame()
    #         # print(one_year_stocks)
           
    #         min_month = one_year_stocks.month.min()
    #         max_month = one_year_stocks.month.max()
        
    #         for month in range(int(min_month), int(max_month)+1):
    #             one_month_stocks = one_year_stocks.where(
    #                 cond= one_year_stocks['month'] == month).dropna()
    #             one_month_stocks = Utils.add_first_last_perc_diff(
    #                 one_month_stocks)
               
                
    #             sectors_data = one_month_stocks.groupby(
    #                 by=one_month_stocks['sector']).mean()
                                   
                   
    #             sectors_month_stats["sector"] = sectors_data.index
    #             sectors_month_stats[month] = sectors_data.flpd.tolist()
    #             sectors_month_stats.set_index('sector', inplace=True)
            
    #         sec_year_stats[year] = sectors_month_stats.copy()
                
            
    #     if show_chart:
    #         self.show_sectors_stats(sec_year_stats, False, plt)
    #     return sectors_month_stats
     
    # def show_sectors_stats(self, data, save_image=False, plt = None):
    #     if plt is None:
    #         return False
        
    #     fig, axs = plt.subplots(len(data))
    #     fig.suptitle('Sector overview')
    #     print(axs)
    #     plt = self.sector_stats_to_plt(plt, data, axs)
    #     if save_image:
    #         return plt
    #     else:
    #         plt.show()
    
    # def sector_stats_to_plt(self,plt, data, axs):
        
    #     i = 0
    #     for key in data.keys():
    #         plotdata = data[key]
            
    #         # plt.subplots(len(data))
    #         # axs.bar(plotdata, height = 2)
    #         plotdata.plot(kind="bar", ax=axs[i])
            
    #         axs[i].set_title("Year " + str(key))
    #         axs[i].grid(True)
    #         # plt.title("Year " + str(key))
    #         plt.xticks(rotation=45)
            
    #         plt.xlabel("Sector")
    #         plt.ylabel("Change")
    #         axs[i].legend().set_visible(False)
            
    #         # for i, j in zip(data[key].index, data[key].flpd):
    #         #     axs[i].annotate(str(j), xy=(i, j))
    #         i += 1
    
    #     return plt
        
        
    
    # def top_sectors(self, table_name, loosers):
        
    #     self.stocks = self.classify_sectors_uptrend(table_name)
    #     self.stocks = self.stocks.sort_values(by='flpd', ascending=loosers)
    #     print(self.stocks.head())
    #     self.stocks.plot(kind="barh",use_index=True,y="flpd",legend = False)
    #     plt.show()


    # def top_stocks(self, show_stocks_num = 20, from_top = 0, table_name = None, top_losers = True):
      
    #     self.stocks =  self.load_data(table_name = table_name, sectors = self.sectors)
      
    #     self.stocks = FinI.add_change(self.stocks)
    #     self.stocks = Utils.add_first_last_perc_diff(self.stocks)
    #     # print("STOCKS: " + str(self.stocks))
    #     if len(self.stocks) > 0:
    #         stocks = self.stocks.groupby(by="sym").mean()
    #         stocks = stocks.sort_values(by='flpd', ascending=top_losers)
            
    #         top_stocks = stocks.iloc[from_top:(from_top + show_stocks_num)]
    #         # print(top_stocks)
    #         top_stocks.plot(kind="barh", use_index=True, y="flpd", legend = False)
    #         # self.top_stocks_list = top_stocks.index.tolist()
    #         self.draw_chart_values(top_stocks.flpd)
    #         self.plot_prices( top_stocks)
    #     else:
    #         print('No stocks has been found')
    

    # def draw_chart_values(self, data):
    #     for index, value in enumerate(data):
    #         plt.text(value, index, str(round(value,2)))
    #     plt.show()

    
            
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
                self.show_sym_stats(value.sym)
            
            except AttributeError:
                self.show_sym_stats(key)
                
        plt.show() 
        
   
   
    
      
    
       
    def show_sym_stats(self, sym, save_img = False):
        
        self.db.limit = None
        dfp, financials, sentiment, earnings, spy = self.db.get_all_data("160d", sym)
        dfp = FinI.add_indicators(dfp)
        dfp = FinI.add_fib(dfp, last_rows = 10)
        # print(dfp)
        # last 5 financials
        financials = financials.tail(3)
        days_to_earnings = FinI.days_to_earnings(earnings)
        # print(earnings)
        if days_to_earnings is None:
            earnings = None
        plots_num = 4 if save_img else 4
            
        fig, axs = plt.subplots(plots_num, 1, figsize=(16, 18))
        PlotI.set_margins(plt)
        
      
     
        
        PlotI.plot_spy(axs[0], spy.loc[spy.index.isin(dfp.index)])
        axsp=PlotI.plot_stock_prices(axs[0].twinx(), dfp, sym,alpha = 0.3)
        axsp = PlotI.plot_sma(axsp,dfp,["sma9","sma50"])
        axsp = PlotI.plot_candles( dfp, axsp, body_w=0.5, shadow_w=0.1, alpha=0.5)
        axsp = PlotI.plot_fib( dfp, axsp, alpha=0.4)
        axsp = PlotI.plot_fib( dfp, axsp, alpha=0.4, fib_name = "fb_mid")
        axsp = PlotI.plot_fib( dfp, axsp, alpha=0.4, fib_name="fb_bot")
        
        
        PlotI.plot_boll(axsp, dfp, sym)
        PlotI.plot_weeks(axsp, dfp)
       
       
        PlotI.plot_rsi(axs[1], dfp )
        ax_macd = PlotI.plot_macd_boll(axs[1].twinx(), dfp)
        # ax_macd = PlotI.plot_macd(ax_macd,dfp)
        
        PlotI.plot_volume_bars(axs[1].twinx(), dfp)
 
        last_prices = self.load_data(TableName.MIN15, sym, limit = 20)
        # self.plot_volume(axs[2], last_prices)
        ax = PlotI.plot_candlesticks2(axs[2], last_prices)
        last_prices = FinI.add_fib_from_day_df(last_prices, dfp)
        ax = PlotI.plot_fib(last_prices, axs[2], alpha=0.4,fib_name = "fb_mid")
        # PlotI.plot_boll(ax, last_prices, sym)
        sectors = self.sectors_day_stats()
        PlotI.plot_sector_stats(axs[3], sectors, dfp.iloc[0].sector)
        if(save_img):
            return plt
        
      
        
        # self.plot_spy(axs[2], self.spy)
        #self.plot_yahoo_candles(last_prices)
        # self.plot_volume(axs[2], last_prices)
        # set rotation of tick labels
        
        
        axs[3].text(0.02, 0.9, str(financials.beta.name) +
                    ' | ' + str(financials.beta.to_list()), fontsize=8)
        axs[3].text(0.02, 0.8, str(financials.priceToSalesTrailing12Months.name) +
                    ' | ' + str(financials.priceToSalesTrailing12Months.to_list()), fontsize=8)
        axs[3].text(0.02, 0.7, str(financials.enterpriseToRevenue.name) +
                    ' | ' + str(financials.enterpriseToRevenue.to_list()), fontsize=8)
        axs[3].text(0.02, 0.6, str(financials.profitMargins.name) +
                    ' | ' + str(financials.profitMargins.to_list()), fontsize=8)
        axs[3].text(0.02, 0.5, str(financials.enterpriseToEbitda.name) +
                    ' | ' + str(financials.enterpriseToEbitda.to_list()), fontsize=8)
        axs[3].text(0.02, 0.4, str(financials.trailingEps.name) +
                    ' | ' + str(financials.trailingEps.to_list()), fontsize=8)
        axs[3].text(0.02, 0.3,  str(financials.forwardEps.name) +
                    ' | ' + str(financials.forwardEps.to_list()), fontsize=8)
        axs[3].text(0.02, 0.2, str(financials.priceToBook.name) +
                    ' | ' + str(financials.priceToBook.to_list()), fontsize=8)
        axs[3].text(0.02, 0.1, str(financials.bookValue.name) +
                    ' | ' + str(financials.bookValue.to_list()), fontsize=8)
        axs[3].text(0.4, 0.9, str(financials.shortRatio.name) +
                    ' | ' + str(financials.shortRatio.to_list()), fontsize=8)
        axs[3].text(0.4, 0.8,  str(financials.sharesShortPriorMonth.name) +
                    ' | ' + str(financials.sharesShortPriorMonth.to_list()), fontsize=8)
        axs[3].text(0.4, 0.7, str(financials.pegRatio.name) +
                    ' | ' + str(financials.pegRatio.to_list()), fontsize=8)
        axs[3].text(0.4, 0.6, str(financials.earningsQuarterlyGrowth.name) +
                    ' | ' + str(financials.earningsQuarterlyGrowth.to_list()), fontsize=8)
        axs[3].text(0.4, 0.5, str(financials.bid.name) +
                    ' | ' + str(financials.bid.to_list()), fontsize=8)
        axs[3].text(0.4, 0.4, str(financials.trailingPE.name) +
                    ' | ' + str(financials.trailingPE.to_list()), fontsize=8)
        axs[3].text(0.4, 0.3, str(financials.forwardPE.name) +
                    ' | ' + str(financials.forwardPE.to_list()), fontsize=8)
        axs[3].text(0.4, 0.2, str(financials.industry.to_list()) +
                    ' | ' + str(financials.sector.to_list()), fontsize=8)
        axs[3].text(0.4, 0.1, str(financials.heldPercentInstitutions.name) +
                    ' | ' + str(financials.heldPercentInstitutions.to_list()) +
                    ' ||| ' + str(financials.heldPercentInsiders.name) +
                    ' | ' + str(financials.heldPercentInsiders.to_list()), fontsize=8)
        axs[3].text(0.6, 0.9, str(financials.fiftyDayAverage.name) +
                    ' | ' + str(financials.fiftyDayAverage.to_list()), fontsize=8)
        axs[3].text(0.6, 0.7, str("Last CLose Price: ") +
                    ' | ' + str(last_prices.iloc[-1].close), fontsize=8)
        axs[3].text(0.6, 0.5, str("Days to earn.: ") +
                    ' | ' + str(days_to_earnings.days) + " D" if earnings is not None else str("Days to earn.: NaN "), fontsize=8)
        axs[3].text(0.6, 0.4, str("Earn. est. | act. | surp.:  ") +
                    str(earnings.iloc[-1].epsestimate) + ' | ' + str(earnings.iloc[-1].epsactual) + ' | ' + str(earnings.iloc[-1].epssurprisepct) if earnings is not None else str("Earn est.: NaN "), fontsize=8)
        axs[3].text(0.6, 0.3, str("  Sentiment article/title: ") +
                    str(sentiment.sentiment_summary_avg.to_list()) + '/' + str(sentiment.sentiment_title_avg.to_list()) if sentiment is not None and len(sentiment)>0 else str("  Sentiment: NaN "), fontsize=8)

        axs[3].text(0.02, 0.01, str(
            financials.longBusinessSummary.to_list()), fontsize=8)
        
        # self.plot_candlesticks(last_prices)
        # axs[3].plot([2], [1], 'o')
        # plt.text()
        
        # plt.show()
        return plt
        
  
    # def show_fin_earn_price(self, sym):
    #     if sym is None:
    #         print("Please specify symbol")
    #         exit()
        
    #     if type(sym) is str:
    #         self.show_sym_stats(sym)
    #     else:
    #         for s in sym:
    #             self.show_sym_stats(s)
    #     plt.show()
        
    # def prices_to_groups(self, sym):
    #     # fig, axes = plt.subplots(nrows=1, ncols=1)
       
    #     sub_data = self.load_data(table_name = self.price_table_name, symbol = sym)
    #     # prepare data for price profile
    #     prices = sub_data['close'].values
    #     x = list(prices)
    #     prices = ((x-min(x))/(max(x)-min(x))) * 100 +1
    #     # create price profile
    #     values=[]
    #     counters=[]

    #     for cl in prices:
    #         cl= int(cl)
    #         values.append(cl)
    #         counters.append(values.count(cl))

    #     profile_df = pd.DataFrame({'value': values, 'index': counters})
    #     return profile_df

          
    # def plot_price_vol(self, sym):
        
    #     sub_data = self.load_data(table_name = self.price_table_name, symbol = sym)
    #     sub_data.get('volume_delta')
    #     sub_data.get('open_-2_r')
    #     plt.scatter(sub_data.volume, sub_data.close, alpha=0.25, marker='.',)
    #     plt.grid(axis="y")
    #     plt.yticks(np.arange(min(sub_data.close), max(sub_data.close), step=(max(sub_data.close) - min(sub_data.close))/20))
       
    #     plt.twiny()
    #     plt.plot(sub_data.index, sub_data.close, color="r")
    #     plt.legend([sym])
        
    #     plt.twinx()
    #     plt.plot(sub_data.index, sub_data.volume_delta, color="y", alpha=0.2) 
    #     plt.show()
      
        
    def iterate_by_symbol(self, table_name, call_back_f ):
        df_out = sdf()
        if self.symbols:
            symbols = self.symbols
        else:
            symbols = self.db.get_symbols()
        # print(str(symbols))
        for symbol in symbols:
            print("symbol: " + str(symbol))
            print(table_name)
            sub_data = self.load_data(table_name, symbol)
            if len(sub_data) < 1:
                break

            buyed_stocks = call_back_f(sub_data)
            df_out = df_out.append(buyed_stocks)
        return df_out

    # def compare_spy_sym(self, data, comp_const = - 0.2):
        
    #     spy_perc = Utils.calc_perc(self.spy.iloc[0].close, self.spy.iloc[-1].close)
    #     stock_perc = Utils.calc_perc(data.iloc[0].close, data.iloc[-1].close)
    #     spy_stock_comp = stock_perc - spy_perc 
       
    #     if spy_stock_comp >= comp_const:
    #         cache_comp_list = data.iloc[-1]
    #         cache_comp_list['perc'] = stock_perc
    #         cache_comp_list['spy_perc'] = spy_perc
    #         cache_comp_list['f_price'] = data.iloc[0].close
    #         cache_comp_list['spy_stock_comp'] = spy_stock_comp
    #         self.comp_sym_spy = self.comp_sym_spy.append(cache_comp_list)
            

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
    

    
    def set_spy_sectors_trend(self, time_from = None):
        # ---------------------------set sectors trend --------------------
        self.sectors_trend.append(self.classify_sectors_uptrend(TableName.DAY, time_from = (time_from - timedelta(days=14)), time_to = (time_from - timedelta(days=7))))
        self.sectors_trend.append(self.classify_sectors_uptrend(TableName.DAY, time_from = (time_from - timedelta(days=7))))
                   
        # ----------------------------set spy trend-----------------------------
        self.spy_trend.append(self.load_data(table_name=TableName.DAY, symbols=["SPY"],
                                             time_from = time_from - timedelta(days=14), time_to = time_from - timedelta(days=7)))
        self.spy_trend[0] = Utils.add_first_last_perc_diff(self.spy_trend[0])
        
        self.spy_trend.append(self.load_data(table_name=TableName.DAY, symbols=["SPY"],
                                             time_from=time_from - timedelta(days=7)))
        self.spy_trend[1] = Utils.add_first_last_perc_diff(self.spy_trend[1])
        # print(self.spy_trend)
        
    def test_buy_alg(self, stats_time = None):
        
        if stats_time is None:
            stats_time="40d"
        backtest_time = "14d"    
        
        spy = self.load_data(table_name=TableName.DAY, symbols=["SPY"], time_from = stats_time)
        spy = FinI.add_indicators(spy)
        
        stocks_day = self.load_data(table_name=TableName.DAY,  time_from=stats_time)
        stocks_day["sym"] = stocks_day["sym"].astype('category')
        
        spy_15 = self.load_data(table_name=TableName.MIN15, symbols=["SPY"],   time_from=stats_time)
                
        stocks_15 = self.load_data(
            table_name=TableName.MIN15,  time_from=backtest_time)
        stocks_15["sym"] = stocks_15["sym"].astype('category')

        # print(spy)
        symbols = self.db.get_symbols()
        # iterate over days in market 
        spy2 = spy.tail(20)
        for key, row in spy2.iterrows():
          
            # print(key)
            # iterate over symbols each day
            for symbol in symbols:
                # load stocks for stats
               
                # self.set_spy_sectors_trend(key.replace(hour=15, minute=25))
                
                stocks_day_sym = stocks_day[stocks_day["sym"] == symbol]
                stocks_day_sym = FinI.add_indicators(stocks_day_sym)
                stocks_15_sym = stocks_15[stocks_15["sym"] == symbol]
                
                # spy_15 = self.load_data(table_name=TableName.MIN15, symbols=[
                #     "SPY"], time_from=key.replace(hour=15, minute=25),
                #     time_to=key.replace(hour=22, minute=5))

                # stocks_15_sym = stocks_15[stocks_15["sym"] == symbol & stocks_15.index>key.replace(hour=15, minute=25) & stocks_15.index<key.replace(hour=22, minute=5)] 
                # self.load_data(table_name=TableName.MIN15, symbols=[
                #     symbol], time_from=key.replace(hour=15, minute=25),
                #     time_to=key.replace(hour=22, minute=5))
                    
                
                for spy_row15 in spy_15.iterrows():
                    stock_rows15 = stocks_15_sym.loc[stocks_15_sym.index <= spy_row15[0]]
                    
                    if len(stock_rows15) > 2:
                        print(stock_rows15.iloc[-1].sym + " | " + str(stock_rows15.index[-1]))
                        self.buy_alg(stocks_day_sym,stock_rows15)
                    # buying algoritm
                                      
    def buy_alg(self, stocks_day, stocks_15):

        # print(str(stocks_day))
        if len(stocks_day) > 1 and stocks_day.iloc[-1].sym not in self.buyed_symbols:
            stocks_day = FinI.add_indicators(stocks_day)
            stocks_day.get('change')
            stocks_day['change'] =  round(stocks_day['change'],2)
            
            stocks_day.sort_index(ascending= True, inplace = True)
            # print(stocks_day)
            stocks_day["flpd"] = Utils.calc_flpd(stocks_day)
           
            # earnings, sentiment, financials = self.db.get_fundamentals(stocks_day.iloc[-1]["sym"])
            # stocks_15 = self.load_data(TableName.MIN15,  stocks_day.iloc[-1].sym, limit=5)
            # days_to_earnings = FinI.days_to_earnings(earnings)
            
            # self.check_sma9(stocks, live_stocks = latest_stocks) and \
            # self.spy.iloc[-1].boll < self.spy.iloc[-1].close and \
            #   chi.check_financials(financials) and \
            #     chi.check_sentiment(sentiment):
            #    (chi.check_pre_sma9(stocks_day, live_stocks = stocks_15) or \
            #     chi.check_sma9(stocks_day, live_stocks = stocks_15)  or \
            #     chi.check_boll_sma9_cross(stocks_day,buy=True)
            
            if len(stocks_day) > 2 and \
                stocks_day.iloc[-1]["flpd"] > 0 and \
                chi.check_pre_sma9(stocks_day, live_stocks = stocks_15):
              
                hl = self.get_fib_hl(stocks_day,  stocks_day.iloc[-1].close)
                
               
                # self.stocks = self.stocks.append(stocks_day.iloc[-1])
                sym = stocks_day.iloc[-1].sym
                # send only one buy suggestion per day
                hash_warn = hash(
                    sym + str(stocks_day.index[-1].strftime("%d/%m/%Y")))
                  
                if hash_warn not in self.warning_check_list:
                    self.bs.buy_stock(
                        stocks_day.iloc[-1], 
                        stocks_day.iloc[-1].close, 
                        table_name="buy_sell_lt",
                        profit_loss = {"profit":hl["h"],"loss":hl["l"]})
                        
                    Utils.send_mm_mail("Buy: " +
                                       str(stocks_15.iloc[-1].sym) +" | " + str(stocks_15.index[-1]) +
                                       " | "+str(hl), "details test", plt)
                    
                    # sector_mess, spy_mess, vol_mess = self.get_common_mess(
                    # stocks_day) 
                    # mess = "BUY suggestion: " + str(sym) + \
                    #         self.subject_fund_info(financials, sentiment, earnings)
                    # mess += sector_mess
                    # mess +=  spy_mess
                    # mess +=  vol_mess
                    # curr_price = stocks_day.iloc[-1].close
                    # hl = self.get_fib_hl(stocks_day,  stocks_day.iloc[-1].close)
                    # mess += " | Loss: " + str(Utils.calc_perc(curr_price, hl["l"])) + "% | " + "  " + str(hl['l']) +\
                    #     " Price: " + str(curr_price) + \
                    #     " | " + "Profit: " + str(hl['h']) + \
                    #     "  " + str(Utils.calc_perc(curr_price, hl["h"])) + "% "
                    
                    # details = self.get_fund_mess(
                    #     financials, curr_price, earnings, sentiment, days_to_earnings, stocks_day)
                    # print(stocks_15)
                    # mess += "| B_Day: " + \
                    #     str(stocks_day.iloc[-1].date) + \
                    #     " | " + str(stocks_15.index[-1])
                    # plt = self.show_sym_stats(sym,True)
                    
                    # Utils.send_mm_mail(mess, details, plt)
                    self.warning_check_list.append(hash_warn)
                    
                return stocks_day.iloc[-1]
            
            else:
                return None
                
            
        
        else:    
            return None            
            
    
    def prepare_buy_logic(self, infinite = False):
        
        if not self.price_table_name:
            self.price_table_name = TableName.DAY
        
        if not self.time_from:
            self.time_from = "60d"
          
        # maybe this two rows are not necessary  
        self.spy = self.load_data(table_name=TableName.DAY, symbols=["SPY"])
        self.spy = FinI.add_indicators(self.spy)
        
         # CALLBACK
        
        if infinite:
            try:
                while True:
                    self.set_spy_sectors_trend()
                     #GET BUYED STOVKS
                    self.buyed_symbols = self.bs.get_buyed_symbols()
                    self.iterate_by_symbol(self.db.price_table_name, self.find_stocks_to_buy)
                    
            except KeyboardInterrupt:
                print("-----------Checking stocks for Sale script: Stopped-----------------")
        else:
            self.iterate_by_symbol(
                self.db.price_table_name, self.find_stocks_to_buy)
            
    
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

        if not empty_stocks:
            self.plot_stats(self.stocks.iloc[:30], spy_indicator, spy_change)
         
    
    def subject_fund_info(self, financials, sentiment, earnings):
        eps, pe = chi.eps_pe(financials)
        mess = " | ShortR: " + str(financials.iloc[-1].shortRatio)
        mess += " | TrEps/FwdEps: " + str(eps) + "%"
        mess += " | TrPE/FwdPE: " + str(pe) + "%"
        mess += " | 50_DA: " + str(Utils.calc_perc(financials.iloc[0].fiftyDayAverage, financials.iloc[-1].fiftyDayAverage)) + "%"
        if len(earnings)>0:
            mess += str("Earn. est. | act. | surp.:  ") + str(earnings.iloc[-1].epsestimate) + ' | ' + str(earnings.iloc[-1].epsactual) + ' | ' + str(earnings.iloc[-1].epssurprisepct) + "\n\r"
        if len(sentiment) > 0:
            mess += str("Sentiment article/title: ") + str(sentiment.iloc[-1].sentiment_summary_avg) + '/' + str(sentiment.iloc[-1].sentiment_title_avg) if sentiment is not None and len(sentiment)>0 else str("Sentiment: NaN ") + "\n\r" 
        
        return mess
    
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

        
    # def find_stocks_to_buy(self, stocks):
            
        
    #         if len(stocks) > 0 and stocks.iloc[-1].sym not in self.buyed_symbols:

    #             stocks.get('change')
    #             stocks['change'] =  round(stocks['change'],2)
    #             stocks = FinI.add_indicators(stocks)
    #             # stocks.sort_index(ascending = True, inplace = True)
    #             stocks = Utils.add_first_last_perc_diff(stocks)
    #             # print(stocks.iloc[-1]["volume"])
    #             earnings, sentiment, financials = self.db.get_fundamentals(stocks.iloc[-1]["sym"])
    #             latest_stocks = self.load_data(TableName.MIN15,  stocks.iloc[-1].sym, limit=5)
    #             days_to_earnings = FinI.days_to_earnings(earnings)
    #             # self.check_sma9(stocks, live_stocks = latest_stocks) and \
    #                 # self.spy.iloc[-1].boll < self.spy.iloc[-1].close and \
                
    #             if len(stocks) > 2 and \
    #                 stocks.iloc[-1]["flpd"] > 0 and \
    #                 (chi.check_pre_sma9(stocks, live_stocks = latest_stocks) or \
    #                 chi.check_sma9(stocks, live_stocks = latest_stocks)  or \
    #                 chi.check_boll_sma9_cross(stocks,buy=True)) and \
    #                 chi.check_financials(financials) and \
    #                 chi.check_sentiment(sentiment):
                   
    #                 self.bs.buy_stock(stocks.iloc[-1],stocks.iloc[-1].open, table_name = "buy_sell_lt")
    #                 self.stocks = self.stocks.append(stocks.iloc[-1])
    #                 sym = stocks.iloc[-1].sym
    #                 # send only one buy suggestion per day
    #                 hash_warn = hash(
    #                     sym + str(datetime.today().strftime("%d/%m/%Y")))
                    
    #                 if hash_warn not in self.warning_check_list:
    #                     sector_mess, spy_mess, vol_mess = self.get_common_mess(
    #                     stocks) 
    #                     mess = "BUY suggestion: " + str(sym) + \
    #                             self.subject_fund_info(financials, sentiment, earnings)
    #                     mess += sector_mess
    #                     mess +=  spy_mess
    #                     mess +=  vol_mess
    #                     curr_price = stocks.iloc[-1].close
    #                     hl = self.get_fib_hl(stocks,  stocks.iloc[-1].close)
    #                     mess += " | Loss: " + str(Utils.calc_perc(curr_price, hl["l"])) + "% | " + "  " + str(hl['l']) +\
    #                         " Price: " + str(curr_price) + \
    #                         " | " + "Profit: " + str(hl['h']) + \
    #                         "  " + str(Utils.calc_perc(curr_price, hl["h"])) + "% "
                        
    #                     details = self.get_fund_mess(
    #                         financials, curr_price, earnings, sentiment, days_to_earnings, stocks)
                     
    #                     plt = self.show_sym_stats(sym,True)
                        
    #                     Utils.send_mm_mail(mess, details, plt)
    #                     self.warning_check_list.append(hash_warn)
                        
    #                 return stocks.iloc[-1]
                
    #             else:
    #                 return None
                  
                
            
    #         else:    
    #             return None     

    def get_common_mess(self, stocks):
        sector_mess = self.create_sectors_trend_mess(
                        stocks.iloc[0].sector)
        spy_mess = self.create_spy_trend_mess()
        vol_perc = Utils.calc_perc(stocks.iloc[-1].boll, stocks.iloc[-1].boll_ub)
        vol_mess = " | Vlt: " + str(vol_perc) +"% "
        
        return sector_mess, spy_mess, vol_mess
            
                        
        
    def send_compl_mail(self, curr_price, financials,  earnings, sentiment, days_to_earnings, mess, day_stocks):
        sector_mess, spy_mess, vol_mess = self.get_common_mess(
            day_stocks)
        details = self.get_fund_mess(
                            financials,curr_price , earnings, sentiment, days_to_earnings,day_stocks)
        details += sector_mess +" \r\n" + spy_mess +" \r\n" + vol_mess
        plt = self.show_sym_stats(day_stocks.iloc[0].sym, True)
        Utils.send_mm_mail(mess, details, plt)
    
            
    
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
                self.set_spy_sectors_trend()
                for sym in symbols:
                    
                    stocks = self.load_data(table_name, sym, limit = 10)
                    
                    
                    stock_stats = self.load_data(table_name_stats, sym, time_from="120d")
                    stock_stats = FinI.add_indicators(stock_stats)
                    earnings, sentiment, financials = self.db.get_fundamentals(sym)
                    sector_mess, spy_mess, vol_mess = self.get_common_mess(
                        stock_stats)
                    days_to_earnings = FinI.days_to_earnings(earnings)
                    details = self.get_fund_mess(
                        financials, stocks.iloc[-1].close, earnings, sentiment, days_to_earnings,stock_stats)

                    sector_mess = self.create_sectors_trend_mess(
                        stock_stats.iloc[0].sector)
                    spy_mess = self.create_spy_trend_mess()
                    sect_spy_mess = sector_mess + spy_mess
                    # stocks = stocks.round(2)
                    # stock_stats = stock_stats.round(2)
                   
                    # stocks = FinI.add_indicators(stocks)
                    perc_move = []
                    perc_move = round(Utils.calc_perc(stocks.iloc[-2]['open'], stocks.iloc[-1]['close']),2)
                    hash_warn = hash(sym + str(stocks.iloc[-1].close)  + str(stocks.iloc[-1].index))
                    
                    if perc_move < -3 :
                        
                        if hash_warn not in check_list:
                            mess = "PERC Step Down suggestion to sell: " + \
                                str(sym) + " | perc move: " + str(perc_move) + \
                                "%: " + " price:" + \
                                str(stocks.iloc[-1]['close']) + sect_spy_mess
                                
                            print(mess)
                            self.send_compl_mail(
                                stocks.iloc[-1].close,  financials, earnings, sentiment, days_to_earnings, mess, stock_stats)
                            check_list.append(hash_warn)

                    # if ( stock_stats.iloc[-2].boll_ub_macd < stock_stats.iloc[-2].macd and \
                    #      stock_stats.iloc[-1].boll_ub_macd > stock_stats.iloc[-1].macd):
                    if chi.check_sma9(stock_stats, live_stocks=stocks, buy=False):
                            
                            hash_warn = hash(sym + str(stocks.iloc[-1].close))
                            
                            if hash_warn not in check_list:
                                mess = "!! SMA9 Suggestion to sell: " + str(sym) + " sma9\price: " + \
                                    str(round(stock_stats.iloc[-1]['sma9'], 2)) + " \ " + \
                                    str(stocks.iloc[-1]
                                        ['close']) + sect_spy_mess
                                    
                                print(mess)
                                self.send_compl_mail(
                                     stocks.iloc[-1].close,  financials, earnings, sentiment, days_to_earnings, mess, stock_stats)
                                check_list.append(hash_warn)
                    
                    if chi.check_financials(financials, buy=False):
                            
                        hash_warn = hash(sym + str(financials.iloc[-1].shortRatio) + str(financials.iloc[-1].forwardPE))
                        financials = financials.round(2)
                        if hash_warn not in check_list:
                            mess = "FINANCIALS Suggestion to sell: " + str(sym) + " | " + \
                                " ShortRatio: " + str(financials.iloc[0].shortRatio) +" ->" + str(financials.iloc[-1].shortRatio) + \
                                " perc: " + str(Utils.calc_perc(financials.iloc[0].shortRatio, financials.iloc[-1].shortRatio)) + " %" + \
                                " | Trailing PE " + str(financials.iloc[-1].trailingPE) + \
                                " -> Forward PE " + str(financials.iloc[-1].forwardPE) + \
                                            " perc: " + str(Utils.calc_perc(financials.iloc[-1].trailingPE, financials.iloc[-1].forwardPE)) + " %" + \
                                " || First PE " + str(financials.iloc[0].forwardPE) + \
                                " -> Last PE " + str(financials.iloc[-1].forwardPE) + \
                                " perc: " + str(Utils.calc_perc(financials.iloc[0].forwardPE, financials.iloc[-1].forwardPE)) + " %" +\
                                sect_spy_mess
                                
                            print(mess)
                            self.send_compl_mail(
                                stocks.iloc[-1].close,  financials, earnings, sentiment, days_to_earnings, mess, stock_stats)
                            check_list.append(hash_warn)
                    
                    if chi.check_sentiment(sentiment, buy=False):
                        mess = "SENTIMENT Suggestion to sell: " + str(sym) + " |  Sentiment" + str(sentiment.iloc[0].sentiment_summary_avg) + \
                            " -> " + \
                            str(sentiment.iloc[-1].sentiment_summary_avg) + \
                            " < 0.2" + sect_spy_mess
                                
                        hash_warn = hash(
                            sym + str(sentiment.iloc[-1].sentiment_summary_avg))
                        
                        if hash_warn not in check_list:
                            print(mess)
                            self.send_compl_mail(
                                 stocks.iloc[-1].close,  financials, earnings, sentiment, days_to_earnings, mess, stock_stats)
                            check_list.append(hash_warn)
                    
                    if chi.check_earnings(earnings, buy=False):
                        mess = "EARNINGS Suggestion to sell: " + str(sym) + \
                            " |  Earnings: " + \
                            str(earnings.iloc[-1].epssurprisepct) + \
                            sect_spy_mess
                                
                        hash_warn = hash(
                            sym + str(earnings.iloc[-1].epssurprisepct))
                        
                        if hash_warn not in check_list:
                            print(mess)
                            self.send_compl_mail(
                                stocks, financials,  earnings, sentiment, days_to_earnings, mess)
                            check_list.append(hash_warn)
                    
                    if days_to_earnings is not None and days_to_earnings.days < 3:
                        
                        mess = "Earnings will be published in " + str(days_to_earnings.days)+ " Days: "+ str(sym) + " | "  + sect_spy_mess
                        hash_warn = hash(sym + str(stocks.iloc[-1].index))
                        
                        if hash_warn not in check_list:
                            print("EARNINGS DATE warning: " + str(sym) +
                                  "Earnings will be published in " + str(days_to_earnings.days))
                            self.send_compl_mail(
                                stocks.iloc[-1].close,  financials, earnings, sentiment, days_to_earnings, mess, stock_stats)
                            check_list.append(hash_warn)
                    
                    # fibonachi stepped down
                    fib_levels = chi.check_fib(stock_stats, live_stocks = stocks)
                    
                    if  stocks.iloc[-2].close < fib_levels["t1"] < stocks.iloc[-1].close or \
                        stocks.iloc[-2].close < fib_levels["t2"] < stocks.iloc[-1].close :
                        
                        hash_warn = hash(
                            sym + str(stocks.iloc[-1].close))
                        
                        if hash_warn not in check_list:
                            mess = "FIB T1/T2 up-level  overstepped : " + str(sym) + \
                                 " Price/T1/T2: " + \
                                str(stocks.iloc[-2].close) + "/" + str(fib_levels["t1"]) + "/" + str(fib_levels["t2"]) +\
                                str(stocks.iloc[-1].close) + " Perc: " + \
                                str(Utils.calc_perc(
                                    fib_levels["c1"], stocks.iloc[-1].close)) + sect_spy_mess
                                      
                            print(mess)
                            self.send_compl_mail(
                                stocks.iloc[-1].close,  financials, earnings, sentiment, days_to_earnings, mess, stock_stats)
                            check_list.append(hash_warn)
                    
                    if  stocks.iloc[-2].close > fib_levels["c1"] > stocks.iloc[-1].close:
                        
                        hash_warn = hash(
                            sym + str(fib_levels["c1"]))
                        
                        if hash_warn not in check_list:
                            mess = "FIB level overstepped warning: " + str(sym) + \
                                 " Price/Fib/Price: " + \
                                str(stocks.iloc[-2].close) + "/" + str(fib_levels["c1"]) + "/" + \
                                str(stocks.iloc[-1].close) + " Perc: " + \
                                str(Utils.calc_perc(
                                    fib_levels["c1"], stocks.iloc[-1].close)) + sect_spy_mess
                                      
                            print(mess)
                            self.send_compl_mail(
                                stocks.iloc[-1].close,  financials, earnings, sentiment, days_to_earnings, mess, stock_stats)
                            check_list.append(hash_warn)
                    
                    if  abs(stock_stats.iloc[-1].sma9 - stock_stats.iloc[-1].boll) <= 1:
                        hash_warn = hash(
                            sym + str(stock_stats.iloc[-1].sma9))
                        
                        if hash_warn not in check_list:
                            mess = "SMA9 cross Boll warning: " + str(sym) + \
                                 " sma9(-2)/sma9(-1)/boll/Price: " + \
                                str(round(stock_stats.iloc[-2].sma9,1)) + "/" + str(round(stock_stats.iloc[-1].sma9,1)) + \
                                "/" + str(round(stock_stats.iloc[-1].boll,1)) +  "/" + str(round(stocks.iloc[-1].close,1)) + \
                                sect_spy_mess
                                      
                            print(mess)
                            self.send_compl_mail(
                                stocks.iloc[-1].close,  financials, earnings, sentiment, days_to_earnings, mess, stock_stats)
                            check_list.append(hash_warn)
                                    
                    if chi.check_boll(stock_stats, live_stocks=stocks, buy=False):
                        
                        hash_warn = hash(
                            sym + str(stocks.iloc[-1].close))
                        
                        if hash_warn not in check_list:
                            mess = "!!! BOLL  level overstepped warning: " + str(sym) + \
                                " Price/boll/Price: " + \
                                str(stocks.iloc[-2].close) + "/" + str(round(stock_stats.iloc[-1].boll,2)) + "/" + \
                                str(stocks.iloc[-1].close) + " Perc: " + \
                                str( Utils.calc_perc(stock_stats.iloc[-1].boll, stocks.iloc[-1].close)) + \
                                sect_spy_mess
                                      
                            print(mess)
                            self.send_compl_mail(
                                stocks.iloc[-1].close,  financials, earnings, sentiment, days_to_earnings, mess, stock_stats)
                            check_list.append(hash_warn)

                print('-------------------------Waiting for new data......------------------')
                if table_name == TableName.MIN15:
                    Utils.countdown(900)
                else:
                    Utils.countdown(59)

        except KeyboardInterrupt:
            print("-----------Checking stocks for Sale script: Stopped-----------------")
            
    @staticmethod
    def get_fib_mess(df, close_price):
        mess = ""
        last_fib = -1
        for col in df:
            if col.startswith('fb'):
                if close_price > last_fib and close_price < df.iloc[-1][col]:
                    mess += str(Utils.calc_perc(close_price, last_fib)) + "% | " \
                        "Price: " + str(close_price) + \
                        " | " + str(Utils.calc_perc(close_price,
                                                  df.iloc[-1][col])) + "% \n\r"
                
                mess += str(col) + ": " + str(df.iloc[0][col]) + "\n\r"
                
                last_fib = df.iloc[-1][col]
        return mess
    
    @staticmethod
    def get_fib_hl(df, close_price):
        hl = {"h":None, "l":None}
        last_fib = -1
        
        for col in df:
            if col.startswith('fb'):
                splt = col.split("_")
                if close_price > last_fib and close_price < df.iloc[-1][col]:
                    # next col
                    # df.iloc[:,df.columns.get_indexer(col)+1]
                   hl["l"] = round(df.iloc[-1]["fb_" + splt[1] + "_2"],2)
                   hl["h"] = round(df.iloc[-1]["fb_" + splt[1] + "_10"],2)
                last_fib = df.iloc[-1][col]
        return hl
    
               
    def get_fund_mess(self, financials, curr_price, earnings, sentiment, days_to_earnings, day_stocks):

        cols = ["shortRatio",
                "sharesShortPriorMonth",
                "fiftyDayAverage",
                "beta",
                "priceToSalesTrailing12Months",
                "enterpriseToRevenue",
                "profitMargins",
                "enterpriseToEbitda",
                "trailingEps",
                "forwardEps",
                "priceToBook",
                "bookValue",
                "pegRatio",
                "earningsQuarterlyGrowth",
                "bid",
                "trailingPE",
                "forwardPE",
                "heldPercentInstitutions",
                "heldPercentInsiders"]
        
        mess = "T_EPS->F_EPS: " + \
            str(Utils.calc_perc(
                financials.iloc[0].trailingEps, financials.iloc[0].forwardEps)) + "%\n\r"
        mess += "T_PE->F_PE: " + \
            str(Utils.calc_perc(
                financials.iloc[0].trailingPE, financials.iloc[0].forwardPE)) + "%\n\r"
        for item in cols:
            
            # financials[item].dropna(inplace=True)
            
            first = financials.iloc[0][item] 
            last = financials.iloc[-1][item]
            mess += item +": " +\
                str(first) + " -> " + \
                str(last) + " | " + \
                str(Utils.calc_perc(
                    financials.iloc[0][item], financials.iloc[-1][item])) + "%\n\r"
  
        mess += str(financials.iloc[0].industry) +   ' | ' + str(financials.iloc[0].sector) + "\n\r" + \
        str("Current Price: ") + ' | ' + str(curr_price) + "\n\r"
        mess += str("Days to earn.: ") + ' | ' + str(days_to_earnings) + " D" + "\n\r" 
        if len(earnings)>0:
            mess += str("Earn. est. | act. | surp.:  ") + str(earnings.iloc[-1].epsestimate) + \
                ' | ' + str(earnings.iloc[-1].epsactual) + \
                ' | ' + str(earnings.iloc[-1].epssurprisepct) + "\n\r"
        if len(sentiment) > 0:
            mess += str("Sentiment article/title: ") + str(sentiment.sentiment_summary_avg.to_list()) + \
                '/' + str(sentiment.sentiment_title_avg.to_list()) if sentiment is not None and len(sentiment)>0 else str("Sentiment: NaN ") + "\n\r" 
        if len(financials) > 0:
            mess += str( financials.iloc[0].longBusinessSummary) + "\n\r"
        
        sector_mess, spy_mess, vol_mess = self.get_common_mess(
                        day_stocks)   
        mess += "\n\r" + sector_mess
        mess += "\n\r" + spy_mess
        mess += "\n\r" + vol_mess
        
        hl = self.get_fib_hl(day_stocks, curr_price)
        mess += "\n\r" + "Loss: " + str(Utils.calc_perc(curr_price, hl["l"])) + "% | " + "  " + str(hl['l']) +\
            " Price: " + str(curr_price) + \
            " | " + "Profit: " + str(hl['h']) + \
            "  " + str(Utils.calc_perc(curr_price, hl["h"])) + "% \n\r"
        
        mess += BackTest.get_fib_mess(day_stocks, curr_price) + "\n\r" 
        
        return mess
    
  
            
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
      
    def volume_stats(self, stocks_count = 10):
        # self.stocks = sdf()
        self.db.limit = 2
        stocks = self.iterate_by_symbol(self.db.price_table_name, self.volume_movement)
        
  
        
        stocks = stocks.groupby(by="sym").mean()
        stocks.sort_values(by="vm", inplace=True, ascending=False)
        print(stocks.head(stocks_count))
        
        stocks.head(stocks_count).plot(
            kind="barh", use_index=True, y="vm", legend=False)
        plt.show()
        for key, value in stocks.head(stocks_count).iterrows():
            self.show_sym_stats(key)
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
            date_from = "2d"
        
        if date_to is None:
            date_to = "3d"

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
           self.show_sym_stats(sym)
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
            self.show_sym_stats(row.symbol)
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
                    
