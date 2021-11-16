import sys
sys.path.append('../')

from singleton import Singleton
from datetime import timedelta
import matplotlib.pyplot as plt
from stockstats import StockDataFrame as sdf
import pandas as pd

# from market_app.overview.refreshFinancials import refreshFinancials

import pytz
utc = pytz.UTC
# from buy_sell import BuySell
# from sklearn import linear_model, preprocessing
from pytz import timezone
localtz = timezone('Europe/Prague')
from utils import Utils
from market_db import Database
from plot_i import PlotI
from fin_i import FinI 
from check_indicators import CheckIndicators as chi
from market_db import TableName
import asyncio
import io

# def set_trend_data_dec(func):
#     print("decorator working")
#     def wrapper(*args, **kwargs):
#         # print(args[0].trend_data)
#         tn = kwargs["table_name"] if "table_name" in kwargs and kwargs["table_name"] is not None else TableName.Day
#         if args[0].trend_data is None:
#             args[0].trend_data = sdf.retype(args[0].db.load_data(
#                 table_name = tn, time_from =kwargs["time_from"], time_to =kwargs["time_to"]))
            
#         func(args[0],table_name=tn,
#                 time_from=kwargs["time_from"], time_to=kwargs["time_to"], symbol =kwargs["symbol"])
#         # print(args[0].trend_data)
#     print("get_trend_data_dec()-done")
#     return wrapper

     
class SectorStats():
    def __init__(self):

        self.db = Database()
        self.sectors_trend = []
        self.spy_trend = []
        self.last_date = None
        self.spy = None
        self.stocks = None
        self.in_day_stocks = None
        self.trend_data = None
      
    def show_sectors_stats(self, data, save_image=False, plt = None):
        if plt is None:
            return False
        
        fig, axs = plt.subplots(len(data))
        fig.suptitle('Sector overview')
        # print(axs)
        plt = self.sector_stats_to_plt(plt, data, axs)
        if save_image:
            return plt
        else:
            plt.show()
    
    def sector_stats_to_plt(self, plt, data, axs):
        
        i = 0
        for key in data.keys():
            plotdata = data[key]
            
            # plt.subplots(len(data))
            # axs.bar(plotdata, height = 2)
            plotdata.plot(kind="bar", ax=axs[i])
            
            axs[i].set_title("Year " + str(key))
            axs[i].grid(True)
            # plt.title("Year " + str(key))
            plt.xticks(rotation=45)
            
            plt.xlabel("Sector")
            plt.ylabel("Change")
            axs[i].legend().set_visible(False)
            
            # for i, j in zip(data[key].index, data[key].flpd):
            #     axs[i].annotate(str(j), xy=(i, j))
            i += 1
    
        return plt

    def sectors_day_stats(self, table_name = TableName.DAY, time_from = "-180d", time_to = "0d"):
        # technical, healthcare, financials etc.
        
        sectors = sdf.retype(self.db.load_data(
            table_name, time_from=time_from, time_to=time_to))
        sectors = sdf.retype(sectors)
        # normalize data
        sectors["date"] = sectors.index

        # print(sectors)
        # sectors.get('close_-1_r')
        
        # sectors = Utils.add_first_last_perc_diff(sectors)
        sectors = sectors.groupby(by=["sector","date"]).mean()
        # for item in sectors
        # sectors = sectors.sort_(by = ["sector","date"])
        # print(sectors.where(cond=sectors.index.name == "Basic Materials"))
        res = sdf()
        for index in sectors.index.unique('sector'):
            sec =sectors.iloc[sectors.index.get_level_values('sector') == index]
            sec = sdf.retype(sec)
            sec['close_n'] = (sec.close - sec.close.min()) / (sec.close.max() - sec.close.min())
            sec.get('close_-1_r')
            sec.get('volume_-1_r')
            # print(sec)
            res = res.append(sec)
        # self.show_sec_day_stats(res)
        # print(res)
        return res
        # print(self.stocks.head())  

    
    def get_trend_slice(self, table_name=None, time_from=None, time_to=None, symbol=None, from_db = False):
        
        if time_to is None:
            time_to = "0d"
        if time_from is None:
            time_from="-14d"
            
        # print(time_from + " --- " + time_to)
        if self.trend_data is None or from_db:
            tn = table_name if table_name else TableName.DAY
            self.trend_data = sdf.retype(self.db.load_data(
                table_name=tn, time_from=time_from, time_to=time_to))
            print("--------filling trend_data------------")

        print("get_trend_slice() - done")
        stocks = self.trend_data.copy()

               
        stocks = stocks[stocks.index.to_series().between(self.db.get_date_format(
            time_from) , self.db.get_date_format(time_to))]
        
        # print(str(symbol) + "   SYMBOOOOOL")
        if symbol is not None:
           
            stocks = stocks[stocks.sym == symbol.upper()]
            # print(stocks)
        # print(self.trend_data)
        
        if stocks is not None and len(stocks)> 0:
            return stocks
        else:
            return None
        
    # def get_splited_spy_sectors_uptrends(self, table_name=None, time_from=None, time_to=None):

    #     stocks = self.get_trend_slice(full_stock=True)
    #     df1 = stocks.iloc[:, :round(len(stocks)/2, 0)]
    #     df2 = stocks.iloc[:, round(len(stocks)/2, 0):]
        
    #     self.sectors_trend.append(self.classify_sectors_uptrend(
    #          table_name=table_name, time_from="-14d", time_to="-7d"))
    #     self.sectors_trend.append(self.classify_sectors_uptrend(table_name=table_name, time_from="-7d", time_to="0d"))
        
    #     # ----------------------------set spy trend-----------------------------
        
    #     self.spy_trend.append(self.get_trend_slice(
    #         table_name=table_name, time_from="-14d", time_to="-7d", symbol="SPY"))
    #     self.spy_trend[0] = Utils.add_first_last_perc_diff(self.spy_trend[0])
        
    #     self.spy_trend.append(sdf.retype(self.get_trend_slice(
    #         table_name=table_name, time_from="-7d", time_to="0d", symbol="SPY")))
    #     self.spy_trend[1] = Utils.add_first_last_perc_diff(self.spy_trend[1])
    #     # print(self.spy_trend)
        
    def classify_sectors_uptrend(self, table_name= None, time_from = None, time_to = None, stocks = None, from_db=False, is_industries = False, separator = ""):
        # print(self.trend_data)
        if stocks is None:
            stocks = self.get_trend_slice(
                table_name=table_name, time_from=time_from, time_to=time_to, from_db=from_db)
        # technical, healthcare, financials etc.
        # print(stocks)
        if stocks is not None:
            stocks = Utils.add_first_last_perc_diff(stocks)
            stocks["industry"] = stocks["sector"] + separator + stocks["industry"]
            group_by = "sector" if is_industries is False else "industry"
            stocks = stocks.groupby(by=group_by).mean()
           
            return stocks
        else:
            return stocks

    
       
        # print(self.stocks.head())
                     
    def sectors_uptrend_by_month(self, yf=2017,yt=None, show_chart = True):
        # technical, healthcare, financials etc.
        sectors = self.db.get_sectors()
        sectors_month_stats = pd.DataFrame()
        sec_year_stats = {}
        # print(sectors)
        # self.sectors = [sec]
        stocks = sdf.retype(self.db.load_data(TableName.DAY))
        stocks.index = pd.to_datetime(stocks.index, utc=True)
        stocks['month'] =  stocks.index.to_series().dt.month
        stocks['year'] =  stocks.index.to_series().dt.year
        
        if not yt:
            max_year = stocks.year.max()
        else:
            max_year = yt
        # data contain more sectors since 2016
        for year in range(yf, max_year+1):
            one_year_stocks = stocks.where(cond = stocks['year'] == year).dropna()
            sectors_month_stats = pd.DataFrame()
            # print(one_year_stocks)
           
            min_month = one_year_stocks.month.min()
            max_month = one_year_stocks.month.max()
        
            for month in range(int(min_month), int(max_month)+1):
                one_month_stocks = one_year_stocks.where(
                    cond= one_year_stocks['month'] == month).dropna()
                one_month_stocks = Utils.add_first_last_perc_diff(
                    one_month_stocks)
               
                
                sectors_data = one_month_stocks.groupby(
                    by=one_month_stocks['sector']).mean()
                                   
                   
                sectors_month_stats["sector"] = sectors_data.index
                sectors_month_stats[month] = sectors_data.flpd.tolist()
                sectors_month_stats.set_index('sector', inplace=True)
            
            sec_year_stats[year] = sectors_month_stats.copy()
                
            
        if show_chart:
            self.show_sectors_stats(sec_year_stats, False, plt)
        return sectors_month_stats
     
     
    def set_spy_sectors_trend(self):
        table_name = TableName.DAY
        if self.trend_data is None:
            self.classify_sectors_uptrend(table_name = table_name, time_from = "-14d", time_to = "0")

        df1 = self.trend_data.iloc[ 0:int(round(len(self.trend_data)/2, 0))]
        df2 = self.trend_data.iloc[int(
            round(len(self.trend_data)/2, 0)):len(self.trend_data)]

        # ---------------------------set sectors trend --------------------
        self.sectors_trend.append(self.classify_sectors_uptrend(stocks=df1))
        self.sectors_trend.append(self.classify_sectors_uptrend(stocks=df2))
        
        # ----------------------------set spy trend-----------------------------
        dfs1 = df1[df1.sym == "SPY"]
        dfs2 = df2[df2.sym == "SPY"]
        self.spy_trend.append(Utils.add_first_last_perc_diff(dfs1))
        self.spy_trend.append(Utils.add_first_last_perc_diff(dfs2))
        
        print("set_spy_sectors_trend() - done")
        

# ss =SectorStats()
# ss.set_spy_sectors_trend()
