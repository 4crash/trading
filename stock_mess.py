
import sys
sys.path.append('../')

from alpaca_examples.sector_stats import SectorStats
from alpaca_examples.singleton import Singleton
from datetime import timedelta
import matplotlib.pyplot as plt
from stockstats import StockDataFrame as sdf
import pandas as pd

# from market_app.overview.refreshFinancials import refreshFinancials

import pytz
utc = pytz.UTC
# from alpaca_examples.buy_sell import BuySell
# from sklearn import linear_model, preprocessing
from pytz import timezone
localtz = timezone('Europe/Prague')
from alpaca_examples.utils import Utils
from alpaca_examples.market_db import Database
from alpaca_examples.plot_i import PlotI
from alpaca_examples.fin_i import FinI 
from alpaca_examples.check_indicators import CheckIndicators as chi
from alpaca_examples.market_db import TableName
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
     
class StockMess():
    def __init__(self):

        self.db = Database()
        self.ss = SectorStats()
        
        self.last_date = None
        self.financials = None
        self.earnings = None
        self.sentiment = None
        self.spy = None
        self.stocks = None
        self.in_day_stocks = None
        self.trend_data = None
      
       
     
          
        
    async def mail_stats(self, data, subject, last_date = None, spy = None):
        # print(data)
        if  spy is None:
            spy = sdf.retype(self.db.load_data(
                TableName.DAY, ["spy"], time_from="-60d", time_to="0d"))
            
        for key, value in data.iterrows():
            
           
            if "flpd" in value:
                    subject_flpd = subject +" " + str(round(value.flpd,2)) + "% | "
                  
            try:
                sym = value.sym    
            
            except AttributeError:
                sym = key 
            
            self.set_fundamentals(sym = sym)
            self.set_prices(sym = sym)
            self.mail_sym_stats(sym, subject_flpd,
                                    last_date=last_date)

    def a_mail_sym_stats(self, sym, subject, last_date = None):
        self.mail_sym_stats(sym, subject, last_date=None)
              
    def mail_sym_stats(self, sym, subject, last_date = None):
        
        # self.db.last_date = last_date
        # if self.spy is None:
        #     self.spy = sdf.retype(self.db.load_data(
        #         TableName.DAY, ["spy"], time_from="-60d", time_to="0d"))
            
        # self.stocks = sdf.retype(self.db.load_data(TableName.DAY, sym, time_from="-60d", time_to = "0d"))
        
        # self.stocks = FinI.add_indicators(self.stocks)
        # time_to = last_date
        # self.earnings, self.sentiment, self.financials = self.db.get_fundamentals(self.stocks.iloc[-1]["sym"], 
        #                                     tf={"e": "-5d", 
        #                                         "s": "-20d", 
        #                                         "f":  "-14d"}, 
        #                                     tt={"e": "30d", "s": "0d", "f": "0d"})
        
        mess, curr_price,days_to_earnings = self.get_subj_mess( subject, sym=sym)
        
       
        
        details = self.get_fund_mess(
            self.financials, curr_price, self.earnings, self.sentiment, days_to_earnings, self.stocks)
        
        plt = self.show_sym_stats(sym, True)
        
        Utils.send_mm_mail(mess, details, plt) 
    
    def get_subj_mess(self, subject, sym = None):
       
        mess=""
        
        if self.stocks is None or sym != self.stocks.iloc[0].sym:
            self.stocks = self.ss.get_trend_slice(
                 table_name=None, time_from="-120d", time_to="0d", symbol=sym)
       
        curr_price = self.stocks.iloc[-1].close
        
        # print(self.stocks)
        
        days_to_earnings = FinI.days_to_earnings(self.earnings)
        sector_mess, spy_mess, vol_mess = self.get_common_mess(
                        self.stocks)
        
        mess = subject + " " + str(sym) + \
            self.subject_fund_info(self.financials, self.sentiment,
                                   self.earnings, days_to_earnings = days_to_earnings)
        
        mess += str(sector_mess)
        mess +=  str(spy_mess)
        mess +=  str(vol_mess)
            
            
        self.stocks = FinI.add_levels(self.stocks)
        # hl = FinI.get_fib_hl(self.stocks,  self.stocks.iloc[-1].close)
        pl = self.stocks.price_level.dropna()
        low,high = FinI.get_nearest_values(pl, curr_price)
        mess +=  " Price: " + str(curr_price)
        mess += " | Loss: " + str(Utils.calc_perc(curr_price, low[0])) + "%, " if low is not None and len(low)>0 else ""
        mess += " " + "Prof.: " + str(Utils.calc_perc(curr_price,
                                       high[0])) + "% " if high is not None and len(high) > 0 else " | "
        
     
            
        print("get_subj_mess() - done")
        return mess, curr_price, days_to_earnings
        
    
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
                
                mess += str(col) + ": " + str(round(df.iloc[0][col],2)) + "\n\r"
                
                last_fib = df.iloc[-1][col]
        return mess
    
       
    def show_sym_stats(self, sym, save_img = False ):
        
        # last  financials
        if self.financials is not None and len(self.financials) > 0:
            self.financials = self.financials.tail(5)
        days_to_earnings = FinI.days_to_earnings(self.earnings)
        # print(earnings)
        if days_to_earnings is None:
            self.earnings = None
        plots_num = 4 if save_img else 5
            
        fig, axs = plt.subplots(plots_num, 1, figsize=(16, 18))
        PlotI.set_margins(plt)
        
        if self.spy is not None:
            PlotI.plot_spy(axs[0], self.spy.loc[self.spy.index.isin(self.stocks.index)])
            
        axsp=PlotI.plot_stock_prices(axs[0].twinx(), self.stocks, sym,alpha = 0.3)
        axsp = PlotI.plot_sma(axsp,self.stocks,["sma9","sma50"])
        axsp = PlotI.plot_candles( self.stocks, axsp, body_w=0.5, shadow_w=0.1, alpha=0.5)
        axsp = PlotI.plot_fib( self.stocks, axsp, alpha=0.4)
        axsp = PlotI.plot_fib( self.stocks, axsp, alpha=0.4, fib_name = "fb_mid")
        axsp = PlotI.plot_fib( self.stocks, axsp, alpha=0.4, fib_name="fb_bot")
        
        
        PlotI.plot_boll(axsp, self.stocks, sym)
        PlotI.plot_weeks(axsp, self.stocks)
       
       
        PlotI.plot_rsi(axs[1], self.stocks )
        ax_macd = PlotI.plot_macd_boll(axs[1].twinx(), self.stocks)
        # ax_macd = PlotI.plot_macd(ax_macd,dfp)
        
        PlotI.plot_volume_bars(axs[1].twinx(), self.stocks)
 
        
        if self.in_day_stocks is None:
            self.in_day_stocks = sdf.retype(self.db.load_data(
                TableName.MIN15, sym, time_from="-1d", time_to="0d"))
            if len(self.in_day_stocks) < 1:
                self.in_day_stocks = sdf.retype(self.db.load_data(
                    TableName.MIN15, sym, limit=20))
                
        # self.plot_volume(axs[2], last_prices)

        ax = PlotI.plot_candlesticks2(axs[2], self.in_day_stocks)
        self.in_day_stocks = FinI.add_fib_from_day_df(self.in_day_stocks, self.stocks)
        ax = PlotI.plot_fib(self.in_day_stocks, axs[2], alpha=0.4,fib_name = "fb_mid")
        # PlotI.plot_boll(ax, last_prices, sym)
        
        sectors = self.ss.sectors_day_stats()
        PlotI.plot_sector_stats(axs[3], sectors, self.stocks.iloc[0].sector)
        if(save_img):
            return plt
        
      
        
        # self.plot_spy(axs[2], self.spy)
        #self.plot_yahoo_candles(last_prices)
        # self.plot_volume(axs[2], last_prices)
        # set rotation of tick labels
        
        
        axs[3].text(0.02, 0.9, str(self.financials.beta.name) +
                    ' | ' + str(self.financials.beta.to_list()), fontsize=8)
        axs[3].text(0.02, 0.8, str(self.financials.priceToSalesTrailing12Months.name) +
                    ' | ' + str(self.financials.priceToSalesTrailing12Months.to_list()), fontsize=8)
        axs[3].text(0.02, 0.7, str(self.financials.enterpriseToRevenue.name) +
                    ' | ' + str(self.financials.enterpriseToRevenue.to_list()), fontsize=8)
        axs[3].text(0.02, 0.6, str(self.financials.profitMargins.name) +
                    ' | ' + str(self.financials.profitMargins.to_list()), fontsize=8)
        axs[3].text(0.02, 0.5, str(self.financials.enterpriseToEbitda.name) +
                    ' | ' + str(self.financials.enterpriseToEbitda.to_list()), fontsize=8)
        axs[3].text(0.02, 0.4, str(self.financials.trailingEps.name) +
                    ' | ' + str(self.financials.trailingEps.to_list()), fontsize=8)
        axs[3].text(0.02, 0.3,  str(self.financials.forwardEps.name) +
                    ' | ' + str(self.financials.forwardEps.to_list()), fontsize=8)
        axs[3].text(0.02, 0.2, str(self.financials.priceToBook.name) +
                    ' | ' + str(self.financials.priceToBook.to_list()), fontsize=8)
        axs[3].text(0.02, 0.1, str(self.financials.bookValue.name) +
                    ' | ' + str(self.financials.bookValue.to_list()), fontsize=8)
        axs[3].text(0.4, 0.9, str(self.financials.shortRatio.name) +
                    ' | ' + str(self.financials.shortRatio.to_list()), fontsize=8)
        axs[3].text(0.4, 0.8, str(self.financials.sharesShortPriorMonth.name) +
                    ' | ' + str(self.financials.sharesShortPriorMonth.to_list()), fontsize=8)
        axs[3].text(0.4, 0.7, str(self.financials.pegRatio.name) +
                    ' | ' + str(self.financials.pegRatio.to_list()), fontsize=8)
        axs[3].text(0.4, 0.6, str(self.financials.earningsQuarterlyGrowth.name) +
                    ' | ' + str(self.financials.earningsQuarterlyGrowth.to_list()), fontsize=8)
        axs[3].text(0.4, 0.5, str(self.financials.bid.name) +
                    ' | ' + str(self.financials.bid.to_list()), fontsize=8)
        axs[3].text(0.4, 0.4, str(self.financials.trailingPE.name) +
                    ' | ' + str(self.financials.trailingPE.to_list()), fontsize=8)
        axs[3].text(0.4, 0.3, str(self.financials.forwardPE.name) +
                    ' | ' + str(self.financials.forwardPE.to_list()), fontsize=8)
        axs[3].text(0.4, 0.2, str(self.financials.industry.to_list()) +
                    ' | ' + str(self.financials.sector.to_list()), fontsize=8)
        axs[3].text(0.4, 0.1, str(self.financials.heldPercentInstitutions.name) +
                    ' | ' + str(self.financials.heldPercentInstitutions.to_list()) +
                    ' ||| ' + str(self.financials.heldPercentInsiders.name) +
                    ' | ' + str(self.financials.heldPercentInsiders.to_list()), fontsize=8)
        axs[3].text(0.6, 0.9, str(self.financials.fiftyDayAverage.name) +
                    ' | ' + str(self.financials.fiftyDayAverage.to_list()), fontsize=8)
        axs[3].text(0.6, 0.7, str("Last CLose Price: ") +
                    ' | ' + str(self.in_day_stocks.iloc[-1].close), fontsize=8)
        axs[3].text(0.6, 0.5, str("Days to earn.: ") +
                    ' | ' + str(days_to_earnings.days) + " D" if self.earnings is not None else str("Days to earn.: NaN "), fontsize=8)
        axs[3].text(0.6, 0.4, str("Earn. est. | act. | surp.:  ") +
                    str(self.earnings.iloc[-1].epsestimate) + ' | ' + str(self.earnings.iloc[-1].epsactual) + ' | ' + str(self.earnings.iloc[-1].epssurprisepct) if self.earnings is not None else str("Earn est.: NaN "), fontsize=8)
        axs[3].text(0.6, 0.3, str("  Sentiment article/title: ") +
                    str(self.sentiment.sentiment_summary_avg.to_list()) + '/' + str(self.sentiment.sentiment_title_avg.to_list()) if self.sentiment is not None and len(self.sentiment)>0 else str("  Sentiment: NaN "), fontsize=8)

        axs[3].text(0.02, 0.01, str(
            self.financials.longBusinessSummary.to_list()), fontsize=8)
        
        # self.plot_candlesticks(last_prices)
        # axs[3].plot([2], [1], 'o')
        # plt.text()
        
        # plt.show()
        return plt

    
    
    def subject_fund_info(self, financials, sentiment, earnings, days_to_earnings = None):
        
        mess=""
        if len(financials) > 0:
            eps, pe = chi.eps_pe(financials)
            mess += ''.join([" | ShortR: ", str(financials.iloc[-1].shortRatio),
                        " | TrEps/FwdEps: ", str(eps) + "%", 
                        " | TrPE/FwdPE: ", str(pe) + "%", 
                        " | 50_DA_move: ", str(Utils.calc_perc(financials.iloc[0].fiftyDayAverage, financials.iloc[-1].fiftyDayAverage)) + "%", 
                        " | Beta: ", str(financials.iloc[0].beta), 
                        " | PriceToBook: ", str(financials.iloc[0].priceToBook), 
                        " | DividRate: ", str(financials.iloc[0].dividendRate)])
            
                                                                 
        if earnings is not None and len(earnings)>0:
            mess += ''.join([str(" | Earn. est./act./surp.:  ") , 
                        str(earnings.iloc[-1].epsestimate) , '/' , 
                        str(earnings.iloc[-1].epsactual) + '/' ,
                        str(earnings.iloc[-1].epssurprisepct), ""])
        if days_to_earnings is not None:
            days_to_earnings = str(days_to_earnings).split(",")
            mess += ''.join([" | Earn :", str(days_to_earnings[0])])
        if sentiment is not None and len(sentiment) > 0:
            mess += ''.join([ str(" | Sentiment article/title: "),
                       str(sentiment.iloc[-1].sentiment_summary_avg),
                       '/',
                       str(sentiment.iloc[-1].sentiment_title_avg) if sentiment is not None and len(sentiment)>0 else str("Sentiment: NaN ")
                       ])
        
        return mess
    
    # def plot_sectors(self, plt):
    #     data = self.sectors_uptrend_by_month(yf=2021,yt=2021, show_chart = False)  
    #     plt = self.sector_stats_to_plt(self,plt, data)
    #     return plt

    # def classify_sectors(self, time_from = "7d", table_name = "p_day", loosers = False):
    #     stocks = self.classify_sectors_uptrend(table_name)
    #     stocks = stocks.sort_values(by='flpd', ascending=loosers)
    #     return stocks
    
    def create_sectors_trend_mess(self, sector):
        
        sector_mess = ""
        
        # if self.sectors_trend is None and len(self.sectors_trend) < 1:
        self.ss.set_spy_sectors_trend()
        
        
        sector0 = self.ss.sectors_trend[0][self.ss.sectors_trend[0].index == sector]
        sector1 = self.ss.sectors_trend[1][self.ss.sectors_trend[1].index == sector]
    
    
        if len(sector0) > 0 and len(sector1)>0:
            sector_mess = " | " + str(sector) + ": " + str(round(sector0.iloc[0].flpd,1)) + "% -> " +  str(round(sector1.iloc[0].flpd,1))+ "%"
        print("create_sectors_trend_mess() - done")
        
        return sector_mess          
    
    def create_spy_trend_mess(self):
        spy_mess = ""
        if self.ss.spy_trend is not None and \
            self.ss.spy_trend[0] is not None and \
            len(self.ss.spy_trend) > 2 and \
            len(self.ss.spy_trend[0]) > 0 and \
            len(self.ss.spy_trend[1]) > 0:
                
            print(self.ss.spy_trend[0])
            spy_mess = " | SPY: " + str(round(self.ss.spy_trend[0].flpd.mean(),2)) + "% -> " + str(round(self.ss.spy_trend[1].flpd.mean(),2))+ "%"
        else:
            spy_mess = "No Data"
        return spy_mess


    def get_common_mess(self, stocks):
        # print(stocks.iloc[0].sector)
        vol_mess = vol_perc = spy_mess = sector_mess = ""
        if stocks is not None and len(stocks) > 0:
            sector_mess = self.create_sectors_trend_mess(
                        stocks.iloc[0].sector)

        if stocks is not None and "boll" in stocks:
            vol_perc = Utils.calc_perc(stocks.iloc[-1].boll, stocks.iloc[-1].boll_ub)
            vol_mess = " | Vlt: " + str(vol_perc) + "% "
        else:
            vol_perc = 0
           
        spy_mess = self.create_spy_trend_mess()
           
        print("get_common_mess() - done")
        return sector_mess, spy_mess, vol_mess
            

               
    def get_fund_mess(self, financials, curr_price, earnings, sentiment, days_to_earnings, day_stocks = None):
        mess = "" 
        mess += ''.join("Detail | ")
        financials.sort_values(by=["date"], inplace=True, ascending = True)
        if len(financials) > 0:
            cols = ["date",
                    "shortRatio",
                    "sharesShortPriorMonth",
                    "fiftyDayAverage",
                    "beta",
                    "dividendRate",
                    "exDividendDate",
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
                    "volume",
                    "trailingPE",
                    "forwardPE",
                    "heldPercentInstitutions",
                    "heldPercentInsiders"]
            
            mess += ''.join(["T_EPS->F_EPS: ",
                str(Utils.calc_perc(
                    financials.iloc[0].trailingEps, financials.iloc[0].forwardEps)) , "%\n\r"
             ,"T_PE->F_PE: ",
                str(Utils.calc_perc(
                    financials.iloc[0].trailingPE, financials.iloc[0].forwardPE)) + "%\n\r"])
            
            for item in cols:
                
                # financials[item].dropna(inplace=True)
                
                first = financials.iloc[0][item] 
                last = financials.iloc[-1][item]
                
                mess += ''.join ([item +": ",
                    str(Utils.human_format(first)) , " -> " ,
                    str(Utils.human_format(last)) , " | " ,
                    str(Utils.calc_perc(first, last)) ,  "%\n\r",])

            mess += ''.join ([str(financials.iloc[0].sector) ,
                ' | ', str(financials.iloc[0].industry),"\n\r",
                str("Current Price: ") ,' | ' ,
                str(curr_price),"\n\r",
                str("Days to earn.: ") , ' | ' ,
                str(days_to_earnings) , " D" , "\n\r" ])
                
        if earnings is not None and len(earnings)>0:
            mess += str("Earn. est. | act. | surp.:  ") + str(earnings.iloc[-1].epsestimate) + \
                ' | ' + str(earnings.iloc[-1].epsactual) + \
                ' | ' + str(earnings.iloc[-1].epssurprisepct) + "\n\r"
        if sentiment is not None and  len(sentiment) > 0:
            mess += str("Sentiment article/title: ") + str(sentiment.sentiment_summary_avg.to_list()) + \
                '/' + str(sentiment.sentiment_title_avg.to_list()) if sentiment is not None and len(sentiment)>0 else str("Sentiment: NaN ") + "\n\r" 
        if financials is not None and len(financials) > 0:
            mess += str( financials.iloc[0].longBusinessSummary) + "\n\r"
        
        sector_mess, spy_mess, vol_mess = self.get_common_mess(
                        self.stocks)   
        mess += "\n\r" + sector_mess
        mess += "\n\r" + spy_mess
        mess += "\n\r" + vol_mess
        
        hl = FinI.get_fib_hl(self.stocks, curr_price)
        mess += "\n\r" + "Loss: " + str(Utils.calc_perc(curr_price, hl["l"])) + "% " + "  " + str(hl['l']) +" | "+\
            " Price: " + str(curr_price) + \
            " | " + "Profit: " + str(hl['h']) + \
            "  " + str(Utils.calc_perc(curr_price, hl["h"])) + "% \n\r"
        
        mess += self.get_fib_mess(self.stocks, curr_price) + "\n\r" 
        print("get_fund_mess() - done")
        return mess
    
    def set_fundamentals(self, sym, last_date=None, tf={"e": "-5d", "s": "-20d", "f": "-30d"}, tt={"e": "14d", "s": "2d", "f": "0d"}):
        
        if last_date:
            self.db.last_date = last_date
        print("set_fund_mess() - done")
        self.earnings,  self.sentiment, self.financials = self.db.get_fundamentals(sym=sym,tf=tf, tt=tt)
  
    def set_prices(self,sym=None, last_date=None, time_from="-90d", time_to="0d" ):
        if last_date:
            self.db.last_date = last_date
            
        self.in_day_stocks = sdf.retype(self.db.get_last_n_days(
           sym))
        
        self.stocks = sdf.retype(self.db.load_data(
            TableName.DAY, sym, time_from=time_from, time_to=time_to))
        
        self.stocks = FinI.add_indicators(
            sdf.retype(self.stocks))

        self.stocks = FinI.add_fib(self.stocks, last_rows=10)
        
        self.spy = sdf.retype(self.db.load_data(
            TableName.DAY, ["SPY"], time_from=time_from, time_to=time_to))
        print("set_prices() - done")
       
