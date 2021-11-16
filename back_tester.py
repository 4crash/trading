from datetime import timedelta
import twitterSentiment
import matplotlib.pyplot as plt
from stockstats import StockDataFrame as sdf
import pandas as pd
# from sqlalchemy import create_engine
import sys
import numpy
sys.path.append('../')
import alpaca2Login as al
# from market_app.overview.refreshFinancials import refreshFinancials

import pytz
utc = pytz.UTC
# from buy_sell import BuySell
# from sklearn import linear_model, preprocessing
from pytz import timezone
localtz = timezone('Europe/Prague')
from utils import Utils
from market_db import Database
from buy_sell import BuySell
from fin_i import FinI 
from alpaca_buy_sell import AlpacaBuySell
from check_indicators import CheckIndicators as chi
from market_db import TableName
from stock_mess import StockMess
import asyncio

class BackTest():
    def __init__(self):
        
        # self.best_at_settings = None
        self.stocks = sdf()
        self.spy = sdf()
        self.db = Database()
        # self.processed_data = None
        # self.db_name = "nyse_financials"
        self.price_table_name =TableName.DAY.to_str()
        # self.engine = create_engine('postgresql://postgres:crasher@localhost:5432/'+self.db_name)
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
        self.sm = StockMess()
                     

    
        
   
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
            

    # def classify_sap_index(self, stocks=None):
    #     buy_indicator = None
        
    #     stocks = FinI.add_indicators(stocks)
    #     if stocks is not None and len(stocks) > 0:

    #         if stocks.iloc[-1]["close"] < stocks.iloc[-1]['boll_lb'] or \
    #             stocks.iloc[-1]["close"] < stocks.iloc[-1]['boll']:
            
    #             buy_indicator = 20

    #         elif stocks.iloc[-1]["close"] > stocks.iloc[-1]['boll'] and \
    #             stocks.iloc[-1]["close"] > stocks.iloc[-1]['boll_mid_ub']:
                
    #             buy_indicator = 35

    #         elif  stocks.iloc[-1]["close"] > stocks.iloc[-1]['boll'] and \
    #             stocks.iloc[-1]["close"] < stocks.iloc[-1]['boll_mid_ub']:

    #             buy_indicator = 70

    #         elif  stocks.iloc[-1]["close"] > stocks.iloc[-1]['boll_lb'] :
                
    #             buy_indicator = 80

    #         elif  stocks.iloc[-1]["close"] > stocks.iloc[-1]['boll_mid_lb']:
    #             buy_indicator = 90
            

    #         if  stocks.iloc[-1]['open_-2_r'] < 0:
    #             buy_indicator -= 50
            

    #     return buy_indicator

    # def sap_moving(self, table_name = None):
        
    #     stocks =  self.db.load_spy(table_name)
    #     stocks = FinI.add_indicators(stocks)

    #     if stocks.iloc[-1]["close"] < stocks.iloc[-1]['boll_lb'] or \
    #        stocks.iloc[-1]["close"] < stocks.iloc[-1]['boll'] and stocks.iloc[-1]['open_-2_r'] < 0:
           
    #        buy_indicator = 20

    #     elif stocks.iloc[-1]["close"] > stocks.iloc[-1]['boll'] and \
    #         stocks.iloc[-1]["close"] < stocks.iloc[-1]['boll_mid_ub'] and \
    #         stocks.iloc[-1]['open_-2_r'] > 0:
            
    #         buy_indicator = 70

    #     elif  stocks.iloc[-1]["close"] > stocks.iloc[-1]['boll_lb'] and \
    #         stocks.iloc[-1]['open_-2_r'] > 0:
            
    #         buy_indicator = 80

    #     elif  stocks.iloc[-1]["close"] > stocks.iloc[-1]['boll_mid_lb'] and \
    #         stocks.iloc[-1]['open_-2_r'] > 0:
            
    #         buy_indicator = 90

    #     return buy_indicator
    

    
    # def set_spy_sectors_trend(self, time_from = None):
    #     # ---------------------------set sectors trend --------------------
    #     self.sectors_trend.append(self.classify_sectors_uptrend(TableName.DAY, time_from = (time_from - timedelta(days=14)), time_to = (time_from - timedelta(days=7))))
    #     self.sectors_trend.append(self.classify_sectors_uptrend(TableName.DAY, time_from = (time_from - timedelta(days=7))))
                   
    #     # ----------------------------set spy trend-----------------------------
    #     self.spy_trend.append(self.load_data(table_name=TableName.DAY, symbols=["SPY"],
    #                                          time_from = time_from - timedelta(days=14), time_to = time_from - timedelta(days=7)))
    #     self.spy_trend[0] = Utils.add_first_last_perc_diff(self.spy_trend[0])
        
    #     self.spy_trend.append(self.load_data(table_name=TableName.DAY, symbols=["SPY"],
    #                                          time_from=time_from - timedelta(days=7)))
    #     self.spy_trend[1] = Utils.add_first_last_perc_diff(self.spy_trend[1])
    #     # print(self.spy_trend)
    
    def set_mess_data(self, fin = None,sen = None,earn = None,spy = None,stc = None):
        if earn is not None:
            self.sm.earnings = earn
        if fin is not None:
            self.sm.financials = fin
        if sen is not None:
            self.sm.sentiment = sen
        if spy is not None:
            self.sm.spy = spy
        # if stc_d is not None:
        #     self.sm.in_day_stocks
        if stc is not None:
            self.sm.stocks = stc

    def test_buy_alg(self, stats_time = None):
        
        if stats_time is None:
            stats_time="-60d"
        backtest_time = "-14d"    
        
        self.db.last_date = self.db.get_date_format(backtest_time)
        spy = self.load_data(table_name=TableName.DAY, symbols=["SPY"], time_from = stats_time)
        spy = FinI.add_indicators(spy)
        
        stocks_day = self.load_data(table_name=TableName.DAY,  time_from=stats_time)
        stocks_day["sym"] = stocks_day["sym"].astype('category')
        
        spy_15 = self.load_data(table_name=TableName.MIN15, symbols=["SPY"],   time_from=backtest_time)
                
        stocks_15 = self.load_data(table_name=TableName.MIN15,  time_from=backtest_time)
        stocks_15["sym"] = stocks_15["sym"].astype('category')

        # print(spy)
        symbols = self.db.get_symbols()
        # iterate over days in market 
        spy2 = spy.tail(20)
        print(spy2.sma9)
        for index, spy_row_day in spy2.iterrows():
          
            for index, spy_row15 in spy_15.iterrows():
                
                #check all buyed symbols for each time
                
                # symbols = ["NVDA","GPRO","PLUG","JKS","IMGN","FCX","GE","GM","TIGR"]
                # symbols = ["PLUG", "FCX","GPRO"]
                # iterate over symbols each day
                for symbol in symbols:
                    # load stocks for stats
                    stocks_day_sym = stocks_day[stocks_day["sym"] == symbol]
                    stocks_15_sym = stocks_15[stocks_15["sym"] == symbol]
                    stock_rows15 = stocks_15_sym.loc[stocks_15_sym.index <= index]
                    # print(stock_rows15.iloc[-1].sym + " | " + str(stock_rows15.index[-1]))
                                        
                    if len(stock_rows15) > 1:
                        print(stock_rows15.iloc[-1].sym + " | " + str(stock_rows15.index[-1]))
                        self.buy_alg_test(
                            stocks_day_sym, stock_rows15, spy, spy_row15)
                        # buying algoritm

        
                                         
    def buy_alg_test(self, stocks_day, stocks_15, spy, spy_row15):
        
        # self.stocks = self.stocks.append(stocks_day.iloc[-1])
        self.db.last_date = stocks_15.iloc[-1].name
        print(self.db.last_date)
        sym = stocks_day.iloc[-1].sym
        # send only one buy suggestion per day
        hash_warn = hash(
            sym + str(stocks_15.index[-1].strftime("%d/%m/%Y")))
        
        hl = FinI.get_fib_hl(stocks_day,  stocks_day.iloc[-1].close)
        
        # print(str(stocks_day))
        if len(stocks_day) > 1 :
            
            stocks_day = FinI.add_indicators(stocks_day)
            stocks_day.sort_index(ascending= True, inplace = True)
            stocks_day["flpd"] = Utils.calc_flpd(stocks_day)
            hl = FinI.get_fib_hl(stocks_day, stocks_15.iloc[-1].close)
            #OLD CHECK SELL moved to this Fce
            # self.check_sell(stocks_15.iloc[-1])
            
            earnings, sentiment, financials = self.db.get_fundamentals(
                stocks_day.iloc[-1]["sym"])
            self.set_mess_data(fin=financials, sen=sentiment, earn=earnings, spy=spy, stc=stocks_day)
           
            
            
            # print(spy_row_day.sma9)
            if len(self.bs.buy_sell_open) > 0:
                print("not selled stocks:" + str(self.bs.buy_sell_open[self.bs.buy_sell_open.state == "open"]))
                print("Current SYM: " + str(stocks_15.iloc[-1].sym))
            
            
            #-----------------------------SELL------------------------------------------
            if len(self.bs.buy_sell_open) > 0:
                bs = self.bs.buy_sell_open[(self.bs.buy_sell_open.sym == stocks_15.iloc[-1].sym) & (
                    self.bs.buy_sell_open.state == "open")]
                
                if len(bs)> 0:
                    print("--------------------SELLING----------------------------")
                    for index, row in bs.iterrows():
                        if Utils.calc_perc(stocks_15.iloc[-1].close, hl["h"], 2) <= 1 or \
                            chi.check_sma9(stocks=stocks_day, live_stocks=stocks_15, buy=False):
                            
                            self.bs.buy_sell_open[(self.bs.buy_sell_open.sym == stocks_15.iloc[-1].sym) & (self.bs.buy_sell_open.state == "open")] = self.bs.sell_stock_t(stocks_15.iloc[-1],  price=stocks_15.iloc[-1].close,
                                        table_name="buy_sell_bt",  qty=1, buyed_stock = bs, sell_date=stocks_15.iloc[-1].name)
                                
                            asyncio.run(self.sm.a_mail_sym_stats(sym, "Selling Profit: " + str(stocks_15.iloc[-1].sym) + " | " + str(stocks_15.index[-1]) + " | " + \
                                            " | B.S.:" + str(row["buy"]) + "/" + str(stocks_15.iloc[-1].close) + \
                                            " | " + str(Utils.calc_perc(row["buy"], stocks_15.iloc[-1].close)) +
                                                    "% | " + str(stocks_15.iloc[-1].name) +
                                                    " | ", stocks_15.iloc[-1].name),debug = True)
            
            #-------------------------------------BUY------------------------------
            if hash_warn not in self.warning_check_list \
                and stocks_day.iloc[-1]["flpd"] > 0 \
                and chi.check_financials(financials) \
                and (len(self.bs.buy_sell_open) == 0 or stocks_15.iloc[-1].sym not in self.bs.buy_sell_open[self.bs.buy_sell_open.state == "open"].sym)  \
                and chi.check_pre_sma9(stocks_day, live_stocks = stocks_15):
                

                # and chi.check_sentiment(sentiment) \
                # and chi.check_financials(financials) \
                    
                self.warning_check_list.append(hash_warn)
                self.bs.buy_stock_t(
                    stocks_15.iloc[-1],
                    stocks_15.iloc[-1].close,
                    table_name="buy_sell_lt",
                    profit_loss = {"profit":hl["h"], "loss":hl["l"]})
                
                print("---------------------------BUYiNG-------------------------------")
                print(self.bs.buy_sell_open)
                self.sm.mail_sym_stats(sym, "Buy: " +
                                    str(stocks_15.iloc[-1].sym) +" | " + str(stocks_15.index[-1]) + " | " + \
                                    str(stocks_15.iloc[-1].close) + " | " + \
                    str(hl), stocks_15.iloc[-1].name)
            
            
           
                    
     
    def buy_alg(self, stocks_day, stocks_15, spy_row_day, spy_row15):
        # set last date(date to) for query data
        self.db.last_date = stocks_15.index[-1]
        # self.stocks = self.stocks.append(stocks_day.iloc[-1])
        sym = stocks_day.iloc[-1].sym
        # send only one buy suggestion per day
        hash_warn = hash(
            sym + str(stocks_15.index[-1].strftime("%d/%m/%Y")))
        
        # print(str(stocks_day))
        if len(stocks_day) > 1 and stocks_day.iloc[-1].sym not in self.buyed_symbols:
            
            stocks_day = FinI.add_indicators(stocks_day)
            stocks_day.sort_index(ascending= True, inplace = True)
            stocks_day["flpd"] = Utils.calc_flpd(stocks_day)
            #OLD CHECK SELL moved to this Fce
            # self.check_sell(stocks_15.iloc[-1])
            
            earnings, sentiment, financials = self.db.get_fundamentals(stocks_day.iloc[-1]["sym"])
            # stocks_15 = self.load_data(TableName.MIN15,  stocks_day.iloc[-1].sym, limit=5)
            # days_to_earnings = FinI.days_to_earnings(earnings)
            
            # self.check_sma9(stocks, live_stocks = latest_stocks) and \
            # self.spy.iloc[-1].boll < self.spy.iloc[-1].close and \
            #   chi.check_financials(financials) and \
            #     chi.check_sentiment(sentiment):
            #    (chi.check_pre_sma9(stocks_day, live_stocks = stocks_15) or \
            #     chi.check_sma9(stocks_day, live_stocks = stocks_15)  or \
            #     chi.check_boll_sma9_cross(stocks_day,buy=True)
            
            # print(spy_row_day.sma9)
            if len(self.bs.buy_sell_open) > 0:
                print("not selled stocks:" + str(self.bs.buy_sell_open[self.bs.buy_sell_open.state == "open"]))
                print("C SYM" + str(stocks_15.iloc[-1].sym))
                
            #BUY
            if len(stocks_day) > 2 and \
                (len(self.bs.buy_sell_open) == 0 or stocks_15.iloc[-1].sym not in self.bs.buy_sell_open[self.bs.buy_sell_open.state == "open"].sym)  and \
                stocks_day.iloc[-1]["flpd"] > 0 and \
                chi.check_pre_sma9(stocks_day, live_stocks = stocks_15) and \
                chi.check_financials(financials) and \
                chi.check_sentiment(sentiment):
              
                hl = FinI.get_fib_hl(stocks_day,  stocks_day.iloc[-1].close)
                
               
                  
                if hash_warn not in self.warning_check_list:
                    self.bs.buy_stock_t(
                        stocks_15.iloc[-1],
                        stocks_15.iloc[-1].close,
                        table_name="buy_sell_lt",
                        profit_loss = {"profit":hl["h"], "loss":hl["l"]})
                    
                    self.warning_check_list.append(hash_warn)
                    print("BUYiNG")
                    print(self.bs.buy_sell_open)
                    asyncio.run(self.sm.a_mail_sym_stats(sym, "Buy: " +
                                       str(stocks_15.iloc[-1].sym) +" | " + str(stocks_15.index[-1]) + " | " + \
                                        str(stocks_15.iloc[-1].close) + " | " + \
                                        str(hl), stocks_15.iloc[-1].name))
                    
                
        
        #SELL
        if len(self.bs.buy_sell_open) > 0 and hash_warn not in self.warning_check_list:
            
            print(stocks_15.iloc[-1].sym)
            bs = self.bs.buy_sell_open[(self.bs.buy_sell_open.sym == stocks_15.iloc[-1].sym) & (
                self.bs.buy_sell_open.state == "open")]
            print(bs)
            print(" ---------------------------- ")
            # exit()
            # print(stock_15)
            
            if len(bs) > 0:
                
                for index, row in bs.iterrows():
                    if ("est_profit" in row and row["est_profit"] and Utils.calc_perc(stocks_15.iloc[-1].close, row["est_profit"], 2) <= 2) or \
                            chi.check_sma9(stocks=stocks_day, live_stocks=stocks_15, buy=False):
                        
                        self.bs.buy_sell_open[(self.bs.buy_sell_open.sym == stocks_15.iloc[-1].sym) & (self.bs.buy_sell_open.state == "open")] = self.bs.sell_stock_t(stocks_15.iloc[-1],  price=stocks_15.iloc[-1].close,
                                table_name="buy_sell_bt",  qty=1, buyed_stock = bs, sell_date=stocks_15.iloc[-1].name)
                        
                        asyncio.run(self.sm.a_mail_sym_stats(sym, "Selling Profit: " + str(stocks_15.iloc[-1].sym) + " | " + str(stocks_15.index[-1]) + " | " + " | B.S.:" + str(row["buy"]) + "/" + str(row["est_profit"]) + " | " + str(Utils.calc_perc(row["buy"], row["est_profit"])) +
                                                "% | " + str(stocks_15.iloc[-1].name) +
                                                             " | ", stocks_15.index[-1]))
                    
                # if chi.check_pre_sma9(stocks_day, live_stocks=stocks_15) and \:
                    
                    
                    # sector_mess, spy_mess, vol_mess = self.get_common_mess(
                    # stocks_day) 
                    # mess = "BUY suggestion: " + str(sym) + \
                    #         self.subject_fund_info(financials, sentiment, earnings)
                    # mess += sector_mess
                    # mess +=  spy_mess
                    # mess +=  vol_mess
                    # curr_price = stocks_day.iloc[-1].close
                    # hl = FinI.get_fib_hl(stocks_day,  stocks_day.iloc[-1].close)
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
                    # self.warning_check_list.append(hash_warn)
                
                
                
                return stocks_day.iloc[-1]
            
            else:
                return None
                
            
        
        else:    
            return None 
     
            
    def check_sell(self, stock_15):
        # TODO  select buysell row by symbol and by open position 
        if len(self.bs.buy_sell_open) > 0:
            
            bs = self.bs.buy_sell_open[(self.bs.buy_sell_open.sym == stock_15.sym) & (self.bs.buy_sell_open.state == "open")]
            # print(bs)
            # print(" ---------------------------- ")
            # print(stock_15)
            if len(bs) > 0:
                
                for index, row in bs.iterrows():
                
                    if "est_profit" in row and Utils.calc_perc(stock_15.close,row["est_profit"],2) <= -1 :
                        self.bs.buy_sell_open[(self.bs.buy_sell_open.sym == stock_15.sym) & (self.bs.buy_sell_open.state == "open")] = self.bs.sell_stock_t(stock_15,  price=stock_15.close,
                                table_name="buy_sell_bt",  qty=1, buyed_stock = bs, sell_date=stock_15.name)
                        
                        Utils.send_mm_mail("Selling Profit: " + str(stock_15.sym) + str(stock_15.index) + " | " + " | B.S.:" + str(row["buy"]) + "/" + str(row["est_profit"]) + " | " + str(Utils.calc_perc(row["buy"], row["est_profit"])) +
                                        "% | " + str(stock_15.name) + \
                                        " | ", "details test", None)
                    
                # if "est_loss" in row and row["est_loss"] > stock_15.close:
                #     self.bs.buy_sell[(self.bs.buy_sell.sym == stock_15.sym) & (self.bs.buy_sell.state == "open")] = self.bs.sell_stock_t(stock_15,  sell_price=stock_15.close,
                #                           table_name="buy_sell_bt",  shares_amount=1, buyed_stock=bs, sell_date=stock_15.name)
                    
                #     # print(bs)
                #     Utils.send_mm_mail("Selling loss: " + str(stock_15.sym) + "  | B.S.:" + str(row["buy"]) + "/" + str(row["est_loss"]) + " | " + str(Utils.calc_perc(row["buy"], row["est_loss"])) +
                #                         "% | " + str(stock_15.name) + \
                #                         " | ", "details test", None)
            
        
    # def prepare_buy_logic(self, infinite = False):
        
    #     if not self.price_table_name:
    #         self.price_table_name = TableName.DAY
        
    #     if not self.time_from:
    #         self.time_from = "60d"
          
    #     # maybe this two rows are not necessary  
    #     self.spy = self.load_data(table_name=TableName.DAY, symbols=["SPY"])
    #     self.spy = FinI.add_indicators(self.spy)
        
    #      # CALLBACK
        
    #     if infinite:
    #         try:
    #             while True:
    #                 self.set_spy_sectors_trend()
    #                  #GET BUYED STOVKS
    #                 self.buyed_symbols = self.bs.get_buyed_symbols()
    #                 self.iterate_by_symbol(self.db.price_table_name, self.find_stocks_to_buy)
                    
    #         except KeyboardInterrupt:
    #             print("-----------Checking stocks for Sale script: Stopped-----------------")
    #     else:
    #         self.iterate_by_symbol(
    #             self.db.price_table_name, self.find_stocks_to_buy)
            
    
    #     # self.find_stock_to_buy()
    #     empty_stocks = True
    #     spy_change = 0
    #     spy_stocks = None
    #     if self.stocks is not None and len(self.stocks) > 0:
    #         self.stocks = self.stocks.groupby(by='sym').mean()
    #         self.stocks = self.stocks.sort_values(by='open_-2_r', ascending=False)
        

    #         spy_stocks = self.load_spy(self.db.price_table_name)
    #         spy_indicator = self.classify_sap_index(spy_stocks)
    #         empty_stocks = False
    #     else:
    #         print("No stocks to buy. DONE...")

    #     if spy_stocks is not None and len(spy_stocks) > 1:
    #         spy_change = Utils.calc_perc(
    #             spy_stocks.iloc[-3].close, spy_stocks.iloc[-1].close)
    #         print("Buy indicator by S&P index 0-100: " + str(spy_indicator))
    #         print("S&P move: " + str(spy_change) + "%")

    #     if not empty_stocks:
    #         self.plot_stats(self.stocks.iloc[:30], spy_indicator, spy_change)
         
    
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

    def classify_sectors(self, time_from = "-7d", table_name = "p_day", loosers = False):
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
    #                     hl = FinI.get_fib_hl(stocks,  stocks.iloc[-1].close)
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
                spy_stats = self.load_data(table_name_stats, "SPY", time_from="-120d")
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
                    
                    
                    stock_stats = self.load_data(table_name_stats, sym, time_from="-120d")
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
        
        hl = FinI.get_fib_hl(day_stocks, curr_price)
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
                    
