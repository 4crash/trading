
from alpaca_trade_api.entity_v2 import BarsV2, Bar
from alpaca_trade_api.rest import TimeFrame, TimeFrameUnit
from asyncio.log import logger
import pandas as pd
# import PostgresConnect
from sqlalchemy import create_engine
import sys
sys.path.append('../')
import alpaca2Login as al
# from alpaca_examples.back_tester import BackTest
from utils import Utils
from market_db import Database, TableName
from datetime import datetime, timedelta
import pytz

  
utc=pytz.UTC


class getData():
    def __init__(self):
        self.engine = create_engine('postgresql://postgres:crasher@localhost:5432/nyse_financials')
        self.api = al.alpaca2Login().getApi()
        self.fin_db = "financials"
        self.earnings_db = "earning_dates"
        self.sentiment_db = "sentiment"
        self.nothingToGet = True
        self.db = Database()
        # df = pd.DataFrame()
        

    def  get_bars(self, symbol, interval, limit, after)-> BarsV2:
        # print(after.isoformat())
        # print(self.api.get_bars)
        timeframe = TimeFrame(15,TimeFrameUnit.Minute)
        data = self.api.get_bars(symbol = symbol,timeframe= timeframe,limit = limit, start=after.isoformat())
        # data =  data.df[symbol]
        # print(data[symbol])
        return data
        
    def get_sentiment(self, days_before=0, days_after=30):
        from stocknews import StockNews
        self.get_sentiment_from_csv()
        symbols = self.db.get_symbols()
        # print(symbols)
        sn = StockNews(symbols, wt_key='0d57e2849d73713e95f395c7440380ff')
        df_sum, r_count = sn.summarize()
        print("rows: " + str(r_count))
        self.save_sentiment(df_sum)
        print("get sentiment ends")
        # print(df_new)
        return df_sum
    
    def get_sentiment_from_csv(self):
        df = pd.read_csv("./data/data.csv", sep=";")
        print(df)
        self.save_sentiment(df)
    
    def save_sentiment(self, df_sum):
        """[summary]
         !!!! WARNING this method ends with exiting whole script
        Args:
            df_sum ([type]): [description]
        """
        try:
            dfdb = pd.read_sql(self.sentiment_db, con=self.engine)
            df_new = df_sum.append(dfdb, ignore_index=True)
        except:
            df_new = df_sum
            print("Database sentiment doesnt exists")
            
        try:    
            df_new.drop(columns=['index','open','close','high','low','volume','change'], inplace=True)
        except KeyError :
            print("some columns to drop hasnt been found")
            
        df_new.drop_duplicates(
            subset=["id","stock", "news_dt"], inplace=True, keep="first")
        
        df_new.to_sql(self.sentiment_db, con=self.engine,
                      if_exists='replace', index=True)
        
        print("Sentiment saved")
        exit()
        # return df_new
    
    def get_earnings(self, days_before = 0, days_after = 30):
        from yahoo_earnings_calendar import YahooEarningsCalendar
      
        dfdb = pd.read_sql(self.earnings_db, con=self.engine)
        yec = YahooEarningsCalendar()
        dfd = pd.DataFrame(yec.earnings_between(
            datetime.now() - timedelta(days=days_before), datetime.now() + timedelta(days=days_after)))
        
        df_new = dfd.append(dfdb, ignore_index=True)
        df_new.drop_duplicates(subset=["ticker","startdatetime","startdatetimetype"], inplace=True, keep="first")
        #save DATAFRAME to database
        df_new.to_sql(self.earnings_db, con=self.engine,
                   if_exists='replace', index=False)
        print(df_new)

        return df_new
    # def convertUnixTimeToDate(columns,df):
    #     for col in columns:
    #         df[col] = pd.to_datetime(df[col],unit="s")
    #     return df

    def fill_database(self, dfs,interval, limit, table_name, after):
        self.nothingToGet = True
        for row in dfs['Symbol']:
            if row:
                row = str.strip(row)
                print(row)
                
                # check last date for specific symbol
                dfp = pd.read_sql('select MAX("index") as last_time from '+ table_name + ' where sym = \'' + row + '\' ', con=self.engine)
                
                if dfp.iloc[0].last_time is not None:
                    after = dfp.iloc[0].last_time

                # get data from ALPACA
                bars = self.get_bars(row, interval, limit, after)
                # print(bars)
                df = pd.DataFrame(bars._raw)
                # print(df)
                if len(df) > 0:
                    self.nothingToGet = False
                   
                    # add symbol
                    df["sym"] = row
                    df.rename(columns={"t":'index',
                                       "o": 'open',
                                       "h": 'high',
                                       "l": 'low',
                                       "c":'close',
                                       "v": 'volume'},
                                       inplace=True)
                    # df.o.rename('open', inplace=True)
                    # df.h.rename('high', inplace=True)
                    # df.l.rename('low', inplace=True)
                    # df.c.rename('close', inplace=True)
                    # df.v.rename('volume', inplace=True)
                    df.drop('vw', inplace=True, axis=1)
                    df.drop('n', inplace=True, axis=1)
                    df.set_index('index', inplace=True)
                    
                    # df.drop('level_0', inplace=True, axis=1)
                    print(df)
                    
                   
                    #add sector
                    # dff = pd.read_sql('select sector from '+ self.fin_db + ' where symbol = \'' + row + '\' and sector is not null limit 1', con=self.engine)
                    # if len(dff) > 0 and dff.iloc[0].sector:
                    #     df["sector"] = dff.iloc[0].sector
                    # logger.info(df)
                 
                   
                  
                    #save DATAFRAME to database
                    df.to_sql(table_name, con=self.engine, if_exists='append', index = True)
                    if table_name == TableName.MIN15:
                        self.save_bar_as_last_in_day(df)

    def get_last_working_day(self):
        
        offset = max(0, (datetime.today().weekday() + 6) % 7 - 3)
        timedelta2 = timedelta(offset)
        # most_recent = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta2
        most_recent = datetime.today() - timedelta2
        return most_recent

    def save_bar_as_last_in_day(self,last_bar:pd.DataFrame):
        last_day = self.db.get_last_n_days(
            sym=last_bar.iloc[-1].sym, n_days_back=1, table=TableName.DAY)
        last_day.iloc[-1].close = last_bar.iloc[-1].close
        last_day.iloc[-1].high = last_day.iloc[-1].high if last_day.iloc[-1].high > last_bar.iloc[-1].high else last_bar.iloc[-1].high
        last_day.iloc[-1].low = last_day.iloc[-1].low if last_day.iloc[-1].low < last_bar.iloc[-1].low else last_bar.iloc[-1].low

        last_day.to_sql(TableName.DAY, con=self.engine,
                       if_exists='replace', index=False)
    
    # param interval could be 1,5,15, 0 - for day
    def start_download(self, min_interval, start_date, infinity = False):
        
        dfs = pd.read_csv("./datasets/RevolutStock.csv", delimiter="|")
        print(start_date);
        min_interval = int(min_interval)

        if min_interval > 0:
            db_name = 'p_'+ str(min_interval) +'min'
        else:
            db_name = 'p_day'
        
        while True:
           
            try:
                if min_interval > 0:
                    self.fill_database(dfs,str(min_interval) + "Min",1000, 'p_'+ str(min_interval) +'min', start_date)
                else:
                    self.fill_database(dfs,"day",1000, 'p_day', start_date)
                
                if infinity and self.nothingToGet:
                    break
                
                Utils.countdown(60)
            except KeyboardInterrupt:
                print("END by keybord interruption")
                exit()
            except: 
                print("Unexpected error:", sys.exc_info()[0])
                print("Waiting 3 minutes and try it again")
                Utils.countdown(60)

        
        print("DONE")    
            

           
        






