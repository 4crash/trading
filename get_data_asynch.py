from alpaca_examples.simpleStream import debug
import pandas as pd
# import PostgresConnect
from sqlalchemy import create_engine
import sys
sys.path.append('../')
from market_src import alpaca2Login as al
# from alpaca_examples.back_tester import BackTest
from .utils import Utils
from .market_db import Database
from datetime import datetime, timedelta
import pytz
utc=pytz.UTC
import asyncio
from asgiref.sync import sync_to_async


class getDataAsynch():
    def __init__(self):
        self.engine = create_engine('postgresql://postgres:crasher@localhost:5432/nyse_financials')
        self.api = al.alpaca2Login().getApi()
        self.fin_db = "financials"
        self.earnings_db = "earning_dates"
        self.sentiment_db = "sentiment"
        self.nothingToGet = True
        self.db = Database()
        # df = pd.DataFrame()
        

    async def  get_bars(self, symbol, interval, limit, after):
        # print(after.isoformat())
        seconds_to_sleep = 5
        data = await sync_to_async(self.api.get_barset)(symbol, interval, limit, after=after.isoformat())
        data =  data.df[symbol]
        print(data)
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

    async def fill_database(self, dfs,interval, limit, db_name, after, symbol):
      
            if symbol:
                symbol = str.strip(symbol)
                # print(symbol)
                
                # check last date for specific symbol
                dfp = pd.read_sql('select MAX("index") as last_time from '+ db_name + ' where sym = \'' + symbol + '\' ', con=self.engine)
                
                if dfp.iloc[0].last_time is not None:
                    after = dfp.iloc[0].last_time

                # get data from ALPACA
                df = await self.get_bars(symbol, interval, limit, after)
                print("async test: " + str(symbol))
                if len(df) > 0:
                    self.nothingToGet = False
                   
                    # add symbol
                    df["sym"] = symbol
                    
                    #add sector
                    # dff = pd.read_sql('select sector from '+ self.fin_db + ' where symbol = \'' + row + '\' and sector is not null limit 1', con=self.engine)
                    # if len(dff) > 0 and dff.iloc[0].sector:
                    #     df["sector"] = dff.iloc[0].sector

                    df.index.rename('index', inplace = True)
                  
                    #save DATAFRAME to database
                    df.to_sql(db_name, con=self.engine, if_exists='append', index = True)

    def get_last_working_day(self):
        
        offset = max(0, (datetime.today().weekday() + 6) % 7 - 3)
        timedelta2 = timedelta(offset)
        # most_recent = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta2
        most_recent = datetime.today() - timedelta2
        return most_recent

    # param interval could be 1,5,15, 0 - for day
    def start_download(self, min_interval, start_date):
        
        dfs = pd.read_csv("../datasets/RevolutStock.csv", delimiter="|")
        print(start_date)
        min_interval = int(min_interval)

        if min_interval > 0:
            db_name = 'p_'+ str(min_interval) +'min';
        else:
            db_name = 'p_day'
        
        while True:
           
            try:
                self.nothingToGet = True
                tasks = []
                loop = asyncio.get_event_loop()
                for row in dfs['Symbol']:
                    if min_interval > 0:
                        tasks.append(self.fill_database(dfs,str(min_interval) + "Min",1000, 'p_'+ str(min_interval) +'min', start_date, row))
                    else:
                        tasks.append(self.fill_database(dfs,"day",1000, 'p_day', start_date, row))

                all_groups = asyncio.gather(*tasks)
                results = loop.run_until_complete(all_groups)
                # print(results)
                loop.close()
                
                if self.nothingToGet:
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
            

           
        






