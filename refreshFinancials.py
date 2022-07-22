import yfinance as yf
import pandas as pd
from pandas import json_normalize
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from yahoo_earnings_calendar import YahooEarningsCalendar
from datetime import datetime
from alpaca_examples.utils import Utils
class refreshFinancials():
   
 

    def __init__(self):
        self.table_name = "financials"
        self.db_name = "nyse_financials" 
        self.engine = create_engine('postgresql://postgres:crasher@localhost:5432/' + self.db_name)

    def convertUnixTimeToDate(self, columns, df):
        for col in columns:
            df[col] = pd.to_datetime(df[col], unit="s")
        return df
    
          
    async def getFinancials(self, send):
        dfs = pd.read_csv("./datasets/RevolutStock.csv", delimiter="|")
        dff = pd.DataFrame()
        # dfs = {"Symbol": "AAPL"}
        for row in dfs["Symbol"]:
            row = str.strip(row)
            tickerData = yf.Ticker(row)
            
            # print(json_normalize(tickerData.info))
            if dff.empty:
                dff = json_normalize(tickerData.info)
            else:
                try:
                    dff = dff.concat(json_normalize(
                        tickerData.info), ignore_index=True)
                except :
                    if send:
                        await send({
                            'type': 'websocket.send',
                            'text': str(tickerData) + " - No Info"
                        })
                    else:
                        print(str(tickerData))
            try:
                
                if send:
                    await send({
                        'type': 'websocket.send',
                        'text': str(row) + " - OK"
                        })
                else:
                    print(str(row))
            except:
                print("cannot send websock et to client")
            self.save_to_db(dff)

    def save_to_db(self,dff: pd.DataFrame):
        print(dff.symbol)
        # dff.drop("coinMarketCapLink", axis=1, inplace = True)
        df = dff.iloc[[0, -1]]
        columns = ["dateShortInterest", "exDividendDate", "lastFiscalYearEnd", "nextFiscalYearEnd",
                   "mostRecentQuarter", "sharesShortPreviousMonthDate", "dateShortInterest"]
        df = self.convertUnixTimeToDate(columns, df)
        if "regularMarketTime" in df:
            df = self.convertUnixTimeToDate(["regularMarketTime"],df)

        df['id'] = df.index
        df['date'] = datetime.today().replace(hour=0,minute=0,second=0)

        if self.is_fin_exists(df) is False:
            df.to_sql(self.table_name, con=self.engine,
                  if_exists='append', index=False)
        
    def is_fin_exists(self, df):
        data = pd.DataFrame()
        if len(df) > 0:
            date = df.tail(1)["date"].values[0]
            sym = df.tail(1)["symbol"].values[0]
            data = pd.read_sql_query(
                f"Select date, symbol from financials where date::date='{date}' and symbol='{sym}'", self.engine)
            print(data)
        return False if len(data) < 1 else True
    
    async def start(self, send):
        await self.getFinancials(send)
