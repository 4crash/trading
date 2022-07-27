import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine, false
from sqlalchemy.orm import sessionmaker
from yahoo_earnings_calendar import YahooEarningsCalendar
from datetime import datetime
from utils import Utils
class refreshFinancials():
   
 

    def __init__(self):
        self.table_name = "financials"
        self.db_name = "nyse_financials"
        self.engine = create_engine('postgresql://postgres:crasher@localhost:5432/' + self.db_name)

    def convertUnixTimeToDate(self, columns, df):
        for col in columns:
            if col in df:
                df[col]= pd.to_datetime(df[col], unit="s")
        return df
    
          
    async def getFinancials(self, send):
        dfs = pd.read_csv("./datasets/RevolutStock.csv", delimiter="|")
        dff = pd.DataFrame()
        # dfs = {"Symbol": "AAPL"}
        existing_financials = self.get_fin_exists()
        symbols = ' '.join( dfs["Symbol"])
        tickerData = yf.Tickers(symbols)

        for row,value in tickerData.tickers.items():
            # row = str.strip(row)
            print(row)
            if len(existing_financials) == 0 or row not in existing_financials.index:
                # tickerData = yf.Tickers("AAPL HIMX AMD MU")
                # print(tickerData.tickers['AAPL'].info)
                # exit()
                # print(tickerData)
                # print(json_normalize(tickerData.info))
                # if dff.empty:
                print('read ticker')
                print(value.info)
                print('------------json normalize')
                dff = pd.json_normalize(value.info)
                print('--------- json end normalize-----------')
                # else:
                #     print('normalizing')
                #     new_df = json_normalize(
                #             tickerData.info)
                #     print('contactinating')
                #     dff = pd.concat([dff,new_df])
                # else:
                #     try:
                #         dff = dff.concat(json_normalize(
                #             tickerData.info), ignore_index=True)
                #     except :
                #         if send:
                #             await send({
                #                 'type': 'websocket.send',
                #                 'text': str(tickerData) + " - No Info"
                #             })
                #         else:
                #             print(str(tickerData))
                # try:
                    
                #     if send:
                #         await send({
                #             'type': 'websocket.send',
                #             'text': str(row) + " - OK"
                #             })
                #     else:
                #         print(str(row))
                # except:
                #     print("cannot send websock et to client")
                print(f'Lets stuck all this shit into DB {len(dff)}')
                self.save_to_db(dff)

    def save_to_db(self,dff: pd.DataFrame):
        
        # dff.drop("coinMarketCapLink", axis=1, inplace = True)
        # df = dff.iloc[[0, -1]]
       
        
        columns = ["dateShortInterest", "exDividendDate", "lastFiscalYearEnd", "nextFiscalYearEnd",
                   "mostRecentQuarter", "sharesShortPreviousMonthDate", "dateShortInterest","regularMarketTime"]
        # print(dff)
        df = self.convertUnixTimeToDate(columns, dff)
       
        df.loc[:,'id'] = df.index
        df.loc[:,'date'] = datetime.today().replace(hour=0,minute=0,second=0,microsecond=0)       
        df.to_sql(self.table_name, con=self.engine,
                if_exists='append', index=False)
        
    def get_fin_exists(self):
        data = pd.DataFrame()
        date = datetime.today().replace(hour=0,minute=0,second=0,microsecond=0)
        data = pd.read_sql_query(
            f"Select symbol from financials where date_trunc('day',date)::date='{date}'", self.engine, index_col=['symbol'])
        return data

    async def start(self, send):
        await self.getFinancials(send)
