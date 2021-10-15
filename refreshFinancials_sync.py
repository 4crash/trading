import yfinance as yf
import pandas as pd
from pandas import json_normalize
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

class refreshFinancialsSync():
    engine = create_engine('postgresql://postgres:crasher@localhost:5432/nyse_financials')
    dbName = "financials"

    def __init__(self):
        
        df = self.getFinancials()
        columns = ["dateShortInterest","exDividendDate","lastFiscalYearEnd","nextFiscalYearEnd","mostRecentQuarter","sharesShortPreviousMonthDate","dateShortInterest"]
        df = self.convertUnixTimeToDate(columns, df)
        df['id'] = df.index
        df.to_sql(self.dbName, con=self.engine, if_exists='append', index=False)
        # self.engine.execute("ALTER TABLE "+self.dbName+" ADD COLUMN id SERIAL PRIMARY KEY;")
        # df.head()

    def getFinancials(self):
        dfs = pd.read_csv("../datasets/RevolutStock.csv", delimiter="|")
        dff = pd.DataFrame()
        # dfs = {"Symbol": "AAPL"}
        for row in dfs["Symbol"]:
            row = str.strip(row)
            tickerData = yf.Ticker(row)
            print(json_normalize(tickerData.info))
            
            if dff.empty:
                dff = json_normalize(tickerData.info)
            else:
                try:
                    dff = dff.append(json_normalize(tickerData.info), ignore_index=True)
                except:
                    print(tickerData)
                    print(" hasnot info")
            #print(dff.tail(1))
            try:
                print(row)
            
            except:
                pass

        return dff

    def convertUnixTimeToDate(self, columns, df):
        for col in columns:
            df[col] = pd.to_datetime(df[col],unit="s")
        return df






