import alpaca_trade_api as tradeapi
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import market_src.alpaca2Login as al
import json
# %config InlineBackend.figure_format = 'retina'
# %config IPCompleter.greedy=True



def  get_bars(symbols):
    output
    for s in symbols:
        data = api. get_barset(s, 'day', limit=1000)
        data =  data.df[s]['close']
    output.add(data)
     
    return output
    
def selectMarketBarset(api, marketName):
    active_assets = api.list_assets(status='active')
    market_assets = [a for a in active_assets if a.exchange == marketName and a.tradable==True]
    return  market_assets
    
    
def changeByDays():
    week_open = aapl_bars[0].o
    week_close = aapl_bars[-1].c
    percent_change = (week_close - week_open) / week_open * 100

def processStockListToDF(data):
    df = None

    for v in data:
        for attr, value in v.__dict__.items():
            if df is None:
                df = pd.json_normalize(value)
            else:
                df = df.append(pd.json_normalize(value))
    return df

    
def writeDFToCSVFile(name, df):
     df.to_csv(name)


def readFromFile(name):
    f = open(name, "r")
    return f.read()
    
    

api = al.alpaca2Login().getApi()
account = api.get_account()
#api.list_positions()

pos_list = [x.symbol for x  in api.list_positions()]

nasdaq = selectMarketBarset(api,"NYSE")
writeDFToCSVFile("nasdaqList.csv", processStockListToDF(nasdaq))

#print(readFromFile("nasdaqList.csv"))