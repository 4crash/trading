# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
import asyncio
from datetime import datetime
import numpy as np
# import datetime
from plotly.subplots import make_subplots
import streamlit as st
import sys
sys.path.append('../../')
from alpaca_examples.market_db import Database, TableName
from alpaca_examples.utils import Utils
from alpaca_examples.fin_i import FinI
from alpaca_examples.plot_p import PlotP
from alpaca_examples.stock_mess import StockMess
from alpaca_examples.stock_whisperer import StockWhisperer
import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
from stockstats import StockDataFrame as sdf
import time
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
class RunData():
    selected_stock = None
    timetick = None
    time_from = None
    time_to = None
    selected = {}
    action_type = None
    types = ["sector-stats", "stock-detail",
                  "stocks-stats"]
    
    def __init__(self):
        self.db = Database()
        self.symbols = self.db.get_symbols(TableName.DAY)
        self.sectors = self.db.get_sectors(TableName.DAY)
        self.sm = StockMess()
        self.app = self.get_home_page()
        self.sw = StockWhisperer()
        self.submit = None
        
        
        # self.fig = None
        # print(self.df)
       
    # def load_data(self, option, time_from = "-180d", time_to = None):
    #     df =  self.db.load_data(
    #         "p_day", symbols=option, time_from=time_from, time_to = time_to)
    #     df = FinI.add_indicators(sdf.retype(df))
    #     return df
        
        
    def get_home_page(self):
       
        st.set_page_config(layout="wide")
        asyncio.new_event_loop().run_until_complete(RunData.prepare_test_tasks())
        
        
    @staticmethod
    async def prepare_test_tasks():
        tasks = []
        task = asyncio.ensure_future(RunData.async_test(0.05))
        tasks.append(task)
        task = asyncio.ensure_future(RunData.async_test(0.1))
        tasks.append(task)
        await asyncio.gather(*tasks, return_exceptions=True)
        
    @staticmethod
    async def async_test(sleep):
        
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        last_rows = np.random.randn(1, 1)
        chart = st.line_chart(last_rows)
        
        for i in range(1, 101):
            new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
            status_text.text("%i%% Complete" % i)
            chart.add_rows(new_rows)
            progress_bar.progress(i)
            last_rows = new_rows
            await asyncio.sleep(sleep)

        progress_bar.empty()
   
    


if __name__ == '__main__':
    
    rd = RunData()
    # rd.app.run_server(debug=True)

