# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
import asyncio
from datetime import datetime
# import datetime
from plotly.subplots import make_subplots
import streamlit as st
import sys
sys.path.append('../')
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
import threading

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
class SlAs():
    selected_stock = None
    timetick = None
    def __init__(self):
        self.db = Database()
        self.symbols = self.db.get_symbols(TableName.DAY)
        self.sectors = self.db.get_sectors(TableName.DAY)
        self.sm = StockMess()
        self.app = self.get_home_page()
        self.sw = StockWhisperer()
        self.submit = None
        
        
        
    def get_home_page(self):
        st.set_page_config(layout="wide")
        
        t1 = threading.Thread(target=SlAs.watch, args=(st,))
        t1.start()

        # loop = asyncio.get_event_loop()
        print('after async')
        test = st.text_area("websocket", self.timetick)
      
        st.write("some text")
        test = st.text_area("websocket2", self.timetick)
       
        
     
    @staticmethod
    def watch(sti):
        i = 0
        while True:
            print('thread')
            Utils.countdown(5)
            i+=1
            st.write(str(i))
           


        # test = st.empty()
    
    async def watch2(self, test):
        while True:
            print('async')
            Utils.countdown(5)
            test = str(datetime.now())
            # test.markdown(
            #     f"""
            #     <p class="time">
            #         {str(datetime.now())}
            #     </p>
            #     """, unsafe_allow_html=True)
            try:
                await asyncio.sleep(1)
            except KeyboardInterrupt:
                print("ending async")
                exit()


        test = st.empty()
    
    @staticmethod
    async def time_tick():
        timetick = 0
        while True:
            timetick += 1
            await asyncio.sleep(1)
       
   


if __name__ == '__main__':
    
    rd = SlAs()
    # rd.app.run_server(debug=True)

