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

from streamlit.commands.page_config import set_page_config
sys.path.append('../')
from market_db import Database, TableName
from utils import Utils
from fin_i import FinI
from plot_p import PlotP
from stock_mess import StockMess
from sector_stats import SectorStats

from stock_whisperer import StockWhisperer
import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
from stockstats import StockDataFrame as sdf
import time
import requests
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
    sw = StockWhisperer()
    inday_days = 2

    def __init__(self):
        self.db = Database()
        self.symbols = self.db.get_symbols(TableName.DAY)
        self.sectors = self.db.get_sectors(TableName.DAY)
        self.sm = StockMess()
        self.ss = SectorStats()
        self.app = self.get_home_page()
        self.submit = None


        # self.fig = None
        # print(self.df)

    # def load_data(self, option, time_from = "-180d", time_to = None):
    #     df =  self.db.load_data(
    #         "p_day", symbols=option, time_from=time_from, time_to = time_to)
    #     df = FinI.add_indicators(sdf.retype(df))
    #     return df

    def testing_mess(self):
        self.sm.stocks = self.db.load_data(
            table_name=TableName.DAY, symbols=["PLUG"], time_from=self.time_from, time_to=self.time_to,)
        self.sm.get_subj_mess(
            "Base Fund: ", "PLUG")

    def get_home_page(self):

        query_params = st.experimental_get_query_params()
        if len(query_params) > 0 and query_params.get("type")[0] == "detail":
            st.set_page_config(initial_sidebar_state="collapsed",
                               page_title=query_params.get("sym")[0], layout="wide")
        else:
            st.set_page_config(layout="wide")
            
        self.hide_footer()
        self.left_menu()
        # self.testing_mess()
        
        self.action_router(query_params)

    async def prepare_detail_tasks(self, sym = None):
        tasks = []
        
        tasks.append(asyncio.ensure_future(self.get_price_detail(sym=sym)))
        tasks.append(asyncio.ensure_future(
            self.get_inday_price_graph(sym=sym)))
        tasks.append(asyncio.ensure_future(self.get_fund_detail(sym=sym)))

        await asyncio.gather(*tasks, return_exceptions=True)



    def left_menu(self):

        self.time_from = st.sidebar.text_input("Time from now -1m, -1h, -1d,", value = "-120d")
        self.time_to = st.sidebar.text_input("Time to 1m, 1h, 1d,")

        self.selected_sector = st.sidebar.selectbox(
            'Select sector',
            self.sectors)

        self.selected_stock = st.sidebar.selectbox(
            'Select stock',
            self.symbols)


        self.inday_days = st.sidebar.number_input(
            'inday chart days',
            value=2)
        
        if self.selected_stock == 0:
            self.selected_stock = "TSLA"

        
        # loop = asyncio.get_event_loop()
        # task = None
        # try:
        #     task = loop.create_task(self.watch())
        # except KeyboardInterrupt:
        #     task.cancel()


        # print('after async')
        # st.sidebar.text_area("websocket",self.timetick)



        # self.action_type = st.sidebar.selectbox(
        #     'Seeect Type',
        #     self.types)

        # self.submit = st.sidebar.button("Submit", "submit")

    def action_router(self, query_params = None):

        
        if st.sidebar.button('stock-detail') or (len(query_params) > 0 and query_params.get("type")[0]== "detail"):
 
            asyncio.new_event_loop().run_until_complete(
                self.prepare_detail_tasks(sym = query_params.get("sym")[0] if len(query_params) > 0 else None ))


        if st.sidebar.button('top-sectors'):
            self.top_sectors(time_from = self.time_from, time_to = self.time_to)
        if st.sidebar.button('top-industries'):
            self.top_sectors(time_from=self.time_from, time_to=self.time_to, is_industries= True)

        if st.sidebar.button('top-stocks'):
            self.top_stocks(time_from = self.time_from, time_to = self.time_to)

        if st.sidebar.button('bottom-stocks'):
            self.top_stocks(time_from = self.time_from, time_to = self.time_to, ascending = True)

        if st.sidebar.button("find-stocks-to-buy"):
            self.find_stocks_to_buy()

        if st.sidebar.button("Last-Financials"):
            self.last_financials()


    def last_financials(self):
        
        df = self.db.get_last_financials()
        df.dropna(inplace = True, axis='columns', how="all")
        df.drop(columns=['zip','city','phone','longBusinessSummary','companyOfficers','maxAge','address1','previousClose', \
            'regularMarketOpen','regularMarketDayHigh','navPrice','totalAssets','regularMarketPreviousClose','open','yield', \
                         'priceHint', 'currency', 'dayLow', 'ask', 'askSize','website','longName', \
                             'exchange'], inplace=True)
        df.set_index(df.symbol, inplace=True)
        st.dataframe(df, height=2000)

    def find_stocks_to_buy(self):
        st.title("Stocks to buy")
        self.sw.time_from = self.time_from
        if self.sectors is not None:
            self.sw.sectors = [self.sectors]
            
        self.sw = StockWhisperer()
        
        try:
            output = self.sw.find_stocks(TableName.DAY, False)
        except KeyboardInterrupt:
            st.write("stopped by keyboard")
            
        stocks = output[["close", "open", "high",
                          "low", "volume", "amount", "flpd","sym"]]
        # print(stocks)
        self.print_stocks_list(stocks, True,
                          from_top=0, show_stocks_num=50)

    def show_sectors(self, stocks, is_industries= False):
        
        fig = px.bar(stocks, y='flpd', x=stocks.index)
        # fig.update_traces(texttemplate=stocks.sector, textposition='inside')
        fig.update_layout(uniformtext_minsize=8,
                          uniformtext_mode='hide', barmode='group')

      
        st.plotly_chart(fig, use_container_width=True,
                        use_container_height=True, template="plotly_dark",)
        # st.bar_chart(stocks)
          
    def top_sectors(self, ascending = False, time_from = None, time_to= None, is_industries = False):
        # st.write(time_to)
        # st.write(time_from)
        title = "Top industries" if is_industries else "Top sectors"
        separator = " | "
        st.title(title + str(time_from))

        if not time_from:
            time_from = "-7d"

        table = TableName.DAY
        if time_from.find("m")>-1 or time_from.find("m")>-1:
            table = TableName.MIN15

        
        stocks = self.ss.classify_sectors_uptrend(table_name = table, time_from=time_from, time_to=time_to, from_db = True, is_industries=is_industries, separator = separator)
        if stocks is not None and len(stocks) > 0:
            stocks = stocks.sort_values(by='flpd', ascending=ascending)
           
            stocks.drop(columns=["close","open","high","low", "volume","amount"], inplace=True)
            self.show_sectors(stocks,is_industries = is_industries)
            
            for idx, row in stocks.iterrows():
                st.write(str(idx) + ": " + str(round(row.flpd, 2)) + "%")
                self.top_stocks(group=idx, ascending=ascending,
                                time_from=time_from, time_to=time_to, table=table, is_industries = is_industries, separator = separator)
        else:
            st.write("No data")

    def top_stocks(self, group= None, ascending= False, time_from = None, time_to= None, table = None, from_top = 0
                   ,show_stocks_num=20, is_industries = False, separator = None):
        st.empty()
        table = TableName.DAY
        if time_from.find("m")>-1:
            table = TableName.MIN15

        subject = "Loosers: " if ascending else "Gainers: "
        if isinstance(group,str):
            industry = str.split(group, separator)
            group = industry[1] if len(industry)>1 else industry[0]
            group = [group]
        if is_industries:
           
            stocks = self.db.load_data(
                table_name=table, industries=group,  time_from=time_from, time_to=time_to,)
        else:
            stocks = self.db.load_data(
                table_name=table, sectors=group,  time_from=time_from, time_to=time_to,)
            
        # self.stocks = FinI.add_change(self.stocks)
        stocks = Utils.add_first_last_perc_diff(stocks)
        self.print_stocks_list(stocks, ascending,
                          from_top=from_top, show_stocks_num=show_stocks_num)
       
    def print_stocks_list(self, stocks, ascending, from_top=0, show_stocks_num=20):
        if stocks is not None and len(stocks) > 0:

            stocks = stocks.groupby(by="sym").mean()
            stocks = stocks.sort_values(by='flpd', ascending=ascending)

            top_stocks = stocks.iloc[from_top:(from_top + show_stocks_num)]
            top_stocks = self.fill_with_mess(top_stocks)
            reduced_top_stocks = top_stocks.drop(
                columns=["open", "high", "low", "amount"])
            reduced_top_stocks["detail"] = '<a href="/?type=detail&sym=' + \
                reduced_top_stocks.index + '" target="_blank"> ' + \
                    reduced_top_stocks.index + '</a>'
            reduced_top_stocks["news"] = '<a href="https://finance.yahoo.com/quote/' + \
                reduced_top_stocks.index + '" target="_blank"> yahoo </a>'
            reduced_top_stocks["rating"] = '<a href="https://zacks.com/stock/quote/' + \
                reduced_top_stocks.index + '" target="_blank"> zacks </a>'

            st.write(reduced_top_stocks.to_html(
                escape=False, index=False), unsafe_allow_html=True)

            # selected_indices = st.multiselect('Select rows:', stocks.index)
            # selected_rows = stocks.loc[selected_indices]
            # self.top_stocks_list = top_stocks.index.tolist()
            # self.draw_chart_values(top_stocks)

            #send_mails
            # asyncio.run(self.sm.mail_stats(top_stocks, subject))
        else:
            print('No stocks has been found')
        
    def fill_with_mess(self, stocks):
        mess_block = []
        for index, row in stocks.iterrows():
            self.sm.set_fundamentals(index)
            # st.write(index)
            mess, curr_price, days_to_earnings = self.sm.get_subj_mess(
                "", index)
            mess_block.append(mess)
            # print(stocks.loc[index])

        stocks["fund"] = mess_block
        return stocks



    def set_time_to(self):
        if self.time_from is None:
            time_from = "-180d"
        else:
            time_from = self.time_from

        return time_from

    async def get_inday_price_graph(self, sym=None):
       
        if sym is None:
            sym = self.selected_stock
        
        inday_df = self.db.get_last_n_days(
               sym, n_days_back=self.inday_days, table=TableName.MIN15)
        
        inday_df = FinI.add_indicators(sdf.retype(inday_df))
                                 
        await self.inday_price_graph(sym ,inday_df)
        

    async def get_price_detail(self, sym = None):

        if sym is None:
            sym = self.selected_stock
        time_from = self.set_time_to()
        df = self.db.load_data(
            table_name=TableName.DAY, symbols=sym, time_from=time_from, time_to=self.time_to,)
        m_df_spy = self.db.load_data(
            table_name=TableName.DAY,  time_from=time_from, time_to=self.time_to, symbols=["SPY"])
        m_df_spy["oc_mean"] = (m_df_spy.close + m_df_spy.open)/2
        # self.stocks = FinI.add_change(self.stocks)
        df = FinI.add_indicators(sdf.retype(df))
        await self.price_graph(sym, df,m_df_spy= m_df_spy)

     

        # self.macd_rsi_graph(option, df)
        self._max_width_()
        # await asyncio.sleep(0.001)

    async def get_fund_detail(self, sym=None):
        if sym is None:
            sym = self.selected_stock
        self.sm.set_fundamentals(sym)

        time_from = self.set_time_to()
        self.sm.stocks = self.db.load_data(
            table_name=TableName.DAY, symbols=sym, time_from=time_from, time_to=self.time_to,)
       
        mess, curr_price, days_to_earnings = self.sm.get_subj_mess(
            "Base Fund: ", sym)
        st.write(mess)
        
        await st.write(self.sm.get_fund_mess(
            self.sm.financials, curr_price, self.sm.earnings, self.sm.sentiment, days_to_earnings,  self.sm.stocks))
        
    def macd_rsi_graph(self, option, df):

        data = PlotP.plot_rsi(df, ax="y")
        data += PlotP.plot_macd_boll(df=df, ax="y2")


        fig = go.Figure(data=data)


        fig.update_layout(autosize=True, height= 400, yaxis=dict(
            title="RSI",
            titlefont=dict(
                color="green"
            ),
            tickfont=dict(
                color="green"
            )
        ),
            yaxis2=dict(
            title="macd",
            titlefont=dict(
                color="#8888ff"
            ),
            tickfont=dict(
                color="#8888ff"
            ),
            anchor="free",
            overlaying="y",
            side="left",
            position=0
        ))

        # Create figure with secondary y-axis
        st.plotly_chart(fig, use_container_width=True)

    def gainers(self):
        pass

    def hide_footer(self):

        hide_streamlit_style = """
            <style>
            #MainMenu {visibility: visible;}
            footer {visibility: hidden;}
            </style>
            """
        st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    # def sector_stats(self, time_from, time_to, loosers):
    #     stocks = self.ss.classify_sectors_uptrend(table_name =TableName.DAY)
    #     stocks = stocks.sort_values(by='flpd', ascending=loosers)

    #     if False:
    #         self.sw.sectors = [self.sw.stocks.iloc[:-1].index[0]]
    #         self.sw.top_stocks(table_name=None, top_losers=loosers)
    

    def inday_price_graph(self, option, df, ax = "y2" ):
        st.write("In day chart: -" + str(self.inday_days) + "d")
         
        sets = [{'x': df["index"], 'open': df.open, 'close': df.close,
                 'high': df.high, 'low': df.low, 'yaxis':  ax, "hovertext":"", 'type': 'candlestick'}]

        sets += [{'x': df["index"], 'y': df.boll, 'yaxis':  ax, 'type': 'scatter',
                'mode': 'lines', 'line': {'width': 1, 'color': 'green'}, 'name': 'Boll'}]

        sets += [{'x': df["index"], 'y':  df.boll + df.boll_2, 'yaxis':  ax, 'type': 'scatter',
                  'mode': 'lines', 'line': {'width': 0.3, 'color': 'green', "dash": "dot"}, 'name': '-'}]

        sets += [{'x': df["index"], 'y': df.boll + df.boll_3, 'yaxis':  ax, 'type': 'scatter',
                  'mode': 'lines', 'line': {'width': 0.3, 'color': 'green', "dash": "dot"}, 'name': '-'}]

        # sets += [{'x': df["index"], 'y': df.boll + df.boll_5, 'yaxis':  ax, 'type': 'scatter',
        #           'mode': 'lines', 'line': {'width': 0.3, 'color': 'green'}, 'name': '-'}]

        sets += [{'x': df["index"], 'y': df.boll + df.boll_6, 'yaxis':  ax, 'type': 'scatter',
                  'mode': 'lines', 'line': {'width': 0.3, 'color': 'green', "dash": "dot"}, 'name': '-'}]

        sets += [{'x': df["index"], 'y': df.boll - df.boll_2, 'yaxis':  ax, 'type': 'scatter',
                  'mode': 'lines', 'line': {'width': 0.3, 'color': 'green', "dash": "dot"}, 'name': '-'}]

        sets += [{'x': df["index"], 'y': df.boll - df.boll_3, 'yaxis':  ax, 'type': 'scatter',
                  'mode': 'lines', 'line': {'width': 0.3, 'color': 'green', "dash": "dot"}, 'name': '-'}]

        # sets += [{'x': df["index"], 'y': df.boll - df.boll_5, 'yaxis':  ax, 'type': 'scatter',
        #           'mode': 'lines', 'line': {'width': 0.3, 'color': 'green'}, 'name': '-'}]

        sets += [{'x': df["index"], 'y': df.boll - df.boll_6, 'yaxis':  ax, 'type': 'scatter',
                  'mode': 'lines', 'line': {'width': 0.3, 'color': 'green', "dash": "dot"}, 'name': '-'}]

        sets += [{'x': df["index"], 'y': df.boll_ub, 'yaxis':  ax, 'type': 'scatter',
                'mode': 'lines', 'line': {'width': 1, 'color': 'green'}, 'name': 'Boll UP'}]
        sets  += [{'x': df["index"], 'y': df.boll, 'yaxis':  ax,
                'type': 'scatter', "fill": 'tonexty', 'line': {'width': 0, } ,"fillcolor": 'rgba(128, 255, 128,0.2)'}]
        sets += [{'x': df["index"], 'y': df.boll_lb, 'yaxis':  ax, 'type': 'scatter',
                'mode': 'lines', 'line': {'width': 1, 'color': 'green'}, 'name': 'Boll Down'}]
        sets += [{'x': df["index"], 'y': df.boll, 'yaxis':  ax,
                'type': 'scatter', "fill": 'tonexty',  'line': {'width': 0,},"fillcolor": 'rgba(128, 255, 128,0.2)'}]

        sets += [{'x': df["index"], 'y': df.sma9, 'yaxis':  ax, 'type': 'scatter',
                  'mode': 'lines', 'line': {'width': 1, 'color': 'blue'}, 'name': 'sma9'}]

        sets += [{'x': df["index"], 'y': df.sma50, 'yaxis':  ax, 'type': 'scatter',
                  'mode': 'lines', 'line': {'width': 1, 'color': 'darkblue'}, 'name': 'sma50'}]

        sets += [{'x': df["index"], 'y': df.sma100, 'yaxis':  ax, 'type': 'scatter',
                  'mode': 'lines', 'line': {'width': 1, 'color': 'orange'}, 'name': 'sma100'}]

        df = FinI.add_levels(df)
        for i, r in df.iterrows():
            if r.price_level is not None:
                sets += [{'x': [r["index"], df.iloc[-1]["index"]], 'y': [r.price_level,r.price_level], 'yaxis':  ax, 'type': 'scatter',
                          'mode': 'lines', 'line': {'width': 1, 'color': 'brown', "dash": "dot"}, 'name': ''}]
        # print(levels)
        # sets += [{'x': df["index"], 'y': df.price_level, 'yaxis':  ax, 'type': 'scatter',
        #           'mode': 'lines', 'line': {'width': 1, 'color': 'orange'}, 'name': 'sma100'}]


        # data += PlotP.plot_rsi(df, ax="y2")
        # data += PlotP.plot_macd_boll(df=df, ax="y3")
        # st.area(df, x=df["index"], y=df.boll, color="continent",
        #         line_group="country")
        # fig = go.Figure(data=sets)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.009, horizontal_spacing=0.009, row_width=[0.1, 0.5], 
                            specs=[[{"secondary_y": True}], [{"secondary_y": True}]]
                            )
        fig.add_traces(data=sets, cols=1, rows=1)
      
        # data = PlotP.plot_rsi(df, ax="y")
        # fig.add_traces(data, 3, 1)
        # data = PlotP.plot_macd_boll(df=df, ax="y2")
        # fig.add_traces(data, 3, 1)
        df.loc[df.open > df.close, "vol_color"] = "red"
        df.loc[df.open <= df.close, "vol_color"] = "green"
        # print(df.vol_color)
        fig.add_trace({'x': df["index"], 'y': df.volume,
                       'type': 'bar', 'name': 'Volume', 'marker_color': df.vol_color}, 2, 1, secondary_y=False, )


        # fig['layout']['margin'] = {'l': 30, 'r': 10, 'b': 50, 't': 25}
        fig.update_yaxes(showgrid=True, zeroline=False, showticklabels=True,
                         showspikes=True, spikemode='across', spikesnap='cursor', showline=True, spikedash='dash', spikethickness=0.5)


        fig.update_xaxes(showgrid=True, zeroline=False, rangeslider_visible=False, showticklabels=True,
                         showspikes=True,  showline=True, spikemode='across', spikesnap='cursor', spikedash='dash', spikethickness=0.5)

        fig.update_layout(autosize=True, height=600,
                          hoverdistance=1, hovermode='y', spikedistance=10000
                      
                          )
        st.plotly_chart(fig, use_container_width=True,
                        use_container_height=True, template="plotly_dark",)


        # st.line_chart(df.close)

    def price_graph(self, option, df, m_df_spy = None, ax = "y2" ):
        st.write("Days chart: " + str(self.time_from))

        sets = [{'x': df.index, 'open': df.open, 'close': df.close,
                 'high': df.high, 'low': df.low, 'yaxis':  ax, "hovertext":"", 
                 'type': 'candlestick'}]

        sets += [{'x': df.index, 'y': df.boll, 'yaxis':  ax, 'type': 'scatter',
                'mode': 'lines', 'line': {'width': 1, 'color': 'green'}, 'name': 'Boll'}]

        sets += [{'x': df.index, 'y':  df.boll + df.boll_2, 'yaxis':  ax, 'type': 'scatter',
                  'mode': 'lines', 'line': {'width': 0.3, 'color': 'green', "dash": "dot"}, 'name': '-'}]

        sets += [{'x': df.index, 'y': df.boll + df.boll_3, 'yaxis':  ax, 'type': 'scatter',
                  'mode': 'lines', 'line': {'width': 0.3, 'color': 'green', "dash": "dot"}, 'name': '-'}]

        # sets += [{'x': df.index, 'y': df.boll + df.boll_5, 'yaxis':  ax, 'type': 'scatter',
        #           'mode': 'lines', 'line': {'width': 0.3, 'color': 'green'}, 'name': '-'}]

        sets += [{'x': df.index, 'y': df.boll + df.boll_6, 'yaxis':  ax, 'type': 'scatter',
                  'mode': 'lines', 'line': {'width': 0.3, 'color': 'green', "dash": "dot"}, 'name': '-'}]

        sets += [{'x': df.index, 'y': df.boll - df.boll_2, 'yaxis':  ax, 'type': 'scatter',
                  'mode': 'lines', 'line': {'width': 0.3, 'color': 'green', "dash": "dot"}, 'name': '-'}]

        sets += [{'x': df.index, 'y': df.boll - df.boll_3, 'yaxis':  ax, 'type': 'scatter',
                  'mode': 'lines', 'line': {'width': 0.3, 'color': 'green', "dash": "dot"}, 'name': '-'}]

        # sets += [{'x': df.index, 'y': df.boll - df.boll_5, 'yaxis':  ax, 'type': 'scatter',
        #           'mode': 'lines', 'line': {'width': 0.3, 'color': 'green'}, 'name': '-'}]

        sets += [{'x': df.index, 'y': df.boll - df.boll_6, 'yaxis':  ax, 'type': 'scatter',
                  'mode': 'lines', 'line': {'width': 0.3, 'color': 'green', "dash": "dot"}, 'name': '-'}]

        sets += [{'x': df.index, 'y': df.boll_ub, 'yaxis':  ax, 'type': 'scatter',
                'mode': 'lines', 'line': {'width': 1, 'color': 'green'}, 'name': 'Boll UP'}]
        sets  += [{'x': df.index, 'y': df.boll, 'yaxis':  ax,
                'type': 'scatter', "fill": 'tonexty', 'line': {'width': 0, } ,"fillcolor": 'rgba(128, 255, 128,0.2)'}]
        sets += [{'x': df.index, 'y': df.boll_lb, 'yaxis':  ax, 'type': 'scatter',
                'mode': 'lines', 'line': {'width': 1, 'color': 'green'}, 'name': 'Boll Down'}]
        sets += [{'x': df.index, 'y': df.boll, 'yaxis':  ax,
                'type': 'scatter', "fill": 'tonexty',  'line': {'width': 0,},"fillcolor": 'rgba(255, 128, 128,0.2)'}]

        sets += [{'x': df.index, 'y': df.sma9, 'yaxis':  ax, 'type': 'scatter',
                  'mode': 'lines', 'line': {'width': 1, 'color': 'blue'}, 'name': 'sma9'}]

        sets += [{'x': df.index, 'y': df.sma50, 'yaxis':  ax, 'type': 'scatter',
                  'mode': 'lines', 'line': {'width': 1, 'color': 'darkblue'}, 'name': 'sma50'}]

        sets += [{'x': df.index, 'y': df.sma100, 'yaxis':  ax, 'type': 'scatter',
                  'mode': 'lines', 'line': {'width': 1, 'color': 'orange'}, 'name': 'sma100'}]

    

        df = FinI.add_levels(df)
        for i, r in df.iterrows():
            if r.price_level is not None:
                sets += [{'x': [i, df.iloc[-1].name], 'y': [r.price_level,r.price_level], 'yaxis':  ax, 'type': 'scatter',
                          'mode': 'lines', 'line': {'width': 1, 'color': 'brown', "dash": "dot"}, 'name': ''}]
        # print(levels)
        # sets += [{'x': df.index, 'y': df.price_level, 'yaxis':  ax, 'type': 'scatter',
        #           'mode': 'lines', 'line': {'width': 1, 'color': 'orange'}, 'name': 'sma100'}]


        # data += PlotP.plot_rsi(df, ax="y2")
        # data += PlotP.plot_macd_boll(df=df, ax="y3")
        # st.area(df, x=df.index, y=df.boll, color="continent",
        #         line_group="country")
        # fig = go.Figure(data=sets)
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            vertical_spacing=0.009, horizontal_spacing=0.009, row_width=[0.2, 0.1, 0.5], specs=[[{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}]])
        fig.add_traces(data=sets, cols=1, rows=1)
        if m_df_spy is not None:
            fig.add_trace({'x': m_df_spy.index, 'y': m_df_spy.oc_mean,  'type': 'scatter', 'yaxis':  "y",
                       'mode': 'lines', 'line': {'width': 1, 'color': 'red'}, 'name': 'SPY'}, 1, 1, secondary_y=True,)

        # data = PlotP.plot_rsi(df, ax="y")
        # fig.add_traces(data, 3, 1)
        mb = PlotP.plot_macd_boll(df=df, ax="y")
        fig.add_traces(data=mb, rows=3, cols=1,
                       )
        
        rsi = PlotP.plot_rsi(df=df, ax="y3")
        fig.add_traces(data=rsi, rows=3, cols=1,
                       secondary_ys=[True, True, True, True, True])
        
        df.loc[df.open > df.close, "vol_color"] = "red"
        df.loc[df.open <= df.close, "vol_color"] = "green"
        # print(df.vol_color)
        
        fig.add_trace({'x': df.index, 'y': df.volume,
                       'type': 'bar', 'name': 'Volume', 'marker_color': df.vol_color}, 2, 1, secondary_y=False, )


        # fig['layout']['margin'] = {'l': 30, 'r': 10, 'b': 50, 't': 25}
        fig.update_yaxes(showgrid=True, zeroline=False, showticklabels=True,
                         showspikes=True, spikemode='across', spikesnap='cursor', showline=True, spikedash='dash', spikethickness=0.5)


        fig.update_xaxes(showgrid=True, zeroline=False, rangeslider_visible=False, showticklabels=True,
                         showspikes=True,  showline=True, spikemode='across', spikesnap='cursor', spikedash='dash', spikethickness=0.5)

        fig.update_layout(autosize=True, height=600,
                          hoverdistance=1, hovermode='y', spikedistance=10000,
                      
                          )
        st.plotly_chart(fig, use_container_width=True,
                        use_container_height=True, template="plotly_dark",)


        # st.line_chart(df.close)




    def _max_width_(self):
        max_width_str = f"max-width: 2000px;max-height:1000px;"
        st.markdown(
            f"""
        <style>
        .reportview-container .main .block-container{{
            {max_width_str}
        }}
        </style>
        """,
            unsafe_allow_html=True,
        )



if __name__ == '__main__':

    rd = RunData()
    # rd.app.run_server(debug=True)

