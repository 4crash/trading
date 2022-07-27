# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import sys
from typing import List, Optional
import pytz
sys.path.insert(1, '.')
sys.path.append('../')
import streamlit as st

from market_lstm import MarketLSTM

from pypfopt import expected_returns
from pypfopt import risk_models
from pypfopt import EfficientFrontier
from TFANN import ANNR
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import time
from stockstats import StockDataFrame as sdf
import pandas as pd
import plotly.express as px
import dash_html_components as html
import dash_core_components as dcc
import dash
import plotly.graph_objects as go
from stock_whisperer import StockWhisperer
from stock_mess import StockMess
from plot_p import PlotP
from fin_i import FinI
from utils import Utils
from market_db import Database, TableName
from check_indicators import CheckIndicators as chi
from buy_sell import BuySell
import asyncio
from datetime import datetime, timedelta
import numpy as np
# import datetime
from plotly.subplots import make_subplots
import logging
import ta
import seaborn as sns
import matplotlib.pyplot as plt
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options


class SlBt():
    selected_stock = None
    timetick = None
    time_from = None
    time_to = None
    selected = {}
    action_type = None
    types = ["sector-stats", "stock-detail",
             "stocks-stats"]
    bs = BuySell()
    db = Database()
    sm = StockMess()
    sw = StockWhisperer()
    submit = None
    warning_check_list = []
    bt_day_gain = {}
    dft = pd.DataFrame()

    def __init__(self):
        self.symbols = self.db.get_symbols(TableName.DAY)
        self.sectors = self.db.get_sectors(TableName.DAY)
        self.app = self.get_home_page()
        
        # self.fig = None
        # logging.info(self.df)

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
        # logging.info(st)
        st.set_page_config(layout="wide")
        self.hide_footer()
        self.left_menu()
        # self.testing_mess()

        self.action_router()

    

    def left_menu(self):

        self.time_from = st.sidebar.text_input(
            "Time from now -1m, -1h, -1d,", value="-120d") 
        self.time_to = st.sidebar.text_input("Time to 1m, 1h, 1d,")

        self.portfolio = st.sidebar.text_input(
            'Portfolio', value= 'GPS,FCX,CYH,VALE,F,AMAT,MU,ABT,HIMX,SBSW'
            )

        self.selected_stock = st.sidebar.selectbox(
            'Select stock',
            self.symbols)

        # if self.selected_stock == 0:
        #     self.selected_stock = "TSLA"



    def action_router(self):

        if st.sidebar.button('test buy'):
            self.buy_alg()

        if st.sidebar.button('Machine learning'):
            self.prepare_data_for_ml()

        if st.sidebar.button('Portfolio opt'):
            self.portfolio_opt()

        if st.sidebar.button('Up-down stats'):
            self.up_down_stats()

        if st.sidebar.button('Logistic regresion'):
            self.logistic_regresion()

        if st.sidebar.button('L.R. Find Best Buy Candidate'):
            self.lr_best_candidate()

    def lr_best_candidate(self):
        df_best_buy = pd.DataFrame()
        symbols = self.db.get_symbols()
        for sym in symbols:
            try:
                st.write(f"filling: {sym}")
                df_lr_raw = self.logistic_regression_raw(sym)
                if df_lr_raw is not None:
                    df_best_buy =  df_best_buy.append(df_lr_raw.tail(1))
                else:
                     st.write(f"No DATA: {sym}")

            except Exception as e:
                st.write(e) 
                
        if "prob_1" in df_best_buy:        
            st.dataframe(df_best_buy.sort_values(by="prob_1"))
        
    def logistic_regression_raw(self, symbol="SPY")-> Optional[pd.DataFrame]:
        #TODO finish thos functioo for find best buy for stock with best Linear Regression probability params
        st.write(self.time_from)
        time_from = datetime.today() - timedelta(minutes=Utils.convert_to_minutes(self.time_from.replace('-','')))
        df = self.db.load_data(
            table_name=TableName.DAY,  time_from=time_from, symbols=[symbol])
        # m_df_spy = self.db.load_data(
        #     table_name=TableName.DAY,  time_from=self.time_from, symbols=["SPY"])
        if len(df) < 1:
            return None
            
        
        df['open-close'] = df['close'] - df['open'].shift(1)
        df['close-close'] = df['close'].shift(-1) - df['close']
        # wrong close close only for research 
        df['close-close-prev'] = df['close'] - df['close'].shift(1)

        df['S_9'] = df['close'].rolling(window=9).mean()
        df['S_20'] = df['close'].rolling(window=20).mean()
        # df['S_50'] = df['close'].rolling(window=50).mean()
        # df['S_200'] = df['close'].rolling(window=200).mean()
        df['Corr_9'] = df['close'].rolling(window=9).corr(df['S_9'])
        df['Corr_20'] = df['close'].rolling(window=9).corr(df['S_20'])
        df['RSI'] = ta.momentum.rsi(close=df['close'])
      

        y = np.where(df['close'].shift(-1) > df['close'], 1, -1)
        df = df[["Corr_9", "open-close", "close-close-prev", "RSI", "S_9"]]
        df = df.dropna()
        X = df.iloc[:, :30]
        # st.write(len(y))
        # st.write(len(X))
        split = int(0.7*len(df))

        X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

        # We will instantiate the logistic regression in Python using ‘LogisticRegression’
        # function and fit the model on the training dataset using ‘fit’ function.
        model = LogisticRegression()
        if len(y_train) > 5:
            model = model.fit(X_train, y_train)

        # Examine coeficients
        # pd.DataFrame(zip(X.columns, np.transpose(model.coef_)))
        # st.write("Examine The Coefficients")
        # st.write(pd.DataFrame(zip(X.columns, np.transpose(model.coef_))))

        #We will calculate the probabilities of the class for the test dataset using ‘predict_proba’ function.
        probability = model.predict_proba(X_test)
        df['Predicted_Signal'] = model.predict(X)
        df = df.tail(len(probability))
        df["prob_0"] = probability[:,0]
        df["prob_1"] = probability[:,1]
        df["sym"] = symbol
        return df
        
        

    def logistic_regresion(self,symbols = None):
        
        if symbols is None:
            symbols = [self.selected_stock]
        # LOGISTIC REGRESSION  sucesfully finish this logic
        st.write(f"Logistic regression: {self.selected_stock}")

        df = self.db.load_data(
            table_name=TableName.DAY,  time_from=self.time_from, symbols=symbols)

        m_df_spy = self.db.load_data(
            table_name=TableName.DAY,  time_from=self.time_from, symbols=["SPY"])

        df_best_buy = pd.DataFrame()
        df_lr_raw = self.logistic_regression_raw()
        st.write("logistic regression RAW SPY")
        if df_lr_raw is not None and df_lr_raw.empty is not None:
            st.dataframe(df_lr_raw)
        
          # remove volume, industry, symbol cols
        # df = df.iloc[:,:4]
        # df =  df.dropna()
        # df = df.iloc[:, :4]
        # df = df.retype(spy)
        df['open-close'] = df['close'] - df['open'].shift(1)
        df['close-close'] = df['close'].shift(-1) - df['close']
        # wrong close close only for research 
        df['close-close-prev'] = df['close'] - df['close'].shift(1)
        # df['close-close-1'] = df['close'] - df['close'].shift(1)
        # df['close-close-2'] = df['close'].shift(1) - df['close'].shift(2)
        # df['close-close-3'] = df['close'].shift(2) - df['close'].shift(3)
        # df['close-close-4'] = df['close'].shift(3)- df['close'].shift(4)
        df, m_df_spy = FinI.get_sizes(df, m_df_spy)
        # df = FinI.add_weekday(df)
        # df = FinI.add_week_of_month(df)
        df = FinI.add_yearweek(df)
 
        # df = FinI.add_levels(df)
        # only first nine rows be aware of this

        # df = FinI.add_indicators(df)
        # df["MACD"] = ta.trend.macd(close=df['close'])
        # df['S_9'] = df['close'].rolling(window=9).mean()
        # df['S_20'] = df['close'].rolling(window=20).mean()
        # df['S_50'] = df['close'].rolling(window=50).mean()

        # df['Corr'] = df['close'].rolling(window=10).corr(df['S_9'])
        # df['RSI'] = ta.momentum.rsi(close= df['close'], n=9)
        # Initialize Bollinger Bands Indicator
     

        # Add Bollinger Bands features
        # df['bb_bbm'] = indicator_bb.bollinger_mavg()
        # df['bb_bbh'] = indicator_bb.bollinger_hband()
        # df['bb_bbl'] = indicator_bb.bollinger_lband()
        
        df['S_9'] = df['close'].rolling(window=9).mean()
        df['S_20'] = df['close'].rolling(window=20).mean()
        # df['S_50'] = df['close'].rolling(window=50).mean()
        # df['S_200'] = df['close'].rolling(window=200).mean()
        df['Corr_9'] = df['close'].rolling(window=9).corr(df['S_9'])
        df['Corr_20'] = df['close'].rolling(window=9).corr(df['S_20'])
        df['RSI'] = ta.momentum.rsi(close=df['close'])
        # df["MACD"] = ta.trend.macd(close=df['close'])
        df = df.fillna(0)
        st.write("Correlation Matrix - Heatmap")
        self.get_heatmap(df)
        # remove open, high, low
        # df = df.iloc[:,3:]
        y = np.where(df['close'].shift(-1) > df['close'], 1, -1)
        st.write("Rows:")
        st.write(len(df))
        # df = df.drop(columns= ["date",
        #                        "yearweek",
        #                        "size_boll_ub",
        #                        "size_boll_lb",
        #                        "size_boll",
        #                        "up_down_row",
        #                        "green_red_row",
        #                        "price_level",
        #                        "weekday",
        #                        "boll",
        #                        "sma9", 
        #                        "close_20_sma",
        #                        "size_top",
        #                        "size_btm",
        #                        "close_20_mstd", 
        #                        "size_top-2",
        #                        "size_top-1",
        #                        "size_btm-2",
        #                        "size_btm-1",
        #                        "boll_ub",
        #                        "boll_lb",
        #                        "size_btm-3",
        #                        "size_top-3",
        #                        "size_sma9",
        #                        "size_sma20",
        #                         "sma20",
        #                        "open",
        #                        "close",
        #                        "high",
        #                        "low",
        #                        "size_body-1",
        #                        "size_body-2",
        #                        "size_body-3",
        #                        "week_in_month",
        #                        "size_body",
        #                        "size_prev_chng",
        #                        "symbol",
        #                        "sector",
        #                        "sym",
        #                        "industry",
        #                        "amount",
        #                        "volume",
                                #  "open-close",
                                # "close-close"
        #                        ])
        df_strategy = df.copy()

        # relatively good negative prediction
        # df = df[["size_sma9"]]
        # values by corelation matrix 
        # df = df[["size_sma20","size_sma9","up_down_row"]]
        # df = df[["green_red_row", "size_sma9", "up_down_row", "size_body"]]
        # df = df[["green_red_row", "size_sma9", "up_down_row", "size_body","volume"]]
        # origin
        # df = df[["Corr_9","Corr_20", "open-close", "close-close-prev", "RSI","sma9"]]
        df = df[["Corr_9", "open-close", "close-close-prev", "RSI","sma9"]]
        # overnight, negative precision 
        # df = df[["green_red_row", "size_body-1","size_body-2","size_body-3", "size_prev_chng","weekday"]]


        # df = df[["close", "sma9", "sma20"]]
        st.write("Dataframe")
        st.dataframe(df)
        X = df.iloc[:, :30]
        st.write(len(y))
        st.write(len(X))
        # split to test dataset
        
        split = int(0.7*len(df))
     
        X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

        # We will instantiate the logistic regression in Python using ‘LogisticRegression’
        # function and fit the model on the training dataset using ‘fit’ function.
        model = LogisticRegression()
        model = model.fit(X_train, y_train)
        
        # Examine coeficients
        pd.DataFrame(zip(X.columns, np.transpose(model.coef_)))
        st.write("Examine The Coefficients")
        st.write(pd.DataFrame(zip(X.columns, np.transpose(model.coef_))))
        
        #We will calculate the probabilities of the class for the test dataset using ‘predict_proba’ function.
        st.write("probability")
        probability = model.predict_proba(X_test)
        st.write(probability)

        st.write("predicted")
        predicted = model.predict(X_test)
        st.write(predicted)

        st.write("Y test")
        st.write(y_test)
        st.write(len(X_test))
        st.write("Confusion matrix")
        conf_matrix = metrics.confusion_matrix(y_test, predicted)
        st.write(conf_matrix)    

        st.write("classification report")
        st.write(metrics.classification_report(y_test, predicted))

        st.write("model accuracy")
        st.write(model.score(X_test, y_test))

        st.write("cross validation accuracy")
        cross_val = cross_val_score(LogisticRegression(), X, y, scoring='accuracy', cv=10)
        st.write(cross_val)
        st.write(cross_val.mean())

        st.write("Trading Strategy")

        df['Predicted_Signal'] = model.predict(X)
        st.write("Predicted signals")
        st.dataframe(df)

        #STRARTEGY count
        st.dataframe(df_strategy)


        # df_strategy = df_strategy.tail(len(df_strategy) - split)
        df_strategy["Predicted_Signal"] = model.predict(X)
        price_change = df_strategy.iloc[len(
            df_strategy)-1].close - df_strategy.iloc[0].close

        st.write("Perfect gain")
        df_strategy["perfect_gain"] = df_strategy["close-close"].where(
            df_strategy["close-close"] > 0).sum()
        st.write(df_strategy["perfect_gain"].iloc[0])
        
        st.write("Invest Gain")
        st.write(price_change)

        st.write("L.G. gain")
        df_strategy["lg_gain"] = df_strategy["close-close"].where(
        df_strategy["Predicted_Signal"] > 0).sum()
        st.write(df_strategy["lg_gain"].iloc[0])
        
        gap = -2
        st.write("sma9 gain: " + str(gap))
        df_strategy["sma9_gain"] = df_strategy["close-close"].where(
        df_strategy["size_sma9"] > gap).sum()
        st.write(df_strategy["sma9_gain"].iloc[0])

        st.write("sma9 gain 0")
        df_strategy["sma9_gain"] = df_strategy["close-close"].where(
            df_strategy["size_sma9"] > 0).sum()
        st.write(df_strategy["sma9_gain"].iloc[0])

        st.write("sma20 gain")
        df_strategy["sma20_gain"] = df_strategy["close-close"].where(
        df_strategy["size_sma20"] > -2).sum()
        st.write(df_strategy["sma20_gain"].iloc[0])
        # df['Nifty_returns'] = np.log(df['close']/df['close'].shift(1))
        # Cumulative_Nifty_returns = np.cumsum(df[split:]['Nifty_returns'])

        # df['Startegy_returns'] = df['Nifty_returns']* df['Predicted_Signal'].shift(1)
        # Cumulative_Strategy_returns = np.cumsum(df[split:]['Startegy_returns'])

        # st.line_chart(Cumulative_Nifty_returns)
        # st.line_chart(Cumulative_Strategy_returns)
     


    def up_down_stats(self):
       
        m_df = self.db.load_data(
            table_name=TableName.DAY,  time_from=self.time_from, symbols=[self.selected_stock])

        m_df_spy = self.db.load_data(
            table_name=TableName.DAY,  time_from=self.time_from, symbols=["SPY"])
        
        m_df, m_df_spy = FinI.get_sizes(m_df, m_df_spy)
        
        self.candle_chart( m_df, m_df_spy)
        # above_sma9, under_sma9, above_boll,under_boll = {"1":0,"2":0,"3":0,"4":0,"5":0,"6":0}
        ind = pd.DataFrame()
        ws = pd.DataFrame()
        u = 0
        pred = pd.DataFrame()
        
        for index, row in m_df.iterrows():
            
            if row.weekday == 0:
                i = 1
                
                while m_df.iloc[m_df.index.get_loc(index) + i].weekday == m_df.iloc[m_df.index.get_loc(index) + 1 +i ].weekday-1:
                    rw = m_df.iloc[m_df.index.get_loc(index) + i]
                    
                    st.write(str(rw.date) + " - " + str(rw.size_body))
                    i +=1
            st.write(row.week_in_month)   
                    
            
            if True or row.size_sma9 > 0 and row.size_boll > 0:
                if row.size_body > 0:
                    # st.write(u)
                    if u < 0:
                      ind = ind.append([u])
                      u = 0
                      
                    u += 1
                elif row.size_body < 0:
                    if u > 0:
                      ind = ind.append([u])
                    #   st.write(ind)
                      u = 0

                    u -=1
        # 
        st.dataframe(ind)
        st.dataframe(ind.groupby(ind.columns[0]).size())
                
           

    # def get_size_data(self, stocks = None):
        
    #     if stocks is None:
    #         stocks = self.selected_stock
    #     m_df = self.db.load_data(
    #         table_name=TableName.DAY,  time_from=self.time_from, symbols=[stocks])

    #     m_df_spy = self.db.load_data(
    #         table_name=TableName.DAY,  time_from=self.time_from, symbols=["SPY"])
    #     m_df_spy["oc_mean"] = ((m_df_spy.close + m_df_spy.open)/2)
    #     m_df = sdf.retype(m_df)
    #     m_df.get("boll")
    #     m_df = FinI.add_sma(9, m_df)
    #     m_df = FinI.add_weekday(m_df)
    #     m_df = FinI.add_week_of_month(m_df)
    #     m_df = FinI.add_levels(m_df)

    #     m_df["size_top"] = m_df.apply(lambda row: Utils.calc_perc(
    #         row.open, row.high) if row.open > row.close else Utils.calc_perc(row.close, row.high), axis=1)


    #     m_df["size_btm"] = m_df.apply(lambda row: Utils.calc_perc(
    #         row.low, row.close) if row.open > row.close else Utils.calc_perc(row.low, row.open), axis=1)

    #     m_df["size_body"] = m_df.apply(lambda row: Utils.calc_perc(row.open, row.close), axis=1)
    #     m_df["size_sma9"] = m_df.apply(lambda row: Utils.calc_perc(row.sma9, row.close), axis=1)
    #     m_df["size_boll"] = m_df.apply(
    #         lambda row: Utils.calc_perc(row.boll, row.close), axis=1)
    #     m_df["size_boll_ub"] = m_df.apply(
    #         lambda row: Utils.calc_perc(row.boll_ub, row.close), axis=1)
    #     m_df["size_boll_lb"] = m_df.apply(
    #         lambda row: Utils.calc_perc(row.boll_lb, row.close), axis=1)

    #     m_df["size_top-1"] = m_df.shift(1).size_top

    #     m_df["size_btm-1"] = m_df.shift(1).size_btm

    #     m_df["size_body-1"] = m_df.shift(1).size_body

    #     m_df["size_top-2"] = m_df.shift(2).size_top

    #     m_df["size_btm-2"] = m_df.shift(2).size_btm

    #     m_df["size_body-2"] = m_df.shift(2).size_body

    #     m_df["size_top-3"] = m_df.shift(3).size_top

    #     m_df["size_btm-3"] = m_df.shift(3).size_btm

    #     m_df["size_body-3"] = m_df.shift(3).size_body
        
    #     m_df["size_prev_chng"] = (
    #         m_df.open - m_df.shift(1).close) / (m_df.shift(1).close/100)

    #     return m_df, m_df_spy
        
        
    def prepare_data_for_ml(self):
        
        m_df = self.db.load_data(
            table_name=TableName.DAY,  time_from=self.time_from, symbols=[self.selected_stock])

        m_df_spy = self.db.load_data(
            table_name=TableName.DAY,  time_from=self.time_from, symbols=["SPY"])
        
        m_df, m_df_spy = FinI.get_sizes(m_df, m_df_spy)

        
        st.dataframe(m_df)
        self.process_data(m_df)
        self.candle_chart(m_df, m_df_spy)
        self.show_sizes(m_df)
        self.get_heatmap(m_df)
        self.nn_pred(m_df)

    def show_sizes(self,df):
        
        sets = [{'x': df.index, 'y': df.size_body,  'type': 'scatter',
                  'mode': 'lines', 'line': {'width': 1, 'color': 'blue'}, 'name': 'body'}]
        sets += [{'x': df.index, 'y': df.size_prev_chng,  'type': 'scatter',
                  'mode': 'lines', 'line': {'width': 1, 'color': 'green'}, 'name': 'prev day'}]
        sets += [{'x': df.index, 'y': df.weekday,  'type': 'scatter',
                  'mode': 'lines+markers', 'line': {'width': 1, 'color': 'brown'}, 'name': 'weeks'}]
        sets += [{'x': df.index, 'y': df.week_in_month,  'type': 'scatter',
                  'mode': 'lines+markers', 'line': {'width': 1, 'color': 'orange'}, 'name': 'weeks'}]
        
        fig = make_subplots(rows=1, cols=1, shared_xaxes=True,
                            vertical_spacing=0.009, horizontal_spacing=0.009, row_width=[0.5],
                            specs=[[{"secondary_y": True}]]
                            )
        fig.add_traces(data=sets, cols=1, rows=1)


        fig.update_yaxes(showgrid=True, zeroline=False, showticklabels=True,
                         showspikes=True, spikemode='across', spikesnap='cursor', showline=True, spikedash='dash', spikethickness=0.5)

        fig.update_xaxes(showgrid=True, zeroline=False, rangeslider_visible=False, showticklabels=True,
                         showspikes=True,  showline=True, spikemode='across', spikesnap='cursor', spikedash='dash', spikethickness=0.5)

        fig.update_layout(autosize=True, height=600,
                          hoverdistance=1, hovermode='x', spikedistance=1000

                          )
        st.plotly_chart(fig, use_container_width=True,
                        use_container_height=True, template="plotly_dark",)

    def get_heatmap(self, df):
        # fig = px.imshow( df,
        #     x= ["size_body", "size_btm", "size_top", "size_body-1", "size_top-1",  "size_btm-1", "size_prev_chng"],
        #     y=["size_body", "size_btm", "size_top", "size_body-1", "size_top-1",  "size_btm-1", "size_prev_chng"],
        #     )
      # fig = px.scatter_matrix(df)
        # fig.show()

        df_c = pd.DataFrame(df, columns=["size_body", "size_prev_chng", "weekday","week_in_month", "size_btm", "size_top", 'volume',
                                         "size_body-1", "size_top-1", "size_btm-1",
                                         "size_body-2", "size_top-2", "size_btm-2", 
                                         "size_body-3",  "size_top-3", "size_btm-3", 
                                         "size_sma9","size_sma20", "size_boll", "size_boll_ub", "size_boll_lb", "green_red_row", "up_down_row"])
        fig, ax = plt.subplots()
        sns.heatmap(df_c.corr(), ax=ax)
        st.write(fig)
        # st.write(df_c.corr())
      

    def portfolio_opt(self):
        portfolio = self.portfolio.split(",")
        df = self.db.load_data(TableName.DAY, time_from=self.time_from, symbols=portfolio)
        # df = FinI.add_date_col(df)
        df = df[["close","sym"]]
        df2 = df.copy()
        df2 = df2.drop(columns=["close","sym"])
        for sym in portfolio:
            df2[sym] = df[df["sym"] == sym]["close"]
        
        df2 = df2.drop_duplicates()
        
        # df = df.transpose()
        # df.index = pd.to_datetime(df['date']).astype(int) / 10**9
        st.write(df2)
        # Calculate expected returns and sample covariance
        mu = expected_returns.mean_historical_return(df2)
        S = risk_models.sample_cov(df2)

        # Optimize for maximal Sharp ratio
        ef = EfficientFrontier(mu, S)
        raw_weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()
        # ef.save_weights_to_file("weights.csv")  # saves to file
        st.write(cleaned_weights)
        mu, sigma, sharpe = ef.portfolio_performance(verbose=True)
        st.write("Expected annual return: {:.1f}%".format(100 * mu))
        st.write("Annual volatility: {:.1f}%".format(100 * sigma))
        st.write("Sharpe Ratio: {:.2f}".format(sharpe))

    def process_data(self, m_df):
        from sklearn import datasets
        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier(n_neighbors=6)
        # knn.fit(m_df.iloc[0,len(m_df/2)], m_df[len(m_df)/2,len(m_df)])
        
    def nn_pred(self, df):
        # st.dataframe(df[["close"]])
        #Number of neurons in the input, output, and hidden layers

        stock_data = df[["close"]]
        lstm = MarketLSTM(stock_data)
        
        # stock_data["seconds"] = stock_data.apply(lambda row: row.index.astype(int) / 10**9)
        # stock_data = stock_data[["seconds", "close"]]
        # stock_data = scale(stock_data)
        # #gets the price and dates from the matrix
        # prices = stock_data[:, 1].reshape(-1, 1)
        # dates = stock_data[:, 0].reshape(-1, 1)
        
        # chart_data = pd.DataFrame([dates[:,0],prices[:,0]])
        # chart_data = chart_data.transpose()
        # self.nn_train(dates,prices)

    def nn_train(self, dates, prices):
        pass
 
    def nn_train_OLD(self, dates, prices):
        input = 1
        output = 1
        hidden = 50
        #array of layers, 3 hidden and 1 output, along with the tanh activation function 
        layers = [('F', hidden), ('AF', 'tanh'), ('F', hidden), ('AF', 'tanh'), ('F', hidden), ('AF', 'tanh'), ('F', output)]
        #construct the model and dictate params
        mlpr = ANNR([input], layers, batchSize = 256, maxIter = 20000, tol = 0.2, reg = 1e-4, verbose = True)

        #number of days for the hold-out period used to access progress
        holdDays = 5
        totalDays = len(dates)
        #fit the model to the data "Learning"
        mlpr.fit(dates[0:(totalDays-holdDays)], prices[0:(totalDays-holdDays)])
        

        # stocks_day["sym"] = stocks_day["sym"].astype('category')

    def candle_chart(self, df, m_df_spy):
        sets = [{'x': df.index, 'open': df.open, 'close': df.close, 'yaxis':  "y1",
                 'high': df.high, 'low': df.low,  "hovertext":"", 'type': 'candlestick', "opacity": 1, 'line': {'width': 0.5, }}]
        sets += [{'x': df.index, 'y': df.boll_ub,  'type': 'scatter', 'yaxis':  "y1",
                  'mode': 'lines', 'line': {'width': 1, 'color': 'green'}, 'name': 'Boll UP'}]
        sets += [{'x': df.index, 'y': df.boll, 'yaxis':  "y1",
                  'type': 'scatter', "fill": 'tonexty', 'line': {'width': 0, }, "fillcolor": 'rgba(128, 255, 128,0.2)'}]
        sets += [{'x': df.index, 'y': df.boll_lb,  'type': 'scatter', 'yaxis':  "y1",
                  'mode': 'lines', 'line': {'width': 1, 'color': 'green'}, 'name': 'Boll Down'}]
        sets += [{'x': df.index, 'y': df.boll, 'yaxis':  "y1",
                  'type': 'scatter', "fill": 'tonexty',  'line': {'width': 0, }, "fillcolor": 'rgba(255, 128, 128,0.2)'}]

        sets += [{'x': df.index, 'y': df.boll_ub,  'type': 'scatter', 'yaxis':  "y1",
                  'mode': 'lines', 'line': {'width': 1, 'color': 'green'}, 'name': 'Boll UP'}]
        sets += [{'x': df.index, 'y': df.sma9,  'type': 'scatter', 'yaxis':  "y1",
                  'mode': 'lines', 'line': {'width': 1, 'color': 'blue'}, 'name': 'Boll Down'}]

        for i, r in df.iterrows():
            if r.price_level is not None:
                sets += [{'x': [r.date, df.iloc[-1].date], 'y': [r.price_level, r.price_level], 'type': 'scatter', 'yaxis':  "y1",
                          'mode': 'lines', 'line': {'width': 1, 'color': 'brown', "dash": "dot"}, 'name': ''}]

       
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.009, horizontal_spacing=0.009, row_width=[0.1, 0.5],
                            specs=[[{"secondary_y": True}],
                                   [{"secondary_y": True}]]
                            )
        fig.add_traces(data=sets, cols=1, rows=1)

        # sets = []
         
        fig.add_trace({'x': m_df_spy.index, 'y': m_df_spy.oc_mean,  'type': 'scatter', 'yaxis':  "y2",
                            'mode': 'lines', 'line': {'width': 1, 'color': 'red'}, 'name': 'SPY'}, 1, 1, secondary_y=True,)

        df.loc[df.open > df.close, "vol_color"] = "red"
        df.loc[df.open <= df.close, "vol_color"] = "green"
        fig.add_trace({'x': df.index, 'y': df.volume, 
                       'type': 'bar', 'name': 'Volume', 'marker_color': df.vol_color}, 2, 1, secondary_y=False, )

        fig.update_yaxes(showgrid=True, zeroline=False, showticklabels=True,
                         showspikes=True, spikemode='across', spikesnap='cursor', showline=True, spikedash='dash', spikethickness=0.5)

        fig.update_xaxes(showgrid=True, zeroline=False, rangeslider_visible=False, showticklabels=True,
                         showspikes=True,  showline=True, spikemode='across', spikesnap='cursor', spikedash='dash', spikethickness=0.5)

        fig.update_layout(autosize=True, height=600,
                          hoverdistance=1, hovermode='y', spikedistance=10000

                          )
        st.plotly_chart(fig, use_container_width=True,
                        use_container_height=True, template="plotly_dark",)
        

    def buy_alg(self, stats_time = None):
        
        if stats_time is None:
            stats_time="-60d"
        backtest_time = "-14d"    
        
        self.db.last_date = self.db.get_date_format(backtest_time)
        spy = self.db.load_data(table_name=TableName.DAY_FS, symbols=["SPY"], time_from = stats_time)
        # spy = sdf.retype(spy)
        # spy = FinI.add_indicators(spy)
        
        stocks_day = self.db.load_data(table_name=TableName.DAY_FS,  time_from=stats_time)
        stocks_day["sym"] = stocks_day["sym"].astype('category')
        
        
                
        stocks_15 = self.db.load_data(table_name=TableName.MIN15,  time_from=backtest_time)
        stocks_15["sym"] = stocks_15["sym"].astype('category')
        spy_15 = stocks_15[stocks_15["sym"] == "SPY"]

        # logging.info(spy)
        symbols = self.db.get_symbols()
        
        
        # for testing performance reason here are only few stocks
        symbols = ["INTC","BYND","ZM","NKE","HIMX","JKS","ENPH","DUK","GE","DIS","LEVI","NVAX","SLCA","GPS"]
        # iterate over days in market 
        spy2 = spy.tail(20)
        
        # retype to stocks dataframe
        if not isinstance(stocks_day, sdf):
            stocks_day = sdf.retype(stocks_day)
       
        for index, spy_row_day in spy2.iterrows():
            st.write("spy: " + str(spy_row_day))
            st.write("DATE" + str(index))
        
            for symbol in symbols:
                # load stocks for stats
                stocks_day_sym = stocks_day[stocks_day["sym"] == symbol]
                # stocks_day_sym = FinI.add_indicators(stocks_day_sym)
                # stocks_day_sym = FinI.add_sma(9, stocks_day_sym)
                # stocks_day_sym = FinI.add_sma(50, stocks_day_sym)
                # stocks_day_sym.get('boll')
                # stocks_day_sym.get('volume_delta')
                # stocks_day_sym.get('macd')
                # stocks_day_sym.get('kdjk')
                # stocks_day_sym.get('open_-2_r')
                # logging.info(" -------------------------------------------------  --------------")
                # logging.info(stocks_day_sym)
                # stocks_day_sym = FinI.add_day_types(stocks_day_sym)
                # stocks_day_sym = FinI.add_levels(stocks_day_sym)
                stocks_15_sym = stocks_15[stocks_15["sym"] == symbol]
                stock_rows15 = stocks_15_sym.loc[stocks_15_sym.index <= pytz.utc.localize(index)]
                # logging.info(stock_rows15.iloc[-1].sym + " | " + str(stock_rows15.index[-1]))
                                    
                if len(stock_rows15) > 1:
                    self.back_buy_best_stock(stocks_day_sym, index)
                    # st.write((stock_rows15.iloc[-1].sym + " | " + str(stock_rows15.index[-1])))
                    logging.info(stock_rows15.iloc[-1].sym + " | " + str(stock_rows15.index[-1]))
                    self.buy_sell(
                        stocks_day_sym, stock_rows15, spy, spy_15, spy_row_day)

            st.write(self.dft)

    def back_buy_best_stock(self, df, index, days_back = -2):
        # self.bt_day_gain[index]
        df2 = pd.DataFrame()
        if df is not None and len(df) > 0:
            dff = df[df.index <= index]
            if len(dff)>1:
                gain = Utils.calc_perc(
                    dff.iloc[days_back].close, dff.iloc[-1].close)
                df2 = dff.iloc[days_back]
                df2["gain"] = gain
                df2["days_back"] = days_back
              
                self.dft = self.dft.append(df2)
                # st.write(self.dft)

                      
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
               
    def buy_sell(self, stocks_day, stocks_15, spy, spy_row15, spy_row_day):
        """ By sell backtrading simaltion with specific strategy

        Args:
            stocks_day ([type]): [description]
            stocks_15 ([type]): [description]
            spy ([type]): [description]
            spy_row15 ([type]): [description]
            spy_row_day ([type]): [description]
        """

        # self.stocks = self.stocks.append(stocks_day.iloc[-1])
        self.db.last_date = stocks_15.iloc[-1].name
        logging.info(self.db.last_date)
        sym = stocks_day.iloc[-1].sym
        # send only one buy suggestion per day
        hash_warn = hash(
            sym + str(stocks_15.index[-1].strftime("%d/%m/%Y")))

        hl = FinI.get_fib_hl(stocks_day,  stocks_day.iloc[-1].close)

        # logging.info(str(stocks_day))
        if len(stocks_day) > 1:
            # stocks_day = sdf.retype(stocks_day)
            # stocks_day = FinI.add_indicators(stocks_day)
            stocks_day.sort_index(ascending=True, inplace=True)
            stocks_day["flpd"] = Utils.calc_flpd(stocks_day)
            hl = FinI.get_fib_hl(stocks_day, stocks_15.iloc[-1].close)
            #OLD CHECK SELL moved to this Fce
            # self.check_sell(stocks_15.iloc[-1])

            earnings, sentiment, financials = self.db.get_fundamentals(
                stocks_day.iloc[-1]["sym"])
            self.set_mess_data(fin=financials, sen=sentiment,
                               earn=earnings, spy=spy, stc=stocks_day)

            # st.write("BUYSELL OPEN trades amount:" + str(len(self.bs.buy_sell_open)))
            #-----------------------------SELL------------------------------------------
            if len(self.bs.buy_sell_open) > 0:

                logging.info("not selled stocks:" +
                      str(self.bs.buy_sell_open[self.bs.buy_sell_open.state == "open"]))
                logging.info("Current SYM: " + str(stocks_15.iloc[-1].sym))
                
                
                bs = self.bs.buy_sell_open[self.bs.buy_sell_open.sym == stocks_15.iloc[-1].sym]
                
                
                if len(bs) > 0:
                    st.write("--------------------SELLING----------------------------")
                    for index, row in bs.iterrows():
                        if Utils.calc_perc(stocks_15.iloc[-1].close, hl["h"], 2) <= 1 or \
                                chi.check_sma9(stocks=stocks_day, live_stocks=stocks_15, buy=False):

                            # self.bs.buy_sell_open[(self.bs.buy_sell_open.sym == stocks_15.iloc[-1].sym) & (self.bs.buy_sell_open.state == "open")] = self.bs.sell_stock_t(stocks_15.iloc[-1],  sell_price=stocks_15.iloc[-1].close,
                            sold_stock = self.bs.sell_stock_t(sym=stocks_15.iloc[-1].sym, 
                                                              price=stocks_15.iloc[-1].close, 
                                                              sell_date=stocks_15.iloc[-1].name)
                            st.write(sold_stock)
                            # st.write(sym + " | Selling Profit: " + str(stocks_15.iloc[-1].sym) + " | " + str(stocks_15.index[-1]) + " | " +
                            #                                      " | B.S.:" + str(row["buy"]) + "/" + str(stocks_15.iloc[-1].close) +
                            #                                      " | " + str(Utils.calc_perc(row["buy"], stocks_15.iloc[-1].close)) +
                            #                                      "% | " + str(stocks_15.iloc[-1].name) +
                            #                                      " | ", stocks_15.iloc[-1].name)


                            # asyncio.run(self.sm.a_mail_sym_stats(sym, "Selling Profit: " + str(stocks_15.iloc[-1].sym) + " | " + str(stocks_15.index[-1]) + " | " +
                            #                                      " | B.S.:" + str(row["buy"]) + "/" + str(stocks_15.iloc[-1].close) +
                            #                                      " | " + str(Utils.calc_perc(row["buy"], stocks_15.iloc[-1].close)) +
                            #                                      "% | " + str(stocks_15.iloc[-1].name) +
                            #                                      " | ", stocks_15.iloc[-1].name), debug=True)

            #------------------------------------------------------------BUY---------------------------------------------
            if hash_warn not in self.warning_check_list \
                    and stocks_day.iloc[-1]["flpd"] > 0 \
                    and chi.check_financials(financials) \
                    and (len(self.bs.buy_sell_open) == 0 or stocks_15.iloc[-1].sym not in self.bs.buy_sell_open[self.bs.buy_sell_open.state == "open"].sym)  \
                    and chi.check_pre_sma9(stocks_day, live_stocks=stocks_15):

                # and chi.check_sentiment(sentiment) \
                # and chi.check_financials(financials) \

                self.warning_check_list.append(hash_warn)
                self.bs.buy_stock_t(
                    stocks_15.iloc[-1],
                    stocks_15.iloc[-1].close,
                    table_name="buy_sell_lt",
                    profit_loss={"profit": hl["h"], "loss": hl["l"]})

                st.write("---------------------------BUYiNG-------------------------------")
                # logging.info(self.bs.buy_sell)
                
                st.write(sym + " | Buy: " +
                                                     str(stocks_15.iloc[-1].sym) + " | " + str(stocks_15.index[-1]) + " | " +
                                                     str(stocks_15.iloc[-1].close) + " | " +
                                                     str(hl), stocks_15.iloc[-1].name)
                # asyncio.run(self.sm.a_mail_sym_stats(sym, "Buy: " +
                #                                      str(stocks_15.iloc[-1].sym) + " | " + str(stocks_15.index[-1]) + " | " +
                #                                      str(stocks_15.iloc[-1].close) + " | " +
                #                                      str(hl), stocks_15.iloc[-1].name))
   
    def set_time_to(self):
        if self.time_from is None:
            time_from = "-180d"
        else:
            time_from = self.time_from

        return time_from

   

    def hide_footer(self):

        hide_streamlit_style = """
            <style>
            #MainMenu {visibility: visible;}
            footer {visibility: hidden;}
            </style>
            """
        st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    

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

    rd = SlBt()
    # rd.app.run_server(debug=True)
