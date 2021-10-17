# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
from dash.dependencies import Input, Output
import sys
sys.path.append('../')
from alpaca_examples.market_db import Database
from alpaca_examples.utils import Utils
from alpaca_examples.fin_i import FinI
import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
from stockstats import StockDataFrame as sdf

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
class RunDash():
    def __init__(self):
        self.db = Database()
        self.df = self.db.load_data("p_day", symbols=['SPY'],time_from="-120d")
        self.df = sdf.retype(self.df)
        self.df = FinI.add_indicators(self.df)
        self.app = self.get_home_page(self.df)
        self.fig = None
        # print(self.df)
       

    def get_home_page(self, df):
        # print(df.index)
        self.fig = go.Figure()
        symbols = ["NVDA","PLUG","AMD","SPY"]


        self.fig.add_trace(
            go.Candlestick(x=df.index,
                            open=df['open'],
                            high=df['high'],
                            low=df['low'],
                            close=df['close']),)

        self.fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df.sma9
            ))
      
   
        app.layout = html.Div(children=[
            html.H1(children='Stock prices'),

            dcc.Dropdown(
                id='buyed_symbols',
                options=[{'label': i, 'value': i}
                         for i in symbols],
                value='symbol'
            ),
            html.Br(),
            html.Div(id="output"),
            
            dcc.Graph(
                id='price-graph',
                figure=self.fig
            )
        ])
        return app
    
@app.callback(
    Output(component_id='price-graph', component_property='figure'),
    Input(component_id='buyed_symbols', component_property='value')
) 

def update_figure(input_value):
    if len(input_value) > 1:
        input_value = "SPY"
        
    db = Database()
    df = db.load_data("p_day", symbols=input_value, time_from="-120d")
    df = sdf.retype(df)
    df = FinI.add_indicators(df)
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(x=df.index,
                       open=df['open'],
                       high=df['high'],
                       low=df['low'],
                       close=df['close']),)
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df.sma9
        ))
    
    fig.update_layout(transition_duration=500)

    return fig


if __name__ == '__main__':
    
    rd = RunDash()
    rd.app.run_server(debug=True)

