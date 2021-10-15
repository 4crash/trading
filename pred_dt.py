import matplotlib as mpl

import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, BaseDecisionTree
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from alpaca_examples.market_db import Database, TableName
from alpaca_examples.back_tester import BackTest
from alpaca_examples.fin_i import FinI
import ta
from stockstats import StockDataFrame as sdf
# from matplotlib import inline


class PredDT(object):
    """
    docstring
    """
    def __init__(self):
        """
        docstring
        """
        self.db = Database()
        self.df = pd.DataFrame()
        self.bt = BackTest()
    
    def dt_prediction(self):
        self.db.set_time_from("730d")
        self.df = self.db.load_data(TableName.DAY,symbols='SPY')
        # print(self.df)
        self.add_sma(30)
        self.add_sma(100)
        print(self.df)
        pass
    def add_sma(self, days):
        self.df["sma"+str(days)] = self.df['close'].rolling(window=days).mean()
        
    def plot_data(self):

        self.df.plot(kind="line", use_index=True,
                y=["close","sma30","sma100"], legend=False, color=["b","orange","y"], linewidth=2, alpha=0.7)
        plt.show()

    def heatmap(self):
        self.db.set_time_from("600d")
        self.df = self.db.load_data(TableName.DAY, symbols='CHWY')
        # calculate the correlation matrix
        
        # self.df = ta.add_all_ta_features(self.df,open="open",close="close",high="high",low="low", volume="volume")
        stocks = sdf.retype(self.df)
        stocks = FinI.add_indicators(stocks)
        stocks.get("macd")
        stocks.get("kdjk")
        stocks.get("boll")
        stocks.get("sma")
        print(stocks)
        # stocks = FinI.add_day_types(stocks)
        
        stocks.drop(columns=["open", "high", "low",
                           "sym", "sector", ], inplace=True)
        print(stocks[['volume',"yearweek"]])
        corr = stocks.corr()
        corr = corr.round(1)
        # print(corr)

        # plot the heatmap
        sns.heatmap(corr, annot=True, cmap="coolwarm",
                    xticklabels=corr.columns,
                    yticklabels=corr.columns)
        plt.show()
    
    def prediction(self):
        df = self.df[['close']]
        future_days = 25
        # create the future date set X and convert it ti numpy array and remove last 'x' rows/days
        df['Prediction'] = df[['close']].shift(-future_days)
        X=np.array(df.drop(['Prediction'],1))[:-future_days]
        # create the target data cell and get all of the target data 
        y=np.array(df['Prediction'])[:-future_days]
        # print(y)
        # lets split the data into 75% training and 25 %tresting
        x_train,x_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
        
        # create decision tree regresor
        tree = DecisionTreeRegressor().fit(x_train, y_train)
        # create regresssion model
        lr = LinearRegression().fit(x_train, y_train)
        # get the last x rows of the future data set
        x_future = df.drop(['Prediction'],1)[:-future_days]
        x_future = x_future.tail(future_days)
        x_future = np.array(x_future)
        
        tree_prediction = tree.predict(x_future)
        print(tree_prediction)
        lr_prediction = lr.predict(x_future)
        print()
        print(lr_prediction)
        
        # visualise predicted data
        predictions = tree_prediction
        valid = df[X.shape[0]:]
        valid['Predictions'] = predictions
        plt.figure(figsize=(16,8))
        plt.title('days')
        plt.xlabel('days')
        plt.gca().fmt_xdata = mpl.dates.DateFormatter('%Y-%m-%d')
        plt.grid(axis="both")
        plt.ylabel("prediction")
        plt.plot(df['close'])
        plt.plot(valid[['close','Predictions']])
        plt.legend(['Orig','Val', 'Pred'])
        plt.show()
        
        # print(y)
        
        # print(X)

pred = PredDT()
pred.heatmap()
# pred.dt_prediction()
# pred.plot_data()
# pred.prediction()
