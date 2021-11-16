import sys
args = sys.argv
sys.path.append("../")
from market_db import Database, TableName
from typing import Tuple
import numpy
import matplotlib.pyplot as plt
from numpy.core.defchararray import array
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


class MarketLSTM():

    def __init__(self, df) -> None:
        self.df = df
        numpy.random.seed(7)
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.example1(self.df)

    def example1(self, dataset):
        look_back = 3
        dataset = self.prepare_data(dataset)
        train, test = self.split_data(dataset)
        trainX, trainY = self.create_dataset(train, look_back)
        testX, testY = self.create_dataset(test, look_back)

        # reshape input to be [samples, time steps, features]
        trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

        self.create_lstm(look_back, trainX, trainY, testX, testY, dataset)

    def create_lstm(self, look_back, trainX, trainY, testX, testY, dataset):

        # create and fit the LSTM network
        model = Sequential()
        model.add(LSTM(4, input_shape=(1, look_back)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(trainX, trainY, epochs=100, batch_size=30, verbose=2)

        # make predictions
        trainPredict = model.predict(trainX)
        testPredict = model.predict(testX)
        # invert predictions
        trainPredict = self.scaler.inverse_transform(trainPredict)
        trainY = self.scaler.inverse_transform([trainY])
        testPredict = self.scaler.inverse_transform(testPredict)
        testY = self.scaler.inverse_transform([testY])
        # calculate root mean squared error
        trainScore = math.sqrt(mean_squared_error(
            trainY[0], trainPredict[:, 0]))
        print('Train Score: %.2f RMSE' % (trainScore))
        testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
        print('Test Score: %.2f RMSE' % (testScore))

        # shift train predictions for plotting
        trainPredictPlot = numpy.empty_like(dataset)
        trainPredictPlot[:, :] = numpy.nan
        trainPredictPlot[look_back:len(
            trainPredict)+look_back, :] = trainPredict

        # shift test predictions for plotting
        testPredictPlot = numpy.empty_like(dataset)
        testPredictPlot[:, :] = numpy.nan
        testPredictPlot[len(trainPredict)+(look_back*2) +
                        1:len(dataset)-1, :] = testPredict

        # plot baseline and predictions
        plt.plot(self.scaler.inverse_transform(dataset))
        plt.plot(trainPredictPlot)
        plt.plot(testPredictPlot)
        plt.show()

    def prepare_data(self, dataset) -> None:

        # normalize the dataset

        dataset = self.scaler.fit_transform(dataset)
        return dataset

    def split_data(self, dataset):
        train_size_perc = 0.67
        # split into train and test sets
        train_size = int(len(dataset) * train_size_perc)
        test_size = len(dataset) - train_size
        train, test = dataset[0:train_size,
                              :], dataset[train_size:len(dataset), :]
        print(len(train), len(test))
        return train, test

    # convert an array of values into a dataset matrix

    def create_dataset(self, dataset: list, look_back=1) -> Tuple[numpy.array, numpy.array]:
        dataX, dataY = [], []
        for i in range(len(dataset)-look_back-1):
            a = dataset[i:(i+look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return numpy.array(dataX), numpy.array(dataY)

# Manual run from command line
for arg in args:
    if arg=="run":
        db = Database()
        m_df = db.load_data(
            table_name=TableName.DAY,  time_from="-380d", symbols=["TSLA"])
        m_df_spy = db.load_data(
            table_name=TableName.DAY,  time_from="-380d", symbols=["SPY"])

        lstm = MarketLSTM(m_df_spy[["close"]])
