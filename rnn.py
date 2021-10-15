# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import timedelta
from sklearn.preprocessing import RobustScaler
plt.style.use("bmh")
from stockstats import StockDataFrame as Sdf
# Technical Analysis library
# import ta
# Neural Network library
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

import sys
sys.path.append('../')

from alpaca_examples.back_tester import BackTest, TableName
from alpaca_examples.indicators import Indicators
# Loading in the Data
import ta

class Rnn():
    n_per_in = 8
    n_per_out = 3
    df = pd.DataFrame()
    close_scaler = RobustScaler()
    model = Sequential()
    
        
    def __init__(self):
        
        # print(data)
        self.df = self.preprocess_data()
        self.df.dropna(inplace=True)
        self.df = self.df.fillna(self.df.mean())
        self.df = self.df.loc[(self.df!=0).any(axis=1)]
        self.n_features = self.df.shape[1]
        
        # print(self.df)
        self.X, self.y = self.split_sequence(self.df.to_numpy(), self.n_per_in, self.n_per_out)
       
  
   
    def preprocess_data(self):
        
        
        bt = BackTest()
        bt.db.set_time_from(60*24*1000)
        data = bt.load_data(TableName.DAY , 'SPY')
        ## Datetime conversion
        data['Date'] = pd.to_datetime(data.index, utc=True)

        # Setting the index;
        data.set_index('Date', inplace=True)
        # Only using the last 1000 days of data to get a more accurate representation of the current market climate
        # data = ta.add_all_ta_features(data, open="open", high="high", low="low", close="close", volume="volume", fillna=True)
        data = Sdf.retype(data)
        data.get("boll")
        data.get("close_9_sma")
        data.drop(['open', 'high', 'low', 'sym', 'sector','amount'], axis=1, inplace=True)
        # new_type = data.iloc[2:2]
        # print(type(new_type))
        # data['volume_fi'].round(5)
        # data = data.astype().round(3)
        print(data)
        # exit()
    
        data = data.tail(1000)
        # print(data.head())
        # close_scaler = RobustScaler()
        self.close_scaler.fit(data[['close']])
        
        # Normalizing/Scaling the DF
        scaler = RobustScaler()
        
        data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)
        
        return data
    
    def split_sequence(self, seq, n_steps_in, n_steps_out):
        """
        Splits the multivariate time sequence
        """
        
        # Creating a list for both variables
        X, y = [], []
        
        for i in range(len(seq)):
            
            # Finding the end of the current sequence
            end = i + n_steps_in
            out_end = end + n_steps_out
            
            # Breaking out of the loop if we have exceeded the dataset's length
            if out_end > len(seq):
                break
            
            # Splitting the sequences into: x = past prices and indicators, y = prices ahead
            seq_x, seq_y = seq[i:end, :], seq[end:out_end, 0]
            
            X.append(seq_x)
            y.append(seq_y)
        
        return np.array(X), np.array(y)
    
    
    def visualize_training_results(self, results):
        """
        Plots the loss and accuracy for the training and testing data
        """
        history = results.history
        plt.figure(figsize=(16,5))
        plt.plot(history['val_loss'])
        plt.plot(history['loss'])
        plt.legend(['val_loss', 'loss'])
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()
        
        plt.figure(figsize=(16,5))
        plt.plot(history['val_accuracy'])
        plt.plot(history['accuracy'])
        plt.legend(['val_accuracy', 'accuracy'])
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.show()
        
        
    def layer_maker(self, n_layers, n_nodes, activation, drop=None, d_rate=.5):
        """
        Creates a specified number of hidden layers for an RNN
        Optional: Adds regularization option - the dropout layer to prevent potential overfitting (if necessary)
        """
        
        # Creating the specified number of hidden layers with the specified number of nodes
        for x in range(1,n_layers+1):
            self.model.add(LSTM(n_nodes, activation=activation, return_sequences=True))

            # Adds a Dropout layer after every Nth hidden layer (the 'drop' variable)
            try:
                if x % drop == 0:
                    self.model.add(Dropout(d_rate))
            except:
                pass
            
            
    def validater(self, n_per_in, n_per_out):
        """
        Runs a 'For' loop to iterate through the length of the DF and create predicted values for every stated interval
        Returns a DF containing the predicted values for the model with the corresponding index values based on a business day frequency
        """
        
        # Creating an empty DF to store the predictions
        predictions = pd.DataFrame(index=self.df.index, columns=[self.df.columns[0]])

        for i in range(self.n_per_in, len(self.df)-self.n_per_in, self.n_per_out):
            # Creating rolling intervals to predict off of
            x = self.df[-i - self.n_per_in:-i]

            # Predicting using rolling intervals
            yhat = self.model.predict(np.array(x).reshape(1, self.n_per_in, self.n_features))
            print(yhat)
            # Transforming values back to their normal prices
            yhat = self.close_scaler.inverse_transform(yhat[0])[0]

            # DF to store the values and append later, frequency uses business days
            pred_df = pd.DataFrame(yhat, 
                                index=pd.date_range(start=x.index[-1], 
                                                    periods=len(yhat), 
                                                    freq="B"),
                                columns=[x.columns[0]])

            # Updating the predictions DF
            predictions.update(pred_df)
            
        return predictions


    def val_rmse(self, df1, df2):
        """
        Calculates the root mean square error between the two Dataframes
        """
        df = df1.copy()
        
        # Adding a new column with the closing prices from the second DF
        df['close2'] = df2.close
        
        # Dropping the NaN values
        df.dropna(inplace=True)
        
        # Adding another column containing the difference between the two DFs' closing prices
        df['diff'] = df.close - df.close2
        
        # Squaring the difference and getting the mean
        rms = (df[['diff']]**2).mean()
        
        # Returning the sqaure root of the root mean square
        return float(np.sqrt(rms))
      
    def create_nn(self):
        ## Creating the NN

        # Instatiating the model
        # model = Sequential()

        # Activation
        activ = "tanh"

        # Input layer
        self.model.add(LSTM(80, 
                      activation=activ, 
                      return_sequences=True, 
                      input_shape=(self.n_per_in, self.n_features)))

        # Hidden layers
        self.layer_maker(n_layers=2, 
                    n_nodes=30, 
                    activation=activ)

        # Final Hidden layer
        self.model.add(LSTM(60, activation=activ))

        # Output layer
        self.model.add(Dense(self.n_per_out))

        # Model summary
        self.model.summary()

        # Compiling the data with selected specifications
        self.model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

        ## Fitting and Training
        # print(len(self.X))
        # print(len(self.y))
        res = self.model.fit(self.X, self.y, epochs=50, batch_size=128, validation_split=0.1)
        print(res)
        return res
       
    def trans_act_val_to_orig_price(self):
      """
      docstring
      Transforming the actual values to their original price
      """
      actual = pd.DataFrame(self.close_scaler.inverse_transform(self.df[["close"]]), 
                      index=self.df.index, 
                      columns=[self.df.columns[0]])

      # Getting a DF of the predicted values to validate against
      predictions = self.validater(self.n_per_in, self.n_per_out)

      # Printing the RMSE
      print("RMSE:", self.val_rmse(actual, predictions))
          
      # Plotting
      plt.figure(figsize=(16,6))

      # Plotting those predictions
      plt.plot(predictions, label='Predicted')

      # Plotting the actual values
      plt.plot(actual, label='Actual')

      plt.title(f"Predicted vs Actual Closing Prices")
      plt.ylabel("Price")
      plt.legend()
      plt.xlim('2019-12', '2020-12')
      plt.show()
      
    def predict_future(self):
      # Predicting off of the most recent days from the original DF
      yhat = self.model.predict(np.array(self.df.tail(self.n_per_in)).reshape(1, self.n_per_in, self.n_features))

      # Transforming the predicted values back to their original format
      yhat = self.close_scaler.inverse_transform(yhat[0])[0]

      # Creating a DF of the predicted prices
      preds = pd.DataFrame(yhat, 
                          index=pd.date_range(start=self.df.index[-1]+timedelta(days=1), 
                                              periods=len(yhat), 
                                              freq="B"), 
                          columns=[self.df.columns[0]])

      # Number of periods back to plot the actual values
      pers = self.n_per_in

      # Transforming the actual values to their original price
      actual = pd.DataFrame(self.close_scaler.inverse_transform(self.df[["close"]].tail(pers)), 
                            index=self.df.close.tail(pers).index, 
                            columns=[self.df.columns[0]]).append(preds.head(1))

      # Printing the predicted prices
      print(preds)

      # Plotting
      plt.figure(figsize=(16,6))
      plt.plot(actual, label="Actual Prices")
      plt.plot(preds, label="Predicted Prices")
      plt.ylabel("Price")
      plt.xlabel("Dates")
      plt.title(f"Forecasting the next {len(yhat)} days")
      plt.legend()
      plt.show()
    
    def plot_current_data(self):
      
        
        plt.figure(figsize=(16,5))
        plt.plot(self.df['close'])
        # plt.plot(self.df['volume'])
        plt.legend(['close', 'volume'])
        plt.title('Loss')
        plt.xlabel('time')
        plt.ylabel('price')
        plt.show()
      
net = Rnn()
# net.visualize_training_results(net.create_nn())
# net.trans_act_val_to_orig_price()
net.predict_future()
net.plot_current_data()
