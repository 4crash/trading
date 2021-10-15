
# import libraries

import sys
sys.path.append('../')
from alpaca_examples.market_db import Database
import pandas as pd  # Import Pandas for data manipulation using dataframes
# import numpy as np  # Import Numpy for data statistical analysis
# import matplotlib.pyplot as plt  # Import matplotlib for  data vis
# import random
# # # import seaborn as sns
from fbprophet import Prophet
import matplotlib.pyplot as plt
import matplotlib as mpl
# from alpaca_examples.database import Database
# import stocker

class MPred():
    
    data = pd.DataFrame()
    def __init__(self):
        """
        docstring
        """
        self.db = Database()
        
        
    def prepare_prop(self, sym):
        # self.db.set_time_from("365d")
        data = self.db.load_data("p_day",symbols=sym)
        data["ds"] = pd.to_datetime(data.index, utc=True).tz_localize(None)
        data["y"] = data.close 
        print(data)
        m = Prophet(daily_seasonality = True) # the Prophet class (model)
        m.fit(data) # fit the model using all data
        self.plot_prediction(m, sym)
        
    def plot_prediction(self,m, sym):
        # we need to specify the number of days in future
        future = m.make_future_dataframe(periods=365)
        prediction = m.predict(future)
        m.plot(prediction)
        plt.title("Prediction using the Prophet: " + str(sym))
        plt.gca().fmt_xdata = mpl.dates.DateFormatter('%Y-%m-%d')
        plt.xlabel("Date")
        plt.ylabel("Close Stock Price")
        plt.show()
        
  
prop = MPred()
prop.prepare_prop(sys.argv[1])

     # stocker.predict.tomorrow('TSLA')
# self.m = Prophet(weekly_seasonality=False, yearly_seasonality=False)
# self.m.add_seasonality('self_define_cycle', period=8,
#           fourier_order=8, mode='additive')

# self.db = Database()
# def load_data(self, table_name='p_day', symbol='SPY'):

#     table_name = self.checkTableName(table_name)
#     result = self.stock_stats.retype(pd.read_sql(
#         'select * from '+table_name + ' where sym = \'' + symbol + '\' order by index ', con=self.engine, index_col='index'))
# def cycle_analysis(self, data, split_date, cycle, mode='additive', forecast_plot=False, print_ind=False):
#     training = data[:split_date].iloc[:-1, ]
#     testing = data[split_date:]
#     predict_period = len(pd.date_range(split_date, max(data.index)))
#     df = training.reset_index()
#     df.columns = ['ds', 'y']
#     m = Prophet(weekly_seasonality=False,
#                 yearly_seasonality=False, daily_seasonality=False)
#     m.add_seasonality('self_define_cycle', period=cycle,
#                     fourier_order=8, mode=mode)
#     m.fit(df)
#     future = m.make_future_dataframe(periods=predict_period)
#     forecast = m.predict(future)
#     if forecast_plot:
#         m.plot(forecast)
#         plt.plot(testing.index, testing.values,
#                 '.', color='#ff3333', alpha=0.6)
#         plt.xlabel('Date', fontsize=12, fontweight='bold', color='gray')
#         plt.ylabel('Price', fontsize=12, fontweight='bold', color='gray')
#         plt.show()
#     ret = max(forecast.self_define_cycle)-min(forecast.self_define_cycle)
#     model_tb = forecast['yhat']
#     model_tb.index = forecast['ds'].map(lambda x: x.strftime("%Y-%m-%d"))
#     out_tb = pd.concat([testing, model_tb], axis=1)
#     out_tb = out_tb[~out_tb.iloc[:, 0].isnull()]
#     out_tb = out_tb[~out_tb.iloc[:, 1].isnull()]
#     mse = mean_squared_error(out_tb.iloc[:, 0], out_tb.iloc[:, 1])
#     rep = [ret, mse]
#     if print_ind:
#         print "Projected return per cycle: {}".format(round(rep[0], 2))
#         print "MSE: {}".format(round(rep[1], 4))
#     return rep
