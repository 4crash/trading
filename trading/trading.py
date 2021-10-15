
from datetime import datetime
import sys
from time import sleep
# sys.path.insert(1, '.')
sys.path.append('../../')
import logging
from alpaca_examples.market_db import TableName, Database 
from sklearn.linear_model import LogisticRegression
import numpy as np
from alpaca_examples.buy_sell import BuySell, OpenTradesSource, TradingProviders, TradingStrategies
from alpaca_examples.fin_i import FinI
from alpaca_examples.utils import Utils
import pandas as pd
import ta
logging.basicConfig(level = logging.DEBUG)



class Trading():
    
    def __init__(self):
        self.db = Database()
        self.bs = BuySell(trading_strategy=TradingStrategies.LOGISTIC_REGRESSION,
                 trading_provider=TradingProviders.ALPACA,
                 fill_open_trades=OpenTradesSource.DB)
        self.bs.close_alpaca_postition = True
        
    
    def lr_best_candidate(self):
        logging.warning("start")
        df_best_buy = pd.DataFrame()
        symbols = self.db.get_symbols()
        prob_perc = 0.9
        for sym in symbols:
            try:
                logging.info(f"filling: {sym}")
                df_lr_raw = self.logistic_regression_raw(self.db, sym)
                df_best_buy = df_best_buy.append(df_lr_raw.tail(1))
                logging.info(df_lr_raw.tail(1).iloc[0].prob_1)
                if df_lr_raw.tail(1).iloc[0].prob_1 > prob_perc:
                   self.bs.buy_stock_t(stock=df_lr_raw.tail(
                       1).iloc[0], price=df_lr_raw.tail(1).iloc[0].close, qty=1)
                   logging.info("Buy" + sym)
                elif df_lr_raw.tail(1).iloc[0].prob_1 < prob_perc and sym in self.bs.buy_sell_open["sym"]:
                    self.bs.sell_stock_t(sym=sym, price=df_lr_raw.tail(1).iloc[0].close, qty=1)
                    logging.info("Sell" + sym)
                   
            except Exception as e:
                logging.info(e)

        if len(df_best_buy) > 0:
            logging.info(df_best_buy.sort_values(by="prob_1"))
        else:
            logging.error("no data")
            


    @staticmethod
    def logistic_regression_raw(db:Database ,symbol="SPY"):
        df = db.load_data(
            table_name=TableName.DAY,  time_from="-90D", symbols=[symbol])

        # m_df_spy = self.db.load_data(
        #     table_name=TableName.DAY,  time_from=self.time_from, symbols=["SPY"])

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
        df = df[["Corr_9", "open-close", "close-close-prev", "RSI", "S_9","close"]]
        # df = df[["Corr_9", "open-close", "close-close-prev", "RSI", "S_9"]]
        df = df.dropna()
        X = df.iloc[:, :30]
        # st.write(len(y))
        # st.write(len(X))
        split = int(0.7*len(df))

        X_train, X_test, y_train, y_test = X[:
                                            split], X[split:], y[:split], y[split:]

        # We will instantiate the logistic regression in Python using ‘LogisticRegression’
        # function and fit the model on the training dataset using ‘fit’ function.
        model = LogisticRegression()
        model = model.fit(X_train, y_train)

        # Examine coeficients
        # pd.DataFrame(zip(X.columns, np.transpose(model.coef_)))
        # st.write("Examine The Coefficients")
        # st.write(pd.DataFrame(zip(X.columns, np.transpose(model.coef_))))

        #We will calculate the probabilities of the class for the test dataset using ‘predict_proba’ function.
        probability = model.predict_proba(X_test)
        df['Predicted_Signal'] = model.predict(X)
        df = df.tail(len(probability))
        df["prob_0"] = probability[:, 0]
        df["prob_1"] = probability[:, 1]
        df["sym"] = symbol
        return df

tr = Trading()
tr.lr_best_candidate()
while True:
    if True or datetime.today().weekday not in [5,6] and datetime.today().hour in [16,21] and datetime.today().minute == 45:
        tr.lr_best_candidate()
    Utils.countdown(30)
