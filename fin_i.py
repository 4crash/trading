from typing import List, Tuple
from numpy.core.defchararray import array
import pandas as pd
import ta
import numpy as np
from stockstats import StockDataFrame as sdf
from datetime import datetime, timedelta
from utils import Utils

class FinI(object):

    @staticmethod
    def add_bollinger_bands(df):

        # Initialize Bollinger Bands Indicator
        indicator_bb = ta.volatility.BollingerBands(
            close=df["close"], n=20, ndev=2)

        # Add Bollinger Bands features
        df['bb_bbm'] = indicator_bb.bollinger_mavg()
        df['bb_bbh'] = indicator_bb.bollinger_hband()
        df['bb_bbl'] = indicator_bb.bollinger_lband()

        # Add Bollinger Band high indicator
        df['bb_bbhi'] = indicator_bb.bollinger_hband_indicator()

        # Add Bollinger Band low indicator
        df['bb_bbli'] = indicator_bb.bollinger_lband_indicator()
        return df

    @staticmethod
    def add_weekday(df):
        df = FinI.add_date_col(df)
        if "weekday" not in df:
            df["weekday"] = df.date.dt.dayofweek
        return df

    @staticmethod
    def add_date_col(df):

        if "date" not in df:
            df['date'] = pd.to_datetime(df.index, utc=True)

        return df

    @staticmethod
    def add_change(df, dec_places=1):
        # get change from stockstats
        print(type(df))
        df.get('change')

        df['change'] = round(df['change'], dec_places)
        return df

    @staticmethod
    def add_yearweek(df):
        df = FinI.add_date_col(df)
        df["yearweek"] = df.date.dt.isocalendar().week
        # df['monthday'] = df.index.to_series().dt.day
        # df['month'] =  df.index.to_series().dt.month
        return df

    @staticmethod
    def add_sma(days, df):
        df["sma"+str(days)] = df['close'].rolling(window=days).mean()
        return df

    @staticmethod
    def add_indicators(df):
        # add columns with indicators
        if not isinstance(df, sdf):
            df = sdf.retype(df)
        if len(df) > 0:
            df.get('boll')
            df.get('volume_delta')
            # sma 9, 20, 30, 50, 100, 2000
            df = FinI.add_sma(9, df)
            df = FinI.add_sma(30, df)
            df = FinI.add_sma(100, df)
            df = FinI.add_sma(200, df)
            df = FinI.add_sma(20, df)
            df = FinI.add_sma(50, df)

            if "boll" in df:
                df['boll_mid_ub'] = (df['boll_ub'] - df['boll'])/2 + df['boll']
                df['boll_mid_lb'] = df['boll'] - (df['boll'] - df['boll_lb'])/2
            elif "bb_bbm" in df:
                df['boll_mid_ub'] = (
                    df['bb_bbh'] - df['bb_bbm'])/2 + df['bb_bbm']
                df['boll_mid_lb'] = df['bb_bbm'] - \
                    (df['bb_bbm'] - df['bb_bbl'])/2

            df = FinI.add_fib_to_bb(df)

            # print(df[['close','boll','boll_ub','boll_2','boll_3']])

            df.get('macd')
            df.get('kdjk')
            # data.get('open_2_d')
            df.get('open_-2_r')
            # print(data)
            FinI.add_boll(df, col_name="macd")
            df = FinI.classify_rsi_oscilator(df)
            # df = FinI.add_weekday(df)
             # df = self.add_bollinger_bands(df)
            # df = FinI.add_yearweek(df)
            df = FinI.add_change(df)
            df = FinI.add_oc_change(df, 2)
            df = FinI.add_day_types(df)
            df = FinI.add_fib(df)
            df = FinI.add_levels(df)
            return df
        else:
            print('No rows in data for adding indicators')
            return None
        # Utils.add_first_last_perc_diff(data)
        # print(data)

    @staticmethod
    def add_oc_change(df, round_num=2):
        """
        perc change high low, close open -> ch_hl,ch_oc
        """

        if df is not None:
            df["ch_oc"] = round((df["close"]-df["open"]) /
                                (df["close"]/100), round_num)
            df["ch_hl"] = round((df["high"]-df["low"]) /
                                (df["low"]/100), round_num)

            return df

        else:
            return None

    @staticmethod
    def classify_rsi_oscilator(data):

        if len(data) == 0:
            print("no stocks found")

        if 'kdjk' not in data:
            data.get('kdjk')

        # add points for low or high RSI oscilator and
        data['valRSIclsf'] = ((data.kdjk + data.kdjd)/2)
        data['upRSIclsf'] = (data.kdjk - data.kdjd)
        # self.stocks['SellRSIOscClassify'] =
        return data

    @staticmethod
    def add_fib_to_bb(df):

        df.get('boll')

        df['boll_2'] = ((df['boll_ub'] - df['boll'])/100) * 23.6
        df['boll_3'] = ((df['boll_ub'] - df['boll'])/100) * 38.2
        df['boll_5'] = ((df['boll_ub'] - df['boll'])/100) * 5
        df['boll_6'] = ((df['boll_ub'] - df['boll'])/100) * 61.8
        df['boll_10'] = (df['boll_ub'] - df['boll'])

        return df

    @staticmethod
    def add_fib_from_day_df(df, df_days):
        for col in df_days:
            if col.startswith('fb'):
                df[col] = df_days.iloc[-1][col]
        print(df)
        return df

    @staticmethod
    def add_fib(df, last_rows=10):
        min = df.iloc[-last_rows:].low.min()
        max = df.iloc[-last_rows:].high.max()
        range = max - min

        one_perc = range/100
        bottom = min-(one_perc * 100)

        df['fb_bot_2'] = one_perc * 23.6 + bottom
        df['fb_bot_3'] = one_perc * 38.2 + bottom
        df['fb_bot_5'] = one_perc * 50 + bottom
        df['fb_bot_6'] = one_perc * 61.8 + bottom
        df['fb_bot_7'] = one_perc * 78.6 + bottom
        df['fb_bot_10'] = one_perc * 100 + bottom

        df['fb_mid_2'] = one_perc * 23.6 + min
        df['fb_mid_3'] = one_perc * 38.2 + min
        df['fb_mid_5'] = one_perc * 50 + min
        df['fb_mid_6'] = one_perc * 61.8 + min
        df['fb_mid_7'] = one_perc * 78.6 + min
        df['fb_mid_10'] = one_perc * 100 + min

        df['fb_top_2'] = one_perc * 23.6 + df['fb_mid_10']
        df['fb_top_3'] = one_perc * 38.2 + df['fb_mid_10']
        df['fb_top_5'] = one_perc * 50 + df['fb_mid_10']
        df['fb_top_6'] = one_perc * 61.8 + df['fb_mid_10']
        df['fb_top_7'] = one_perc * 78.6 + df['fb_mid_10']
        df['fb_top_10'] = one_perc * 100 + df['fb_mid_10']

        # print(df)
        return df

    @staticmethod
    def add_week_of_month(df):
        df['week_in_month'] = pd.to_numeric(df.date.dt.day/7)
        df['week_in_month'] = df['week_in_month'].apply(
            lambda x: np.math.ceil(x))
        return df

    @staticmethod
    def add_day_types(df):
        df = sdf.retype(df)
        df = FinI.add_date_col(df)
        df["weekday"] = df.date.dt.dayofweek
        df['monthday'] = df.date.dt.day
        df['month'] = df.date.dt.month
        df['year'] = df.date.dt.year
        df = FinI.add_week_of_month(df)
        df = FinI.add_yearweek(df)
        df = FinI.add_change(df, 2)
        df = FinI.add_weekday(df)

        return df

    @staticmethod
    def add_boll(df, boll_per=8, boll_std_times=0.8, col_name="close"):
        """ Get Bollinger bands.

        boll_ub means the upper band of the Bollinger bands
        boll_lb means the lower band of the Bollinger bands
        boll_ub = MA + Kσ
        boll_lb = MA − Kσ
        M = BOLL_PERIOD
        K = BOLL_STD_TIMES
        :param df: data
        :return: None
        """
        moving_avg = df[str(col_name) + '_{}_sma'.format(boll_per)]
        moving_std = df[str(col_name) + '_{}_mstd'.format(boll_per)]
        df['boll_' + str(col_name)] = moving_avg
        moving_avg = list(map(np.float64, moving_avg))
        moving_std = list(map(np.float64, moving_std))
        # noinspection PyTypeChecker
        df['boll_ub_' + str(col_name)] = np.add(moving_avg,
                               np.multiply(boll_std_times, moving_std))
        # noinspection PyTypeChecker
        df['boll_lb_' + str(col_name)] = np.subtract(moving_avg,
                                    np.multiply(boll_std_times,
                                                moving_std))

    @staticmethod
    def days_to_earnings(earnings):

        if earnings is not None and len(earnings) > 0:
            # print(earnings)
            days_to_earnings = datetime.fromisoformat(
                earnings.iloc[-1].startdatetime[:-1]) - datetime.today()
            # print(str(days_to_earnings))
        else:
            days_to_earnings = None
        print("days_to_earnings() - done")
        return days_to_earnings

    @staticmethod
    def get_fib_hl(df, close_price):
        hl = {"h": None, "l": None}
        last_fib = -1
        if isinstance(close_price, float):
            for col in df:
                if col.startswith('fb'):
                    splt = col.split("_")
                    print(close_price)
                    print(df.iloc[-1][col])
                    if close_price > last_fib and close_price < df.iloc[-1][col]:
                        # next col
                        # df.iloc[:,df.columns.get_indexer(col)+1]
                        hl["l"] = round(df.iloc[-1]["fb_" + splt[1] + "_2"], 2)
                        hl["h"] = round(
                            df.iloc[-1]["fb_" + splt[1] + "_10"], 2)
                    last_fib = df.iloc[-1][col]
        return hl

    @staticmethod
    def isSupport(df:pd.DataFrame, i:int)-> bool:
        """checking support

        Args:
            df (pd.DataFrame): dataframe where is calculated resistance level
            i (int): specific index in dataframe

        Returns:
            bool: return true or false if its support or not
        """
        support = df['low'][i] < df['low'][i-1] and df['low'][i] < df['low'][i +
            1] and df['low'][i+1] < df['low'][i+2] and df['low'][i-1] < df['low'][i-2]
        return support

    @staticmethod  
    def isResistance(df:pd.DataFrame, i:int)-> bool:
        """checking resistance

        Args:
            df (pd.DataFrame): dataframe where is calculated resistance level
            i (int): specific index in dataframe

        Returns:
            bool: return true or false if its resistance or not
        """
        resistance = df['high'][i] > df['high'][i-1] and df['high'][i] > df['high'][i +
            1] and df['high'][i+1] > df['high'][i+2] and df['high'][i-1] > df['high'][i-2]
        return resistance

    @staticmethod
    def isFarFromLevel(l, levels, s):
        return np.sum([abs(l-x) < s for x in levels]) == 0

    @staticmethod
    def get_nearest_values(price_levels: list, price: float)-> Tuple[list, list]:
        """returns top and bottom values splitted by price arg

        Args:
            price_levels (list):  values array 
            price (float): splitting value

        Returns:
            Tuple[list, list]: return sorted low desc,high asc values
        """
        price_levels = np.array(price_levels)
        price_levels = price_levels[price_levels != None]
        # price_levels = price_levels[~np.isnan(price_levels)]
        
        low:np
        high:np
        low = price_levels[np.where(price_levels <= price)]
        high = price_levels[np.where(price_levels > price)]
        return sorted(low, reverse=True), sorted(high)
    
    @staticmethod
    def add_levels(df: pd.DataFrame)->pd.DataFrame:
        """Add price resistance and supports into price_level named field

        Args:
            df (pd.DataFrame): prices 

        Returns:
            pd.DataFrame: prices with price_level field
        """

        s = np.mean(df['high'] - df['low'])
        #  for level in levels:
        #     plt.hlines(level[1],xmin=df['Date'][level[0]],\
        #        xmax=max(df['Date']),colors='blue')

        levels:List = []

        df["price_level"] = None
        i = 0
        for idx, r in df.iterrows():
            if i > 1 and i < (len(df) - 2):
                if FinI.isSupport(df, i):
                    l = df['low'][i]
                    if FinI.isFarFromLevel(l, levels, s):
                        levels.append((i, l))
                        df.loc[idx, "price_level"] = l
                        # print(df.loc[idx])

                elif FinI.isResistance(df, i):
                    l = df['high'][i]
                    if FinI.isFarFromLevel(l, levels, s):
                        levels.append((i, l))
                        df.loc[idx, "price_level"] = l
                        # print(df.loc[idx])

            i += 1

        # print(df)
        return df
        # return levels

    @staticmethod
    def get_sizes(m_df, m_df_spy = None):
        """return size indicators and updown in a row for open close and for candle to candle specific column
            SPY count only mean between open close value

        Args:
            m_df ([type]): [description]
            m_df_spy ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
          
        if len(m_df_spy) > 0 :
            m_df_spy["oc_mean"] = ((m_df_spy.close + m_df_spy.open)/2)
            
        m_df = sdf.retype(m_df)
        m_df.get("boll")
        m_df = FinI.add_sma(9, m_df)
        m_df = FinI.add_sma(20, m_df)
        m_df = FinI.add_weekday(m_df)
        m_df = FinI.add_week_of_month(m_df)
        m_df = FinI.add_levels(m_df)

        m_df["size_top"] = m_df.apply(lambda row: Utils.calc_perc(
            row.open, row.high) if row.open > row.close else Utils.calc_perc(row.close, row.high), axis=1)


        m_df["size_btm"] = m_df.apply(lambda row: Utils.calc_perc(
            row.low, row.close) if row.open > row.close else Utils.calc_perc(row.low, row.open), axis=1)

        m_df["size_body"] = m_df.apply(lambda row: Utils.calc_perc(row.open, row.close), axis=1)
        m_df["size_sma9"] = m_df.apply(lambda row: Utils.calc_perc(row.sma9, row.close), axis=1)
        m_df["size_sma20"] = m_df.apply(lambda row: Utils.calc_perc(row.sma20, row.close), axis=1)
        m_df["size_boll"] = m_df.apply(
            lambda row: Utils.calc_perc(row.boll, row.close), axis=1)
        m_df["size_boll_ub"] = m_df.apply(
            lambda row: Utils.calc_perc(row.boll_ub, row.close), axis=1)
        m_df["size_boll_lb"] = m_df.apply(
            lambda row: Utils.calc_perc(row.boll_lb, row.close), axis=1)

        m_df["size_top-1"] = m_df.shift(1).size_top

        m_df["size_btm-1"] = m_df.shift(1).size_btm

        m_df["size_body-1"] = m_df.shift(1).size_body

        m_df["size_top-2"] = m_df.shift(2).size_top

        m_df["size_btm-2"] = m_df.shift(2).size_btm

        m_df["size_body-2"] = m_df.shift(2).size_body

        m_df["size_top-3"] = m_df.shift(3).size_top

        m_df["size_btm-3"] = m_df.shift(3).size_btm

        m_df["size_body-3"] = m_df.shift(3).size_body
        
        m_df["size_prev_chng"] = (
            m_df.open - m_df.shift(1).close) / (m_df.shift(1).close/100)

        m_df = FinI.get_up_down_sum_in_row(m_df)
        m_df = FinI.get_green_red_sum_in_row(m_df)

        return m_df, m_df_spy

    @staticmethod
    def get_green_red_sum_in_row(df:pd.DataFrame, shift=0):
        counter = 0
        up_down = []
        # values 0 or (-1 for not current row only previus)
        for i,data in enumerate(df.values):
            if i==1:
                up_down.append(counter)
            else:
                if df["open"][i+shift] > df["close"][i+shift]:
                    if i > 2 and df["open"][i-1+shift] < df["close"][i-1+shift]:
                        counter = 0
                    counter += 1
                else:
                    if i > 2 and df["open"][i-1+shift] > df["close"][i-1+shift]:
                        counter = 0
                    counter -= 1

                up_down.append(counter)

            # print(str(counter) + " - " +
            #       str(df["open"][i+shift]-df["close"][i+shift]))
            # if df[index].close < df[index).shift(1).close:
        df["green_red_row"] = up_down
        return df

    @staticmethod
    def get_up_down_sum_in_row(df:pd.DataFrame, column = "close", shift=0):

        counter = 0
        up_down = []
        # values 0 or (-1 for not current row only previus)
        for i,data in enumerate(df.values):
            if i==1:
                up_down.append(counter)
            else:
                if df[column][i-1+shift] > df[column][i+shift]:
                    if i > 2 and df[column][i-2+shift] < df[column][i-1+shift]:
                        counter = 0
                    counter -= 1
                else:
                    if i > 2 and df[column][i-2+shift] > df[column][i-1+shift]:
                        counter = 0
                    counter += 1

                up_down.append(counter)

            # print(str(counter) + " - " +
            #       str(df[column][i-1+shift]-df[column][i+shift]))
        df["up_down_row"] = up_down
        return df
                
            
        
