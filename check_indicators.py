
from sys import path
from typing import List, Optional, Tuple

import pandas as pd
path.append("../")
from utils import Utils

def min_3_rows(func):
    def wrapper(*args, **kwargs):
        if "stocks" in kwargs and kwargs['stocks'].shape[0] > 3 or \
            len(args) > 0 :
            return func(*args, **kwargs)
        else:
            return False
    return wrapper


class CheckIndicators():
  
    
    @staticmethod
    @min_3_rows
    def check_macd(stocks, buy=True):
        if buy:
            # stocks["macd"].max()/3 > stocks.iloc[-1]["macd"] and \
            if ((stocks.iloc[-2].boll_lb_macd > stocks.iloc[-2].macd and
                  stocks.iloc[-1].boll_lb_macd < stocks.iloc[-1].macd) or
                 (stocks.iloc[-2].boll_ub_macd > stocks.iloc[-2].macd and
                  stocks.iloc[-1].boll_ub_macd < stocks.iloc[-1].macd)):
                return True
            else:
                return False
        else:
            if stocks.iloc[-2].boll_ub_macd < stocks.iloc[-2].macd and \
                stocks.iloc[-1].boll_ub_macd > stocks.iloc[-1].macd:
                return True
            else:
                return False

    @staticmethod
    def check_rsi(stocks, buy=True, kdjk_bottom_level = 40, kdjk_top_level = 70):
        if buy:
            if stocks.iloc[-1].kdjk < kdjk_bottom_level:
                # stocks.iloc[-2].kdjk < stocks.iloc[-1].kdjk and \
                # stocks.iloc[-1].kdjd <= stocks.iloc[-1].kdjk:
                return True
            else:
                return False
        else:
            if  stocks.iloc[-1].kdjk > kdjk_top_level:
                # stocks.iloc[-2].kdjd > stocks.iloc[-2].kdjk and \
                # stocks.iloc[-1].kdjd >= stocks.iloc[-1].kdjk:
                return True
            else:
                return False

   
    @staticmethod
    @min_3_rows
    def check_sma( stocks, live_stocks=None, buy=True, params = 9):

        latest = live_stocks if live_stocks!= None and len(live_stocks) > 1 else stocks
        if buy:
            if stocks.iloc[-2][f"sma{params}"] > latest.iloc[-2].close and \
                stocks.iloc[-1][f"sma{params}"] < latest.iloc[-1].close:
                return True
            else:
                return False
        else:
            if stocks.iloc[-2][f"sma{params}"] < latest.iloc[-2].close and \
                    stocks.iloc[-1][f"sma{params}"] > latest.iloc[-1].close:
                return True
            else:
                return False
            
    @staticmethod
    @min_3_rows
    def check_boll(stocks, live_stocks=None, buy=True, params=""):

        latest = live_stocks if live_stocks and len(live_stocks) > 1 else stocks
        if buy:
            if stocks.iloc[-2].close > stocks.iloc[-1][f"boll{params}"] > latest.iloc[-1].close:
                return True
            else:
                return False
        else:
            if stocks.iloc[-2].close < stocks.iloc[-1][f"boll{params}"] < latest.iloc[-1].close:
                return True
            else:
                return False
            
    @staticmethod
    def check_fib(stocks, live_stocks=None, buy=True):
        """ FIXIT doesnt return true false and doesnt acces buy variable

        Args:
            stocks ([type]): [description]
            live_stocks ([type], optional): [description]. Defaults to None.
            buy (bool, optional): [description]. Defaults to True.

        Returns:
            [type]: [description]
        """

        latest = live_stocks if live_stocks and len(live_stocks) > 1 else stocks
        # check nearest fibonachi up, down
        levels = {"c1": 99999, "c2": 99999, "t1": 99999, "t2": 99999}
        # fib =  stocks.where(lambda col: print(col), axis=1)
        # print(fib)
        for col in stocks:
            if col.startswith('fb'):
                # index = abs(df['values'] - value).idxmin()
                if abs(latest.iloc[-1].close - stocks.iloc[-1][col]) < abs(latest.iloc[-1].close - levels["c1"]):
                    print(stocks.iloc[-1][col])
                    print(abs(latest.iloc[-1].close - stocks.iloc[-1][col]))
                    levels["c1"] = stocks.iloc[-1][col]

                # elif abs(latest.iloc[-1].close - stocks.iloc[-1][col]) < levels["c2"]:
                #     levels["c2"] = stocks.iloc[-1][col]

                if abs(latest.iloc[-1].close - stocks.iloc[-1][col]) < abs(latest.iloc[-1].close - levels["c2"]) and \
                    stocks.iloc[-1][col] != levels["c1"]:
                    levels["c2"] = stocks.iloc[-1][col]
                    print(str(col) + " | " + str(stocks.iloc[-1][col]))

                if latest.iloc[-1].close < stocks.iloc[-1][col] and \
                        stocks.iloc[-1][col] < levels["t1"]:

                    levels["t1"] = stocks.iloc[-1][col]
                    print(str(col) + " | " + str(stocks.iloc[-1][col]))

                if latest.iloc[-1].close < stocks.iloc[-1][col] and \
                        stocks.iloc[-1][col] < levels["t2"] and \
                        stocks.iloc[-1][col] != levels["t1"]:

                    levels["t2"] = stocks.iloc[-1][col]
                    print(str(col) + " | " + str(stocks.iloc[-1][col]))

        print(levels)
        print(latest.iloc[-1].close)

        return levels
    
    @staticmethod
    @min_3_rows
    def check_hammer(stocks: pd.DataFrame, buy:bool=True,params=1.9) -> bool:
        how_many_times_bigger = params
        # get  low value of the body
        open_close_low = stocks.iloc[-1].open if stocks.iloc[-1].open > stocks.iloc[-1].close else stocks.iloc[-1].close
        open_close_high = stocks.iloc[-1].close if stocks.iloc[-1].open > stocks.iloc[-1].close else stocks.iloc[-1].open
        if buy:
            if stocks.iloc[-3].close > stocks.iloc[-2].close > stocks.iloc[-1].close and \
                    abs(stocks.iloc[-1].open - stocks.iloc[-1].close) < (abs(open_close_low - stocks.iloc[-1].low)/how_many_times_bigger):
                return True
            else:
                return False
        else:
            if stocks.iloc[-3].close < stocks.iloc[-2].close < stocks.iloc[-1].close and \
                    abs(stocks.iloc[-1].open - stocks.iloc[-1].close) < (abs(open_close_high - stocks.iloc[-1].high)/how_many_times_bigger):
                return True
            else:
                return False

    
    @staticmethod
    @min_3_rows
    def check_star( stocks: pd.DataFrame, live_stocks=None, buy:bool=True, params=1.9) -> bool:
        how_many_times_bigger = params
        # get  low value of the body
        open_close_low = stocks.iloc[-1].open if stocks.iloc[-1].open > stocks.iloc[-1].close else stocks.iloc[-1].close
        open_close_high = stocks.iloc[-1].close if stocks.iloc[-1].open > stocks.iloc[-1].close else stocks.iloc[-1].open
        if buy:
            if stocks.iloc[-3].close > stocks.iloc[-2].close > stocks.iloc[-1].close and \
                abs(stocks.iloc[-1].open - stocks.iloc[-1].close) < (abs(open_close_low - stocks.iloc[-1].low)/how_many_times_bigger) and \
                abs(stocks.iloc[-1].open - stocks.iloc[-1].close) < (abs(open_close_high - stocks.iloc[-1].high)/how_many_times_bigger):
                return True
            else:
                return False
        else:
            if stocks.iloc[-3].close < stocks.iloc[-2].close < stocks.iloc[-1].close and \
                abs(stocks.iloc[-1].open - stocks.iloc[-1].close) < (abs(open_close_low - stocks.iloc[-1].low)/how_many_times_bigger) and \
                abs(stocks.iloc[-1].open - stocks.iloc[-1].close) < (abs(open_close_high - stocks.iloc[-1].high)/how_many_times_bigger):
                return True
            else:
                return False
            
    @staticmethod
    def check_bullish_engulfing( stocks, live_stocks=None, buy=True):
        # todo fill it
        pass
      
    @staticmethod
    @min_3_rows
    def check_pre_sma( stocks, live_stocks=None, buy=True, params=9):

        latest = live_stocks if live_stocks and len(live_stocks) > 1 else stocks

        if buy:
            if stocks.iloc[-2].close < latest.iloc[-1].close and \
                    stocks.iloc[-1][f"sma{params}"] > latest.iloc[-1].close:
                return True
            else:
                return False
        else:
            if stocks.iloc[-2].close > latest.iloc[-1].close and \
                    stocks.iloc[-1][f"sma{params}"] < latest.iloc[-1].close:
                return True
            else:
                return False
    @staticmethod
    @min_3_rows
    def check_candles(stocks: pd.DataFrame, buy=True, params=3)-> Optional[bool]:
        result = CheckIndicators.check_candles_in_row(stocks=stocks)

        # if buy, check descendent price in num days counted by params
        if buy:
            if params and (result[0]) > params:
                return True
            else:
                return False
         # if sell, check ascendant price in num days counted by params
        else:
            if params and (result[1]) > params:
                return True
            else:
                return False
            
        
    @staticmethod
    def check_candles_in_row(stocks: pd.DataFrame) -> List[int]:
        """ count last red, green candles in a row retrospectively

        Args:
            stocks (pd.DataFrame): [description]
            check_down (bool, optional): [description]. Defaults to True.

        Returns:
            Tuple[int, int]: last red candles sum, last green candles sum. One of them should be 0
        """

        red = green = 1
        
        while stocks.iloc[-(red+1)].close >= stocks.iloc[-red].close:
            red += 1
            if red >= len(stocks):
                break
        while stocks.iloc[-(green+1)].close <= stocks.iloc[-green].close:
            green += 1
            if green >= len(stocks):
                break

        return [red-1,green-1]

    @staticmethod
    @min_3_rows
    def check_perc_move(stocks, buy=True, params=15):
        doji = Utils.calc_perc(stocks.iloc[-2].close, stocks.iloc[-1].close)
        if buy:
            if doji < -params:
                return True
            else:
                return False
        else:
            if doji > params:
                return True
            else:
                return False
            
         
    @staticmethod
    @min_3_rows
    def check_boll_sma_cross(stocks, buy=True, params=9):

        # latest = live_stocks if len(live_stocks) > 0 else stocks

        try:
            if buy:
                if abs(stocks.iloc[-1][f"sma{params}"] - stocks.iloc[-1].boll) <= 1 and \
                        stocks.iloc[-3][f"sma{params}"] < stocks.iloc[-1][f"sma{params}"]:
                    return True
                else:
                    return False
            else:
                if abs(stocks.iloc[-1][f"sma{params}"] - stocks.iloc[-1].boll) <= 1 and \
                        stocks.iloc[-3][f"sma{params}"] > stocks.iloc[-1][f"sma{params}"]:
                    return True
                else:
                    return False
        except KeyError as e:
            print("boll_sma_cross index -3 doesnt exists" + str(e))
            return False
        
    @staticmethod
    def check_financials(financials, buy=True, max_short_ratio = 3):

        # print(str(financials.iloc[0].shortRatio) + " | " + str( (financials.iloc[0].date)))
        # print(str(financials.iloc[-1].shortRatio) + " | " + str( (financials.iloc[-1].date)))
        
        if len(financials) > 0:
            financials.shortRatio.replace(
                to_replace=[None], value=0, inplace=True)
            financials.forwardPE.replace(
                to_replace=[None], value=0, inplace=True)
            financials.trailingPE.replace(
                to_replace=[None], value=0, inplace=True)

        if buy:
            if (len(financials) > 0 and
                (financials.iloc[0].shortRatio >= financials.iloc[-1].shortRatio and
                 financials.iloc[-1].shortRatio <= max_short_ratio)) and \
                (len(financials) > 0 and
                 financials.iloc[-1].forwardPE is not None and
                 financials.iloc[-1].trailingPE is not None and
                 CheckIndicators.check_eps_pe(financials) and
                 float(financials.iloc[0].forwardPE) >= float(financials.iloc[-1].forwardPE) or
                 (financials.iloc[-1].trailingPE != 0 and float(
                     financials.iloc[-1].trailingPE) >= float(financials.iloc[-1].forwardPE))
                 ):
                return True
            else:
                return False
        else:
            if (len(financials) > 0 and
                (financials.iloc[0].shortRatio < financials.iloc[-1].shortRatio or
                 financials.iloc[-1].shortRatio > max_short_ratio)) and \
                (len(financials) > 0 and
                 financials.iloc[-1].forwardPE is not None and
                 financials.iloc[-1].trailingPE is not None and
                 financials.iloc[-1].trailingPE != 0 and
                 financials.iloc[0].forwardPE != 0 and
                 (float(financials.iloc[0].forwardPE) < float(financials.iloc[-1].forwardPE) or
                  float(financials.iloc[-1].trailingPE) < float(financials.iloc[-1].forwardPE))):
                return True
            else:
                return False
            
    @staticmethod
    def check_eps_pe(financials, check_eps=True, check_pe = True):
        eps, pe = CheckIndicators.eps_pe(financials)
        res = False
        res_pe = res_eps = False
        if eps is not None and eps > 0 and check_eps:
            res_eps = True

        if pe is not None and pe < 0 and check_pe:
            res_pe = True

        if check_eps and check_pe:
            res = True if res_pe and res_eps else False
        elif check_eps:
            res = res_eps
        elif check_pe:
            res = check_pe

        return res
    
    @staticmethod
    def check_sentiment( sentiment, buy=True):

        if len(sentiment) > 0:
            sentiment.sentiment_summary_avg.replace(
                to_replace=[None], value=0, inplace=True)

        if buy:
            if len(sentiment) < 1 or \
               (sentiment.iloc[0].sentiment_summary_avg <= sentiment.iloc[-1].sentiment_summary_avg or
                    sentiment.iloc[-1].sentiment_summary_avg > 0.4):

                return True
            else:
                return False
        else:
            if len(sentiment) > 0 and \
                (sentiment.iloc[0].sentiment_summary_avg > sentiment.iloc[-1].sentiment_summary_avg or
                 sentiment.iloc[-1].sentiment_summary_avg < 0.2):

                return True
            else:
                return False
    @staticmethod
    def check_earnings( earnings, buy=True):

        if buy:
            if len(earnings) < 1 or \
                    earnings.iloc[-1].epssurprisepct is not None and \
                    earnings.iloc[-1].epssurprisepct > 0:

                return True
            else:
                return False
        else:
            if len(earnings) > 0 and \
                    earnings.iloc[-1].epssurprisepct is not None and \
                    earnings.iloc[-1].epssurprisepct < 0:

                return True
            else:
                return False
            
    @staticmethod
    def eps_pe( financials):

        if len(financials) < 1 or financials.iloc[-1].trailingPE == 0 or financials.iloc[-1].forwardPE == 0:
            pe = None
            eps = None
        else:
            pe = Utils.calc_perc(
                financials.iloc[-1].trailingPE, financials.iloc[-1].forwardPE)
            eps = Utils.calc_perc(
                financials.iloc[-1].trailingEps, financials.iloc[-1].forwardEps)

        return eps, pe
