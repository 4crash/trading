from stockstats import StockDataFrame as sdf
import sys
sys.path.append('../')
from alpaca_examples.buy_sell import BuySell
from alpaca_examples.market_db import Database
from alpaca_examples.back_tester import BackTest
class Trader(object):
    """
    docstring
    """
    def __init__(self):
        """
        docstring
        """
        self.bs = BuySell()
        self.db = Database()
        self.bt = BackTest()
        
    def load_data(self, table_name = None, symbols = None, sectors = None, limit = None):
        symbols = symbols if symbols is not None else self.symbols
        return sdf.retype(self.db.load_data(table_name,symbols=symbols,sectors=sectors, limit=limit))
    
    def test_boll_rsi_macd(self, table_name=None, buy_now=False):

        self.bs.buyed_stocks = 0
        self.bs.money = self.bs.credit["start_amount"]
        spy_stocks = self.load_data(table_name=table_name, symbols="SPY")
        symbols = self.db.get_symbols()
        print(str(symbols))
        for symbol in symbols:
            print("symbol: " + str(symbol))

            stck = self.load_data(table_name=table_name, symbols=symbol)
            if len(stck) < 1:
                break

            stck = self.bt.add_indicators(stck)
            print("calculating percent change:" + str(symbol))
            # stck = self.stocks.loc[self.stocks.sym ==symbol[0]].sort_values(by='index')
            buy_now_process = buy_now
            self.symbols = symbol[0]
            cross_bollinger = 0
            # self.prev_stock = stck.iloc[0]
            # self.first_stock = stck.iloc[0]

            # self.sell_marks = self.sell_marks.iloc[0:0]
            # self.buy_marks = self.buy_marks.iloc[0:0]
            self.transactions = 0
            self.profit_perc = 0

            for inx in range(50,len(stck)):

                # def check_boll():
                """
                docstring
                """
                if stck.iloc[inx].boll >= stck.iloc[inx-1].close and \
                        stck.iloc[inx]['boll'] <= stck.iloc[inx].close:

                    print("go up " + str(stck.iloc[inx].name) + " - boll:" + str(stck.iloc[inx-1].boll) + " -prev: " + str(
                        stck.iloc[inx-1].close) + " - curr:" + str(stck.iloc[inx].close))
                    cross_bollinger = 1

                elif stck.iloc[inx]['boll'] <= stck.iloc[inx-1]['close'] and \
                        stck.iloc[inx]['boll'] >= stck.iloc[inx]['close']:

                    print("go down " + str(stck.iloc[inx].name) + " - boll:" + str(stck.iloc[inx-1].boll) + " -prev: " + str(
                        stck.iloc[inx-1].close) + " - curr:" + str(stck.iloc[inx]['close']))
                    cross_bollinger = -1

                else:
                    cross_bollinger = 0

                if self.bs.buyed_stocks == 0 and \
                    (cross_bollinger == 1) or \
                        buy_now_process:

                    self.bs.buy(stck)
                    buy_now_process = False

                #comment this block for selling at the end of the date
                if self.bs.buyed_stocks != 0 and \
                    (cross_bollinger == -1 or
                     (stck['boll_mid_lb'] <= stck.iloc[inx-1]['close'] and stck.iloc[inx]['boll_mid_lb'] > stck.iloc[inx]['close'])):
                    self.bs.sell(stck.iloc[inx])

                # if  self.buyed_stocks == 0  and \
                #     (cross_bollinger == 1 or \
                #     (stock['boll_mid_ub'] >= self.prev_stock['close'] and stock['boll_mid_ub'] < stock['close']) or \
                #     (stock['boll_mid_lb'] >= self.prev_stock['close'] and stock['boll_mid_lb'] < stock['close'])):
                #     self.buy_stock(stock)

                # if   self.buyed_stocks != 0 and \
                #      (cross_bollinger == -1 or \
                #      (stock['boll_mid_ub'] <= self.prev_stock['close'] and stock['boll_mid_ub'] > stock['close']) or \
                #      (stock['boll_mid_lb'] <= self.prev_stock['close'] and stock['boll_mid_lb'] > stock['close'])):
                #     self.sell_stock(stock)

                # if  self.buyed_stocks == 0  and \
                #     (cross_bollinger == 1 or \
                #     (stock['boll_mid_ub'] >= self.prev_stock['close'] and stock['boll_mid_ub'] < stock['close'])):
                #     self.buy_stock(stock)

                # if   self.buyed_stocks != 0 and \
                #      (cross_bollinger == -1 or \
                #      (stock['boll_mid_ub'] <= self.prev_stock['close'] and stock['boll_mid_ub'] > stock['close'])):
                #     self.sell_stock(stock)

                # self.prev_stock = stck
            # check_boll()
            if self.transactions > 0:
                self.show_stats(symbol)
                # self.plot_stats(stck, spy_stocks)

            else:
                print("Theres no transactions please change BUY/SELL params")
    
tr = Trader()
tr.test_boll_rsi_macd()