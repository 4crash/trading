

class TradingAlg(object):
    """
    docstring
    """

    def boll_rsi_macd(self, table_name=None, buy_now=False):

        self.buyed_stocks = 0
        self.money = self.startCredit
        spy_stocks = self.load_data(table_name=table_name, symbols="SPY")
        if self.symbols:
            symbols = self.symbols
        else:
            symbols = self.db.get_symbols()
        print(str(symbols))
        for symbol in symbols:
            print("symbol: " + str(symbol))

            sub_data = self.load_data(table_name=table_name, symbols=symbol)
            if len(sub_data) < 1:
                break

            sub_data = self.add_indicators(sub_data)
            print("calculating percent change:" + str(symbol))
            # sub_data = self.stocks.loc[self.stocks.sym ==symbol[0]].sort_values(by='index')
            buy_now_process = buy_now
            self.symbols = symbol[0]
            cross_bollinger = 0
            self.prev_stock = sub_data.iloc[0]
            self.first_stock = sub_data.iloc[0]

            self.sell_marks = self.sell_marks.iloc[0:0]
            self.buy_marks = self.buy_marks.iloc[0:0]
            self.transactions = 0
            self.profit_perc = 0

            for index, stock in sub_data.iterrows():

                if stock['boll'] >= self.prev_stock['close'] and \
                   stock['boll'] <= stock['close']:
                   print("go up " + str(stock.name) + " - boll:" + str(self.prev_stock['boll']) + " -prev: " + str(
                       self.prev_stock['close']) + " - curr:" + str(stock['close']))
                   cross_bollinger = 1

                elif stock['boll'] <= self.prev_stock['close'] and \
                        stock['boll'] >= stock['close']:
                   print("go down " + str(stock.name) + " - boll:" + str(self.prev_stock['boll']) + " -prev: " + str(
                       self.prev_stock['close']) + " - curr:" + str(stock['close']))
                   cross_bollinger = -1

                else:
                    cross_bollinger = 0

                if self.buyed_stocks == 0 and \
                    (cross_bollinger == 1) or \
                        buy_now_process:
                    self.buy_stock(stock)
                    buy_now_process = False

                #comment this block for selling at the end of the date
                if self.buyed_stocks != 0 and \
                    (cross_bollinger == -1 or
                     (stock['boll_mid_lb'] <= self.prev_stock['close'] and stock['boll_mid_lb'] > stock['close'])):
                    self.sell_stock(stock)

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

                self.prev_stock = stock

            if self.transactions > 0:
                self.show_stats(symbol)
                # self.plot_stats(sub_data, spy_stocks)

            else:
                print("Theres no transactions please change BUY/SELL params")
                
    def test_rsi_params(self):

        for buyBand in range(self.rsi_params['buyBand']["min"], self.rsi_params['buyBand']["max"], self.rsi_params['buyBand']["step"]):
            for sellBand in range(self.rsi_params['sellBand']["min"], self.rsi_params['sellBand']["max"], self.rsi_params['sellBand']["step"]):
                for buySignalGap in numpy.arange(self.rsi_params['buySignalGap']["min"], self.rsi_params['buySignalGap']["max"], self.rsi_params['buySignalGap']["step"]):
                    for sellSignalGap in numpy.arange(self.rsi_params['sellSignalGap']["min"], self.rsi_params['sellSignalGap']["max"], self.rsi_params['sellSignalGap']["step"]):
                        self.transactions = 0
                        # self.last_buyed_stock['close'] = 0
                        self.trading_simulation_rsi(
                            buyBand, sellBand, buySignalGap, sellSignalGap)
                        
    def trading_simulation_rsi(self, buyBand, sellBand, buySignalGap, sellSignalGap):
        self.buyed_stocks = 0
        self.money = self.startCredit
        self.first_stock = self.stocks.iloc[0]

        for index, stock in self.stocks.iterrows():
            
            if stock.valRSIclsf <= sellBand and stock.upRSIclsf >= sellSignalGap and self.buyed_stocks == 0:
            # if stock.valRSIclsf <= sellBand and stock.upRSIclsf >= sellSignalGap and self.money > stock.close:
               self.buy_stock(stock)

            if stock.valRSIclsf >= buyBand and stock.upRSIclsf <= buySignalGap and self.buyed_stocks != 0 and stock.close >= self.last_buyed_stock['close']:
            #if stock.valRSIclsf > 60 and stock.upRSIclsf < 0.2 and self.buyed_stocks != 0:
                self.sell_stock(stock)

            self.prev_stock = stock
        # print("last_buyed_stock['close']: " + str(self.last_buyed_stock['close']) + "Transactions: " + str(self.transactions))
        # print("PARAMS buyBand: " + str(buyBand) + " sellBand:" + str(sellBand) + " buySignalGap:" + str(buySignalGap) + " sellSignalGap:" + str(sellSignalGap))
        
        if self.transactions > 0:
            afterSellingStockMoney = round(self.money + (self.buyed_stocks * self.last_buyed_stock['close']),2)
            tradingGainPercent = (afterSellingStockMoney - self.startCredit) / ( self.startCredit/100)/self.share_amount
            
            if  tradingGainPercent > self.best_rsi_params:
                self.best_rsi_params = tradingGainPercent
               
                self.best_settings[self.symbols[0]] = {}
                self.best_settings[self.symbols[0]]['buyBand'] = buyBand
                self.best_settings[self.symbols[0]]['sellBand'] = sellBand
                self.best_settings[self.symbols[0]]['buySignalGap'] = buySignalGap
                self.best_settings[self.symbols[0]]['sellSignalGap'] = sellSignalGap
                print("RSI PARAMS: buyBand: " + str(buyBand) + " sellBand:" + str(sellBand) + " buySignalGap:" + str(buySignalGap) + " sellSignalGap:" + str(sellSignalGap))
                print("RSI Gain: " + str(round(tradingGainPercent,2) ) + "%" + " | Transactions: " + str(self.transactions)+ " | StocksNum: " + str(self.buyed_stocks)+ " | Money: " + str(afterSellingStockMoney))

        
