
import alpaca_trade_api as tradeapi
from alpaca_trade_api.polygon import StreamConn
import threading
import time
import datetime
import sys
sys.path.append('../')
import market_src.alpaca2Login as al
import logging
logging.basicConfig(level=logging.INFO)
class AlpacaBuySell():
    
    def __init__(self, alpaca = None):
        
        if alpaca:
            self.alpaca = alpaca
        else:
            self.alpaca = al.alpaca2Login().getApi()
   
    
    def run(self):
        # Wait for market to open.
        print("Waiting for market to open...")
        tAMO = threading.Thread(target=self.awaitMarketOpen)
        tAMO.start()
        tAMO.join()
        print("Market opened.")

         # Rebalance the portfolio every minute, making necessary trades.
        while True:

            # Figure out when the market will close so we can prepare to sell beforehand.
            clock = self.alpaca.get_clock()
            closingTime = clock.next_close.replace(tzinfo=datetime.timezone.utc).timestamp()
            currTime = clock.timestamp.replace(tzinfo=datetime.timezone.utc).timestamp()
            self.timeToClose = closingTime - currTime

            if(self.timeToClose < (60 * 15)):
                # Close all positions when 15 minutes til market close.
                logging.info("Market closing soon.  Closing positions.")

                positions = self.alpaca.list_positions()
                for position in positions:
                    if(position.side == 'long'):
                        orderSide = 'sell'
                    else:
                        orderSide = 'buy'
                    qty = abs(int(float(position.qty)))
                    respSO = []
                    tSubmitOrder = threading.Thread(target=self.submitOrder(qty, position.symbol, orderSide, respSO))
                    tSubmitOrder.start()
                    tSubmitOrder.join()

                # Run script again after market close for next trading day.
                logging.info("Sleeping until market close (15 minutes).")
                time.sleep(60 * 15)
            else:
                # Rebalance the portfolio.
                tRebalance = threading.Thread(target=self.rebalance)
                tRebalance.start()
                tRebalance.join()
                time.sleep(60)

      # Wait for market to open.
    def awaitMarketOpen(self):
        isOpen = self.alpaca.get_clock().is_open
        while(not isOpen):
            clock = self.alpaca.get_clock()
            openingTime = clock.next_open.replace(
                tzinfo=datetime.timezone.utc).timestamp()
            currTime = clock.timestamp.replace(
                tzinfo=datetime.timezone.utc).timestamp()
            timeToOpen = int((openingTime - currTime) / 60)
            logging.info(str(timeToOpen) + " minutes til market open.")
            time.sleep(60)
            isOpen = self.alpaca.get_clock().is_open
    
        # Submit a batch order that returns completed and uncompleted orders.
    def sendBatchOrder(self, qty, stocks, side, resp):
        executed = []
        incomplete = []
        for stock in stocks:
            if(self.blacklist.isdisjoint({stock})):
                respSO = []
                tSubmitOrder = threading.Thread(target=self.submitOrder, args=[qty, stock, side, respSO])
                tSubmitOrder.start()
                tSubmitOrder.join()
                if(not respSO[0]):
                # Stock order did not go through, add it to incomplete.
                    incomplete.append(stock)
                else:
                    executed.append(stock)
                    respSO.clear()
            resp.append([executed, incomplete])

    # Submit an order if quantity is above 0.
    def submitOrder(self, qty, stock, side, resp = None):
        if resp is None:
            resp= []
        
        if(qty > 0):
            try:
                self.alpaca.submit_order(stock, qty, side, "market", "day")
                logging.info("Market order of | " + str(qty) + " " + stock + " " + side + " | completed.")
                resp.append(True)
            except Exception as e:
                logging.error("Order of | " + str(qty) + " " + stock + " " + side + " | did not go through.")
                resp.append(False)
            else:
                logging.error("Quantity is 0, order of | " + str(qty) + " " + stock + " " + side + " | not completed.")
                resp.append(True)
    
    def get_positions(self):
        """[summary]

        Returns:
            alpaca positions: opened positions on alpaca
            alpaca orders: unfilled orders on alpaca
        """
        positions = self.alpaca.list_positions()
        orders = self.alpaca.list_orders()
        logging.info(positions)
        logging.info(orders)
        # for position in positions:
        #     # print(position)
        #     print(position.symbol)

        # for order in orders:
        #     # print(position)
        #     print(order.symbol)
           
        return positions, orders
    
    def close_postions(self, sym:str):
        return self.alpaca.close_position(sym)

    def close_all_postions(self):
        return self.alpaca.close_all_positions()
        
    async def stream_conn(self):
        conn = StreamConn()


        @conn.on(r'^trade_updates$')
        async def on_account_updates(conn, channel, account):
            logging.info('account', account)


        @conn.on(r'^status$')
        async def on_status(conn, channel, data):
            logging.info('polygon status update', data)


        @conn.on(r'^AM$')
        async def on_minute_bars(conn, channel, bar):
            logging.info('bars', bar)


        @conn.on(r'^A$')
        async def on_second_bars(conn, channel, bar):
            logging.info('bars', bar)

        # blocks forever
        conn.run(['trade_updates', 'AM.*'])

        # if Data API streaming is enabled
        # conn.run(['trade_updates', 'alpacadatav1/AM.SPY'])
        

    
# abs = AlpacaBuySell()
# abs.get_positions()
# respSO = []
# abs.submitOrder(1, 'HD', 'sell', respSO)
