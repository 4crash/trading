
import sys
sys.path.append('../')
from market_app.overview.refreshFinancials import refreshFinancials
import asyncio
from alpaca_examples.get_data import getData
from alpaca_examples.stock_whisperer import StockWhisperer
from alpaca_examples.market_db import TableName
from datetime import datetime, timedelta
from pytz import timezone
import pytz
utc = pytz.UTC

localtz = timezone('Europe/Prague')


sw = StockWhisperer()

args = sys.argv

leave_first = 0
# default values
sw.db.price_table_name = TableName.MIN15
loosers = False
ts = None
sym = None
for arg in args:
    
    if args[0] != arg:
        k = arg.split("=")[0]
        v = arg.split("=")[1]
        if k == "t":
            itype = v
            
        if k == "s" or  k == "sym":
            sw.symbols = sym = v.upper().split(',')
            
            
        if k == "tf":
            sw.db.set_time_from(v)
            
        if k == "tt":
            sw.db.set_time_to(v)
            
        if k =="tb" or k=="tab":
            sw.db.price_table_name = v
       
        if k == "l":
            loosers = True
        
        if k=="se":
            sw.sectors = v.split(',')
        
        if k=="i":
            interval = v
            
        if k=="lim":
            sw.db.limit = v
            
        if k=="ts":
            ts = v
        # else:
        #     print(" specify interval could be 1,5,15 - minutes, 0 - for day")
    

    # return time_from,time_to,symbols,sectors, loosers


# time_from, time_to, symbols, sectors,loosers = proccess_inputs(sys.argv)

# download financials
if itype == "df":
    rf = refreshFinancials()
    asyncio.run(rf.start(send=None))

# FIND BEST RSI SETTINGS BY BACKTESTING BOLL MACD RSI
# elif itype == "fbrm" or itype=="tr":
#     from alpaca_examples.back_trader import BackTrader
#     btr = BackTrader()
#     # bt.stock_stats.KDJ_WINDOW = 14
#     if not btr.db.time_from:
#         btr.db.set_time_from("365d")
        
#     btr.trading_alg(table_name=TableName.DAY, buy_now=False) 

# back test strategy 1 bollinger bands with fibb levels
elif itype == "ttr":
    from alpaca_examples.back_trader import BackTrader
    btr = BackTrader()
    if not btr.db.time_from:
        btr.db.set_time_from("-365d")
        
    btr.test_strategy_1()
    
# test trading strategy
elif itype == "ttrs":
    from alpaca_examples.back_trader import BackTrader
    btr = BackTrader()
    btr.symbols = sym
    # bt.stock_stats.KDJ_WINDOW = 14
    if not btr.db.time_from:
        btr.db.set_time_from("-365d")
    if ts is None:
        ts = "sma9"
    btr.trading_alg(table_name=TableName.DAY, buy_now=True, strategy_name=ts)
    
# TOP STOCKS
elif itype == "tss":
    sw.top_stocks(table_name=sw.db.price_table_name, top_losers=loosers)

#TOP SECTORS AND TOP STOCKS IN SECTOR WITH THE MOST GAIN
elif itype == "ts":
    sw.top_sectors(table_name=None, loosers=loosers)
    sw.stocks = sw.stocks.sort_values(by="flpd", ascending=loosers)
    print(sw.stocks.iloc[:-1].index[0])
    sw.sectors = [sw.stocks.iloc[:-1].index[0]]
    sw.top_stocks(table_name=None, top_losers=loosers)


# FIND BEST RSI SETTINGS BY BACKTESTING
elif itype == "frp":
    if sw.db.time_from is None:
        sw.db.set_time_from("100d")
    if sw.symbols:
        sw.start_rsi_test(TableName.DAY, 1, sw.symbols)
    else:
        print('Please fill stock symbol as second param. ')
        exit()

# FIND TOP STOCK AND TRADE THEM
elif itype == "tss_fbrm":

  
    sw.top_stocks(show_stocks_num=10, from_top=20, top_losers=False)

    # TRADING TOP GAINERS
    sw.symbols = sw.top_stocks_list
    # bt.stock_stats.KDJ_WINDOW = 14
    sw.trading_alg(buy_now=True, strategy_name=ts)

# find stocks to buy with low RSI, bollinger and MACD
elif itype == "fstb":
    # bt = BackTest()

    # bt.sectors = ['Technology']
    # bt.sectors = ['Consumer Cyclical', 'Real Estate']
 
    
    sw.prepare_buy_logic(infinite = True)

# price volume show the most trading volume at the price level
elif itype == "pv":

    # bt.sectors = ['Technology']
    # bt.sectors = ['Consumer Cyclical', 'Real Estate']
    sw.plot_price_vol(sw.symbols)
  
    # ANALAYZE TOP GAINERS
    # bt.time_from = utc.localize(datetime.now().replace(hour=15,minute=29,second=0) - timedelta(hours=480))
    # bt.time_to = utc.localize(datetime.now().replace(hour=21,minute=29,second=0) - timedelta(hours=1))
    # print(str(bt.time_from))
    
# check buyed stocks for sale
elif itype == "chsfs":
    
    sw.check_stocks_for_sale()
    #  bt.save_chart()

# test the best moment when stock should be buyed
elif itype == "tst":

    # sw.sectors_day_stats(table_name = TableName.DAY)
    #  bt.save_chart()
    
    from alpaca_examples.back_tester import BackTest
    bt = BackTest()
    bt.test_buy_alg()



# Compare spy sym 
elif itype == "tssc":

    sw.spy = sw.load_spy(sw.db.price_table_name)
    sw.iterate_by_symbol(TableName.DAY,None, sw.compare_spy_sym)
    sw.comp_sym_spy = sw.comp_sym_spy.sort_values(
        by="spy_stock_comp", ascending=False)

    sw.comp_sym_spy.iloc[:10].plot(
        kind="barh", x="sym", y="spy_stock_comp", legend=True)
    sw.draw_chart_values(sw.comp_sym_spy.iloc[:10].spy_stock_comp)

# day week month statistics
# elif itype == "ds":
#     pass

# uptrend sectors in each month
elif itype == "tspy":
    sw.ss.sectors_uptrend_by_month()

# find bottom of the stocks when spy is going down
elif itype == "fb":
    sw.find_bottom()
# download ernings
elif itype == "de":
    gd = getData()
    gd.get_earnings()
# download prices
elif itype == "dd":
    gd = getData()
    gd.start_download(interval, utc.localize(
            datetime.now() - timedelta(days=365)))
    
# show earning dates
elif itype == "se":
    sw.show_earning_dates(sw.db.time_from, sw.db.time_to, symbols = sw.symbols)
    
# show stats for specific stocks 
elif itype == "ss":
    sw.show_fin_earn_price(sym = sw.symbols)

# show stats for buyed stocks
elif itype == "bs":
    sw.show_fin_earn_price(sym=sw.buyed_stocks_list)

# download sentiment
elif itype == "ds":
    gd = getData()
    gd.get_sentiment()

# save sentiment from csv which has been filled from web
elif itype == "dsc":
    gd = getData()
    gd.get_sentiment_from_csv()

# overnight stats
elif itype == "on":
    from alpaca_examples.back_trader import BackTrader
    btr = BackTrader()
    btr.symbols = sw.symbols
    btr.calculate_overnight(time_from="-120d")

# stats by volume increase
elif itype == "vm":
   sw.volume_stats()
    # bt.volume_move(table_name=TableName.DAY)
    
# short ratio from previous days
elif itype == "sr":
    
    if sw.db.limit is not None:
        limit = sw.db.limit
    else:
        limit = 5
    sw.find_best_short_ratio(sw.db.time_from, sw.db.time_to, limit = limit )
   # bt.volume_move(table_name=TableName.DAY)
else:
    print('Please use one of these params: ts- test strategy name, de-get earnings dates, dd-download prices,ss-showstats, e, fstb year month sector,ttrs, ttr, frp, ts, fbrm, rf, tss + h + (gainers/loosers) 0/1 + tn , fbrm + sym, pv + sym + tn, chsfs - check stocks for sell, ds -weekdaystats hours symbol, vm - volume movement, on - overnight, ds - get sentiment, ge - get eranings')

# SAP INDEX WHICH HAS ACTUALLY SPY SYMBOL
# bt.sap.get('macd')
# bt.sap.get('kdjk')
# bt.sap.get('boll')
# bt.sap.get('change')
# print(bt.sap.head())


# bt.classifyFundamental()
# bt.classifySPYIndex()


# print(bt.stocks.tail())
# f1 = plt.subplot2grid((6, 4), (1, 0), rowspan=6, colspan=4, axisbg='#07000d')

# candlestick_ohlc(f1, bt.stocks.values, width=.6, colorup='#53c156', colordown='#ff1717')
# mpf.plot(bt.stocks, type='candle', volume=True)
# plt.show()
