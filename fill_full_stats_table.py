from datetime import datetime
from sys import path
path.append("../")
import pandas as pd
from alpaca_examples.fin_i import FinI
from alpaca_examples.stock_whisperer import StockWhisperer
from alpaca_examples.market_db import Database, TableName



class FillFullStatsTable():

    def __init__(self) -> None:
        db = Database()
        stocks: pd.DataFrame
        stocks_fs = db.load_data(
            table_name=TableName.DAY_FS, limit=1)
        db_mode = "append"
        

        if stocks_fs is not None and len(stocks_fs) > 0:
            stocks = db.load_data(table_name=TableName.DAY, time_from = stocks_fs.index[0])
        
        else:
            stocks = db.load_data(table_name=TableName.DAY, time_from = "-500d")
            db_mode = "replace"

        
        
        symbols = db.get_symbols()
        for sym in symbols:
            if len(stocks) > 0:
                stocks_sym = FinI.add_indicators(stocks[stocks.sym == sym])
                # stocks.columns.drop(["symbol","sector","industry"])
                # for idx, sym_s in stocks_sym.iterrows():

                #     sql_string = "".join(["select * from financials where symbol = '",
                #                           str(sym_s.sym),
                #                           "' and date::date <= date '",
                #                           str(idx.replace(hour=23, minute=59)),
                #                           "' order by date desc limit 1"])
                #     print(sql_string)
                #     financials = db.sql_select_query_data(sql_string)
                #     new_row = pd.concat([stocks_sym, financials], axis=1)
                #     new_row.to_sql(name="p_day_fs",
                #                    if_exists="append", con=db.engine)
                if stocks_sym is not None:
                    stocks_sym.to_sql(
                        name="p_day_fs", if_exists=db_mode, con=db.engine)
                print(stocks_sym)
        # sw = StockWhisperer()
        # output = sw.find_stocks(TableName.DAY, False)


FillFullStatsTable()
