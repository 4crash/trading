
import sys
from typing import List
sys.path.append('../')
from alpaca_examples.utils import Utils
from alpaca_examples.singleton import Singleton
from pytz import timezone
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime, timedelta
from enum import Enum
import pytz
from sqlalchemy.exc import NoSuchTableError

utc = pytz.UTC
localtz = timezone('Europe/Prague')


class TableName(Enum):
    MIN15 = "p_15min"
    MIN1 = "p_1min"
    DAY = "p_day"
    DAY_FS = "p_day_fs"

    def to_str(self):
        return str(self.value)

# @dataclass
# class TimeStruct():
#     time_from = {"ds":"1d","dl":"60d","e": "2d", "s": "20d", "f": "7d"}
#     time_to = {"ds": "1d", "dl": "60d", "e": "2d", "s": "20d", "f": "7d"}
class Database(Singleton):
    
    def __init__(self):
        """
        docstring
        """
        self.db_name = "nyse_financials"
        self.price_table_name = TableName.DAY.to_str()
        self.financials_table_name = "financials"
        self.earnings_table_name = "earning_dates"
        self.sentiment_table_name = "sentiment"
        self.engine = create_engine(
            'postgresql://postgres:crasher@localhost:5432/'+self.db_name)
        # self.symbols = None
        self.time_from = None
        self.time_to = None
        self.stock_stats = pd.DataFrame()
        self.limit = None
        self.last_date = None
        # self.sectors = []

    def query_sym(self, symbols, col_name="sym",  where_part=''):
        if symbols is not None and type(symbols) != str:

            if len(symbols) > 0:
                if where_part != '':
                    where_part += ' and '
                where_part += col_name + ' in ('
                for sym in symbols:
                    where_part += '\'' + str(sym.upper()) + '\','

                where_part = where_part[:-1]
                where_part += ')'

        elif symbols and type(symbols) == str:
            where_part += col_name + '=\'' + symbols.upper() + '\' '
        
        return where_part
    
    # def check_time_to(self, time_to = None):
    #     return self.last_date  if self.last_date else time_to

    def assemblyWhereQuery(self, sectors=[], industries = [], symbols=None, time_from=None, time_to=None):
        where_part = self.query_sym(symbols = symbols)
       
        
        if time_from or self.time_from:
            time_from = self.get_date_format(
                time_from) if time_from else self.get_date_format(self.time_from)
            if where_part != '':
                where_part += ' and '
            where_part += ' index >= \'' + str(time_from) + '\''

        if time_to or self.time_to or self.last_date:
            if self.last_date is None:
                time_to = self.get_date_format(
                    time_to) if time_to else self.get_date_format(self.time_to)
            else:
                time_to = self.last_date
                
            if where_part != '':
                where_part += ' and '
            where_part += ' index <= \'' + str(time_to) + '\''

        if sectors is not None and type(symbols) != str and len(sectors) > 0:
            if where_part != '':
                where_part += ' and '
            where_part += ' sectors.sector in ('
            for sector in sectors:
                where_part += '\'' + str(sector) + '\','

            where_part = where_part[:-1]
            where_part += ')'
        elif sectors and type(sectors) == str:
            where_part += ' sectors.sector=\'' + sectors + '\' '

        if industries is not None and type(symbols) != str and len(industries) > 0:
            if where_part != '':
                where_part += ' and '
            where_part += ' industry in ('
            for industry in industries:
                where_part += '\'' + str(industry) + '\','

            where_part = where_part[:-1]
            where_part += ')'
        elif industries and type(industries) == str:
            where_part += ' industry=\'' + industries + '\' '

                
        if where_part != '':
            where_part = " where " + where_part
            
        
        return where_part
    
    def get_symbols(self, table_name = None)-> List:
        
        result = pd.read_sql(
            'select symbol as sym from sectors group by symbol order by symbol', con=self.engine).sym.tolist()
        # print("symbols: " + str(result))
        return result
    
    def get_sectors(self, table_name=None) -> List:
        
        result = pd.read_sql(
            'select sector from sectors  group by sector order by sector', con=self.engine).sector.tolist()
        
        return result
    
    def get_industries(self, table_name=None) -> List:
        
        result = pd.read_sql(
            'select industry from sectors  group by industry order by industry', con=self.engine).sector.tolist()
        
        return result   
    def checkTableName(self, table_name):
        
        if not table_name and self.price_table_name:
            table_name = self.price_table_name
        elif not table_name:
            print("Please set the DB table name.")
            exit()
        # print(type(table_name))
        if isinstance(table_name, Enum):
            
            table_name = table_name.to_str()
            
        return table_name
    
    def get_limit(self, limit):
        limit = limit if limit is not None else self.limit
        return ' DESC limit ' + str(limit)  if limit is not None else ""
            
    def load_data(self, table_name=None, symbols=None, sectors=None, industries = None, limit=None, time_from=None, time_to=None):
       
        table_name  = self.checkTableName(table_name)
        sql_string = 'select * from '+table_name + ' right join sectors on sym=sectors.symbol ' + self.assemblyWhereQuery(symbols=symbols, sectors=sectors, industries = industries, time_from=time_from, time_to=time_to) + \
            '  order by index ' + self.get_limit(limit)
        
        print(sql_string)
        result = pd.read_sql(sql_string, con=self.engine, index_col='index')
      
        
        if len(result) == 0:
            print("NO DATA: check DB or extend timespan for: " + str(symbols))
            Utils.countdown(1) 
            
            
        result.sort_index(inplace=True)
        result = self.addAmountColumnToData(result)
        return result
    
    def addAmountColumnToData(self, data):
        data["amount"] = ((data.high + data.low + data.open + data.close) / 4) * data.volume
        return data
      
    def load_spy(self, table_name = None, limit=None, time_from = None, time_to = None):
        
        table_name  = self.checkTableName(table_name)
        data = pd.read_sql('select * from ' + table_name + self.assemblyWhereQuery(symbols='SPY', time_from=time_from,
                                                                                   time_to=time_to) + ' order by index ' + self.get_limit(limit), con=self.engine, index_col='index')
        # print('select * from ' + table_name + self.assemblyWhereQuery(symbols='SPY') + ' order by index')
        data = self.addAmountColumnToData(data)
        data.sort_index(inplace=True)
        return data
    
    def get_financials(self, symbol = None, type="financials", date_from='0d', date_to='0d', limit = None):
        # stats like EBIDTa, SHORT OPTIONS, EARNINGS, DIVIDENDS, DEBT etc.
        
       
        date_from = self.get_date_format(date_from)
        
        if self.last_date is None:
            date_to = self.get_date_format( date_to)
        else:
            date_to = self.last_date

        
        # print("DATE TO B: " + str(date_to))
        # print("LAST DATE " + str(self.last_date))
        # print("DATE TO " + str(date_to))
        
        sym_q = ""
        
        if type == "financials":
            table = self.financials_table_name
            
            if symbol:
                sym_q = self.query_sym(symbols=symbol, col_name="symbol") + ' and '
            
            where = ' where ' + sym_q + ' date > \'' + str(
                date_from) + '\' and date <= \'' + str(date_to) + '\''
            
            where +=  ' order by date desc'
            
        elif type == "earnings":
            table = self.earnings_table_name
            if symbol is not None:
                sym_q = self.query_sym(symbols=symbol, col_name="ticker") + ' and '
                
            where = ' where ' + sym_q +' startdatetime > \'' + str(
                date_from) + '\' and startdatetime <= \'' + str(date_to) + '\''
            
            where += ' order by startdatetime desc'
            
        elif type == "sentiment":
            table = self.sentiment_table_name
            if symbol is not None:
                 sym_q = self.query_sym(symbols=symbol, col_name="stock") + ' and '
                
            where = ' where ' + sym_q + ' news_dt > \'' + str(
                    date_from) + '\' and news_dt <= \'' + str(date_to) + '\''
            
            where += ' order by news_dt desc'
            
        sql_string = 'select * from ' + table + where + self.get_limit(limit)
        print(sql_string)
        output = pd.read_sql(sql_string, con=self.engine)
        if type == "financials":
           output =  output.sort_values(by="date", ascending=True)

        return output
    
    def get_last_financials(self):
        sql_string = 'select  DISTINCT ON (symbol) * from financials order by symbol, date desc '
        print(sql_string)
        output = pd.read_sql(sql_string, con=self.engine)
        output = output.sort_values(by=["sector", "symbol"], ascending=True)
        return output

    def get_last_n_days(self, sym, n_days_back = 1, table = TableName.MIN15):
        table_name = self.checkTableName(table)
        sql_string = "select * from "+ str(table_name) +" where date(index) >= " + \
            "(select date(index) - INTERVAL '" + \
            str(n_days_back) + " DAY' from p_15min order by index desc limit 1)" + \
            "and sym = '" + \
            str(sym) + "'"
        print(sql_string)
        output = pd.read_sql(sql_string, con=self.engine)
        output = output.sort_values(by=["index"], ascending=True)
        return output
        

    def get_date_format(self, date_str):
        print("db-get_date_format-LAST DATE  " + str(self.last_date))
        last_date = self.last_date if self.last_date else datetime.now()

        if last_date.tzinfo is None:
            last_date = localtz.localize(last_date)
            
        # date_str = "0d" if date_str is None else date_str
        
        if type(date_str) != str:
            return date_str
        else:
            add = True
            if date_str.find("-") > -1:
                add = False
                date_str = date_str.replace("-", "")
            
                
        if add:
            return last_date + \
                timedelta(minutes=Utils.convert_to_minutes(date_str))
        else:
            return last_date - \
                timedelta(minutes=Utils.convert_to_minutes(date_str))
           
        
    def set_time_from(self, data):
       
        self.time_from = self.get_date_format(data)
        return self.time_from
        
            
    
    def set_time_to(self, data):
        
        self.time_to = self.get_date_format(data)
        return self.time_to
    
    def save_data(self, table_name, data, if_exists):
        
        try:
           return  data.to_sql(table_name, con = self.engine, if_exists = if_exists)
        except NoSuchTableError:
            return pd.DataFrame()
    
    def get_data(self, table_name):
        
        try:
            return pd.read_sql(table_name, con=self.engine)
        except NoSuchTableError:
            return pd.DataFrame()
    
    def get_all_data(self,time_from = None, sym = None, table_name = TableName.DAY, time_to=None):
        print(time_from)
        if time_from is not None and type(time_from) is str:
            self.set_time_from(time_from)
        dfp = self.load_data(table_name, symbols=sym, time_from = time_from, time_to=time_to)
        
        if len(dfp) < 1:
            print("No data !!!")
            # exit()
            
        
        if dfp is None or len(dfp) < 1:
            return None
  
        spy = self.load_spy(table_name = TableName.DAY)
    
        # financials = self.db.get_financials(sym)
        # financials = financials.round(2)
        # sentiment = None
        # earnings = self.db.get_financials(
        #         symbol=sym, type="earnings", date_from="2d", date_to="30d")
        
        
        # sentiment = self.db.get_financials(
        #     symbol=sym, type="sentiment", date_from="20d", date_to="30d")
        # sentiment = sentiment.groupby(by="stock").mean().round(2) if sentiment is not None and len(sentiment) > 0 else sentiment
        
        earnings, sentiment, financials = self.get_fundamentals(sym)
        print("get_all_data() - done")
        return dfp, financials, sentiment, earnings, spy
        
    def get_fundamentals(self, sym = None, tf={"e": "-8d", "s": "-20d", "f": "-7d"}, tt={"e": "30d", "s": "30d", "f": "1d"}):
        """
        docstring
        """
        earnings = self.get_financials(
            symbol=sym, type="earnings", date_from=tf["e"], date_to=tt["e"])
        # print(earnings)
        sentiment = self.get_financials(
            symbol=sym, type="sentiment", date_from=tf["s"], date_to=tt["s"])
        # print(sentiment)
        financials = self.get_financials(
            symbol=sym, type="financials", date_from=tf["f"], date_to=tt["f"])
        # print(financials)
        print("get_fundamentals() - done")

        return earnings.round(2), sentiment.round(2), financials.round(2)

    def sql_select_query_data(self, sql_string=None):

        # print(sql_string)
        result = pd.read_sql(sql_string, con=self.engine)
        return result
