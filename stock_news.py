from stocknews import StockNews as SN



class StockNews2():
    def __init__(self):
        """
        docstring
        """
        pass
    
    def get_news(self):
        stocks = ['JKS', 'SPWR', 'GPS',"TER","AXTA"]
        sn = SN(stocks, wt_key='0d57e2849d73713e95f395c7440380ff')
        df = sn.summarize()
        print(df)

sn = StockNews2()
sn.get_news()