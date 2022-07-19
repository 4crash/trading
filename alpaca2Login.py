import alpaca_trade_api as tradeapi

class alpaca2Login():
    __shared_instance = None
    api = None

    def __init__(self):
        """virtual private constructor"""
        if self.__shared_instance is not None:
            raise Exception ("This class is a singleton class !") 
        else:
            self.api = tradeapi.REST('PKO3I1RA5WNUJRL049BR', 'DaQLVBDCfFtZRCDxsmjkiYdvUFqfQGpzvH9dyVpD',
                                     'https://paper-api.alpaca.markets', api_version='v2')  # or use ENV Vars shown below
            self.__shared_instance = self

    @staticmethod
    def getInstance(): 
        """Static Access Method"""
        if self.__shared_instance is None:
            alpacaLogin() 
        return self.__shared_instance
    
    
    def getApi(self):
        return self.api

