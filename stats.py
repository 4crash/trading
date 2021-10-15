from scipy.stats import linregress
import sys
sys.path.append('../')
from alpaca_examples.market_db import Database, TableName
import statsmodels.api as sm
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

warnings.filterwarnings('ignore')


class Stats(object):
    """
    docstring
    """
    def __init__(self):
        """
        docstring
        """
        self.db = Database()
        self.df = pd.DataFrame()
    def get_data(self):
        self.db.set_time_from("-120d")
        self.df = self.db.load_data(TableName.DAY, symbols="jks")
        return self.df
        
    def get_volume_realtions(self):
        df = self.get_data()
        print(df.corr())
        # print(df)
        sns.pairplot(df, kind="scatter")
        plt.show()
    
    def linear_relations(self):
        df = self.get_data()
        print(linregress(df['close'], df['volume']))
        plt.scatter(df['close'], df['volume'])
        plt.show()

st = Stats()
st.get_volume_realtions()
