import pandas as pd
from matplotlib.dates import date2num
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

class Plot(object):
    """
    docstring
    """
    @staticmethod
    def candles(self, df, ax, body_w=0.1, shadow_w=0.01):
        width = body_w
        width2 = shadow_w
        pricesup = df[df.close >= df.open]
        pricesdown = df[df.close < df.open]

        ax.bar(pricesup.index, pricesup.close-pricesup.open,
               width, bottom=pricesup.open, color='g')
        ax.bar(pricesup.index, pricesup.high-pricesup.close,
               width2, bottom=pricesup.close, color='g')
        ax.bar(pricesup.index, pricesup.low-pricesup.open,
               width2, bottom=pricesup.open, color='g')

        ax.bar(pricesdown.index, pricesdown.close-pricesdown.open,
               width, bottom=pricesdown.open, color='r')
        ax.bar(pricesdown.index, pricesdown.high-pricesdown.open,
               width2, bottom=pricesdown.open, color='r')
        ax.bar(pricesdown.index, pricesdown.low-pricesdown.close,
               width2, bottom=pricesdown.close, color='r')
        ax.grid()
        return ax

    @staticmethod
    def plot_volume(self,ax,df):
        """
        docstring
        """
        
        ax.set_ylabel(' volume', color="brown")
        ax.tick_params(axis='y', labelcolor="brown")
        ax.tick_params(axis='x',reset=True)
       
        hours = mdates.HourLocator(interval=1)
        h_fmt = mdates.DateFormatter('%Hh')
        ax.xaxis.set_major_locator(hours)
        # ax.set_xticklabels(['{:.0f}'.format(k) for k in ax.get_xticks()])
        
        df.plot(kind="line", use_index=True,
                        y="volume", legend=False, ax=ax, color='brown', linewidth=2, alpha=0.2)
     
        ax.grid()
        # df.plot(kind='line', linestyle="-",
        #         y=["close", "open"], ax=ax, grid=True, linewidth=0.5, color=["red", "green"])
       
        for tick in ax.get_xticklabels():
            tick.set_rotation(0)
        
        return ax

    @staticmethod
    def plot_spy(self, ax, df):

        ax.set_ylabel('S&P pt.', color="tab:red")
        ax.tick_params(axis='y', labelcolor="tab:red")
        # axs[0].tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=True)
        # ax.set_xticks([])
        # ax.tick_params(axis='x', reset=True)
        # sliced_spy = self.spy.loc[self.spy.index.isin(df.index)]
        df.plot(kind="line", use_index=True,
                y="close", legend=False, ax=ax, color='tab:red', linewidth=0.5, alpha=0.5)
        return ax

    @staticmethod
    def plot_stock_prices(self, ax, df, sym):
        ax.set_ylabel(str(sym) + ' price', color="tab:blue")
        ax.tick_params(axis='y', labelcolor="tab:blue")
        ax.set_xticks(df.index)

        # interval = mdates.DayLocator(interval=7)
        # h_fmt = mdates.DateFormatter('%dd')
        # ax.xaxis.set_major_locator(interval)
        # ax.set_xticklabels(['{:.0f}'.format(k) for k in ax.get_xticks()])
        df.plot(kind="line", use_index=True,
                y="close", legend=False, title=str(sym + " | S&P"), ax=ax, grid=True, linewidth=1, color="tab:blue")
        return ax
