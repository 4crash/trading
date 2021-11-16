import logging
import matplotlib.dates as mdates
import mplfinance as mpf
import sys
import numpy
sys.path.append('../')
from utils import Utils
from fin_i import FinI
from matplotlib.dates import date2num
import pandas as pd

class PlotI(object):
    @staticmethod
    def plot_rsi( ax, df):
        """
        docstring
        """
        df.get("kdjk")
        df.plot(kind='line', linestyle="-",
                y=["kdjk", "kdjd"], ax=ax, grid=True, linewidth=0.5, color=["blue", "red"])
        ax.fill_between(df.index, df.kdjk, df.kdjd, where=df.kdjk > df.kdjd,
                        facecolor='blue', interpolate=True, alpha=0.2)
        ax.fill_between(df.index, df.kdjk, df.kdjd, where=df.kdjd >= df.kdjk,
                        facecolor='red', interpolate=True, alpha=0.2)
        for tick in ax.get_xticklabels():
            tick.set_rotation(0)


    @staticmethod
    def plot_macd( ax, df):
        """
        docstring
        """
        df.get("macd")
        df.plot(kind='line', linestyle="-",
                y=["macds", "macd"], ax=ax, grid=True, linewidth=0.5, color=["orange", "green"])
        ax.fill_between(df.index, df.macds, df.macd, where=df.macds > df.macd,
                        facecolor='orange', interpolate=True, alpha=0.2)
        ax.fill_between(df.index, df.macds, df.macd, where=df.macd >= df.macds,
                        facecolor='green', interpolate=True, alpha=0.2)
        for tick in ax.get_xticklabels():
            tick.set_rotation(0)

        return ax

    @staticmethod
    def plot_volume_bars( ax, df, width=0.85):
        """
        docstring
        """
        import numpy as np
        
        # pricesup = df[df.close >= df.open]
        # pricesdown = df[df.close < df.open]
        df = FinI.add_oc_change(df)
        # alpha = float((df.ch_oc / 4) * 0.1)
        # df['alpha'] = float((df.ch_oc / 4) * 0.1)
        # print(df["ch_oc"])
        # ax.bar(pricesup.index, pricesup.volume, width,
        #        bottom=0, color='g', alpha=rgba_colors)
        # ax.bar(pricesdown.index, pricesdown.volume,
        #        width, bottom=0, color='r', alpha=rgba_colors)
        for index, row in df.iterrows():
            # print(str())
            color = "g" if row.open < row.close else "r"
            alpha = Utils.zero_one_vals(
                df.ch_oc.min(), df.ch_oc.max(), row.ch_oc )
            alpha = alpha if row.open < row.close else abs(alpha -1)
            ax.bar(index, row.volume, width,
                   bottom=0, color=color, alpha=float(alpha))
        # df.plot(kind='line', linestyle="o",
        #                 y=["volume"], ax=ax, grid=True, linewidth=0.5, color=["green"])
        # ax.fill_between(df.index, df.macds, df.macd, where=df.macds > df.macd,
        #                     facecolor='orange', interpolate=True, alpha=0.2)
        # ax.fill_between(df.index, df.macds, df.macd, where=df.macd >= df.macds,
        #                     facecolor='green', interpolate=True, alpha=0.2)
        for tick in ax.get_xticklabels():
            tick.set_rotation(0)

        return ax

    @staticmethod
    def plot_boll( ax, df, sym, fib=True):
        """
        docstring
        """

        # df = self.add_bollinger_bands(df)
        df = FinI.add_indicators(df)

        df.plot(kind="line", use_index=True,
                y=["boll", "boll_lb", "boll_ub"], legend=False, ax=ax, grid=True, linewidth=0.7, color=["brown", "orange", "green"], alpha=0.7)

        ax.fill_between(df.index, df.boll_lb, df.boll_ub, where=df.boll > df.close,
                        facecolor='red', interpolate=True, alpha=0.2)
        ax.fill_between(df.index, df.boll_lb, df.boll_ub, where=df.boll <= df.close,
                        facecolor='green', interpolate=True, alpha=0.2)

        def fill_bands(boll, bmin, bmax, alpha=0.2, color="white"):
            nonlocal ax
            ax.fill_between(df.index, boll + bmin, boll + bmax,
                            facecolor=color, interpolate=True, alpha=alpha)
            ax.fill_between(df.index, boll - bmin, boll - bmax,
                            facecolor=color, interpolate=True, alpha=alpha)
            return ax

        if fib:

            # print("slnfvadklngůdnfsld s")
            ax = fill_bands(df.boll, 0, df.boll_2, alpha=0.1, color="white")
            ax = fill_bands(df.boll, df.boll_2, df.boll_3,
                            alpha=0.2, color="white")
            ax = fill_bands(df.boll, df.boll_3, df.boll_5,
                            alpha=0.1, color="blue")
            ax = fill_bands(df.boll, df.boll_5, df.boll_6,
                            alpha=0.2, color="green")
            ax = fill_bands(df.boll, df.boll_6, df.boll_10,
                            alpha=0, color="brown")

        else:
            ax.fill_between(df.index, df.boll_mid_lb, df.boll_lb,
                            facecolor='brown', interpolate=True, alpha=0.2)
            ax.fill_between(df.index, df.boll_mid_ub, df.boll_ub,
                            facecolor='brown', interpolate=True, alpha=0.2)

        for tick in ax.get_xticklabels():
            tick.set_rotation(0)

        return ax
    
    @staticmethod
    def plot_macd_boll( ax, df):
        """
        docstring
        """

        # df = self.add_bollinger_bands(df)
        if "boll" not in df or "macd" not in df:
            df = FinI.add_indicators(df)

        df.plot(kind="line", use_index=True,
                y=[ "macd", "boll_lb_macd", "boll_ub_macd"], legend=False, ax=ax, grid=True, linewidth=0.7, color=["black", "red", "green"], alpha=0.7)

        ax.fill_between(df.index, df.boll_lb_macd, df.macd, where=df.boll_lb_macd > df.macd,
                        facecolor='red', interpolate=True, alpha=0.3)
        ax.fill_between(df.index, df.boll_ub_macd, df.macd, where=df.boll_ub_macd < df.macd,
                        facecolor='green', interpolate=True, alpha=0.3)
        
        ax.fill_between(df.index, df.boll_ub_macd, df.boll_lb_macd,
                        facecolor='blue', interpolate=True, alpha=0.3)


        for tick in ax.get_xticklabels():
            tick.set_rotation(0)

        return ax
    
    @staticmethod
    def plot_candlesticks2( ax, df):
        """
        docstring
        """
        # import matplotlib.dates as mdates
        # hours = mdates.HourLocator(interval=1)
        df["date"] = df.index
        # # print(df)
        # h_fmt = mdates.DateFormatter('%Hh')
        # ax.xaxis.set_major_locator(hours)
        # ax.xaxis.set_major_formatter(h_fmt)

        # df.plot(kind='line', linestyle="dotted", marker="o",
        #         y=["close", "open"], ax=ax, grid=True, linewidth=0.5, color=["red", "green"])

        # ax.fill_between(df.index, df.close, df.open, where=df.close > df.open,
        #                 facecolor='green', interpolate=True, alpha=0.2)
        # ax.fill_between(df.index, df.close, df.open, where=df.open >= df.close,
        #                 facecolor='red', interpolate=True, alpha=0.2)

        for val in range(0, len(df)):
            perc = Utils.calc_perc(
                df.iloc[val].open, df.iloc[val].close)

            ax.text(df.iloc[val].date, ax.get_ylim()[0], str(
                round(perc, 1)) + '%', fontsize=8, color="green" if perc >= 0 else "red")

            ax.text(df.iloc[val-1].date, df.iloc[val].close, str(
                round(df.iloc[val].close, 1)), fontsize=6, color="brown")

        sum_perc = Utils.calc_perc(
            df.iloc[0].open, df.iloc[-1].close)

        ax.text(df.iloc[-1].date, ax.get_ylim()[1], str(
            round(sum_perc, 1)) + '%' + ' sum', fontsize=8, color="green" if sum_perc >= 0 else "red")

        for tick in ax.get_xticklabels():
            tick.set_rotation(0)

        ax = PlotI.plot_candles(df, ax, 0.01, 0.001)

        return ax

    @staticmethod
    def plot_candles( df, ax, body_w=0.1, shadow_w=0.01, alpha=0.7):
        width = body_w
        width2 = shadow_w
        pricesup = df[df.close >= df.open]
        pricesdown = df[df.close < df.open]

        ax.bar(pricesup.index, pricesup.close-pricesup.open,
               width, bottom=pricesup.open, color='g', alpha=alpha)
        ax.bar(pricesup.index, pricesup.high-pricesup.close, width2,
               bottom=pricesup.close, color='g', alpha=alpha)
        ax.bar(pricesup.index, pricesup.low-pricesup.open, width2,
               bottom=pricesup.open, color='g', alpha=alpha)

        ax.bar(pricesdown.index, pricesdown.close-pricesdown.open,
               width, bottom=pricesdown.open, color='r', alpha=alpha)
        ax.bar(pricesdown.index, pricesdown.high-pricesdown.open,
               width2, bottom=pricesdown.open, color='r', alpha=alpha)
        ax.bar(pricesdown.index, pricesdown.low-pricesdown.close,
               width2, bottom=pricesdown.close, color='r', alpha=alpha)
        ax.grid()
        return ax

    @staticmethod
    def plot_fib( df, ax, alpha=0.2, fib_name = "fb_top"):
        
               
        ax.fill_between(df.index, df.low.min(), df[fib_name+"_2"],
                        facecolor='gray', interpolate=True, alpha=alpha)
        ax.fill_between(df.index, df[fib_name+"_2"], df[fib_name+"_3"],
                        facecolor='blue', interpolate=True, alpha=alpha)
        ax.fill_between(df.index, df[fib_name+"_3"], df[fib_name+"_5"],
                        facecolor='purple', interpolate=True, alpha=alpha)
        ax.fill_between(df.index, df[fib_name+"_5"], df[fib_name+"_6"],
                        facecolor='green', interpolate=True, alpha=alpha)
        ax.fill_between(df.index, df[fib_name+"_6"], df[fib_name+"_7"],
                        facecolor='gray', interpolate=True, alpha=alpha)
        
        return ax
    
    @staticmethod
    def plot_overnight( ax, df):
        """
        docstring
        """

        hours = mdates.DayLocator(interval=1)
        df["date"] = df.index
        # print(df)
        h_fmt = mdates.DateFormatter('%d - %m')
        ax.xaxis.set_major_locator(hours)
        ax.xaxis.set_major_formatter(h_fmt)

        # df.plot(kind='line', linestyle="None", marker="o",
        #         y=["close", "open"], ax=ax, grid=True, linewidth=0.5, color=["red", "green"])
        ax = PlotI.plot_candles(df, ax, 0.5, 0.05)

        # ax.fill_between(df.index, df.close, df.open, where=df.close > df.open,
        #                 facecolor='green', interpolate=True, alpha=0.2)
        # ax.fill_between(df.index, df.close, df.open, where=df.open >= df.close,
        #                 facecolor='red', interpolate=True, alpha=0.2)

        if "weekday" not in df:
            df['date'] = pd.to_datetime(df.index, utc=True)
            # df.index = pd.to_datetime(df.index, utc=True)
            df["weekday"] = df.date.dt.dayofweek
        for val in range(0, len(df)-1):
            print
            try:
                if df.iloc[val].weekday != (df.iloc[val+1].weekday-1):
                    last_week_day = df.iloc[val].weekday
                    weekend_percents = Utils.calc_perc(
                        df.iloc[val].close, df.iloc[val+1].open)
                    ax.text(date2num(df.iloc[val].date), ax.get_ylim()[1], str(
                        round(weekend_percents, 2)) + '%', fontsize=8, color="green" if weekend_percents >= 0 else "red")

                if (df.iloc[val-1].weekday - df.iloc[val].weekday) > 1:
                    first_week_day = df.iloc[val].weekday

                    days_to_end = 1
                    while (df.iloc[val+days_to_end].weekday-df.iloc[val].weekday) > 0:
                        days_to_end += 1

                    week_percents = Utils.calc_perc(
                        df.iloc[val].open, df.iloc[val + days_to_end-1].close)
                    color = "green" if week_percents >= 0 else "red"

                    # change in week
                    ax.text(date2num(df.iloc[val].date), ax.get_ylim()[1], str(
                        round(week_percents, 2)) + '%', fontsize=8, color=color)

                    ax.axvspan(date2num(df.iloc[val].date),
                               date2num(df.iloc[val+days_to_end-1].date),  alpha=0.05, color=color)

                    # print(df.iloc[val].date.day_name() + " - " + df.iloc[val+days_to_end-1].date.day_name())

            except IndexError:
                print("plot_overnight Index error")

            try:
                perc = Utils.calc_perc(
                    df.iloc[val].close, df.iloc[val+1].open)
            except IndexError:
                print("indexError")

            ax.text(df.iloc[val].date, ax.get_ylim()[0], str(
                round(perc, 2)) + '%', fontsize=8, color="green" if perc >= 0 else "red")

            ax.text(df.iloc[val].date, df.iloc[val].close, str(
                round(df.iloc[val].close, 1)), fontsize=8, color="brown")

        # # sum_perc = Utils.calc_perc(
        # #         df.iloc[0].open, df.iloc[-1].close)

        # # ax.text(df.iloc[-1].date, ax.get_ylim()[1], str(
        # #     round(sum_perc, 1)) + '%' + ' sum', fontsize=8, color="green" if sum_perc >= 0 else "red")

        # for tick in ax.get_xticklabels():
        #     tick.set_rotation(0)

        return ax

    # def plot_volume(ax,df):
    #     """
    #     docstring
    #     """

    #     ax.set_ylabel(' volume', color="brown")
    #     ax.tick_params(axis='y', labelcolor="brown")
    #     ax.tick_params(axis='x',reset=True)

    #     hours = mdates.HourLocator(interval=1)
    #     h_fmt = mdates.DateFormatter('%Hh')
    #     ax.xaxis.set_major_locator(hours)
    #     # ax.set_xticklabels(['{:.0f}'.format(k) for k in ax.get_xticks()])

    #     df.plot(kind="line", use_index=True,
    #                     y="volume", legend=False, ax=ax, color='brown', linewidth=2, alpha=0.2)

    #     ax.grid()
    #     # df.plot(kind='line', linestyle="-",
    #     #         y=["close", "open"], ax=ax, grid=True, linewidth=0.5, color=["red", "green"])

    #     for tick in ax.get_xticklabels():
    #         tick.set_rotation(0)

    #     return ax
    @staticmethod
    def plot_yahoo_candles( df):
        mpf.plot(df, type='candle', style='yahoo', volume=True)

    @staticmethod
    def plot_spy( ax, df, alpha = 0.7):

        ax.set_ylabel('S&P pt.', color="tab:red")
        ax.tick_params(axis='y', labelcolor="tab:red")
        # axs[0].tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=True)
        # ax.set_xticks([])
        # ax.tick_params(axis='x', reset=True)
        # sliced_spy = self.spy.loc[self.spy.index.isin(df.index)]
        df.plot(kind="line", use_index=True,
                y="close", legend=False, ax=ax, color='tab:red', linewidth=0.5, alpha=alpha)
        return ax
    @staticmethod
    def set_margins(plt):
        plt.subplots_adjust(left=0.03, bottom=0.03, right=0.97,
                            top=0.97, wspace=0.1, hspace=0.1)
    @staticmethod
    def plot_sma( ax, df, sma_values, alpha=0.9):

        df.plot(kind="line", use_index=True,
                y=sma_values, legend=False, ax=ax, grid=True, linewidth=1, alpha=alpha)
        return ax

    @staticmethod
    def plot_stock_prices( ax, df, sym, alpha):
        ax.set_ylabel(str(sym) + ' price', color="tab:blue")
        ax.tick_params(axis='y', labelcolor="tab:blue")
        ax.set_xticks(df.index)

        # interval = mdates.DayLocator(interval=7)
        # ax.xaxis.set_major_locator(interval)
        h_fmt = mdates.DateFormatter('%dd')

        ax.xaxis.set_major_formatter(h_fmt)
        # ax.set_xticklabels(ax.get_xticks(), fontsize=4)00
        # ax.set_xticklabels([k.DateFormatter('%d') for k in ax.get_xticks()])

        df.plot(kind="line", use_index=True,
                y="close", legend=False, title=str(sym) + " | S&P", ax=ax, grid=True, linewidth=1, color="tab:gray", alpha=alpha)
        return ax
    
    @staticmethod
    def plot_weeks( ax, df):

        df = FinI.add_weekday(df)

        # print(df)
        # ax.axvspan(date2num(datetime(2020,11,9)), date2num(datetime(2020,11,12)),
        #    label="March", color="crimson", alpha=0.3)
        color = "gray"
        for val in range(0, len(df)-1):

            # in progress version with hollidays
            last_week_day = None
            first_week_day = None
            weekend_percents = 0
            week_percents = 0

            try:
                if df.iloc[val].weekday != (df.iloc[val+1].weekday-1):
                    last_week_day = df.iloc[val].weekday
                    weekend_percents = Utils.calc_perc(
                        df.iloc[val].close, df.iloc[val+1].open)
                    ax.text(date2num(df.iloc[val].date), ax.get_ylim()[1], str(
                        round(weekend_percents, 1)) + '%', fontsize=8, color="green" if weekend_percents >= 0 else "red")

                if (df.iloc[val-1].weekday - df.iloc[val].weekday) > 1:
                    first_week_day = df.iloc[val].weekday

                    days_to_end = 1
                    while (df.iloc[val+days_to_end].weekday-df.iloc[val].weekday) > 0:
                        days_to_end += 1

                    week_percents = Utils.calc_perc(
                        df.iloc[val].open, df.iloc[val + days_to_end-1].close)
                    color = "green" if week_percents >= 0 else "red"

                    # change in week
                    ax.text(date2num(df.iloc[val].date), ax.get_ylim()[0], str(
                        round(week_percents, 1)) + '%', fontsize=8, color=color)

                    ax.axvspan(date2num(df.iloc[val].date),
                               date2num(df.iloc[val+days_to_end-1].date),  alpha=0.05, color=color)

                    # print(df.iloc[val].date.day_name() + " - " + df.iloc[val+days_to_end-1].date.day_name())

            except IndexError:
                print("plot_weeks Index error")

        return ax

    @staticmethod
    def plot_bs(ax, df):
        # ax.set_ylabel(str(sym) + ' price', color="tab:blue")
        # ax.tick_params(axis='y', labelcolor="tab:blue")
        # ax.set_xticks(df.index)
        # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        #     print(df)
        try:
            ax = df.loc[df['buy'] > 0].plot(
                marker='o', y='buy', ax=ax, color='g', linestyle="None", label="buy", alpha = 0.5)
            ax = df.loc[df['sell'] > 0].plot(
                marker='o',x="sell_date", y='sell', ax=ax, color='r', linestyle="None", label="sell", alpha = 0.5)
        except TypeError:
            logging.info("No sell data")

        # if len(self.sell_marks)>0:
        #     ax3 = self.sell_marks.plot(marker='o' ,y='close', ax=ax2, color='r', linestyle="None", label = "sell", grid=True)
        # df.plot(kind="line", use_index=True,
        #                 y="close", legend=False,  grid=True, linewidth=1, color="tab:blue")
        return ax

    @staticmethod
    def plot_sector_stats( ax, df, sector = None):
        """
        docstring
        """
        # print(df)
        for index in df.index.unique('sector'):
            sec = df.iloc[df.index.get_level_values(
                'sector') == index]
            alpha = 1 if sector == index else 0.2
            sec.unstack(level=0).plot(kind="line", 
                                      y=["close_n"], legend=True, label=[index], marker='o', ax=ax, grid=True, linewidth=2, alpha=alpha)
            
           

        # ax.fill_between(df.index, df.boll_lb_macd, df.macd, where=df.boll_lb_macd > df.macd,
        #                 facecolor='red', interpolate=True, alpha=0.3)
        # ax.fill_between(df.index, df.boll_ub_macd, df.macd, where=df.boll_ub_macd < df.macd,
        #                 facecolor='green', interpolate=True, alpha=0.3)
        
        # ax.fill_between(df.index, df.boll_ub_macd, df.boll_lb_macd,
        #                 facecolor='blue', interpolate=True, alpha=0.3)

        # ax.tick_params(axis='y', labelcolor="tab:blue")
        locator = mdates.AutoDateLocator(minticks=1)
        formatter = mdates.ConciseDateFormatter(locator)
        formatter.zero_formats[3] = '%d-%b'

        formatter.offset_formats = ['',
                                    '%Y',
                                    '%b %Y',
                                    '%d %b %Y',
                                    '%d %b %Y',
                                     ]
        # formatter.formats = '%d'
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
            # ax.set_xticks(sec.iloc[sec.index.get_level_values('date')])
        # for tick in ax.get_xticklabels():
        #     tick.set_rotation(0)

        return ax
