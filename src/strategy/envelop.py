import pandas as pd
from ta.trend import SMAIndicator, EMAIndicator

class indicator_envelop:
    @staticmethod
    def ma(data:pd.DataFrame, x:int)-> pd.Series:
        """
        Moving Average Ratio

        This method calculates the moving average.

        :param data: DataFrame containing 'open' prices.
        :type data: pd.DataFrame
        :param x: Window size for the moving average.
        :type x: int
        :return: Series containing the moving average 
        :rtype: pd.Series
        """
        ma_x = SMAIndicator(data['open'], window=x).sma_indicator()
        return ma_x

    @staticmethod
    def ema(data:pd.DataFrame, x:int, y:int)-> pd.Series:
        """
        Exponential Moving Average Ratio

        This method calculates the moving average.

        :param data: DataFrame containing 'open' prices.
        :type data: pd.DataFrame
        :param x: Window size for the exponential moving average.
        :type x: int
        :return: Series containing the exponential moving average 
        :rtype: pd.Series
        """
        ema = EMAIndicator(data['open'], window=x).ema_indicator()
        return ema