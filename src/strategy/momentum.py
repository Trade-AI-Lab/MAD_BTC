from ta.trend import SMAIndicator, EMAIndicator
import pandas as pd

class indicator_momentum:
    @staticmethod
    def ma_ratio(data:pd.DataFrame, x:int, y:int)-> pd.Series:
        """
        Moving Average Ratio

        This method calculates the ratio of two simple moving averages.

        :param data: DataFrame containing 'open' prices.
        :type data: pd.DataFrame
        :param x: Window size for the first moving average.
        :type x: int
        :param y: Window size for the second moving average.
        :type y: int
        :return: Series containing the moving average ratio.
        :rtype: pd.Series
        """
        ma_x = SMAIndicator(data['open'], window=x).sma_indicator()
        ma_y = SMAIndicator(data['open'], window=y).sma_indicator()
        return (ma_x / ma_y)

    @staticmethod
    def ema_ratio(data:pd.DataFrame, x:int, y:int)-> pd.Series:
        """
        Exponential Moving Average Ratio

        This method calculates the ratio of two exponential moving averages.

        :param data: DataFrame containing 'open' prices.
        :type data: pd.DataFrame
        :param x: Window size for the first exponential moving average.
        :type x: int
        :param y: Window size for the second exponential moving average.
        :type y: int
        :return: Series containing the exponential moving average ratio.
        :rtype: pd.Series
        """
        ema_x = EMAIndicator(data['open'], window=x).ema_indicator()
        ema_y = EMAIndicator(data['open'], window=y).ema_indicator()
        return (ema_x / ema_y)
    

