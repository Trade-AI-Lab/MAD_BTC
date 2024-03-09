from ta.trend import SMAIndicator, EMAIndicator
from ta.volatility import DonchianChannel
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
    
    @staticmethod
    def donchian(data:pd.DataFrame,window:int)->tuple[pd.Series, pd.Series, pd.Series]:
        """
        Donchian Channel

        This method calculates the high, middle, and low bands of the Donchian Channel.

        :param data: DataFrame containing 'high', 'low', and 'close' prices.
        :type data: pd.DataFrame
        :param window: Window size for the Donchian Channel.
        :type window: int
        :return: Tuple containing the high, middle, and low bands of the Donchian Channel.
        :rtype: tuple[pd.Series, pd.Series, pd.Series]
        """
        donchian_object = DonchianChannel(data["high"],data["low"],data["close"],window=window)
        return donchian_object.donchian_channel_hband(),donchian_object.donchian_channel_mband(),donchian_object.donchian_channel_lband()

