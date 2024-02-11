from ta.trend import SMAIndicator, EMAIndicator
class indicator_momentum:
    @staticmethod
    def ma_ratio(data, x, y):
        ma_x = SMAIndicator(data['open'], window=x).sma_indicator()
        ma_y = SMAIndicator(data['open'], window=y).sma_indicator()
        return (ma_x / ma_y)

    @staticmethod
    def ema_ratio(data, x, y):
        ema_x = EMAIndicator(data['open'], window=x).ema_indicator()
        ema_y = EMAIndicator(data['open'], window=y).ema_indicator()
        return (ema_x / ema_y)