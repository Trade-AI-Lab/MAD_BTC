import ta
import pandas as pd
from ta.trend import SMAIndicator, EMAIndicator

def tema(data, period):
    ema_data = EMAIndicator(data, window=period).ema_indicator()
    ema_ema_data = EMAIndicator(ema_data, window=period).ema_indicator()
    ema_ema_ema_data = EMAIndicator(ema_ema_data, window=period).ema_indicator()
    return 3 * ema_ema_ema_data - 3 * ema_ema_data + ema_data

def dema(data, period):
    ema_data = EMAIndicator(data, window=period).ema_indicator()
    return 2 * ema_data - EMAIndicator(ema_data, window=period).ema_indicator()

def zlema(data, period):
    ema_data = data + (data - data.shift((period - 1) // 2))
    return EMAIndicator(ema_data, window=period).ema_indicator()


def general_ema(data, order, period):
    list_ema = [data]
    final_ema = 0

    for i in range(1, order + 1):
        # Determine the coefficient based on the order and iteration
        if i == 1:
            coeff = order
        elif i == order:
            coeff = 1 if order % 2 == 1 else -1
        else:
            coeff = (-2 * (order - i) - 1) if i % 2 == 0 else (2 * (order - i) + 1)
        
        # Calculate EMA and append to the list
        ema_value = EMAIndicator(list_ema[-1], window=period).ema_indicator()
        list_ema.append(ema_value)
        
        # Update final EMA
        final_ema += coeff * ema_value

    return final_ema

class Trix():
    """ Trix indicator

        Args:
            close(pd.Series): dataframe 'close' columns,
            trix_length(int): the window length for each mooving average of the trix,
            trix_signal_length(int): the window length for the signal line
    """

    def __init__(
        self,
        close: pd.Series,
        trix_length: int = 9,
        trix_signal_length: int = 21,
        trix_signal_type: str = "sma" # or ema
    ):
        self.close = close
        self.trix_length = trix_length
        self.trix_signal_length = trix_signal_length
        self.trix_signal_type = trix_signal_type
        self._run()

    def _run(self):
        ema_data = EMAIndicator(self.close, window=self.trix_length).ema_indicator()
        ema_ema_data = EMAIndicator(ema_data, window=self.trix_length).ema_indicator()
        self.trix_line = EMAIndicator(ema_ema_data, window=self.trix_length).ema_indicator()     
        self.trix_pct_line = self.trix_line.pct_change()*100

        if self.trix_signal_type == "sma":
            self.trix_signal_line = ta.trend.sma_indicator(
                close=self.trix_pct_line, window=self.trix_signal_length)
        elif self.trix_signal_type == "ema":
            self.trix_signal_line = ta.trend.ema_indicator(
                close=self.trix_pct_line, window=self.trix_signal_length)
            
        self.trix_histo = self.trix_pct_line - self.trix_signal_line

    def get_trix_line(self) -> pd.Series:
        return pd.Series(self.trix_line, name="trix_line")

    def get_trix_pct_line(self) -> pd.Series:
        return pd.Series(self.trix_pct_line, name="trix_pct_line")

    def get_trix_signal_line(self) -> pd.Series:
        return pd.Series(self.trix_signal_line, name="trix_signal_line")

    def get_trix_histo(self) -> pd.Series:
        return pd.Series(self.trix_histo, name="trix_histo")

if __name__=="__main__":
    """
    df['trix'] = trix_obj.get_trix_pct_line()
    df['trix_signal'] = trix_obj.get_trix_signal_line()
    df['trix_hist'] = df['trix'] - df['trix_signal']
    if params['long_ma_type'] == "ema":
        df["long_ma"] = ta.trend.ema_indicator(df["close"], window=params['long_ma_length'])
    else:
        df["long_ma"] = ta.trend.sma_indicator(df["close"], window=params['long_ma_length'])
    if self.use_long:
            df.loc[
                (df["trix_hist"] > 0) &
                (df['close'] > df["long_ma"])
                , "open_long_market"
            ] = True

            df.loc[
                (df["trix_hist"] < 0)
                , "close_long_market",
            ] = True

        if self.use_short:
            df.loc[
                (df["trix_hist"] < 0) &
                (df['close'] < df["long_ma"])
                , "open_short_market"
            ] = True

            df.loc[
                (df["trix_hist"] > 0)
                , "close_short_market",
            ] = True

        self.df = df 
        """ 