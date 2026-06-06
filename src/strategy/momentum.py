from ta.trend import SMAIndicator, EMAIndicator, ADXIndicator
import numpy as np
import pandas as pd


class indicator_momentum:
    @staticmethod
    def vol_normalized_mad(
        data: pd.DataFrame,
        lookback: int,
        k: int = 4,
        vol_window: int = None,
        column_name: str = "open",
    ) -> pd.Series:
        """
        Volatility-normalised Moving Average Distance — a scale-invariant MAD.

        Classic MAD (MA_x / MA_y) lives on a per-asset scale: the same (x, y)
        means different things on BTC vs a high-vol altcoin, which is exactly
        why the optimiser had to refit (x, y) per market (overfitting). Here the
        MA distance is expressed in *units of realised volatility*:

            z = (EMA_x / EMA_y - 1) / sigma

        where sigma is the rolling std of log-returns. z is dimensionless and
        comparable across assets, so a single threshold (e.g. z > 1 = "one vol
        unit of trend") has the same meaning everywhere.

        This collapses three hyperparameters (x, y, ratio_method) to ONE
        (lookback), with the long window fixed at y = k * lookback and the vol
        window tied to it. Fewer free parameters -> lower probability of
        backtest overfitting (Bailey & Lopez de Prado).

        :param data: DataFrame with the price column and (implicitly) returns.
        :param lookback: Short EMA window x; long window is k * lookback.
        :param k: Fixed short/long ratio (default 4, the classic family).
        :param vol_window: Realised-vol window; defaults to k * lookback.
        :param column_name: Price column to use.
        :return: Dimensionless vol-normalised MAD series z.
        """
        x = lookback
        y = k * lookback
        vw = vol_window if vol_window is not None else y

        price = data[column_name]
        ema_x = EMAIndicator(price, window=x).ema_indicator()
        ema_y = EMAIndicator(price, window=y).ema_indicator()
        mad = ema_x / ema_y - 1.0

        log_ret = np.log(price / price.shift(1))
        sigma = log_ret.rolling(vw).std()
        # Guard the division: never divide by ~0 vol.
        return mad / sigma.clip(lower=1e-8)

    @staticmethod
    def adx(data: pd.DataFrame, window: int = 14) -> pd.Series:
        """
        Average Directional Index — trend-strength regime filter.

        ADX measures how strongly a market is trending, independent of
        direction. Conventionally ADX > 25 ≈ trending, < 20 ≈ ranging.
        Used to gate the (directional) MAD crossover so it only trades when a
        trend is actually present.

        :param data: DataFrame containing 'high', 'low', 'close' columns.
        :param window: Lookback window for the ADX computation.
        :return: Series of ADX values (0–100).
        """
        return ADXIndicator(
            high=data["high"],
            low=data["low"],
            close=data["close"],
            window=window,
        ).adx()

    @staticmethod
    def indicator(
        data: pd.Series,
        x: int,
        method: str = "ma",
    ) -> pd.Series:
        if method == "ma":
            return SMAIndicator(data, window=x).sma_indicator()
        elif method == "ema":
            return EMAIndicator(data, window=x).ema_indicator()
        elif method == "zlema":
            ema_data = data + (data - data.shift((x - 1) // 2))
            return EMAIndicator(ema_data, window=x).ema_indicator()
        elif method == "dema":
            ema_data = EMAIndicator(data, window=x).ema_indicator()
            return 2 * ema_data - EMAIndicator(ema_data, window=x).ema_indicator()
        elif method == "tema":
            ema_data = EMAIndicator(data, window=x).ema_indicator()
            ema_ema_data = EMAIndicator(ema_data, window=x).ema_indicator()
            ema_ema_ema_data = EMAIndicator(ema_ema_data, window=x).ema_indicator()
            return 3 * ema_ema_ema_data - 3 * ema_ema_data + ema_data
        else:
            raise "Not known method"

    @staticmethod
    def ma_ratio(
        data: pd.DataFrame, x: int, y: int, column_name: str = "open"
    ) -> pd.Series:
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
        ma_x = SMAIndicator(data[column_name], window=x).sma_indicator()
        ma_y = SMAIndicator(data[column_name], window=y).sma_indicator()
        return ma_x / ma_y

    @staticmethod
    def ema_ratio(
        data: pd.DataFrame, x: int, y: int, column_name: str = "open"
    ) -> pd.Series:
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
        ema_x = EMAIndicator(data[column_name], window=x).ema_indicator()
        ema_y = EMAIndicator(data[column_name], window=y).ema_indicator()
        return ema_x / ema_y

    @staticmethod
    def zlema_ratio(
        data: pd.DataFrame, x: int, y: int, column_name: str = "open"
    ) -> pd.Series:
        """
        Zl Exponential Moving Average Ratio

        This method calculates the ratio of two ZL exponential moving averages.

        :param data: DataFrame containing 'open' prices.
        :type data: pd.DataFrame
        :param x: Window size for the first exponential moving average.
        :type x: int
        :param y: Window size for the second exponential moving average.
        :type y: int
        :return: Series containing the exponential moving average ratio.
        :rtype: pd.Series
        """

        def zlema(data, period):
            ema_data = data + (data - data.shift((period - 1) // 2))
            return EMAIndicator(ema_data, window=period).ema_indicator()

        zlema_x = zlema(data[column_name], period=x)
        zlema_y = zlema(data[column_name], period=y)
        return zlema_x / zlema_y

    @staticmethod
    def dema_ratio(
        data: pd.DataFrame, x: int, y: int, column_name: str = "open"
    ) -> pd.Series:
        """
        Double Exponential Moving Average Ratio

        This method calculates the ratio of two Double exponential moving averages.

        :param data: DataFrame containing 'open' prices.
        :type data: pd.DataFrame
        :param x: Window size for the first exponential moving average.
        :type x: int
        :param y: Window size for the second exponential moving average.
        :type y: int
        :return: Series containing the exponential moving average ratio.
        :rtype: pd.Series
        """

        def dema(data, period):
            ema_data = EMAIndicator(data, window=period).ema_indicator()
            return 2 * ema_data - EMAIndicator(ema_data, window=period).ema_indicator()

        dema_x = dema(data[column_name], period=x)
        dema_y = dema(data[column_name], period=y)
        return dema_x / dema_y

    @staticmethod
    def tema_ratio(
        data: pd.DataFrame, x: int, y: int, column_name: str = "open"
    ) -> pd.Series:
        """
        Triple Exponential Moving Average Ratio

        This method calculates the ratio of two Triple exponential moving averages.

        :param data: DataFrame containing 'open' prices.
        :type data: pd.DataFrame
        :param x: Window size for the first exponential moving average.
        :type x: int
        :param y: Window size for the second exponential moving average.
        :type y: int
        :return: Series containing the exponential moving average ratio.
        :rtype: pd.Series
        """

        def tema(data, period):
            ema_data = EMAIndicator(data, window=period).ema_indicator()
            ema_ema_data = EMAIndicator(ema_data, window=period).ema_indicator()
            ema_ema_ema_data = EMAIndicator(ema_ema_data, window=period).ema_indicator()
            return 3 * ema_ema_ema_data - 3 * ema_ema_data + ema_data

        dema_x = tema(data[column_name], period=x)
        dema_y = tema(data[column_name], period=y)
        return dema_x / dema_y

    @staticmethod
    def general_ema_ratio(
        data: pd.DataFrame,
        x: int,
        y: int,
        order_x=3,
        order_y=3,
        column_name: str = "open",
    ) -> pd.Series:
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
                    coeff = (
                        (-2 * (order - i) - 1) if i % 2 == 0 else (2 * (order - i) + 1)
                    )

                # Calculate EMA and append to the list
                ema_value = EMAIndicator(list_ema[-1], window=period).ema_indicator()
                list_ema.append(ema_value)

                # Update final EMA
                final_ema += coeff * ema_value

            return final_ema

        general_ema_x = general_ema(data[column_name], order=order_x, period=x)
        general_ema_y = general_ema(data[column_name], order=order_y, period=y)
        return general_ema_x / general_ema_y


"""  
if __name__=="__main__":
    import pandas as pd
    df = pd.read_csv(r"data\harmonia-20.1-82-2022-market-23months.csv.zip")
    import plotly.graph_objs as go
    period = 10
    order = 6
    list_trace = []
    for i in range(1,order):
        list_trace.append(go.Scatter(x=df['open_date'], y=indicator_momentum.general_ema(data=df["open_price"],order=i,period=period), mode='lines', name=f'Open Price EMA_{i}_{period}'))
    list_trace.append(go.Scatter(x=df['open_date'], y=df["open_price"], mode='lines', name='Open Price'))
    list_trace.append(go.Scatter(x=df['open_date'], y=df["high"], mode='lines', name='High'))
    list_trace.append(go.Scatter(x=df['open_date'], y=df["low"], mode='lines', name='Low'))
    fig = go.Figure(data=list_trace)
    fig.update_layout(title='Open Price and TEMA over Time', xaxis_title='Time', yaxis_title='Price')
    fig.show()
"""
