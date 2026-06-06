import os
import sys
import pandas as pd

absolute_path = os.path.abspath(__file__)[
    : os.path.abspath(__file__).find("MAD_BTC") + len("MAD_BTC")
]
sys.path.append(absolute_path)
from src.utils.utils_df import open_dataframe, filter_dataframe_by_date_range
from src.strategy.momentum import indicator_momentum
from src.strategy.envelop import indicator_envelop
from src.strategy_utils.strategy_comparator import evalutation
from src.strategy_utils.renderer_strategy import renderer, save, monthly_return
from src.strategy_utils.save_experiences_strategy import (
    create_save_one_experience,
    create_save_experiences,
)
from src.utils.utils_signature import create_signature
from ta.momentum import RSIIndicator


class trading_strategy:
    """
    Trading Strategy

    This class represents a generic trading strategy.
    """

    def __init__(self, config: dict) -> None:
        """
        Initialize Trading Strategy

        This method initializes the trading strategy.

        :param config: Configuration parameters for the strategy.
        :type config: dict
        """
        self.config = config
        self._suppress_save = False
        # TODO this should be passed as parameter from somewhere in case the path changes at some point. We should also check somewhere if it exists
        self.PATH = (
            absolute_path
            + rf"/data/{config['exchange']}-{config['crypto_currency']}-{config['time']}.pkl"
        )
        self.parent_df = open_dataframe(self.PATH)
        self.child_df = None
        self.df_filtered = None
        self.strategy_folder_path = None
        self.time_eccefpsn = create_save_experiences(self.config)
        self.evolution_df = None
        self.signature_strategy = create_signature(self.config["strategy"])
        self.begin_date = (
            self.config["begin_date"] if "begin_date" in self.config.keys() else None
        )
        self.end_date = (
            self.config["end_date"] if "end_date" in self.config.keys() else None
        )

    def get_computed_df(self) -> pd.DataFrame:
        return self.child_df

    def loop_simulation(self) -> None:
        """
        loop_simulation loop_simulation represents a trading strategy implementation that simulates buying and selling of a cryptocurrency.

        It initializes a portfolio with a specified initial capital, defines a fee percentage for transactions,
        and sets initial values for cryptocurrency holdings and position.
        The strategy iterates through a filtered DataFrame of market data, executing buy and sell signals based on certain conditions.
        It maintains track of the portfolio, position, and other relevant information over time.
        The resulting evolution of the portfolio and associated metrics are stored in a Pandas DataFrame.
        The code also creates a folder to save the strategy's results and visualizations.
        The final results are saved, and the code appears to be part of a larger trading strategy analysis or backtesting framework.
        """

        portfolio = self.config["initial_capital"]
        # the fee should be specified as config parameter
        fee_percentage = 0.1 / 100  # 0.1%
        crypto, position = 0, 0
        list_portfolio, list_buy_hold, list_time, list_position = [], [], [], []

        self.child_df.set_index("open_date", inplace=True)
        self.df_filtered.set_index("open_date", inplace=True)

        for index, row in self.df_filtered.iterrows():
            row = self.df_filtered.loc[index]
            if row["buy_signal"] and position != 1 and portfolio > 0:
                position = 1
                list_portfolio.append(portfolio)
                list_position.append(position)
                crypto = portfolio / (row["open"] * (1 + fee_percentage))
                portfolio = 0

                list_buy_hold.append(row["buy_and_hold"])
                list_time.append(index)

            elif row["sell_signal"] and position != -1 and crypto > 0:
                position = -1
                portfolio = crypto * row["open"] * (1 - fee_percentage)
                crypto = 0

                list_portfolio.append(portfolio)
                list_position.append(position)
                list_buy_hold.append(row["buy_and_hold"])
                list_time.append(index)

            elif crypto == 0 and portfolio == 0:
                break

        if crypto > 0:
            # Calculate the final portfolio value based on remaining position in crypto
            final_portfolio_value = (
                crypto * self.child_df.iloc[-1]["open"] * (1 - fee_percentage)
            )
            list_portfolio.append(final_portfolio_value)
            list_position.append(position)  # Assuming you maintain the same position
            list_buy_hold.append(self.child_df["buy_and_hold"].iloc[-1])
            list_time.append(self.child_df.index[-1])

        elif crypto == 0:
            list_portfolio.append(
                list_portfolio[-1] if len(list_portfolio) > 0 else portfolio
            )
            list_position.append(
                list_position[-1] if len(list_position) > 0 else 0
            )  # Assuming you maintain the same position
            list_buy_hold.append(self.child_df["buy_and_hold"].iloc[-1])
            list_time.append(self.child_df.index[-1])

        self.evolution_df = pd.DataFrame(
            {
                "Time": list_time,
                "Portfolio": list_portfolio,
                "Buy_and_Hold": list_buy_hold,
                "Position": list_position,
            }
        )
        self.save_data()

    def save_data(self):
        if self.evolution_df is not None:
            relative_portfolio_value = self.evolution_df["Portfolio"].pct_change()
            relative_buy_and_hold_value = self.evolution_df["Buy_and_Hold"].pct_change()

            self.evolution_df["Relative_Portfolio_VS_Relative_Buy&Hold"] = (
                relative_portfolio_value - relative_buy_and_hold_value
            )
            self.evolution_df[f"Portfolio/Buy_and_Hold"] = (
                self.evolution_df["Portfolio"] / self.evolution_df["Buy_and_Hold"]
            )
            self.evolution_df[f"RSI_Portfolio/Buy_and_Hold"] = (
                RSIIndicator(self.evolution_df[f"Portfolio/Buy_and_Hold"]).rsi() / 100
            )
            value = round(
                (self.evolution_df["Relative_Portfolio_VS_Relative_Buy&Hold"]).mean()
                * 100,
                4,
            )

            if value > 0 and not self._suppress_save:
                self.strategy_folder_path = create_save_one_experience(
                    self.config, self.time_eccefpsn
                )
                if self.title_image is None:
                    save("No strategy", self.strategy_folder_path, self.evolution_df)
                else:
                    save(self.title_image, self.strategy_folder_path, self.evolution_df)
                    # can cause error not quite important monthly_return(df_evolution = self.evolution_df,saving=True, path=self.strategy_folder_path,title_image=self.title_image+"monthly_return")

    def renderer_strat(self):
        self.loop_simulation()
        renderer(self.evolution_df, verbose=True)

    def compute_alpha_dates(self, df):
        df["time_diffs"] = df["open_date"].diff(-1)
        time_delta = df["open_date"].diff(1).iloc[-1]
        return df["time_diffs"] == -time_delta

    def renderer(self) -> None:
        """
        Renderer

        This method renders the evolution of the strategy.

        :return: None
        :rtype: None
        """
        renderer(self.evolution_df, verbose=True)

    def compute_simulation(self) -> pd.DataFrame:
        """
        compute_simulation _summary_

        _extended_summary_

        :return: _description_
        :rtype: pd.DataFrame
        """
        self.compute_needed_features()
        self.strategy_rules()
        self.loop_simulation()
        return self.evolution_df

    def compute_needed_features(self) -> None:
        """
        Compute Needed Features for Trading Strategy.

        This method calculates the necessary features for the Trading Strategy...

        """
        self.child_df = filter_dataframe_by_date_range(
            self.parent_df.copy(deep=True), self.begin_date, self.end_date
        )
        self.child_df["buy_and_hold"] = (
            self.child_df["open"]
            / self.child_df["open"].iloc[0]
            * self.config["initial_capital"]
        )

    def strategy_rules(self):
        """
        Define Strategy Rules.

        This method defines the rules for the momentum strategy, including buy and sell signals.

        """
        raise "To be implemented"


class momentum_strategy(trading_strategy):
    """
    Momentum Trading Strategy

    This class represents a momentum trading strategy based on the Moving Average Distance (MAD) concept.

    :param trading_strategy: The base class for trading strategies.
    :type trading_strategy: trading_strategy
    """

    def __init__(self, config) -> None:
        """
        Initialize the Momentum Trading Strategy.

        :param config: Configuration parameters for the strategy.
        :type config: dict
        """
        super().__init__(config)

    def compute_needed_features(self):
        """
        Compute Needed Features for Momentum Strategy.

        This method calculates the necessary features for the momentum strategy, including the MAD ratio.

        """
        super().compute_needed_features()
        # Price column the MAD ratio is computed on. Defaults to "open" so the
        # signal at bar t uses only information available at t's open, matching
        # the trade-on-open execution engine (no lookahead). Set
        # strategy.price_column = "close" to follow the close-based MAD of the
        # original paper.
        price_column = self.config["strategy"].get("price_column", "open")
        self.strategy_name = f"{self.config['strategy']['ratio_method']}_{self.config['strategy']['x']}_{self.config['strategy']['y']}"
        self.title_image = f"{self.config['exchange']}-{self.config['crypto_currency']}-{self.config['time']}_portfolio_vs_buy_hold_{self.strategy_name}_{self.config['initial_capital']}$"
        self.child_df[self.strategy_name] = getattr(
            indicator_momentum, self.config["strategy"]["ratio_method"]
        )(
            self.child_df,
            self.config["strategy"]["x"],
            self.config["strategy"]["y"],
            column_name=price_column,
        )
        self.child_df = self.child_df.dropna().reset_index(drop=True)

    def strategy_rules(self):
        """
        Define Strategy Rules.

        This method defines the rules for the momentum strategy, including buy and sell signals.

        """
        self.compute_needed_features() if self.child_df is None else None
        self.child_df["buy_signal"] = (self.child_df[self.strategy_name] > 1) & (
            self.child_df[self.strategy_name].shift(1) <= 1
        )
        self.child_df["sell_signal"] = (self.child_df[self.strategy_name] < 1) & (
            self.child_df[self.strategy_name].shift(1) >= 1
        )
        self.df_filtered = self.child_df[
            (self.child_df["buy_signal"] > 0) | (self.child_df["sell_signal"] > 0)
        ].reset_index(drop=True)

    def change_config(self, config):
        """
        Change Configuration.

        This method allows the user to change the configuration, but the object needs to be reinitialized.

        :param config: New configuration for the strategy.
        :type config: dict
        """
        if (
            self.PATH
            == absolute_path
            + rf"/data/{config['exchange']}-{config['crypto_currency']}-{config['time']}.pkl"
        ):
            if self.signature_strategy != create_signature(self.config["strategy"]):
                self.child_df = None
        else:
            super(momentum_strategy, self).__init__(config)


class band_momentum_strategy(momentum_strategy):
    """
    Neutral-band (hysteresis) MAD strategy.

    Plain MAD flips on every crossing of ratio == 1, which produces whipsaw
    trades when the ratio wobbles right on the threshold. This subclass adds a
    single dead-band parameter `band` (delta): a long is only opened when the
    ratio rises *above* 1 + delta, and only closed when it falls *below*
    1 - delta. Between those bounds the position is held. This targets the exact
    threshold-whipsaw weakness with one extra parameter (vs the two ADX params
    of regime_momentum_strategy).

    Extra config under config['strategy']:
        band:  half-width of the neutral zone around 1.0 (float, default 0.0,
               which reduces to plain momentum_strategy).
    """

    def strategy_rules(self):
        self.compute_needed_features() if self.child_df is None else None
        band = self.config["strategy"].get("band", 0.0)
        ratio = self.child_df[self.strategy_name]
        self.child_df["buy_signal"] = (ratio > 1 + band) & (
            ratio.shift(1) <= 1 + band
        )
        self.child_df["sell_signal"] = (ratio < 1 - band) & (
            ratio.shift(1) >= 1 - band
        )
        self.df_filtered = self.child_df[
            (self.child_df["buy_signal"] > 0) | (self.child_df["sell_signal"] > 0)
        ].reset_index(drop=True)


class regime_momentum_strategy(momentum_strategy):
    """
    Regime-Gated Momentum (MAD) Strategy.

    Plain MAD is a *trend* signal: it crosses 1 constantly in sideways markets
    and bleeds money to fees and whipsaws. This subclass gates the MAD crossover
    with a trend-strength regime filter (ADX): a buy/sell signal is only acted
    on when ADX exceeds a threshold, i.e. when a trend is actually present.
    Outside a trend the strategy holds its current position (no new entries).

    It reuses every bit of momentum_strategy (MAD computation, simulation, save)
    and only adds one indicator and one parameter (adx_threshold), keeping the
    parameter budget — and therefore the overfitting risk — minimal.

    Extra config under config['strategy']:
        adx_window:     ADX lookback (int, default 14)
        adx_threshold:  Minimum ADX to allow a trade (float, default 25)
    """

    def compute_needed_features(self):
        super().compute_needed_features()
        adx_window = self.config["strategy"].get("adx_window", 14)
        # Recompute on the date-filtered child_df. super() has already dropped
        # NaNs from the MAD warm-up; ADX adds its own warm-up NaNs which we drop.
        self.child_df["adx"] = indicator_momentum.adx(self.child_df, window=adx_window)
        self.child_df = self.child_df.dropna().reset_index(drop=True)

    def strategy_rules(self):
        self.compute_needed_features() if self.child_df is None else None
        adx_threshold = self.config["strategy"].get("adx_threshold", 25)
        trending = self.child_df["adx"] > adx_threshold

        crossover_up = (self.child_df[self.strategy_name] > 1) & (
            self.child_df[self.strategy_name].shift(1) <= 1
        )
        crossover_down = (self.child_df[self.strategy_name] < 1) & (
            self.child_df[self.strategy_name].shift(1) >= 1
        )
        # Only act on a crossover when the market is in a trending regime.
        self.child_df["buy_signal"] = crossover_up & trending
        self.child_df["sell_signal"] = crossover_down & trending
        self.df_filtered = self.child_df[
            (self.child_df["buy_signal"] > 0) | (self.child_df["sell_signal"] > 0)
        ].reset_index(drop=True)


class signal_momentum_strategy(trading_strategy):
    """
    Momentum Trading Strategy

    This class represents a momentum trading strategy based on the Moving Average Distance (MAD) concept.

    :param trading_strategy: The base class for trading strategies.
    :type trading_strategy: trading_strategy
    """

    def __init__(self, config) -> None:
        """
        Initialize the Momentum Trading Strategy.

        :param config: Configuration parameters for the strategy.
        :type config: dict
        """
        super().__init__(config)

    def compute_needed_features(self):
        """
        Compute Needed Features for Momentum Strategy.

        This method calculates the necessary features for the momentum strategy, including the MAD ratio.

        """
        super().compute_needed_features()
        self.strategy_name = f"{self.config['strategy']['ratio_method']}_{self.config['strategy']['x']}_{self.config['strategy']['y']}_{self.config['strategy']['method']}_{self.config['strategy']['z']}"
        self.title_image = f"{self.config['exchange']}-{self.config['crypto_currency']}-{self.config['time']}_portfolio_vs_buy_hold_{self.strategy_name}_{self.config['initial_capital']}$"
        ratio = getattr(indicator_momentum, self.config["strategy"]["ratio_method"])(
            self.child_df, self.config["strategy"]["x"], self.config["strategy"]["y"]
        )
        signal = indicator_momentum.indicator(
            data=ratio,
            x=self.config["strategy"]["z"],
            method=self.config["strategy"]["method"],
        )
        self.child_df[self.strategy_name] = ratio / signal

        self.child_df = self.child_df.dropna().reset_index(drop=True)

    def strategy_rules(self):
        """
        Define Strategy Rules.

        This method defines the rules for the momentum strategy, including buy and sell signals.

        """
        self.compute_needed_features() if self.child_df is None else None
        self.child_df["buy_signal"] = (self.child_df[self.strategy_name] > 1) & (
            self.child_df[self.strategy_name].shift(1) <= 1
        )
        self.child_df["sell_signal"] = (self.child_df[self.strategy_name] < 1) & (
            self.child_df[self.strategy_name].shift(1) >= 1
        )
        self.df_filtered = self.child_df[
            (self.child_df["buy_signal"] > 0) | (self.child_df["sell_signal"] > 0)
        ].reset_index(drop=True)

    def change_config(self, config):
        """
        Change Configuration.

        This method allows the user to change the configuration, but the object needs to be reinitialized.

        :param config: New configuration for the strategy.
        :type config: dict
        """
        if (
            self.PATH
            == absolute_path
            + rf"/data/{config['exchange']}-{config['crypto_currency']}-{config['time']}.pkl"
        ):
            if self.signature_strategy != create_signature(self.config["strategy"]):
                self.child_df = None
        else:
            super(signal_momentum_strategy, self).__init__(config)


class envelop_strategy(trading_strategy):
    """
    Envelop Trading Strategy

    This class represents a envelop trading strategy.

    :param trading_strategy: The base class for trading strategies.
    :type trading_strategy: trading_strategy
    """

    def __init__(self, config) -> None:
        """
        Initialize the Momentum Trading Strategy.

        :param config: Configuration parameters for the strategy.
        :type config: dict
        """
        super().__init__(config)

    def compute_needed_features(self):
        """
        Compute Needed Features for Momentum Strategy.

        This method calculates the necessary features for the momentum strategy, including the MAD ratio.

        """
        super().compute_needed_features()
        envelop = self.config["strategy"]["envelop"]
        self.strategy_name = f"{self.config['strategy']['method']}_{self.envelope_from_list_to_str(envelop)}"
        self.title_image = f"{self.config['exchange']}-{self.config['crypto_currency']}-{self.config['time']}_portfolio_vs_buy_hold_{self.strategy_name}_{self.config['initial_capital']}$"
        self.child_df[self.strategy_name] = getattr(
            indicator_envelop, self.config["strategy"]["method"]
        )(self.child_df, self.config["strategy"]["x"])  # refers to the MA or EMA
        self.child_df = self.child_df.dropna().reset_index(drop=True)
        self.child_df["alpha_date"] = self.compute_alpha_dates(self.child_df)
        self.list_col_buy_signal = []
        self.list_col_sell_signal = []

        for percent_envelop in self.config["strategy"]["envelop"]:
            percent_portfolio = self.config["strategy"]["envelop"][percent_envelop]
            percent_envelop = float(percent_envelop)
            # Buy
            self.child_df[f"envelop_up_{percent_envelop}"] = self.child_df[
                self.strategy_name
            ] * (1 + percent_envelop)  # E and/or MA(1+x%)
            buy_signal = f"buy_signal_{percent_envelop}"
            self.child_df[buy_signal] = (
                self.child_df["high"] > self.child_df[f"envelop_up_{percent_envelop}"]
            ).astype(int) * percent_portfolio
            self.list_col_buy_signal.append(buy_signal)

            # Sell
            self.child_df[f"envelop_down_{percent_envelop}"] = self.child_df[
                self.strategy_name
            ] * (1 - percent_envelop)  # E and/or MA(1-x%)
            sell_signal = f"sell_signal_{percent_envelop}"
            self.child_df[sell_signal] = (
                self.child_df[f"envelop_down_{percent_envelop}"] > self.child_df["low"]
            ).astype(int) * percent_portfolio
            self.list_col_sell_signal.append(sell_signal)

    def strategy_rules(self):
        """
        Define Strategy Rules.

        This method defines the rules for the momentum strategy, including buy and sell signals.

        """
        self.compute_needed_features() if self.child_df is None else None
        self.child_df["buy_signal"] = self.child_df[self.list_col_buy_signal].sum(
            axis=1
        )
        self.child_df["sell_signal"] = self.child_df[self.list_col_sell_signal].sum(
            axis=1
        )
        self.df_filtered = self.child_df[
            (self.child_df["buy_signal"] > 0) | (self.child_df["sell_signal"] > 0)
        ].reset_index(drop=True)

    def change_config(self, config):
        """
        Change Configuration.

        This method allows the user to change the configuration, but the object needs to be reinitialized.

        :param config: New configuration for the strategy.
        :type config: dict
        """
        if (
            self.PATH
            == absolute_path
            + rf"/data/{config['exchange']}-{config['crypto_currency']}-{config['time']}.pkl"
        ):
            if self.signature_strategy != create_signature(self.config["strategy"]):
                self.child_df = None
        else:
            super(envelop_strategy, self).__init__(config)

    @staticmethod
    def envelope_from_list_to_str(dictionnary: dict):
        liste = [key for key in dictionnary]
        return "_layer_".join(map(str, liste)) + "_layer"

    def check_validate_date(self, time):
        if time == self.last_time:
            self.end_simulation = True
        elif self.child_df["alpha_date"][time] != True:
            self.jumping_simulation = True

    def loop_simulation(self) -> None:
        """
        loop_simulation loop_simulation represents a trading strategy implementation that simulates buying and selling of a cryptocurrency.

        It initializes a portfolio with a specified initial capital, defines a fee percentage for transactions,
        and sets initial values for cryptocurrency holdings and position.
        The strategy iterates through a filtered DataFrame of market data, executing buy and sell signals based on certain conditions.
        It maintains track of the portfolio, position, and other relevant information over time.
        The resulting evolution of the portfolio and associated metrics are stored in a Pandas DataFrame.
        The code also creates a folder to save the strategy's results and visualizations.
        The final results are saved, and the code appears to be part of a larger trading strategy analysis or backtesting framework.
        """

        portfolio = self.config["initial_capital"]
        fee_percentage = 0.06 / 100  # 0.1%
        fee_maker = 0.06 / 100  # 0.1%
        crypto = 0
        position = 0
        crypto_amount = 0
        percent_portfolio = 0
        list_portfolio, list_buy_hold, list_time, list_position = [], [], [], []
        self.last_time = self.child_df["open_date"].iloc[-1]
        self.delta_t = self.child_df["open_date"].diff(1).iloc[-1]
        self.jumping_time = (
            self.df_filtered["open_date"].iloc[0] if len(self.df_filtered) > 0 else None
        )
        self.df_filtered.set_index("open_date", inplace=True)
        self.child_df.set_index("open_date", inplace=True)
        self.end_simulation = False
        self.jumping_simulation = False

        if self.jumping_time is None:
            list_portfolio, list_buy_hold, list_time, list_position = (
                [0.1],
                [0.1],
                [0.1],
                [0.1],
            )
        else:
            for index, row in self.df_filtered.iterrows():
                row = self.df_filtered.loc[index]  # t = time current time
                self.check_validate_date(index)
                if self.jumping_simulation == True:
                    self.jumping_time = time - self.child_df["time_diffs"].loc[index]
                    self.jumping_simulation = False
                elif (
                    index > self.jumping_time
                ):  # if the jumping trade is higher than index you can trade
                    k = 1
                    time = self.delta_t * k + index  # t+delta_t
                    self.check_validate_date(time)
                    if self.end_simulation or self.jumping_simulation:
                        if self.end_simulation:
                            break
                    elif (
                        len(list_portfolio) > 0
                        and list_buy_hold[-1] / list_portfolio[-1] > 5
                    ):
                        break
                    elif row["buy_signal"] or row["sell_signal"]:  # t is index
                        # Begin the trade
                        percent_portfolio = 0
                        position = 1 if row["buy_signal"] else -1
                        for percent_envelop in self.config["strategy"]["envelop"]:
                            percent_portfolio = self.config["strategy"]["envelop"][
                                percent_envelop
                            ]

                            envelop_price = (
                                row[f"envelop_down_{percent_envelop}"]
                                if row["buy_signal"]
                                else row[f"envelop_up_{percent_envelop}"]
                            )
                            crypto_amount = (
                                portfolio
                                * percent_portfolio
                                / (envelop_price * (1 + fee_percentage))
                            )
                            crypto += (
                                crypto_amount if row["buy_signal"] else -crypto_amount
                            )  # crypto is postivie for the buy but negative for the sell
                            if crypto == float("-inf"):
                                raise ValueError("Float value is -inf")
                            portfolio -= (
                                portfolio * percent_portfolio
                                if (envelop_price and row["buy_signal"])
                                else 0
                            )
                            percent_portfolio += (
                                percent_portfolio if envelop_price else 0
                            )  # only needed for the sells

                        col_condition = "low" if row["buy_signal"] else "high"

                        while (
                            self.child_df[self.strategy_name].loc[time]
                            < self.child_df[col_condition].loc[time]
                            and self.end_simulation != True
                            and self.jumping_simulation != True
                        ):
                            k += 1
                            time = self.delta_t * k + index  # time= t +\delta_{t}
                            self.check_validate_date(time)
                        current_price = self.child_df[self.strategy_name].loc[time]
                        current_crypto_amount = portfolio / (
                            current_price * (1 + fee_maker)
                        )  # only for the sell
                        portfolio += (
                            crypto * current_price * (1 - fee_maker)
                            if row["buy_signal"]
                            else (current_crypto_amount - crypto) * current_price
                        )

                        # Trade ended
                        if self.end_simulation:
                            self.jumping_time = time

                        elif self.jumping_simulation:
                            self.jumping_time = (
                                time - self.child_df["time_diffs"].loc[time]
                            )
                            self.jumping_simulation = False

                        else:
                            self.jumping_time = time

                    list_portfolio.append(portfolio)
                    list_position.append(position)
                    list_buy_hold.append(self.child_df["buy_and_hold"].loc[time])
                    list_time.append(time)

                    crypto = 0
                    position = 0
                    crypto_amount = 0
                    percent_portfolio = 0
                    current_crypto_amount = 0

        self.evolution_df = pd.DataFrame(
            {
                "Time": list_time,
                "Portfolio": list_portfolio,
                "Buy_and_Hold": list_buy_hold,
                "Position": list_position,
            }
        )
        # Delegate to the base save_data(): it builds the
        # RSI_Portfolio/Buy_and_Hold column that save()/core_img() require,
        # honours _suppress_save, and only writes profitable runs.
        self.save_data()


if __name__ == "__main__":
    config = {
        "crypto_currency": "BNBUSDT",
        "time": "4h",
        "exchange": "binance",
        "initial_capital": 1000,
        "broker": "binance",
        # "begin_date": "2020:"
        "strategy": {
            "name": "momentum_strategy",
            "x": 10,
            "y": 103,
            "ratio_method": "tema_ratio",
        },
    }
    obj = momentum_strategy(config)
    obj.strategy_rules()
    obj.loop_simulation()
