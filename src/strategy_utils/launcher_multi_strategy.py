import os
import sys
import pandas as pd
from functools import reduce

absolute_path = os.path.abspath(__file__)[
    : os.path.abspath(__file__).find("MAD_BTC") + len("MAD_BTC")
]
sys.path.append(absolute_path)
from src.utils.utils_df import open_dataframe, filter_dataframe_by_date_range
from src.strategy.momentum import indicator_momentum
from src.strategy_utils.renderer_strategy import renderer, save
from src.strategy_utils.save_experiences_strategy import (
    create_save_one_experience,
    create_save_experiences,
)
from src.utils.utils_signature import create_signature
from ta.momentum import RSIIndicator

def from_list_to_str(list_symbols:str)->str:
    symbols_sorted = sorted(list_symbols)
    str_symbols = "_".join(symbols_sorted)
    # Join the sorted elements with underscores
    return str_symbols

class multi_trading_strategy:
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
        list_df = []
        self.dict_best_crypto = {"Name":"BTC",
                            "Value":0}
        for crypto_currency in config["list_crypto_currencies"]:
        # TODO this should be passed as parameter from somewhere in case the path changes at some point. We should also check somewhere if it exists
            PATH = (
                absolute_path
                + rf"/data/{config['exchange']}-{crypto_currency}-{config['time']}.pkl"
            )
            df = open_dataframe(PATH)
            Value = (df["close"].iloc[-1] - df["close"].iloc[0])/df["close"].iloc[0]
            
            if Value>self.dict_best_crypto["Value"]:
                self.dict_best_crypto["Value"] = Value
                self.dict_best_crypto["Name"] = crypto_currency
            df.drop(["volume", "sin_weekly", "sin_daily"],axis=1)
            df.columns = [col + f'_{crypto_currency}' if col != 'open_date' else col 
                                  for col in df.columns]
            list_df.append(df)
            
        self.parent_df  = reduce(lambda left, right: pd.merge(left, right, on='open_date'), list_df)
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

        # the fee should be specified as config parameter
        fee_percentage = 0.1 / 100  # 0.1%
        self.child_df.set_index("open_date", inplace=True)
        self.df_filtered.set_index("open_date", inplace=True)
        list_df = []
        for crypto_currency in self.config["list_crypto_currencies"]:
            portfolio = self.config["initial_capital"]/len(self.config["list_crypto_currencies"])
            crypto, position = 0, 0
            list_portfolio, list_buy_hold, list_time, list_position = [], [], [], []
            for index, row in self.df_filtered.iterrows():
                row = self.df_filtered.loc[index]
                if row["buy_signal"] and position != 1 and portfolio > 0:
                    position = 1
                    list_portfolio.append(portfolio)
                    list_position.append(position)
                    crypto = portfolio / (row[f"open_{crypto_currency}"] * (1 + fee_percentage))
                    portfolio = 0

                    list_buy_hold.append(row[f"buy_and_hold_{crypto_currency}"])
                    list_time.append(index)

                elif row["sell_signal"] and position != -1 and crypto > 0:
                    position = -1
                    portfolio = crypto * row[f"open_{crypto_currency}"] * (1 - fee_percentage)
                    crypto = 0

                    list_portfolio.append(portfolio)
                    list_position.append(position)
                    list_buy_hold.append(row[f"buy_and_hold_{crypto_currency}"])
                    list_time.append(index)

                elif crypto == 0 and portfolio == 0:
                    break

            if crypto > 0:
                # Calculate the final portfolio value based on remaining position in crypto
                final_portfolio_value = (
                    crypto * self.child_df.iloc[-1][f"open_{crypto_currency}"] * (1 - fee_percentage)
                )
                list_portfolio.append(final_portfolio_value)
                list_position.append(position)  # Assuming you maintain the same position
                list_buy_hold.append(self.child_df[f"buy_and_hold_{crypto_currency}"].iloc[-1])
                list_time.append(self.child_df.index[-1])

            elif crypto == 0:
                list_portfolio.append(list_portfolio[-1])
                list_position.append(
                    list_position[-1]
                )  # Assuming you maintain the same position
                list_buy_hold.append(self.child_df[f"buy_and_hold_{crypto_currency}"].iloc[-1])
                list_time.append(self.child_df.index[-1])
            list_df.append(pd.DataFrame(
            {
                "Time": list_time,
                f"Portfolio_{crypto_currency}": list_portfolio,
                f"Buy_and_Hold_{crypto_currency}": list_buy_hold,
                f"Position_{crypto_currency}": list_position,
            }
            )
            )
        
        self.evolution_df = reduce(lambda left, right: pd.merge(left, right, on='Time'), list_df)
        self.evolution_df["Portfolio"] = 0
        self.evolution_df["Buy_and_Hold"] = 0
        for crypto_currency in self.config["list_crypto_currencies"]:
            self.evolution_df["Portfolio"] += self.evolution_df[ f"Portfolio_{crypto_currency}"]
            self.evolution_df["Buy_and_Hold"] += self.evolution_df[ f"Buy_and_Hold_{crypto_currency}"]
            
        self.evolution_df["Portfolio"] /= len(self.config["list_crypto_currencies"])
        self.evolution_df["Buy_and_Hold"] /= len(self.config["list_crypto_currencies"])
        self.save_data()

    def save_data(self):
        if self.evolution_df is not None:
            self.evolution_df["Relative_Portfolio_VS_Relative_Buy&Hold"] = (
                self.evolution_df["Portfolio"].pct_change()
                - self.evolution_df["Buy_and_Hold"].pct_change()
            )
            self.evolution_df[f"Portfolio/Buy_and_Hold"] = (
                self.evolution_df["Portfolio"] / self.evolution_df["Buy_and_Hold"]
            )
            self.evolution_df[f"RSI_Portfolio/Buy_and_Hold"] = (
                RSIIndicator(self.evolution_df[f"Portfolio/Buy_and_Hold"]).rsi() / 100
            )

            if (
                self.evolution_df["Portfolio"].iloc[-1]
                > self.evolution_df["Buy_and_Hold"].iloc[-1]
            ):
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
        name_best_crypto = self.dict_best_crypto["Name"]
        for crypto_currency in self.config["list_crypto_currencies"]:
            self.child_df[f"buy_and_hold_{crypto_currency}"] = (
                self.child_df[f"open_{crypto_currency}"]
                / self.child_df[f"open_{crypto_currency}"].iloc[0]
                * self.config["initial_capital"]
            )
        self.child_df["buy_and_hold"] = (
                self.child_df[f"open_{name_best_crypto}"]
                / self.child_df[f"open_{name_best_crypto}"].iloc[0]
                * self.config["initial_capital"]
            )
        
        self.strategy_name = f'{self.config["strategy"]["ratio_method"]}_{self.config["strategy"]["x"]}_{self.config["strategy"]["y"]}'
        self.title_image = f"{self.config['exchange']}-{from_list_to_str(self.config['list_crypto_currencies'])}-{self.config['time']}_portfolio_vs_buy_hold_{self.strategy_name}_{self.config['initial_capital']}$"
        self.child_df[self.strategy_name] = 0
        for crypto_currency in self.config["list_crypto_currencies"]:
            column_name = f"open_{crypto_currency}"
            serie = getattr(
                    indicator_momentum, self.config["strategy"]["ratio_method"]
                )(self.child_df, self.config["strategy"]["x"], self.config["strategy"]["y"],column_name)
            self.child_df[self.strategy_name] += serie
        self.child_df[self.strategy_name] /= len(self.config["list_crypto_currencies"])
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
        self.__init__(config)
