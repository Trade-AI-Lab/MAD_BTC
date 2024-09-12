import optuna
import os, sys
import plotly.graph_objects as go
import random

absolute_path = os.path.abspath(__file__)[
    : os.path.abspath(__file__).find("MAD_BTC") + len("MAD_BTC")
]
sys.path.append(absolute_path)
from src.utils.utils_df import open_dataframe, filter_dataframe_by_date_range
from src.strategy.momentum import indicator_momentum
from src.utils.utils_json import config_data_available

config = config_data_available("data")

class AnalysisVisualizer:
    def __init__(self, PATH) -> None:
        study_name = PATH.split("/")[-1].split(".")[0]
        storage_name = f"sqlite:///{PATH}"
        study = optuna.study.load_study(study_name=study_name, storage=storage_name)
        list_study_name_split = study_name.split("_")
        for element in list_study_name_split:
            if element in  config["crypto_currency"]:
                crypto_currency = element
            elif element in  config["time"]:
                time = element
            elif element in  config["exchange"]:
                exchange = element
        self.data_df = open_dataframe(
            r"data/" + f"{exchange}-{crypto_currency}-{time}.pkl"
        )

        self.study_df = study.trials_dataframe()
        study_df_sorted = self.study_df.sort_values(by="value", ascending=False)
        self.top_study_df_sorted = study_df_sorted.head(25)
        self.child_data_df = self.data_df.copy()
        self.delta_dt = self.child_data_df["open_date"].diff().iloc[1]
        self.child_data_df.set_index("open_date")

    def compute_needed_features(self, x: int, y: int, ratio_method: str):
        self.strategy_name = f"{ratio_method}_{x}_{y}"
        self.child_data_df[self.strategy_name] = getattr(
            indicator_momentum, ratio_method
        )(self.child_data_df, x, y)

    def strategy_rules(self, x: int, y: int, ratio_method: str):
        """
        Define Strategy Rules.

        This method defines the rules for the momentum strategy, including buy and sell signals.

        """
        self.compute_needed_features(x, y, ratio_method)
        self.child_data_df["buy_signal"] = (
            self.child_data_df[self.strategy_name] > 1
        ) & (self.child_data_df[self.strategy_name].shift(1) <= 1)
        self.child_data_df["sell_signal"] = (
            self.child_data_df[self.strategy_name] < 1
        ) & (self.child_data_df[self.strategy_name].shift(1) >= 1)

    def plot(self, x: int, y: int, ratio_method: str):
        self.strategy_rules(x, y, ratio_method)
        list_index = list(self.child_data_df["buy_signal"].index()) + list(
            self.child_data_df["sell_signal"].index()
        )
        random_integer = random.randint(0, len(list_index))
        random_date = list_index[random_integer]
        self.df = self.child_data_df.iloc[
            -self.delta_dt * 30 + random_date : +self.delta_dt * 30 + random_date
        ]

        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=self.df.index,
                    open=self.df["open"],
                    high=self.df["high"],
                    low=self.df["low"],
                    close=self.df["close"],
                )
            ]
        )
        fig.show()


if __name__ == "__main__":
    PATH = "study/study_momentum_strategy_BTCUSDT_binance_1h.db"
    analysisvisualizer = AnalysisVisualizer(PATH)
