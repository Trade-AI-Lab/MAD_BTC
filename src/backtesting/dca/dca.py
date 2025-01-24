import pandas as pd
import numpy as np
import plotly.graph_objects as go
from ta.trend import SMAIndicator, EMAIndicator
from dataclasses import dataclass
from src.utils.utils_df import open_dataframe
import sys, os

absolute_path = os.path.abspath(__file__)[
    : os.path.abspath(__file__).find("MAD_BTC") + len("MAD_BTC")
]


@dataclass
class Dca_Args:
    exchange: str = "binance"
    cryptocurrency: str = "ETHUSDT"
    time: str = "4h"
    investment: float = 125
    investment_interval: str = "W"
    begin_date: str = "2020-01-01 00:00:00"
    end_date: str = None
    fees: float = 0.1 / 100


class Dca:
    def __init__(self, args):
        self.args = args
        self.df = self._open_dataframe()

    def _open_dataframe(self):
        df = open_dataframe(
            rf"data/{self.args.exchange}-{self.args.cryptocurrency}-{self.args.time}.pkl"
        )
        self.args.end_date = (
            df["open_date"].iloc[-1]
            if self.args.end_date is None
            else self.args.end_date
        )
        df = df[
            (df["open_date"] >= self.args.begin_date)
            & (df["open_date"] <= self.args.end_date)
        ]
        df.set_index("open_date", inplace=True)
        print("Dataframe loaded")
        return df

    def _dca_loop(self):
        total_invested = 0
        total_shares = 0
        portfolio_values = []
        resampled_df = (
            self.df["close"].resample(self.args.investment_interval).first()
        )  # Use the first close price of the period

        investment_fees = self.args.investment * (1 - self.args.fees)
        for price in resampled_df.dropna():
            total_shares += investment_fees / price
            total_invested += self.args.investment
            portfolio_values.append(total_shares * price)

        not_invested_values = (
            np.arange(0, len(portfolio_values)) * self.args.investment
        )
        resampled_dates = resampled_df.index

        return (
            portfolio_values,
            total_invested,
            resampled_df,
            not_invested_values,
            resampled_dates,
        )

    @staticmethod
    def _plot(resampled_dates, not_invested_values, portfolio_values):
        # Plot results with Plotly
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=resampled_dates,
                y=not_invested_values,
                mode="lines",
                name="Not invested",
                line=dict(color="blue"),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=resampled_dates,
                y=portfolio_values,
                mode="lines",
                name="Portfolio Value",
                line=dict(color="green"),
            )
        )

        fig.update_layout(
            title="Dollar-Cost Averaging (DCA) Simulation",
            xaxis_title="Time",
            yaxis_title="Value ($)",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            template="plotly_white",
        )

        fig.show()

    def compute_dca_from_dataframe(self, output: float = False, plot: float = False):
        (
            portfolio_values,
            total_invested,
            resampled_df,
            not_invested_values,
            resampled_dates,
        ) = self._dca_loop()

        return_percentage = (
            (portfolio_values[-1] - total_invested) / total_invested
        ) * 100
        print(f"Total Invested: ${total_invested:.2f}")
        print(f"Final Portfolio Value: ${portfolio_values[-1]:.2f}")
        print(f"Return on Investment: {return_percentage:.2f}%")
        if plot:
            self._plot(resampled_dates, not_invested_values, portfolio_values)
        if output:
            return (
                portfolio_values,
                total_invested,
                resampled_df,
                not_invested_values,
                resampled_dates,
            )


@dataclass
class Dca_Ema_Args(Dca_Args):
    increase_factor: float = 5 / 100  # Default value for increase factor
    window: int = 12  # Default window size for EMA

    def __init__(self, **kwargs):
        # Pass parent attributes to Dca_Args
        self.increase_factor = kwargs.pop("increase_factor", self.increase_factor)
        self.window = kwargs.pop("window", self.window)
        super().__init__(**kwargs)


class Dca_Ema(Dca):
    def _dca_loop(self):
        resampled_df = self.df.resample(
            self.args.investment_interval
        ).first()  # Use the first close price of the period

        resampled_df["indicator"] = SMAIndicator(
            resampled_df["close"], window=self.args.window
        ).sma_indicator()
        total_invested = 0
        total_shares = 0
        portfolio_values = []
        not_invested_values = []
        investment = self.args.investment
        fees = self.args.fees
        increase_factor = self.args.increase_factor
        for index, row in resampled_df.iterrows():
            price = row["close"]
            indicator = row["indicator"]
            if price >= indicator:
                current_investment = investment * (1 + increase_factor)
            elif price <= indicator:
                current_investment = investment * (1 - increase_factor)
            else:
                current_investment = investment
            shares_bought = current_investment * (1 - fees) / price
            total_shares += shares_bought
            total_invested += current_investment
            portfolio_values.append(total_shares * price)
            not_invested_values.append(total_invested)
            return_percentage = (
                (portfolio_values[-1] - total_invested) / total_invested
            ) * 100
        return (
            portfolio_values,
            total_invested,
            resampled_df.index,
            not_invested_values,
            return_percentage,
        )


@dataclass
class Dca_PriceVariation_Args(Dca_Args):
    increase_factor: float = 5 / 100  # Default value for increase factor
    decrease_threshold: float = (
        0.01 / 100
    )  # Default value for price variation threshold

    def __init__(self, **kwargs):
        self.increase_factor = kwargs.pop("increase_factor", self.increase_factor)
        self.decrease_threshold = kwargs.pop(
            "decrease_threshold", self.decrease_threshold
        )
        super().__init__(**kwargs)


class Dca_PriceVariation(Dca):
    def _dca_loop(self):
        resampled_df = self.df.resample(self.args.investment_interval).first()

        total_invested = 0
        total_shares = 0
        portfolio_values = []
        not_invested_values = []
        resampled_df["price_change"] = (
            resampled_df["close"] - resampled_df["open"]
        ) / resampled_df["close"]
        investment = self.args.investment
        fees = self.args.fees
        increase_factor = self.args.increase_factor
        decrease_threshold = self.args.decrease_threshold
        for index, row in resampled_df.iterrows():
            price = row["close"]
            price_change = row["price_change"]
            if price_change >= decrease_threshold:
                current_investment = investment * (1 + increase_factor)
            elif price_change < -decrease_threshold:
                current_investment = investment * (1 - increase_factor)
            else:
                current_investment = investment
            shares_bought = current_investment * (1 - fees) / price
            total_shares += shares_bought
            total_invested += current_investment
            portfolio_values.append(total_shares * price)
            not_invested_values.append(total_invested)

        resampled_dates = resampled_df.index
        return (
            portfolio_values,
            total_invested,
            resampled_df,
            not_invested_values,
            resampled_dates,
        )


if __name__ == "__main__":
    # Initialize the Dca_Args instance for DCA
    dca_args = Dca_Args()
    dca_element = Dca(dca_args)
    dca_element.compute_dca_from_dataframe(output=False, plot=False)

    # Initialize the Dca_Ema_Args instance with correct parameters
    dca_ema_args = Dca_Ema_Args()  # Create an instance of Dca_Ema_Args
    dca_ema_element = Dca_Ema(dca_ema_args)  # Pass it to the Dca_Ema class
    dca_ema_element.compute_dca_from_dataframe(output=False, plot=False)

    dca_pricevariation_args = Dca_PriceVariation_Args(
        increase_factor=7 / 100, decrease_threshold=0.01
    )
    dca_pricevariation_element = Dca_PriceVariation(dca_pricevariation_args)
    dca_pricevariation_element.compute_dca_from_dataframe(output=False, plot=False)
