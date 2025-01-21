import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dataclasses import dataclass
from src.utils.utils_df import open_dataframe
import sys, os

absolute_path = os.path.abspath(__file__)[
    : os.path.abspath(__file__).find("MAD_BTC") + len("MAD_BTC")
]


@dataclass
class Dca_Args:
    exchange: str = "binance"
    cryptocurrency: str = "BTCUSDT"
    time: str = "4h"
    monthly_investment: float = 500
    investment_interval: str = "M"
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

        monthly_investment_fees = self.args.monthly_investment * (1 - self.args.fees)
        for price in resampled_df.dropna():
            total_shares += monthly_investment_fees / price
            total_invested += self.args.monthly_investment
            portfolio_values.append(total_shares * price)

        not_invested_values = (
            np.arange(0, len(portfolio_values)) * self.args.monthly_investment
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
class Dca_Ema_Args:
    exchange: str = "binance"
    cryptocurrency: str = "BTCUSDT"
    time: str = "4h"
    monthly_investment: float = 125
    investment_interval: str = "W"
    begin_date: str = "2020-01-01 00:00:00"
    end_date: str = None
    fees: float = 0.1 / 100
    increase_factor: float = 5 / 100
    decrease_threshold: float = 0.01 / 100


class Dca_Ema(Dca):
    def _dca_loop(self):
        resampled_df = self.df.resample(
            self.args.investment_interval
        ).first()  # Use the first close price of the period

        total_invested = 0
        total_shares = 0
        portfolio_values = []
        not_invested_values = []
        resampled_df["price_change"] = (
            resampled_df["close"] - resampled_df["open"]
        ) / resampled_df["close"]
        monthly_investment = self.args.monthly_investment
        fees = self.args.fees
        increase_factor = self.args.increase_factor
        decrease_threshold = self.args.decrease_threshold
        for index, row in resampled_df.iterrows():
            price = row["close"]
            price_change = row["price_change"]
            if price_change >= decrease_threshold:
                current_investment = monthly_investment * (1 + increase_factor)
            elif price_change <= -decrease_threshold:
                current_investment = monthly_investment * (1 - increase_factor)
            else:
                current_investment = monthly_investment
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
