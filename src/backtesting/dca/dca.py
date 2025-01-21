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
class Args:
    exchange: str = "binance"
    cryptocurrency: str = "BTCUSDT"
    time: str = "4h"
    monthly_investment: float = 500
    investment_interval: str = "M"
    begin_date: str = "2020-01-01 00:00:00"
    end_date: str = None
    fees: float = 0.1 / 100


class dca:
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
        return portfolio_values, total_invested, resampled_df

    def compute_dca_from_dataframe(self, output: float = False, plot: float = False):
        portfolio_values, total_invested, resampled_df = self._dca_loop()
        # Final portfolio value and return
        not_invested = np.arange(0, len(portfolio_values)) * args.monthly_investment
        return_percentage = (
            (portfolio_values[-1] - total_invested) / total_invested
        ) * 100

        final_portfolio_value = portfolio_values[-1]
        resampled_dates = resampled_df.index
        # Print results
        print(f"Total Invested: ${total_invested:.2f}")
        print(f"Final Portfolio Value: ${final_portfolio_value:.2f}")
        print(f"Return on Investment: {return_percentage:.2f}%")
        if plot:
            # Plot results with Plotly
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=resampled_dates,
                    y=not_invested,
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
        if output:
            return (
                portfolio_values,
                total_invested,
                resampled_df.index,
                not_invested,
                return_percentage,
            )


if __name__ == "__main__":
    args = Args
    dca_element = dca(args)
    dca_element.compute_dca_from_dataframe(output=False, plot=True)
