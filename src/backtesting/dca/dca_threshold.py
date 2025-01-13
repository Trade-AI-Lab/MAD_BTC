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
    increase_factor: float = 5 / 100
    decrease_threshold: float = 0.01 / 100


def calculate_dca_from_dataframe(
    df, monthly_investment, interval, fees, decrease_threshold, increase_factor
):
    """
    Calculate DCA portfolio value given a DataFrame with 'Close' prices.

    Args:
        df (pd.DataFrame): DataFrame containing at least a 'Close' column and a datetime index.
        monthly_investment (float): Fixed amount invested per interval.
        interval (str): Resampling frequency (e.g., 'M' for monthly, 'W' for weekly).

    Returns:
        portfolio_values (list): Portfolio value over time.
        total_invested (float): Total amount invested.
    """
    # Resample the DataFrame to the specified interval
    resampled_df = df.resample(
        interval
    ).first()  # Use the first close price of the period

    total_invested = 0
    total_shares = 0
    portfolio_values = []
    not_invested_values = []
    resampled_df["price_change"] = (
        resampled_df["close"] - resampled_df["open"]
    ) / resampled_df["close"]
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

    return_percentage = ((portfolio_values[-1] - total_invested) / total_invested) * 100
    return (
        portfolio_values,
        total_invested,
        resampled_df.index,
        not_invested_values,
        return_percentage,
    )


if __name__ == "__main__":
    args = Args
    df = open_dataframe(rf"data/{args.exchange}-{args.cryptocurrency}-{args.time}.pkl")
    args.end_date = df["open_date"].iloc[-1] if args.end_date is None else args.end_date
    df = df[(df["open_date"] >= args.begin_date) & (df["open_date"] <= args.end_date)]
    df.set_index("open_date", inplace=True)
    # Calculate DCA
    (
        portfolio_values,
        total_invested,
        resampled_dates,
        not_invested_values,
        return_percentage,
    ) = calculate_dca_from_dataframe(
        df,
        args.monthly_investment,
        args.investment_interval,
        fees=args.fees,
        increase_factor=args.increase_factor,
        decrease_threshold=args.decrease_threshold,
    )

    final_portfolio_value = portfolio_values[-1]

    # Print results
    print(f"Total Invested: ${total_invested:.2f}")
    print(f"Final Portfolio Value: ${final_portfolio_value:.2f}")
    print(f"Return on Investment: {return_percentage:.2f}%")

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
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
    )

    fig.show()
