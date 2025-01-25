import pandas as pd
import numpy as np
import plotly.graph_objects as go
from ta.trend import SMAIndicator, EMAIndicator
from dataclasses import dataclass
from src.utils.utils_df import open_dataframe
from itertools import product
import sys, os

absolute_path = os.path.abspath(__file__)[
    : os.path.abspath(__file__).find("MAD_BTC") + len("MAD_BTC")
]


@dataclass
class Dca_Args:
    exchange: str = "binance"
    cryptocurrency: str = "ETHUSDT"
    investment: float = 125
    investment_interval: str = "W"
    begin_date: str = "2020-01-01 00:00:00"
    end_date: str = "2024-12-01 00:00:00"
    fees: float = 0.1 / 100
    day: int = 5
    offset: int = 22


class Dca:
    weekday_map = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]

    def __init__(self, args):
        self.args = args
        self.args.time = "1h"
        self.df = self._open_dataframe()
        self.df.index = self.df.index + pd.Timedelta(hours=args.offset)
        self.resampled_df = self.df.resample(
            args.investment_interval + f"-{self.weekday_map[args.day % 7]}"
        ).first()
        del self.df

    def _open_dataframe(self):
        df = open_dataframe(
            rf"data/{self.args.exchange}-{self.args.cryptocurrency}-{self.args.time}.pkl"
        )
        self.args.end_date = (
            df["open_date"].iloc[-1]
            if self.args.end_date is None
            else self.args.end_date
        )
        self.args.begin_date = (
            df["open_date"].iloc[0]
            if self.args.begin_date is None
            else self.args.begin_date
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
        investment_fees = self.args.investment * (1 - self.args.fees)
        for price in self.resampled_df["close"].dropna():
            total_shares += investment_fees / price
            total_invested += self.args.investment
            portfolio_values.append(total_shares * price)

        not_invested_values = np.arange(0, len(portfolio_values)) * self.args.investment
        resampled_dates = self.resampled_df.index

        return (
            portfolio_values,
            total_invested,
            self.resampled_df,
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
        self.resampled_df["indicator"] = SMAIndicator(
            self.resampled_df["close"], window=self.args.window
        ).sma_indicator()
        total_invested = 0
        total_shares = 0
        portfolio_values = []
        not_invested_values = []
        investment = self.args.investment
        fees = self.args.fees
        increase_factor = self.args.increase_factor
        for index, row in self.resampled_df.iterrows():
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
        resampled_dates = self.resampled_df.index
        return (
            portfolio_values,
            total_invested,
            self.resampled_df,
            not_invested_values,
            resampled_dates,
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
        total_invested = 0
        total_shares = 0
        portfolio_values = []
        not_invested_values = []
        self.resampled_df["price_change"] = (
            self.resampled_df["close"] - self.resampled_df["open"]
        ) / self.resampled_df["close"]
        investment = self.args.investment
        fees = self.args.fees
        increase_factor = self.args.increase_factor
        decrease_threshold = self.args.decrease_threshold
        for index, row in self.resampled_df.iterrows():
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

        resampled_dates = self.resampled_df.index
        return (
            portfolio_values,
            total_invested,
            self.resampled_df,
            not_invested_values,
            resampled_dates,
        )


def process(offset, weekday):
    dca_args = Dca_Args(offset=offset, day=weekday, begin_date=None, end_date=None)
    dca_element = Dca(dca_args)
    portfolio_values, total_invested, _, not_invested_values, dates = (
        dca_element._dca_loop()
    )
    return portfolio_values, total_invested, not_invested_values, dates, offset, weekday


# Function to process a single date range
def run_multiprocessing():
    def process_date_range(begin_date, end_date):
        print(f"Processing date range: {begin_date} to {end_date}")
        liste = []
        for i in range(0, 7):  # Days of the week
            for j in range(0, 24):  # Hours of the day
                dca_args = Dca_Args(
                    offset=j, day=i, begin_date=begin_date, end_date=end_date
                )
                dca_element = Dca(dca_args)
                portfolio_values, total_invested, _, _, _ = dca_element._dca_loop()
                return_percentage = (
                    (portfolio_values[-1] - total_invested) / total_invested
                ) * 100
                liste.append([i, j, return_percentage])

        # Create a DataFrame and generate the heatmap
        df = pd.DataFrame(liste, columns=["day", "hour", "return_percentage"])
        heatmap_data = df.pivot(index="day", columns="hour", values="return_percentage")
        fig = go.Figure(
            data=go.Heatmap(
                z=heatmap_data.values
                / heatmap_data.values.min(),  # Normalized ROI data
                x=heatmap_data.columns,  # Hours (columns)
                y=[
                    "Monday",
                    "Tuesday",
                    "Wednesday",
                    "Thursday",
                    "Friday",
                    "Saturday",
                    "Sunday",
                ],  # Days (rows)
                colorscale="Viridis",  # Choose a color scale
                colorbar_title="ROI (%)",
            )
        )
        fig.update_layout(
            title=f"Return on Investment by Day and Hour\nBegin Date: {begin_date}, End Date: {end_date}",
            xaxis_title="Hour of Day",
            yaxis_title="Day of Week",
            xaxis=dict(tickmode="linear"),
            yaxis=dict(tickmode="array", tickvals=list(range(7))),
        )

        # Save or display the plot
        fig.show()
        print(f"Finished processing date range: {begin_date} to {end_date}")

    begin_date = pd.to_datetime("2020-01-01 00:00:00")
    end_date = begin_date + pd.Timedelta(weeks=int(52 * 1.5))
    final_date = pd.to_datetime("2024-05-26 00:00:00")
    date_ranges = []
    while end_date < final_date:
        date_ranges.append((begin_date, end_date))
        begin_date += pd.Timedelta(weeks=int(4))
        end_date += pd.Timedelta(weeks=int(4))

    # Use multiprocessing to process each date range
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.starmap(process_date_range, date_ranges)


if __name__ == "__main__":
    import multiprocessing

    # Initialize the Dca_Args instance for DCA
    dca_args = Dca_Args()
    dca_element = Dca(dca_args)
    dca_element.compute_dca_from_dataframe(output=False, plot=False)

    # Initialize the Dca_Ema_Args instance with correct parameters
    dca_ema_args = Dca_Ema_Args()  # Create an instance of Dca_Ema_Args
    dca_ema_element = Dca_Ema(dca_ema_args)  # Pass it to the Dca_Ema class
    dca_ema_element.compute_dca_from_dataframe(output=False, plot=False)

    dca_pricevariation_args = Dca_PriceVariation_Args(
        increase_factor=3 / 100, decrease_threshold=0.03
    )
    dca_pricevariation_element = Dca_PriceVariation(dca_pricevariation_args)
    dca_pricevariation_element.compute_dca_from_dataframe(output=False, plot=False)

    day_hour_combinations = [
        (day, hour) for day, hour in product(range(0, 7), range(0, 24))
    ]
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.starmap(process, day_hour_combinations)

    collected_results = []
    for result in results:
        (
            portfolio_values,
            total_invested,
            not_invested_values,
            dates,
            offset,
            weekday,
        ) = result
        return_percentage = (
            (portfolio_values[-1] - total_invested) / total_invested
        ) * 100
        collected_results.append([weekday, offset, return_percentage])
