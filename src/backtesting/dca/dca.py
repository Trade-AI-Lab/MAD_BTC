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
    cryptocurrency: str = "BNBUSDT"
    investment: float = 1
    investment_interval: str = "672h"
    begin_date: str = None
    end_date: str = "2024-05-20 00:00:00"
    fees: float = 0.1 / 100
    day: int = 12
    offset: int = 3


class Dca:
    weekday_map = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]

    def __init__(self, args):
        self.args = args
        self.args.time = "1h"
        self.df = self._open_dataframe()
        self.df.index = self.df.index + pd.Timedelta(hours=args.offset)
        if args.investment_interval == "W" or args.investment_interval == "D":
            self.resampled_df = self.df.resample(
                "W" + f"-{self.weekday_map[args.day % 7]}"
            ).first()
            if args.investment_interval == "D":
                self.resampled_df = self.df.resample(args.investment_interval).first()
        else:
            self.resampled_df = self.df.resample(args.investment_interval).first()
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
    factor: float = 5 / 100  # Default value for increase factor
    window: int = 12  # Default window size for EMA

    def __init__(self, **kwargs):
        # Pass parent attributes to Dca_Args
        self.factor = kwargs.pop("factor", self.factor)
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
        factor = self.args.factor
        for _, row in self.resampled_df.iterrows():
            price = row["close"]
            indicator = row["indicator"]
            if price >= indicator:
                current_investment = investment * (1 + factor)
            else:
                current_investment = investment * (1 - factor)
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
        0.05 / 100
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


def process_dca(offset, weekday, investment_interval, cryptocurrency):
    dca_args = Dca_Args(
        offset=offset,
        day=weekday,
        investment_interval=investment_interval,
        cryptocurrency=cryptocurrency,
    )
    dca_element = Dca(dca_args)
    portfolio_values, total_invested, _, not_invested_values, dates = (
        dca_element._dca_loop()
    )
    return_percentage = ((portfolio_values[-1] - total_invested) / total_invested) * 100
    return weekday, offset, return_percentage, investment_interval


class DcaComputeMultiprocessing:
    def __int__(self, cfg):
        self.cfg = cfg

    def compute_dca_multiprocessing(
        self,
        project="dca",
        entity="corporate-manu-sureli",
        name="dca_strategy",
        list_cryptos=["ADAUSDT", "BNBUSDT", "ETHUSDT", "BTCUSDT"],
        list_investment_interval=["24h", "168h", "672h", "W", "ME", "D"],
        columns=["offset", "weekday", "return_percentage", "investment_interval"],
        name_csv="dca_multiprocessing",
        wandb_active=False,
        plot_figure=False,
    ):
        from tqdm import tqdm

        weekday_map = {
            0: "Monday",
            1: "Tuesday",
            2: "Wednesday",
            3: "Thursday",
            4: "Friday",
            5: "Saturday",
            6: "Sunday",
        }
        if wandb_active:
            wandb.init(project=project, entity=entity, name=name)
        for cryptocurrency in tqdm(list_cryptos):
            # Initialize the Dca_Args instance for DCA
            day_hour_investment_interval_combinations = [
                (day, hour, investment_interval, cryptocurrency)
                for day, hour, investment_interval in product(
                    range(0, 7), range(0, 24), list_investment_interval
                )
            ]
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                results = pool.starmap(
                    process_dca, day_hour_investment_interval_combinations
                )
            df = pd.DataFrame(
                results,
                columns=columns,
            )

            df["weekday"] = df["weekday"].map(weekday_map)
            df.to_csv(path_or_buf=f"./{cryptocurrency}_{name_csv}.csv")
            fig = px.scatter(
                df,
                y="return_percentage",
                x="weekday",
                color="investment_interval",
                title=f"Cryptocurrency_{cryptocurrency}",
            )
            fig.update_traces(marker_size=10)
            if plot_figure:
                fig.show()
            if wandb_active:
                wandb.log({"weekday_scatter_plot": fig})
            del results
            print(f"Finished for {cryptocurrency}")
        if wandb_active:
            wandb.finish()


if __name__ == "__main__":
    import multiprocessing
    import plotly.express as px
    import wandb

    # Initialize the Dca_Ema_Args instance with correct parameters
    dca_ema_args = Dca_Ema_Args()  # Create an instance of Dca_Ema_Args
    dca_ema_element = Dca_Ema(dca_ema_args)  # Pass it to the Dca_Ema class
    dca_ema_element.compute_dca_from_dataframe(output=True, plot=True)

    dca_pricevariation_args = Dca_PriceVariation_Args(
        increase_factor=3 / 100, decrease_threshold=0.03
    )

    dca_pricevariation_element = Dca_PriceVariation(dca_pricevariation_args)
    dca_pricevariation_element.compute_dca_from_dataframe(output=True, plot=True)
    DcaComputeMultiprocessing.compute_dca_multiprocessing()
