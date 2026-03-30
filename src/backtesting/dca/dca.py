import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from ta.trend import SMAIndicator, EMAIndicator
from dataclasses import dataclass
from src.utils.utils_df import open_dataframe
from itertools import product
import multiprocessing
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
    # Removed day and offset


class Dca:
    def __init__(self, args):
        self.args = args
        self.args.time = "1h"
        self.df = self._open_dataframe()

        # Simply resample by the investment interval (no weekday offset needed)
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

    def compute_dca_from_dataframe(self, output: bool = False, plot: bool = False):
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
    factor: float = 1 / 100
    window: int = 12


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
            if pd.isna(indicator):  # Handle initial NaN from SMA window
                current_investment = investment
            elif price >= indicator:
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
    increase_factor: float = 1 / 100
    decrease_threshold: float = 1 / 100


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


# Multiprocessing Worker Function
def process_dca(strategy_class, args_instance):
    """
    Executes a DCA strategy loop given the strategy class and its arguments.
    """
    dca_element = strategy_class(args_instance)
    try:
        portfolio_values, total_invested, _, _, _ = dca_element._dca_loop()
        return_percentage = (
            (portfolio_values[-1] - total_invested) / total_invested
        ) * 100
    except Exception as e:
        # Failsafe for missing data or calculation errors to not break the multiprocessing pool
        return_percentage = 0.0

    return (
        strategy_class.__name__,
        args_instance.cryptocurrency,
        args_instance.investment_interval,
        return_percentage,
    )


class DcaComputeMultiprocessing:
    def __init__(self, cfg=None):  # Fixed __int__ typo
        self.cfg = cfg

    def compute_dca_multiprocessing(
        self,
        project="dca",
        entity="corporate-manu-sureli",
        name="dca_strategy",
        list_cryptos=["ADAUSDT", "BNBUSDT", "ETHUSDT", "BTCUSDT"],
        list_investment_interval=["24h", "48h", "72h", "168h", "672h", "W", "ME", "D"],
        # Provide the strategies you want to compare
        strategies=[
            (Dca, Dca_Args),
            (Dca_Ema, Dca_Ema_Args),
            (Dca_PriceVariation, Dca_PriceVariation_Args),
        ],
        columns=[
            "strategy",
            "cryptocurrency",
            "investment_interval",
            "return_percentage",
        ],
        name_csv="dca_multiprocessing",
        wandb_active=False,
        plot_figure=False,
    ):
        from tqdm import tqdm

        if wandb_active:
            import wandb

            wandb.init(project=project, entity=entity, name=name)

        # Build list of configurations to run
        combinations = []
        for crypto, interval, (strat_class, strat_args_class) in product(
            list_cryptos, list_investment_interval, strategies
        ):
            args_instance = strat_args_class(
                cryptocurrency=crypto, investment_interval=interval
            )
            combinations.append((strat_class, args_instance))

        print(
            f"Running {len(combinations)} combinations across {multiprocessing.cpu_count()} CPU cores..."
        )

        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            # starmap applies the process_dca function to all elements in combinations
            results = pool.starmap(process_dca, combinations)

        # Consolidate results into a single DataFrame
        df = pd.DataFrame(results, columns=columns)
        df.to_csv(path_or_buf=f"./{name_csv}.csv", index=False)

        # Plotly chart: X-axis is interval, colored by Strategy
        for crypto in list_cryptos:
            df_crypto = df[df["cryptocurrency"] == crypto]

            fig = px.scatter(
                df_crypto,
                y="return_percentage",
                x="investment_interval",
                color="strategy",
                title=f"Strategy Comparison for {crypto}",
                labels={
                    "return_percentage": "Return (%)",
                    "investment_interval": "Investment Interval",
                },
            )
            fig.update_traces(marker_size=12)

            if plot_figure:
                fig.show()
            if wandb_active:
                wandb.log({f"{crypto}_scatter_plot": fig})

        if wandb_active:
            wandb.finish()


if __name__ == "__main__":
    # Test Individual Run
    dca_ema_args = Dca_Ema_Args()
    dca_ema_element = Dca_Ema(dca_ema_args)
    dca_ema_element.compute_dca_from_dataframe(output=True, plot=True)

    dca_pricevariation_args = Dca_PriceVariation_Args(
        increase_factor=3 / 100, decrease_threshold=0.03
    )
    dca_pricevariation_element = Dca_PriceVariation(dca_pricevariation_args)
    dca_pricevariation_element.compute_dca_from_dataframe(output=True, plot=True)

    # Test Multiprocessing
    multi_compute = DcaComputeMultiprocessing()
    multi_compute.compute_dca_multiprocessing(plot_figure=True)
