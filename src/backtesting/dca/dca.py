import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from dataclasses import dataclass
from src.utils.utils_df import open_dataframe
from itertools import product
import multiprocessing
import os

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


# ---------------------------------------------------------
# Worker Function for Grid Search
# ---------------------------------------------------------
def process_dca_grid(
    strategy_class, args_class, crypto, interval, hyperparams, baseline_return
):
    """
    Worker function executed by multiprocessing pool.
    Calculates the normalized return: Strategy Return / Baseline Return
    """
    args_instance = args_class(
        cryptocurrency=crypto, investment_interval=interval, **hyperparams
    )
    dca_element = strategy_class(args_instance)

    try:
        portfolio_values, total_invested, _, _, _ = dca_element._dca_loop()
        return_percentage = (
            (portfolio_values[-1] - total_invested) / total_invested
        ) * 100
    except Exception as e:
        return_percentage = 0.0

    # Normalize against the benchmark (prevent division by zero)
    safe_baseline = baseline_return if baseline_return != 0 else 0.0001
    normalized_return = return_percentage / safe_baseline

    result = {
        "strategy": strategy_class.__name__,
        "cryptocurrency": crypto,
        "investment_interval": interval,
        "raw_return_percentage": return_percentage,
        "normalized_return": normalized_return,  # 1.0 = equal to benchmark
    }
    result.update(hyperparams)
    return result


# ---------------------------------------------------------
# Grid Search & Plotting Class
# ---------------------------------------------------------
class DcaHyperparameterSearch:
    def __init__(
        self,
        list_cryptos=["BTCUSDT", "ETHUSDT"],
        list_intervals=["24h", "168h", "W", "ME"],
    ):
        self.list_cryptos = list_cryptos
        self.list_intervals = list_intervals
        self.baselines = self._compute_baselines()

    def _compute_baselines(self):
        """
        Computes the standard DCA returns to act as our benchmark.
        Returns a dictionary: {(crypto, interval): baseline_return_percentage}
        """
        print("Computing Standard DCA Baselines...")
        baselines = {}
        for crypto in self.list_cryptos:
            for interval in self.list_intervals:
                # Initialize standard DCA
                base_args = Dca_Args(
                    cryptocurrency=crypto, investment_interval=interval
                )
                base_dca = Dca(base_args)
                try:
                    p_vals, t_inv, _, _, _ = base_dca._dca_loop()
                    base_return = ((p_vals[-1] - t_inv) / t_inv) * 100
                except Exception:
                    base_return = 0.0

                baselines[(crypto, interval)] = base_return
                print(f"  Benchmark {crypto} ({interval}): {base_return:.2f}%")

        return baselines

    def run_grid_search(self, strategy_class, args_class, param_grid):
        print(f"\nStarting Grid Search for {strategy_class.__name__}...")

        keys, values = zip(*param_grid.items())
        hyperparam_combinations = [dict(zip(keys, v)) for v in product(*values)]

        tasks = []
        for crypto, interval, h_params in product(
            self.list_cryptos, self.list_intervals, hyperparam_combinations
        ):
            # Fetch the specific baseline for this crypto/interval pair
            baseline_return = self.baselines[(crypto, interval)]
            tasks.append(
                (
                    strategy_class,
                    args_class,
                    crypto,
                    interval,
                    h_params,
                    baseline_return,
                )
            )

        print(f"Running {len(tasks)} total simulations...")

        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = pool.starmap(process_dca_grid, tasks)

        df_results = pd.DataFrame(results)
        print(f"Completed grid search for {strategy_class.__name__}.")
        return df_results

    def plot_3d_results(self, df, param_x, param_y, title_suffix=""):
        """
        Plots a 3D scatter based on the Normalized Return.
        """
        for crypto in df["cryptocurrency"].unique():
            df_crypto = df[df["cryptocurrency"] == crypto]

            fig = px.scatter_3d(
                df_crypto,
                x=param_x,
                y=param_y,
                z="investment_interval",
                color="normalized_return",
                title=f"{crypto} {title_suffix} Optimization (Normalized vs Std DCA)",
                labels={
                    param_x: param_x.replace("_", " ").title(),
                    param_y: param_y.replace("_", " ").title(),
                    "investment_interval": "Interval",
                    "normalized_return": "Ratio to Benchmark",
                },
                # Adjust color scale so 1.0 is neutral (white/yellow), below 1 is red, above 1 is green
                color_continuous_scale="RdYlGn",
                color_continuous_midpoint=1.0,
            )

            fig.update_traces(
                marker=dict(
                    size=8, opacity=0.8, line=dict(width=0.5, color="DarkSlateGrey")
                )
            )
            fig.show()


@dataclass
class Dca_RSI_Args(Dca_Args):
    window: int = 14
    oversold_threshold: float = 30
    overbought_threshold: float = 70
    oversold_multiplier: float = 2.0  # Buy 2x amount when oversold
    overbought_multiplier: float = 0.5  # Buy 0.5x amount when overbought

    def __init__(self, **kwargs):
        self.window = kwargs.pop("window", self.window)
        self.oversold_threshold = kwargs.pop(
            "oversold_threshold", self.oversold_threshold
        )
        self.overbought_threshold = kwargs.pop(
            "overbought_threshold", self.overbought_threshold
        )
        self.oversold_multiplier = kwargs.pop(
            "oversold_multiplier", self.oversold_multiplier
        )
        self.overbought_multiplier = kwargs.pop(
            "overbought_multiplier", self.overbought_multiplier
        )
        super().__init__(**kwargs)


class Dca_RSI(Dca):
    def _dca_loop(self):
        # Pre-compute RSI to speed up the loop
        self.resampled_df["rsi"] = RSIIndicator(
            self.resampled_df["close"], window=self.args.window
        ).rsi()

        total_invested = 0
        total_shares = 0
        portfolio_values = []
        not_invested_values = []

        investment = self.args.investment
        fees = self.args.fees

        for _, row in self.resampled_df.iterrows():
            price = row["close"]
            rsi = row["rsi"]

            # Handle initial NaNs before RSI window is filled
            if pd.isna(rsi):
                current_investment = investment
            elif rsi <= self.args.oversold_threshold:
                current_investment = investment * self.args.oversold_multiplier
            elif rsi >= self.args.overbought_threshold:
                current_investment = investment * self.args.overbought_multiplier
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


# ==========================================
# Strategy 2: Dynamic Drawdown Strategy
# ==========================================
@dataclass
class Dca_Drawdown_Args(Dca_Args):
    # factor = 2 means if price drops 50%, we buy (1 + (0.5 * 2)) = 2x normal amount
    drawdown_factor: float = 2.0

    def __init__(self, **kwargs):
        self.drawdown_factor = kwargs.pop("drawdown_factor", self.drawdown_factor)
        super().__init__(**kwargs)


class Dca_Drawdown(Dca):
    def _dca_loop(self):
        # Pre-compute cumulative maximum and drawdown percentage
        self.resampled_df["cummax"] = self.resampled_df["close"].cummax()
        self.resampled_df["drawdown"] = (
            self.resampled_df["cummax"] - self.resampled_df["close"]
        ) / self.resampled_df["cummax"]

        total_invested = 0
        total_shares = 0
        portfolio_values = []
        not_invested_values = []

        investment = self.args.investment
        fees = self.args.fees

        for _, row in self.resampled_df.iterrows():
            price = row["close"]
            drawdown = row["drawdown"]

            # Dynamic multiplier based on how far we are from the ATH
            multiplier = 1.0 + (drawdown * self.args.drawdown_factor)
            current_investment = investment * multiplier

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


# ==========================================
# Strategy 3: Moving Average Ratio (Mayer Multiple)
# ==========================================
@dataclass
class Dca_MARatio_Args(Dca_Args):
    window: int = 200  # Long-term trend
    cheap_threshold: float = 0.8  # Price is 20% below MA
    expensive_threshold: float = 1.5  # Price is 50% above MA
    cheap_multiplier: float = 3.0  # Triple buy when historically cheap
    expensive_multiplier: float = 0.2  # Barely buy when in a bubble

    def __init__(self, **kwargs):
        self.window = kwargs.pop("window", self.window)
        self.cheap_threshold = kwargs.pop("cheap_threshold", self.cheap_threshold)
        self.expensive_threshold = kwargs.pop(
            "expensive_threshold", self.expensive_threshold
        )
        self.cheap_multiplier = kwargs.pop("cheap_multiplier", self.cheap_multiplier)
        self.expensive_multiplier = kwargs.pop(
            "expensive_multiplier", self.expensive_multiplier
        )
        super().__init__(**kwargs)


class Dca_MARatio(Dca):
    def _dca_loop(self):
        # Calculate Long-Term SMA and the Ratio
        self.resampled_df["sma"] = SMAIndicator(
            self.resampled_df["close"], window=self.args.window
        ).sma_indicator()

        self.resampled_df["ma_ratio"] = (
            self.resampled_df["close"] / self.resampled_df["sma"]
        )

        total_invested = 0
        total_shares = 0
        portfolio_values = []
        not_invested_values = []
        investment = self.args.investment
        fees = self.args.fees

        for _, row in self.resampled_df.iterrows():
            price = row["close"]
            ratio = row["ma_ratio"]

            if pd.isna(ratio):
                current_investment = investment
            elif ratio <= self.args.cheap_threshold:
                current_investment = investment * self.args.cheap_multiplier
            elif ratio >= self.args.expensive_threshold:
                current_investment = investment * self.args.expensive_multiplier
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


class DcaComputeMultiprocessing:
    def __init__(self, cfg=None):  # Fixed __int__ typo
        self.cfg = cfg

    def compute_dca_multiprocessing(
        self,
        project="dca",
        entity="corporate-manu-sureli",
        name="dca_strategy",
        list_cryptos=["BNBUSDT", "ETHUSDT", "BTCUSDT"],
        list_investment_interval=[
            "24h",
            "48h",
            "72h",
            "168h",
            "W",
            "ME",
            "D",
        ],
        # Provide the strategies you want to compare
        strategies=[
            (Dca, Dca_Args),
            (Dca_Drawdown, Dca_Drawdown_Args),
            (Dca_RSI, Dca_RSI_Args),
            (Dca_MARatio, Dca_MARatio_Args),
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
    # Test Multiprocessing
    multi_compute = DcaComputeMultiprocessing()
    multi_compute.compute_dca_multiprocessing(plot_figure=True)

    # Initialize the search class

    grid_searcher = DcaHyperparameterSearch(
        list_cryptos=["BTCUSDT", "ETHUSDT"],
        list_intervals=[
            "24h",
            "168h",
            "D",
            "W",
            "ME",
        ],
    )

    # --- 1. Optimize Drawdown DCA ---
    drawdown_grid = {
        # Test how aggressive we should be when the price drops (1x to 5x multiplier scale)
        "drawdown_factor": np.round(np.arange(1.0, 6.0, 1.0), 1).tolist()
    }

    df_drawdown = grid_searcher.run_grid_search(
        Dca_Drawdown, Dca_Drawdown_Args, drawdown_grid
    )

    # We can use a 2D scatter here since there is only one hyperparameter right now
    # Or just use the 3D plot with a dummy variable if you add more parameters later.

    # --- 2. Optimize RSI DCA ---
    rsi_grid = {
        "oversold_threshold": np.arange(10.0, 40.0, 1.0).tolist(),
        "oversold_multiplier": [1.5, 2.0, 3.0, 4.0],
    }

    df_rsi = grid_searcher.run_grid_search(Dca_RSI, Dca_RSI_Args, rsi_grid)

    grid_searcher.plot_3d_results(
        df_rsi,
        param_x="oversold_threshold",
        param_y="oversold_multiplier",
        title_suffix="(RSI Strategy)",
    )

    # --- 3. Optimize MA Ratio (Mayer Multiple) ---
    ma_ratio_grid = {
        "window": [25, 50, 100, 200],  # Test medium to long term trends
        "cheap_threshold": [0.7, 0.8, 0.9],
        "cheap_multiplier": [2.0, 3.0, 4.0],
    }

    df_maratio = grid_searcher.run_grid_search(
        Dca_MARatio, Dca_MARatio_Args, ma_ratio_grid
    )

    grid_searcher.plot_3d_results(
        df_maratio,
        param_x="cheap_threshold",
        param_y="cheap_multiplier",
        title_suffix="(MA Ratio Strategy)",
    )
