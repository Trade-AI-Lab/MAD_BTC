import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from ta.trend import SMAIndicator
from ta.momentum import RSIIndicator
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
    cryptocurrency: str = "ETHUSDT"
    investment: float = 1
    investment_interval: str = "672h"
    begin_date: str = None
    end_date: str = "2024-05-20 00:00:00"
    fees: float = 0.1 / 100


# ==========================================
# Core DCA Class (Now Vectorized)
# ==========================================
class Dca:
    def __init__(self, args):
        self.args = args
        self.args.time = "1h"
        self.df = self._open_dataframe()
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

    def _compute_multipliers(self):
        """Base strategy buys 1x the investment amount. Overridden by subclasses."""
        return pd.Series(1.0, index=self.resampled_df.index)

    def _dca_loop(self):
        # 1. Fetch dynamic multipliers (Vectorized)
        multipliers = self._compute_multipliers().fillna(1.0)

        # 2. Vectorized Portfolio Math (No more iterrows!)
        current_investments = self.args.investment * multipliers
        shares_bought = (
            current_investments * (1 - self.args.fees)
        ) / self.resampled_df["close"]

        total_shares = shares_bought.cumsum()
        total_invested = current_investments.cumsum()
        portfolio_value = total_shares * self.resampled_df["close"]

        not_invested_values = np.arange(0, len(portfolio_value)) * self.args.investment

        return (
            portfolio_value.tolist(),
            total_invested.iloc[-1] if not total_invested.empty else 0,
            self.resampled_df,
            not_invested_values.tolist(),
            self.resampled_df.index,
        )

    def compute_dca_from_dataframe(self, output: bool = False, plot: bool = False):
        portfolio_values, total_invested, res_df, not_invested, dates = self._dca_loop()
        return_pct = ((portfolio_values[-1] - total_invested) / total_invested) * 100

        print(
            f"Total Invested: ${total_invested:.2f} | Final Value: ${portfolio_values[-1]:.2f} | ROI: {return_pct:.2f}%"
        )

        if plot:
            fig = go.Figure()
            fig.add_scatter(
                x=dates,
                y=not_invested,
                mode="lines",
                name="Not invested",
                line=dict(color="blue"),
            )
            fig.add_scatter(
                x=dates,
                y=portfolio_values,
                mode="lines",
                name="Portfolio Value",
                line=dict(color="green"),
            )
            fig.update_layout(
                title="DCA Simulation",
                xaxis_title="Time",
                yaxis_title="Value ($)",
                template="plotly_white",
            )
            fig.show()

        if output:
            return portfolio_values, total_invested, res_df, not_invested, dates


# ==========================================
# Strategy 1: RSI
# ==========================================
@dataclass
class Dca_RSI_Args(Dca_Args):
    window: int = 14
    oversold_threshold: float = 30
    overbought_threshold: float = 70
    oversold_multiplier: float = 2.0
    overbought_multiplier: float = 0.5


class Dca_RSI(Dca):
    def _compute_multipliers(self):
        rsi = RSIIndicator(self.resampled_df["close"], window=self.args.window).rsi()
        conditions = [
            rsi <= self.args.oversold_threshold,
            rsi >= self.args.overbought_threshold,
        ]
        choices = [self.args.oversold_multiplier, self.args.overbought_multiplier]
        return pd.Series(
            np.select(conditions, choices, default=1.0), index=self.resampled_df.index
        )


# ==========================================
# Strategy 2: Dynamic Drawdown
# ==========================================
@dataclass
class Dca_Drawdown_Args(Dca_Args):
    drawdown_factor: float = 2.0


class Dca_Drawdown(Dca):
    def _compute_multipliers(self):
        cummax = self.resampled_df["close"].cummax()
        drawdown = (cummax - self.resampled_df["close"]) / cummax
        return 1.0 + (drawdown * self.args.drawdown_factor)


# ==========================================
# Continuous Strategy 2: Proportional MA Ratio
# ==========================================
@dataclass
class Dca_ContinuousMARatio_Args(Dca_Args):
    window: int = 200
    aggression: float = 2.0  # Squaring the inverse ratio punishes buying tops and rewards buying bottoms


class Dca_ContinuousMARatio(Dca):
    def _compute_multipliers(self):
        sma = SMAIndicator(
            self.resampled_df["close"], window=self.args.window
        ).sma_indicator()
        ma_ratio = self.resampled_df["close"] / sma

        safe_ratio = ma_ratio.clip(lower=0.01)

        # If Ratio is 0.5 (Price is 50% below MA), 1/0.5 = 2.
        # With aggression=2, multiplier becomes 4x.
        multipliers = (1.0 / safe_ratio) ** self.args.aggression

        return multipliers.clip(upper=10.0).fillna(1.0)


# ==========================================
# Strategy 3: Moving Average Ratio
# ==========================================
@dataclass
class Dca_MARatio_Args(Dca_Args):
    window: int = 200
    cheap_threshold: float = 0.8
    expensive_threshold: float = 1.5
    cheap_multiplier: float = 3.0
    expensive_multiplier: float = 0.2


class Dca_MARatio(Dca):
    def _compute_multipliers(self):
        sma = SMAIndicator(
            self.resampled_df["close"], window=self.args.window
        ).sma_indicator()
        ratio = self.resampled_df["close"] / sma
        conditions = [
            ratio <= self.args.cheap_threshold,
            ratio >= self.args.expensive_threshold,
        ]
        choices = [self.args.cheap_multiplier, self.args.expensive_multiplier]
        return pd.Series(
            np.select(conditions, choices, default=1.0), index=self.resampled_df.index
        )


# ==========================================
# Multiprocessing & Grid Search Engine
# ==========================================
def process_dca(strategy_class, args_instance):
    try:
        p_vals, t_inv, _, _, _ = strategy_class(args_instance)._dca_loop()
        return_pct = ((p_vals[-1] - t_inv) / t_inv) * 100
    except Exception:
        return_pct = 0.0
    return (
        strategy_class.__name__,
        args_instance.cryptocurrency,
        args_instance.investment_interval,
        return_pct,
    )


def process_dca_grid(
    strategy_class, args_class, crypto, interval, hyperparams, baseline_return
):
    args_instance = args_class(
        cryptocurrency=crypto, investment_interval=interval, **hyperparams
    )
    try:
        p_vals, t_inv, _, _, _ = strategy_class(args_instance)._dca_loop()
        return_pct = ((p_vals[-1] - t_inv) / t_inv) * 100
    except Exception:
        return_pct = 0.0

    result = {
        "strategy": strategy_class.__name__,
        "cryptocurrency": crypto,
        "investment_interval": interval,
        "raw_return_percentage": return_pct,
        "normalized_return": return_pct
        / (baseline_return if baseline_return != 0 else 0.0001),
    }
    result.update(hyperparams)
    return result


@dataclass
class Dca_Ensemble_Args(Dca_Args):
    window: int = 200
    rsi_window: int = 14
    # Conditions
    crash_drawdown: float = 0.40  # Price must be down 40% from ATH
    cheap_ma_ratio: float = 0.80  # AND price must be 20% below 200-SMA
    # Multipliers
    confluence_multiplier: float = 5.0  # 5x buy when BOTH are met
    bubble_ma_ratio: float = 1.50  # Stop buying if in a bubble
    bubble_multiplier: float = 0.0


class Dca_Ensemble(Dca):
    def _compute_multipliers(self):
        # 1. Calculate MA Ratio
        sma = SMAIndicator(
            self.resampled_df["close"], window=self.args.window
        ).sma_indicator()
        ma_ratio = self.resampled_df["close"] / sma

        # 2. Calculate Drawdown
        cummax = self.resampled_df["close"].cummax()
        drawdown = (cummax - self.resampled_df["close"]) / cummax

        # 3. Build Confluence Conditions
        is_generational_bottom = (drawdown >= self.args.crash_drawdown) & (
            ma_ratio <= self.args.cheap_ma_ratio
        )
        is_bubble_top = ma_ratio >= self.args.bubble_ma_ratio

        # 4. Apply Multipliers Vector-style
        conditions = [is_generational_bottom, is_bubble_top]
        choices = [self.args.confluence_multiplier, self.args.bubble_multiplier]

        return pd.Series(
            np.select(conditions, choices, default=1.0), index=self.resampled_df.index
        )


# ==========================================
# Strategy 4: Adaptive Volatility (Z-Score)
# ==========================================
@dataclass
class Dca_AdaptiveZScore_Args(Dca_Args):
    vol_window: int = 20  # How many periods to look back to gauge current volatility
    z_buy_threshold: float = (
        -2.0
    )  # Buy when price is 2 Standard Deviations BELOW the mean
    buy_multiplier: float = 3.0  # Triple investment when statistically cheap
    z_pause_threshold: float = (
        2.0  # (Optional) Pause buying if 2 StdDevs ABOVE the mean
    )
    pause_multiplier: float = (
        0.0  # Set to 0.5 to just buy half, or 0.0 to pause entirely
    )


# ==========================================
# Continuous Strategy 3: Exponential Z-Score
# ==========================================
@dataclass
class Dca_ContinuousZScore_Args(Dca_Args):
    vol_window: int = 20
    aggression: float = (
        1.0  # 1.0 is standard exponential decay. Lower this to be more conservative.
    )


class Dca_ContinuousZScore(Dca):
    def _compute_multipliers(self):
        rolling_mean = (
            self.resampled_df["close"].rolling(window=self.args.vol_window).mean()
        )
        rolling_std = (
            self.resampled_df["close"].rolling(window=self.args.vol_window).std()
        )

        z_score = (self.resampled_df["close"] - rolling_mean) / rolling_std

        # np.exp(-z_score) naturally inverts the Z-score.
        # Negative Z (Crashes) become massive positive multipliers.
        # Positive Z (Pumps) become fractions near 0.
        multipliers = np.exp(-z_score * self.args.aggression)

        # We must clip this because np.exp can get astronomically high during black swan crashes
        return multipliers.clip(lower=0.0, upper=10.0).fillna(1.0)


class Dca_AdaptiveZScore(Dca):
    def _compute_multipliers(self):
        # 1. Calculate Rolling Mean and Rolling Standard Deviation
        rolling_mean = (
            self.resampled_df["close"].rolling(window=self.args.vol_window).mean()
        )
        rolling_std = (
            self.resampled_df["close"].rolling(window=self.args.vol_window).std()
        )

        # 2. Calculate the Z-Score (How many standard deviations are we from the mean?)
        z_score = (self.resampled_df["close"] - rolling_mean) / rolling_std

        # 3. Dynamic Conditions based purely on current volatility
        is_statistically_cheap = z_score <= self.args.z_buy_threshold
        is_statistically_expensive = z_score >= self.args.z_pause_threshold

        # 4. Apply Multipliers Vector-style
        conditions = [is_statistically_cheap, is_statistically_expensive]
        choices = [self.args.buy_multiplier, self.args.pause_multiplier]

        return pd.Series(
            np.select(conditions, choices, default=1.0), index=self.resampled_df.index
        )


class DcaComputeMultiprocessing:
    def __init__(self, cfg=None):
        self.cfg = cfg

    def compute_dca_multiprocessing(
        self,
        project="dca",
        entity="corporate-manu-sureli",
        name="dca_strategy",
        list_cryptos=["ETHUSDT", "BTCUSDT"],
        list_investment_interval=["24h", "48h", "168h", "W", "ME"],
        strategies=[
            (Dca, Dca_Args),
            (Dca_Drawdown, Dca_Drawdown_Args),
            (Dca_RSI, Dca_RSI_Args),
            (Dca_MARatio, Dca_MARatio_Args),
            (Dca_Ensemble, Dca_Ensemble_Args),
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
        save_csv=False,
    ):

        if wandb_active:
            import wandb

            wandb.init(project=project, entity=entity, name=name)

        combinations = [
            (cls, args(cryptocurrency=c, investment_interval=i))
            for c, i, (cls, args) in product(
                list_cryptos, list_investment_interval, strategies
            )
        ]

        print(
            f"Running {len(combinations)} baseline combinations across {multiprocessing.cpu_count()} CPU cores..."
        )
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            df = pd.DataFrame(pool.starmap(process_dca, combinations), columns=columns)

        for crypto in list_cryptos:
            fig = px.scatter(
                df[df["cryptocurrency"] == crypto],
                y="return_percentage",
                x="investment_interval",
                color="strategy",
                title=f"Strategy Comparison for {crypto}",
            )
            fig.update_traces(marker_size=12)
            if plot_figure:
                fig.show()
            if wandb_active:
                wandb.log({f"{crypto}_scatter_plot": fig})

        if wandb_active:
            wandb.finish()
        if save_csv:
            df.to_csv(path_or_buf=f"./{name_csv}.csv", index=False)


class DcaHyperparameterSearch:
    def __init__(
        self,
        list_cryptos=["BTCUSDT", "ETHUSDT"],
        list_intervals=["24h", "168h", "W", "ME"],
    ):
        self.list_cryptos, self.list_intervals = list_cryptos, list_intervals
        self.baselines = self._compute_baselines()

    def _compute_baselines(self):
        print("Computing Standard DCA Baselines...")
        baselines = {}
        for c, i in product(self.list_cryptos, self.list_intervals):
            try:
                p_vals, t_inv, _, _, _ = Dca(
                    Dca_Args(cryptocurrency=c, investment_interval=i)
                )._dca_loop()
                baselines[(c, i)] = ((p_vals[-1] - t_inv) / t_inv) * 100
            except Exception:
                baselines[(c, i)] = 0.0
            print(f"  Benchmark {c} ({i}): {baselines[(c, i)]:.2f}%")
        return baselines

    def run_grid_search(self, strategy_class, args_class, param_grid):
        print(f"\nStarting Grid Search for {strategy_class.__name__}...")
        keys, values = zip(*param_grid.items())
        tasks = [
            (
                strategy_class,
                args_class,
                c,
                i,
                dict(zip(keys, v)),
                self.baselines[(c, i)],
            )
            for c, i, v in product(
                self.list_cryptos, self.list_intervals, product(*values)
            )
        ]

        print(f"Running {len(tasks)} grid simulations...")
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            return pd.DataFrame(pool.starmap(process_dca_grid, tasks))

    def plot_3d_results(self, df, param_x, param_y, title_suffix=""):
        for crypto in df["cryptocurrency"].unique():
            fig = px.scatter_3d(
                df[df["cryptocurrency"] == crypto],
                x=param_x,
                y=param_y,
                z="investment_interval",
                color="normalized_return",
                title=f"{crypto} {title_suffix} Optimization",
                color_continuous_scale="RdYlGn",
                color_continuous_midpoint=1.0,
            )
            fig.update_traces(
                marker=dict(
                    size=8, opacity=0.8, line=dict(width=0.5, color="DarkSlateGrey")
                )
            )
            fig.show()


if __name__ == "__main__":
    multi_compute = DcaComputeMultiprocessing()
    multi_compute.compute_dca_multiprocessing(
        plot_figure=True
    )  # Turned off by default to jump to grid search

    grid_searcher = DcaHyperparameterSearch(
        list_cryptos=["BTCUSDT", "ETHUSDT"], list_intervals=["D", "W", "ME"]
    )

    # --- 1. Drawdown Strategy ---
    df_drawdown = grid_searcher.run_grid_search(
        Dca_Drawdown,
        Dca_Drawdown_Args,
        {"drawdown_factor": np.round(np.arange(1.0, 6.0, 1.0), 1).tolist()},
    )
    grid_searcher.plot_3d_results(
        df_drawdown,
        param_x="drawdown_factor",
        param_y="drawdown_factor",
        title_suffix="(Drawdown Strategy)",
    )

    # --- 2. RSI Strategy ---
    rsi_grid = {
        "oversold_threshold": np.arange(20.0, 40.0, 5.0).tolist(),
        "oversold_multiplier": [2.0, 3.0, 4.0],
    }
    df_rsi = grid_searcher.run_grid_search(Dca_RSI, Dca_RSI_Args, rsi_grid)
    grid_searcher.plot_3d_results(
        df_rsi,
        param_x="oversold_threshold",
        param_y="oversold_multiplier",
        title_suffix="(RSI Strategy)",
    )

    # --- 3. MA Ratio Strategy ---
    ma_ratio_grid = {
        "window": [25, 50, 100, 150, 200, 250],
        "cheap_threshold": [0.6, 0.7, 0.8],
        "cheap_multiplier": [3.0],
    }
    df_maratio = grid_searcher.run_grid_search(
        Dca_MARatio, Dca_MARatio_Args, ma_ratio_grid
    )
    grid_searcher.plot_3d_results(
        df_maratio,
        param_x="cheap_threshold",
        param_y="window",
        title_suffix="(MA Ratio)",
    )
    # Grid Search the Ensemble Strategy
    ensemble_grid = {
        "crash_drawdown": [
            0.30,
            0.40,
            0.50,
        ],  # Test waiting for 30%, 40%, and 50% crashes
        "confluence_multiplier": [
            3.0,
            5.0,
            7.0,
        ],  # Test how aggressive to buy the bottom
    }

    df_ensemble = grid_searcher.run_grid_search(
        Dca_Ensemble, Dca_Ensemble_Args, ensemble_grid
    )

    # Plot the 3D Results
    grid_searcher.plot_3d_results(
        df_ensemble,
        param_x="crash_drawdown",
        param_y="confluence_multiplier",
        title_suffix="(Ensemble Confluence)",
    )

    ensemble_grid = {
        "z_buy_threshold": [-1.5, -2.0, -2.5, -3.0],
        "vol_window": [20, 50, 100],
    }
    df_ensemble = grid_searcher.run_grid_search(
        Dca_AdaptiveZScore, Dca_AdaptiveZScore_Args, ensemble_grid
    )

    # Plot the 3D Results
    grid_searcher.plot_3d_results(
        df_ensemble,
        param_x="z_buy_threshold",
        param_y="vol_window",
        title_suffix="(Dca_AdaptiveZScore)",
    )

    continuous_z_grid = {
        "vol_window": [20, 50, 100],
        "aggression": np.round(
            np.arange(0.5, 2.5, 0.25), 2
        ).tolist(),  # Testing curve steepness
    }

    df_cont_z = grid_searcher.run_grid_search(
        Dca_ContinuousZScore, Dca_ContinuousZScore_Args, continuous_z_grid
    )

    grid_searcher.plot_3d_results(
        df_cont_z,
        param_x="vol_window",
        param_y="aggression",
        title_suffix="(Continuous Z-Score)",
    )
