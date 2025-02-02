from backtesting import Strategy
from backtesting import Backtest
import pandas as pd
import sys, os
import wandb
from bokeh.embed import file_html
from bokeh.resources import CDN

absolute_path = os.path.abspath(__file__)[
    : os.path.abspath(__file__).find("MAD_BTC") + len("MAD_BTC")
]
sys.path.append(absolute_path)
from src.strategy.momentum import indicator_momentum
from src.utils.utils_df import open_dataframe_backtesting
from dataclasses import dataclass
import multiprocessing
from itertools import product
from tqdm import tqdm
import plotly.express as px


@dataclass
class StrategyConfig:
    ratio_method: str
    x: int
    y: int
    name: str


@dataclass
class Config:
    crypto_currency: str
    exchange: str
    time: str
    strategy: StrategyConfig


# Example of creating an instance of the Config dataclass
config = Config(
    crypto_currency="ETHUSDT",
    exchange="binance",
    time="1h",
    strategy=StrategyConfig(ratio_method="ema_ratio", x=4, y=100, name="momentum"),
)
DF = open_dataframe_backtesting(
    rf"data/{config.exchange}-{config.crypto_currency}-{config.time}.pkl"
)
DF["Buy and Hold Equity"] = (DF["Close"] - DF["Close"].iloc[0]) / DF["Close"].iloc[0]


def function(df, config):
    return (
        getattr(indicator_momentum, config.strategy.ratio_method)(
            df, config.strategy.x, config.strategy.y, column_name="Open"
        )
        - 1
    ).values


class MaRatio(Strategy):
    config = {}

    def init(self):
        # Precompute the two moving averages
        self.ratio = self.I(function, self.data._Data__df, self.config)

    def next(self):
        if not self.position.is_long and not self.position.is_short:
            if self.ratio >= 0:
                self.buy()
            elif self.ratio < 0:
                self.sell()

        elif self.position.is_long or self.position.is_short:
            if self.ratio >= 0 and self.position.is_short:
                self.position.close()
                self.buy()

            elif self.ratio < 0 and self.position.is_long:
                self.position.close()
                self.sell()


def _backtest(config):
    MaRatio.config = config
    bt = Backtest(DF, MaRatio, cash=1000, commission=0.1 / 100)
    stats = bt.run()
    """
    filename_save = rf"experience/backtest/{config.exchange}-{config.crypto_currency}-{config.time}/method_{config.strategy.ratio_method}_x_{config.strategy.x}_y_{config.strategy.y}"
    os.makedirs(os.path.dirname(filename_save), exist_ok=True)
    bt.plot(
        open_browser=False,
        plot_volume=False,
        plot_pl=False,
        filename=filename_save,
    )
    """
    del bt

    return stats  # , filename_save


def _backtest_with_logging(config):
    """
    Run backtest for a given config and log results to W&B.
    """
    stats = _backtest(config)  # Your original backtest logic

    # Convert Timedelta values to seconds (if present)
    def convert_timedelta_to_seconds(value):
        if isinstance(value, pd.Timedelta):
            return value.total_seconds()
        return value

    # Log backtest statistics to W&B
    wandb.log(
        data={
            "backtest_results/Exposure Time [%]": convert_timedelta_to_seconds(
                stats.loc["Exposure Time [%]"]
            ),
            "backtest_results/Equity Final [$]": stats.loc["Equity Final [$]"],
            "backtest_results/Equity Peak [$]": stats.loc["Equity Peak [$]"],
            "backtest_results/Return [%]": stats.loc["Return [%]"],
            "backtest_results/Buy & Hold Return [%]": stats.loc[
                "Buy & Hold Return [%]"
            ],
            "backtest_results/Return (Ann.) [%]": stats.loc["Return (Ann.) [%]"],
            "backtest_results/Volatility (Ann.) [%]": stats.loc[
                "Volatility (Ann.) [%]"
            ],
            "backtest_results/CAGR [%]": stats.loc["CAGR [%]"],
            "backtest_results/Sharpe Ratio": stats.loc["Sharpe Ratio"],
            "backtest_results/Sortino Ratio": stats.loc["Sortino Ratio"],
            "backtest_results/Calmar Ratio": stats.loc["Calmar Ratio"],
            "backtest_results/Max. Drawdown [%]": stats.loc["Max. Drawdown [%]"],
            "backtest_results/Avg. Drawdown [%]": stats.loc["Avg. Drawdown [%]"],
            "backtest_results/Max. Drawdown Duration": convert_timedelta_to_seconds(
                stats.loc["Max. Drawdown Duration"]
            ),
            "backtest_results/Avg. Drawdown Duration": convert_timedelta_to_seconds(
                stats.loc["Avg. Drawdown Duration"]
            ),
            "backtest_results/# Trades": stats.loc["# Trades"],
            "backtest_results/Win Rate [%]": stats.loc["Win Rate [%]"],
            "backtest_results/Profit Factor": stats.loc["Profit Factor"],
            "backtest_results/Expectancy [%]": stats.loc["Expectancy [%]"],
        }
    )
    df = stats["_equity_curve"]
    df["Equity"] = (df["Equity"] - df["Equity"].iloc[0]) / df["Equity"].iloc[0]
    df = df.drop(["DrawdownPct", "DrawdownDuration"], axis=1)
    title = f"Method: {config.strategy.ratio_method}, x: {config.strategy.x}, y: {config.strategy.y} (equity)"
    df = df.rename(columns={"Equity": title})
    return df


def launcher_backtest(
    project="bt_momentum",
    entity="corporate-manu-sureli",
    name=f"bt_{config.crypto_currency}_{config.time}",
    config=config,
):
    # Initialize W&B
    wandb.init(project=project, entity=entity, name=name)

    # Define parameter ranges
    x_range = range(1, 30)
    y_range = range(30, 150)
    ratio_method = ["ma_ratio", "ema_ratio", "zlema_ratio", "dema_ratio", "tema_ratio"]

    # Generate configs for backtests
    configs = [
        Config(
            crypto_currency=config.crypto_currency,
            exchange=config.exchange,
            time=config.time,
            strategy=StrategyConfig(
                ratio_method=method, x=x, y=y, name=config.strategy.name
            ),
        )
        for x, y, method in product(x_range, y_range, ratio_method)
    ]

    # Define the tqdm_map function with a progress bar
    def tqdm_map(func, iterable, num_processes=None, desc=None):
        """
        A wrapper for multiprocessing.Pool.map with tqdm progress bar.
        """
        num_processes = num_processes or multiprocessing.cpu_count()

        with multiprocessing.Pool(processes=num_processes) as pool:
            with tqdm(total=len(iterable), desc=desc) as progress_bar:

                def update(*_):
                    progress_bar.update()

                # Update progress for each result
                for result in pool.imap(func, iterable):
                    update()
                    yield result

    # Run the backtests with tqdm progress bar
    results = list(
        tqdm_map(_backtest_with_logging, configs, desc="Backtesting Progress")
    )
    step = 100
    i = 0  # Track step index

    while len(results) > 0:
        # Take the first 'step' elements and remove them
        sub_results = results[:step]
        results = results[step:]  # Remove used elements (pop from front)
        sub_results.insert(index=0, object=DF["Buy and Hold Equity"])
        # Concatenate and plot
        df = pd.concat(sub_results, axis=1)  # Combine data into a DataFrame
        fig = px.line(
            df,
            x=df.index,
            y=df.columns,
            title="Equity for different strategies",
        )

        # Log with a unique key to accumulate
        wandb.log({f"equity_chart_{i}": wandb.Plotly(fig)})
        del df  # Free memory

        i += 1  # Increment step index

    wandb.finish()


if __name__ == "__main__":
    # plot = _backtest(config)
    launcher_backtest()
