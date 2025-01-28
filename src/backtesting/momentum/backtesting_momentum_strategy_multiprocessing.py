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
        if self.position.is_long or self.position.is_short:
            if self.ratio >= 0 and self.position.is_short:
                self.position.close()
                self.buy()

            elif self.ratio <= 0 and self.position.is_long:
                self.position.close()
                self.sell()

        elif not self.position.is_long and not self.position.is_short:
            if self.ratio >= 0:
                self.buy()
            elif self.ratio <= 0:
                self.sell()


def _backtest(config):
    MaRatio.config = config
    bt = Backtest(DF, MaRatio, cash=1000, commission=0.1 / 100)
    stats = bt.run()
    filename_save = rf"experience/backtest/{config.exchange}-{config.crypto_currency}-{config.time}/method_{config.strategy.ratio_method}_x_{config.strategy.x}_y_{config.strategy.y}"
    bt.plot(
        open_browser=False,
        plot_volume=False,
        plot_pl=False,
        filename=filename_save,
    )
    # Log the HTML file to wandb
    wandb.log({f"plot/": wandb.Html(open(filename_save).read())})

    return stats


def launcher_backtest(
    project="bt_momentum",
    entity="corporate-manu-sureli",
    name=f"bt_{config.crypto_currency}_{config.time}",
    config=config,
):
    wandb.init(project=project, entity=entity, name=name)
    x_range = range(1, 34)
    y_range = range(35, 120)
    ratio_method = ["ma_ratio", "ema_ratio", "zlema_ratio", "dema_ratio", "tema_ratio"]
    configs = [
        Config(
            crypto_currency=config.crypto_currency,
            exchange=config.exchange,
            time=config.time,
            strategy=StrategyConfig(
                ratio_method=ratio_method, x=x, y=y, name=config.strategy.name
            ),
        )
        for x, y, ratio_method in product(x_range, y_range, ratio_method)
    ]
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.starmap(_backtest, configs)


if __name__ == "__main__":
    plot = _backtest(config)
    print(plot)
