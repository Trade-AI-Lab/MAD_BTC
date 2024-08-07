from backtesting import Strategy, Backtest
import pandas as pd
import ray
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search import ConcurrencyLimiter
import sys, os
from ray import train
absolute_path = os.path.abspath(__file__)[: os.path.abspath(__file__).find("MAD_BTC") + len("MAD_BTC")]
sys.path.append(absolute_path)
from src.strategy.momentum import indicator_momentum
from src.utils.utils_df import open_dataframe


def function(df, config):
    return (getattr(
            indicator_momentum, config["strategy"]["ratio_method"]
        )(df, config["strategy"]["x"], config["strategy"]["y"], column_name="Open")-1).values
    
class MaRatio(Strategy):
    config = {
    }
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

def launcher_backtest(config, df):
    MaRatio.config = config
    bt = Backtest(df, MaRatio, cash=1000, commission=0.1/100)
    stats = bt.run()
    return stats.loc['Equity Final [$]']

def objective(config, df):
    equity_final = launcher_backtest(config, df)
    train.report({"equity_final":equity_final})
    
search_space = {
    "crypto_currency": "ADAUSDT",
    "exchange": "binance",
    "time": "1d",
    "strategy": {
        "ratio_method": tune.choice(["ma_ratio", "ema_ratio", "zlema_ratio", "dema_ratio", "tema_ratio"]),
        "x": tune.randint(1, 34),
        "y": tune.randint(35, 120),
        "name": "momentum",
    }
}


def _run(config):
    algo = OptunaSearch()
    os.makedirs("experience/raytune", exist_ok=True)
    os.makedirs(rf"experience/raytune/{config['exchange']}-{config['crypto_currency']}-{config['time']}", exist_ok=True)
    df = open_dataframe(rf"data/{config['exchange']}-{config['crypto_currency']}-{config['time']}.pkl")
    df.set_index("open_date", inplace=True)
    df.columns = [col.capitalize() for col in df.columns]
    
    study_name = (
        "study"
        + f"_{config['strategy']['name']}_{config['crypto_currency']}_{config['exchange']}_{config['time']}_raytune"
    )
    algo.save(f"raytune/{study_name}")
    objective_with_df = tune.with_parameters(objective, df=df)

    tuner = tune.Tuner(
        objective_with_df,
        tune_config=tune.TuneConfig(
            metric='equity_final',
            mode="max",
            search_alg=algo,
            num_samples=500,
        ),
        param_space=search_space,
    )
    results = tuner.fit()

config = {
    "crypto_currency": "ADAUSDT",
    "exchange": "binance",
    "time": "1d",
    "strategy": {
        "ratio_method": "ema_ratio",
        "x": 4,
        "y": 85,
        "name": "momentum",
    }
}

_run(config)
