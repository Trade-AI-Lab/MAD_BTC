from backtesting import Strategy
from backtesting import Backtest
import pandas as pd
import optuna
import sys, os
absolute_path = os.path.abspath(__file__)[
    : os.path.abspath(__file__).find("MAD_BTC") + len("MAD_BTC")
]
sys.path.append(absolute_path)
from src.strategy.momentum import indicator_momentum
from src.utils.utils_df import open_dataframe

def function(df,config):
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
            if self.ratio>=0 and self.position.is_short:
                self.position.close()
                self.buy()

            elif self.ratio<=0 and self.position.is_long:
                self.position.close()

                self.sell()
        elif not self.position.is_long and not self.position.is_short:
            if self.ratio>=0:
                self.buy()
            elif self.ratio<=0:
                self.sell()

config = {
    "crypto_currency":"ETHUSDT",
    "exchange":"binance",
    "time":"1d",
    
        "strategy": {
            "ratio_method": "ema_ratio",  # Ensure this method exists in indicator_momentum
            "x": 4,
            "y": 85,
            "name":"momentum",
        }
    }

def launcher_backtest(trial, df,config):
    config["strategy"]["x"] = trial.suggest_int("x", 1, 34)  # hyperparameters
    config["strategy"]["y"] = trial.suggest_int(
        "y", 35, 120
    )  # hyperparameters
    config["strategy"]["ratio_method"] = trial.suggest_categorical(
        "ratio_method",
        ["ma_ratio", "ema_ratio", "zlema_ratio", "dema_ratio", "tema_ratio"],
    )
    MaRatio.config = config
    bt = Backtest(df, MaRatio, cash=1000, commission=0.1/100)
    stats = bt.run()
    if stats.loc['Buy & Hold Return [%]']< stats.loc['Return [%]']:# and stats.loc['Equity Final [$]']*2>stats.loc['Equity Peak [$]']:
        bt.plot(open_browser=False,filename=rf"experience/backtest/{config['exchange']}-{config['crypto_currency']}-{config['time']}/method_{config['strategy']['ratio_method']}_x_{config['strategy']['x']}_y_{config['strategy']['y']}")
    return stats.loc['Equity Final [$]']

def _run(config):
    df = open_dataframe(rf"data/{config['exchange']}-{config['crypto_currency']}-{config['time']}.pkl")
    os.makedirs("experience/backtest", exist_ok=True)
    os.makedirs(rf"experience/backtest/{config['exchange']}-{config['crypto_currency']}-{config['time']}", exist_ok=True)
    df.set_index("open_date")
    df.columns = [col.capitalize() for col in df.columns]
    study_name = (
        "study"
        + f"_{config['strategy']['name']}_{config['crypto_currency']}_{config['exchange']}_{config['time']}_backtest"
    )  # the study_name depending on the strategy name, crypto,exchange and time
    storage_name = f"sqlite:///{os.path.join(absolute_path, 'study', study_name)}.db"
    study = optuna.create_study(
        storage_name,
        sampler=optuna.samplers.TPESampler(
            multivariate=True, group=True, constant_liar=True, n_startup_trials=150
        ),
        study_name=study_name,
        direction="maximize",
        load_if_exists="True",
    )  # study created else we reload the past strategy
    study.optimize(
        lambda trial, : launcher_backtest(trial, df, config), n_trials=300
    )
_run(config)
