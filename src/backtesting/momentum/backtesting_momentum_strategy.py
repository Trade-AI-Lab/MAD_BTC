from backtesting import Strategy
from backtesting import Backtest
import numpy as np
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
    return (indicator_momentum.general_ema_ratio(data=df,x = config["strategy"]["x"], 
                                                 y=config["strategy"]["y"], column_name="Open", 
                                                 order_x = config['strategy']['order_x'],order_y = config['strategy']['order_y'])-1).values
    
    
class GeneralEMaRatio(Strategy):
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



def launcher_backtest(trial, df,config):
    config["strategy"]["x"] = trial.suggest_int("x", 1, 34)  # hyperparameters
    config["strategy"]["y"] = trial.suggest_int(
        "y", 35, 250
    )  # hyperparameters
    config["strategy"]["order_x"] = trial.suggest_int(
        "order_x", 1,10)
    config["strategy"]["order_y"] = trial.suggest_int(
        "order_y", 1,10)
    GeneralEMaRatio.config = config
    bt = Backtest(df, GeneralEMaRatio, cash=100000, commission=0.1/100)
    stats = bt.run()
    if stats.loc['Buy & Hold Return [%]']< stats.loc['Return [%]'] and stats.loc['Equity Final [$]']==stats.loc['Equity Peak [$]']:
        bt.plot(open_browser=False,filename=rf"experience/backtest/{config['exchange']}-{config['crypto_currency']}-{config['time']}/method_{config['strategy']['ratio_method']}_x_{config['strategy']['x']}_y_{config['strategy']['y']}_orderx_{config['strategy']['order_x']}_ordery_{config['strategy']['order_y']}")
    return (stats.loc['Return [%]']-stats.loc['Buy & Hold Return [%]'])/stats.loc['Buy & Hold Return [%]']*100*1/(1+np.exp(-stats.loc['Max. Drawdown [%]']/100))

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
            multivariate=True, group=True, constant_liar=True, n_startup_trials=100
        ),
        study_name=study_name,
        direction="maximize",
        load_if_exists="True",
    )  # study created else we reload the past strategy
    study.optimize(
        lambda trial, : launcher_backtest(trial, df, config), n_trials=300
    )
config = {
    "crypto_currency":"BNBUSDT",
    "exchange":"binance",
    "time":"15m",
    
        "strategy": {
            "ratio_method": "general_ema_ratio",  # Ensure this method exists in indicator_momentum
            "x": 4,
            "y": 85,
            "order_x":5,
            "order_y":5,
            "name":"momentum",
        }
    }

list_cryptocurrencies = ["MATICUSDT","OCEANUSDT","TRXUSDT","XRPUSDT"]
for crypto_currency in list_cryptocurrencies:
    config["crypto_currency"] = crypto_currency
    _run(config)
"""
if __name__=="__main__":
    config = {
    "crypto_currency":"BTCUSDT",
    "exchange":"binance",
    "time":"4h",
    
        "strategy": {
            "ratio_method": "general_ema_ratio",  # Ensure this method exists in indicator_momentum
            "x": 14,
            "y": 78,
            "order_x":1,
            "order_y":1,
            "name":"momentum",
        }
    }
    df = open_dataframe(rf"data/{config['exchange']}-{config['crypto_currency']}-{config['time']}.pkl")
    os.makedirs("experience/backtest", exist_ok=True)
    os.makedirs(rf"experience/backtest/{config['exchange']}-{config['crypto_currency']}-{config['time']}", exist_ok=True)
    df.set_index("open_date")
    df.columns = [col.capitalize() for col in df.columns]
    GeneralEMaRatio.config = config
    bt = Backtest(df, GeneralEMaRatio, cash=100000, commission=0.1/100)
    stats = bt.run()
    print(stats)
    bt.plot(open_browser=True,resample=False)
"""