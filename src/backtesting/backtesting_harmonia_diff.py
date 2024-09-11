from backtesting import Strategy
from backtesting import Backtest
import numpy as np
import pandas as pd
import optuna
import sys, os
absolute_path = os.path.abspath(__file__)[
    : os.path.abspath(__file__).find("MAD_BTC") + len("MAD_BTC")
]
def function(df,config):
    df["Harmonia_Open"] = (1-df["Harmonia_prediction_2024"]/100)*df["Open"]
    df["EMA_Harmonia_Open"] = df["Harmonia_Open"].ewm(span=config["EMA_Harmonia_Open"]).mean()
    df["EMA_Open"] = df["Open"].ewm(span=config["EMA_Open"]).mean()
    if config["EMA_Open"]>config["EMA_Harmonia_Open"]:
        df["Diff"] = df["EMA_Harmonia_Open"]/df["EMA_Open"]
    else:
        df["Diff"] = df["EMA_Open"]/df["EMA_Harmonia_Open"]
    return df["Diff"].values-1
    
    
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



def launcher_backtest(trial, df, config):
    config["EMA_Harmonia_Open"] = trial.suggest_int("x", 1, 500)  # hyperparameters
    config["EMA_Open"] = trial.suggest_int(
        "y", 1, 500
    )  # hyperparameters
    GeneralEMaRatio.config = config
    bt = Backtest(df, GeneralEMaRatio, cash=10000, commission=0.06/100)
    stats = bt.run()
    if stats.loc['Buy & Hold Return [%]']< stats.loc['Return [%]']:
        bt.plot(open_browser=False,resample=False,filename=rf"experience/backtest/{['EMA_Harmonia_Open']}_harmonia_{['EMA_Open']}_open_comission")
    return (stats.loc['Return [%]']-stats.loc['Buy & Hold Return [%]'])/stats.loc['Buy & Hold Return [%]']*100*1/(1+np.exp(-stats.loc['Max. Drawdown [%]']/100))

def _run(df, config):
    study_name = (
        "study"
        + f"{config['EMA_Harmonia_Open']}_harmonia_{config['EMA_Open']}_open_backtest"
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
        lambda trial, : launcher_backtest(trial, df, config), n_trials=400
    )

def load_df():
    df = pd.read_csv("https://github.com/blockhain-elite-labs/harmonia-v20.1-82-backtests/raw/main/harmonia-20.1-82-2024-market-9months.zip")
    df["open_date"] = pd.to_datetime(df["open_date"])
    df.rename(columns={"harmonia_prediction":"harmonia_prediction_2024"}, inplace=True)
    os.makedirs("experience/backtest", exist_ok=True)
    df = df.set_index("open_date")
    df.columns = [col.capitalize() for col in df.columns]
    df.rename(columns={
        "Open_price": "Open"
    }, inplace=True)
    return df

if __name__=="__main__":
    df = load_df()
    config = {
        "EMA_Harmonia_Open":200,
        "EMA_Open":100,
    }
    GeneralEMaRatio.config = config
    _run(df, config)