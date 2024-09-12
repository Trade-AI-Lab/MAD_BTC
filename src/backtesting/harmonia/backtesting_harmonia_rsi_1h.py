import pandas as pd
import sys, os
absolute_path = os.path.abspath(__file__)[
    : os.path.abspath(__file__).find("MAD_BTC") + len("MAD_BTC")
]
sys.path.append(absolute_path)
from backtesting.lib import TrailingStrategy
from backtesting import Backtest
import optuna

def RSI(array, n):
    """Relative strength index"""
    # Approximate; good enough
    gain = pd.Series(array).diff()
    loss = gain.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    rs = gain.ewm(n).mean() / loss.abs().ewm(n).mean()
    return 100 - 100 / (1 + rs)

class HarmoniaStrategy(TrailingStrategy):
    config = {
        "OVER_TREND":0.27,
        "TRIGGER":0.20,
        "TP":5.5,
        "SL":3.5,
        "RSI":10
    }
    def init(self):
        super().init()
        self.bull_block = False
        self.bear_block = False
        self.set_trailing_sl(self.config["SL"])
        self.set_trailing_tp(self.config["TP"])
        self.rsi = self.I(RSI, self.data.Open_1h, self.config["RSI"])
        

    def next(self):
        super().next()
        harmonia_signal = self.data.Harmonia_prediction[-1]
        if harmonia_signal > self.config["OVER_TREND"]:
            self.bull_block = True
        elif harmonia_signal< -self.config["OVER_TREND"]:
            self.bear_block = True
        if harmonia_signal< self.config["TRIGGER"]:
            self.bull_block = False
        elif harmonia_signal > -self.config["TRIGGER"]:
            self.bear_block = False
        if harmonia_signal > self.config["TRIGGER"] and not self.bull_block and not self.position.is_long and self.rsi[-1]<self.config["RSI_SEUIL"]:
            self.position.close()
            self.buy()
        elif harmonia_signal < -self.config["TRIGGER"] and not self.bear_block and not self.position.is_short and self.rsi[-1]>100-self.config["RSI_SEUIL"]:
            self.position.close()
            self.sell()
"""
df = pd.read_csv("data/harmonia-20.1-82-2022-market-23months.csv.zip")

df["open_date"] = pd.to_datetime(df["open_date"])
df.columns = [col.capitalize() for col in df.columns]
df.rename(columns={
    "Open_date": "Date",
    "Open_price": "Open"
}, inplace=True)

# Set the correct index
df["Date"] = pd.to_datetime(df["Date"])

df.set_index("Date", inplace=True)
df = df.fillna(method='ffill')  # Forward fill NaNs
bt = Backtest(df, HarmoniaStrategy, cash=10000, commission=0.4/100)
stats = bt.run()
bt.plot(resample=False)
"""
config = {
    "OVER_TREND":0.27,
    "TRIGGER":0.2,
    "TP":1,
    "SL":1,
    "RSI":10,
    "RSI_SEUIL":20,
}
def launcher_backtest(trial, df,config):
    config["OVER_TREND"] = 0.01*trial.suggest_int("OVER_TREND", 21, 40)  # hyperparameters
    config["TRIGGER"] = 0.01*trial.suggest_int("TRIGGER", 1, 20)
    config["TP"] = 0.1*trial.suggest_int("TP", 1, 50)
    config["SL"] = 0.1*trial.suggest_int("SL", 1, 50)
    config["RSI"] = trial.suggest_int("RSI", 1, 50)
    config["RSI_SEUIL"] = trial.suggest_categorical("RSI_SEUIL",[15,20,25,30,35,40])
    HarmoniaStrategy.config = config
    bt = Backtest(df, HarmoniaStrategy, cash=10000, commission=0.4/100)
    stats = bt.run()
    if stats.loc['Buy & Hold Return [%]']< stats.loc['Return [%]']: #and stats.loc['Equity Final [$]']*2>stats.loc['Equity Peak [$]']:
        filename = rf"experience/Harmonia_TP_SL_RSI_1h/Harmonia_{config['OVER_TREND']}_{config['TRIGGER']}_{config['TP']}_{config['SL']}_{config['RSI']}_{config['RSI_SEUIL']}_1h"
        bt.plot(open_browser=False, filename=filename, resample=False)
    return stats.loc['Equity Final [$]']

def _run(config):
    df = pd.read_csv("data/harmonia-20.1-82-2022-market-23months.csv.zip")

    df["open_date"] = pd.to_datetime(df["open_date"])
    df.columns = [col.capitalize() for col in df.columns]
    df.rename(columns={
        "Open_date": "Date",
        "Open_price": "Open"
    }, inplace=True)

    # Set the correct index
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df["Open_1h"] = df['Open'].resample('1h').first().ffill()
    df = df.fillna(method='ffill')  # Forward fill NaNs
    df = df.dropna()
    
    os.makedirs("experience/Harmonia_TP_SL_RSI_1h", exist_ok=True)
   
    study_name = (
        "study"
        + f"_Harmonia_TP_SL_RSI_1h_backtest"
    )  # the study_name depending on the strategy name, crypto,exchange and time
    storage_name = f"sqlite:///{os.path.join(absolute_path, 'study', study_name)}.db"
    study = optuna.create_study(
        storage_name,
        sampler=optuna.samplers.TPESampler(
            multivariate=True, group=True, constant_liar=True, n_startup_trials=10
        ),
        study_name=study_name,
        direction="maximize",
        load_if_exists="True",
    )  # study created else we reload the past strategy
    study.optimize(
        lambda trial, : launcher_backtest(trial, df, config), n_trials=500
    )
_run(config)
