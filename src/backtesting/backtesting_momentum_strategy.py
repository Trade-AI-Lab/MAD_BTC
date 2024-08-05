from backtesting import Strategy
from backtesting import Backtest
import pandas as pd
import sys, os
absolute_path = os.path.abspath(__file__)[
    : os.path.abspath(__file__).find("MAD_BTC") + len("MAD_BTC")
]
sys.path.append(absolute_path)
from src.strategy.momentum import indicator_momentum
from src.utils.utils_df import open_dataframe

def function(df,config):
    return getattr(
            indicator_momentum, config["strategy"]["ratio_method"]
        )(df, config["strategy"]["x"], config["strategy"]["y"], column_name="Open").values
class MaRatio(Strategy):
    config = {
        "strategy": {
            "ratio_method": "SMA",  # Example method
            "x": 10,
            "y": 20
        }
    }
    def init(self):
        # Precompute the two moving averages
        self.ratio = self.I(function, self.data._Data__df, self.config)
    
    def next(self):
        if self.ratio>=1:
            self.position.close()
            self.buy()

        elif self.ratio<1:
            self.position.close()
            self.sell()

df = open_dataframe(r"data\binance-ADAUSDT-1d.pkl")
df.set_index("open_date")
df.columns = [col.capitalize() for col in df.columns]
config = {
        "strategy": {
            "ratio_method": "ema_ratio",  # Ensure this method exists in indicator_momentum
            "x": 10,
            "y": 20
        }
    }


def launcher_backtest(df,config):
    MaRatio.config = config
    bt = Backtest(df, MaRatio, cash=10_000, commission=.002)
    stats = bt.run()
    bt.plot()
launcher_backtest(df,config)
