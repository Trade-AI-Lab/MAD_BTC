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

df = open_dataframe(r"data\binance-ADAUSDT-1d.pkl")
df.set_index("open_date")
df.columns = [col.capitalize() for col in df.columns]
config = {
        "strategy": {
            "ratio_method": "ema_ratio",  # Ensure this method exists in indicator_momentum
            "x": 7,
            "y": 94
        }
    }


def launcher_backtest(df,config):
    MaRatio.config = config
    bt = Backtest(df, MaRatio, cash=10_000, commission=0.1/100)
    stats = bt.run()
    print(stats)
    bt.plot()
launcher_backtest(df,config)
