import pandas as pd
from src.utils.utils_df import open_dataframe
from src.utils.utils_signature import create_signature
from src.utils.utils_json import save_dict_to_json
from ta.trend import SMAIndicator, EMAIndicator
import matplotlib.pyplot as plt
import os
import sys
absolute_path = os.path.abspath(__file__)[:os.path.abspath(__file__).find("MAD_BTC")+len("MAD_BTC")]
config = {
    "crypto_currency":"ETHUSDT",
    "time":"4h",
    "exchange":"binance",
    "x":12,
    "y":90,
    "ratio_method":"ema_ratio",
    "initial_capital":1000
    
}
signature_config = create_signature(config)

PATH = absolute_path + rf"/data/{config['exchange']}-{config['crypto_currency']}-{config['time']}.pkl"
df = open_dataframe(PATH)
# Function to calculate MA(x)/MA(y) for open prices
class indicator_momentum:
    @staticmethod
    def ma_ratio(data, x, y):
        ma_x = SMAIndicator(data['open'], window=x).sma_indicator()
        ma_y = SMAIndicator(data['open'], window=y).sma_indicator()
        return (ma_x / ma_y)

    @staticmethod
    def ema_ratio(data, x, y):
        ema_x = EMAIndicator(data['open'], window=x).ema_indicator()
        ema_y = EMAIndicator(data['open'], window=y).ema_indicator()
        return (ema_x / ema_y)



df[f'{config["ratio_method"]}_{config["x"]}_{config["y"]}'] = getattr(indicator_momentum,config["ratio_method"])(df, config["x"],config["y"])

df = df.dropna().reset_index(drop=True)
# Print the resulting dataframe
# Buy and hold strategy

initial_capital = 1000

df['buy_and_hold'] = df['open'] / df['open'][0] * initial_capital

# Your trading strategy
df['buy_signal'] = (df[f'{config["ratio_method"]}_{config["x"]}_{config["y"]}'] > 1) & (df[f'{config["ratio_method"]}_{config["x"]}_{config["y"]}'].shift(1) <= 1)
df['sell_signal'] = (df[f'{config["ratio_method"]}_{config["x"]}_{config["y"]}'] < 1) & (df[f'{config["ratio_method"]}_{config["x"]}_{config["y"]}'].shift(1) >= 1)
df_filtered = df[df['buy_signal'] | df['sell_signal']].reset_index(drop=True)

position = 0
portfolio = initial_capital
crypto = 0
list_portfolio = []
list_buy_hold = []
fee_percentage = 0.1 / 100  # 0.1%
# Iterate through the DataFrame
for i in range(len(df_filtered)):
    if df_filtered.iloc[i]['buy_signal'] and position != 1 and portfolio > 0:
        position = 1
        crypto = portfolio / (df_filtered.iloc[i]['open']*(1 + fee_percentage))
        portfolio = 0

    elif df_filtered.iloc[i]['sell_signal'] and position != -1 and crypto > 0:
        position = -1
        portfolio = crypto * df_filtered.iloc[i]['open'] * (1-fee_percentage)
        crypto = 0

        # Store portfolio values for comparison
        list_portfolio.append(portfolio)
        list_buy_hold.append(df_filtered['buy_and_hold'].iloc[i])

# Create a DataFrame with the evolution data
evolution_df = pd.DataFrame({
    'Portfolio': list_portfolio,
    'Buy_and_Hold': list_buy_hold
})


experience_folder_path = absolute_path + "/experience"
os.makedirs(experience_folder_path,exist_ok=True)
strategy_folder_path = experience_folder_path+f"/{signature_config}"
os.makedirs(strategy_folder_path,exist_ok=True)
save_dict_to_json(config, strategy_folder_path, 'strategy.json')



# Plotting
plt.figure(figsize=(10, 6))
plt.plot(evolution_df.index, evolution_df['Portfolio'], label='Portfolio', marker='o')
plt.plot(evolution_df.index, evolution_df['Buy_and_Hold'], label='Buy and Hold', marker='x')
plt.xlabel('Timesteps')
plt.ylabel('Value')
plt.title('Portfolio vs Buy and Hold Comparison')
plt.legend()
plt.grid(True)
plt.savefig(strategy_folder_path+"/"+f"{config['exchange']}-{config['crypto_currency']}-{config['time']}_portfolio_vs_buy_hold_{config['ratio_method']}_{config['x']}_{config['y']}_{initial_capital}$.png")
plt.show()