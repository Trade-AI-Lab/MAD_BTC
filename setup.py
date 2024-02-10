import pandas as pd
from src.utils.utils_df import open_dataframe
from ta.trend import SMAIndicator
import matplotlib.pyplot as plt

PATH = r"data/binance-ETHUSDT-4h.pkl"
df = open_dataframe(PATH)
# Function to calculate MA(x)/MA(y) for open prices
def ma_ratio(data, x, y):
    ma_x = SMAIndicator(data['open'], window=x).sma_indicator()
    ma_y = SMAIndicator(data['open'], window=y).sma_indicator()
    
    return (ma_x / ma_y)
x,y = 12,90    #x,y = 23 ,160 1h # x,y = 12 ,12*22 5m
# Calculate MA(5)/MA(10) for open prices
df[f'ma_ratio_{x}_{y}'] = ma_ratio(df, x, y)


df = df.dropna().reset_index(drop=True)
# Print the resulting dataframe
# Buy and hold strategy

initial_capital = df['open'][0]

df['buy_and_hold'] = df['open'] / df['open'][0] * initial_capital

# Your trading strategy
df['buy_signal'] = (df[f'ma_ratio_{x}_{y}'] > 1) & (df[f'ma_ratio_{x}_{y}'].shift(1) <= 1)
df['sell_signal'] = (df[f'ma_ratio_{x}_{y}'] < 1) & (df[f'ma_ratio_{x}_{y}'].shift(1) >= 1)
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

    elif df_filtered.iloc[i]['sell_signal'] and position != -1 and crypto > 0:
        position = -1
        portfolio = crypto * df_filtered.iloc[i]['open'] * (1-fee_percentage)

    # Store portfolio values for comparison
    list_portfolio.append(portfolio)
    list_buy_hold.append(df_filtered['buy_and_hold'].iloc[i])

# Create a DataFrame with the evolution data
evolution_df = pd.DataFrame({
    'Portfolio': list_portfolio,
    'Buy_and_Hold': list_buy_hold
})

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(evolution_df.index, evolution_df['Portfolio'], label='Portfolio', marker='o')
plt.plot(evolution_df.index, evolution_df['Buy_and_Hold'], label='Buy and Hold', marker='x')
plt.xlabel('Day')
plt.ylabel('Value')
plt.title('Portfolio vs Buy and Hold Comparison')
plt.legend()
plt.grid(True)
plt.show()