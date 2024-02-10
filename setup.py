import pandas as pd
from src.utils.utils_df import open_dataframe
from ta.trend import SMAIndicator
import matplotlib.pyplot as plt

PATH = r"data/binance-BTCUSDT-5m.pkl"
df = open_dataframe(PATH)
# Function to calculate MA(x)/MA(y) for open prices
def ma_ratio(data, x, y):
    ma_x = SMAIndicator(data['open'], window=x).sma_indicator()
    ma_y = SMAIndicator(data['open'], window=y).sma_indicator()
    
    return (ma_x / ma_y)
x,y = 12 ,12*22
# Calculate MA(5)/MA(10) for open prices
df[f'ma_ratio_{x}_{y}'] = ma_ratio(df, x, y)


df = df.dropna().reset_index(drop=True)
# Print the resulting dataframe
# Initial capital
initial_capital = 1000

# Buy signal: If ma_ratio is greater than 1 and was less than or equal to 1 in the previous row
df['buy_signal'] = (df[f'ma_ratio_{x}_{y}'] > 1) & (df[f'ma_ratio_{x}_{y}'].shift(1) <= 1)

# Sell signal: If ma_ratio is less than 1 and was greater than or equal to 1 in the previous row
df['sell_signal'] = (df[f'ma_ratio_{x}_{y}'] < 1) & (df[f'ma_ratio_{x}_{y}'].shift(1) >= 1)

# Keep rows where either buy_signal or sell_signal is true
df_filtered = df[df['buy_signal'] | df['sell_signal']]

# Reset the index
df_filtered = df_filtered.reset_index(drop=True)
print(df_filtered["open_date"].head())
position = 0
portfolio = 1000
crypto = 0
list_portfolio = []
list_price = []
for i in range(len(df_filtered)):
    if df_filtered.iloc[i]['buy_signal'] and position != 1 and portfolio>0:
        position = 1
        crypto = portfolio / df_filtered.iloc[i]['open']

    elif df_filtered.iloc[i]['sell_signal'] and position != -1 and crypto>0:
        position = -1
        portfolio = crypto * df_filtered.iloc[i]['open']
        list_portfolio.append(portfolio)
        list_price.append(df_filtered.iloc[i]['open'])
    elif portfolio == 0 and crypto == 0:
        print("negative")

print(crypto,portfolio, 1000 / df.iloc[0]['open'])

evolution_df = pd.DataFrame({
    'Asset': list_price,
    'Portfolio': list_portfolio
})

# Plotting# Plotting with secondary y-axis
fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:red'
ax1.set_xlabel('Day')
ax1.set_ylabel('Asset Evolution', color=color)
ax1.plot(evolution_df.index, evolution_df['Asset'], color=color, marker='o')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Portfolio Evolution', color=color)
ax2.plot(evolution_df.index, evolution_df['Portfolio'], color=color, marker='x')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Asset vs Portfolio Evolution')
fig.tight_layout()
plt.show()



