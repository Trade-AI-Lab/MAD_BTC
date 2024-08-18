
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
df = pd.read_csv("https://github.com/blockhain-elite-labs/harmonia-v20.1-82-backtests/raw/main/harmonia-20.1-82-2024-market-9months.zip")
print(df.columns)
df["open_date"] = pd.to_datetime(df["open_date"])
df.rename(columns={"harmonia_prediction":"harmonia_prediction_2024"}, inplace=True)
df = df.set_index("open_date")
df.columns = [col.capitalize() for col in df.columns]
df.rename(columns={
    "Open_price": "Open"
}, inplace=True)
hours  = 8
commission = 0.1/100
# Example configuration
config = {
    "TP": 0.01,   # Take Profit (1%)
    "SL": 0.005,  # Stop Loss (0.5%)
    "TRIGGER": 0.14,
    "OVER_TREND": 0.3,
    "TIME_LIMIT": datetime.timedelta(hours=hours).total_seconds()  # Time limit of 8 hours in seconds
}

# Initialize variables
positions = []
trades = []
percentage_returns = [0]  # Start with 0% return
date_history = []

# Assuming df is already loaded with the necessary columns
# Example: df = pd.read_csv('your_data.csv', parse_dates=True, index_col='Date')
def save(index, price, position, trades):
    position['status'] = 'closed'
    position['exit_time'] = index
    position['exit_price'] = price
    trades.append(position)
    positions.remove(position)

for index, row in df.iterrows():
    price = row['Open']
    high = row['High']
    low = row['Low']
    harmonia_signal = row['Harmonia_prediction_2024']

    # Check for open positions and update their status
    for position in positions[:]:
        trade_duration = (index - position['entry_time']).total_seconds()

        # Check for time limit
        if trade_duration >= config['TIME_LIMIT']:
            save(index=index, price=price,position=position, trades=trades)
            continue

        # Check for stop loss or take profit
        if position['type'] == 'long':
            if low <= position['sl']:
                save(index=index, price=position['sl'] ,position=position, trades=trades)
            elif high >= position['tp']:
                save(index=index, price=position['tp'] ,position=position, trades=trades)
            
            
        elif position['type'] == 'short':
            if high >= position['sl']:
                save(index=index, price=position['sl'] ,position=position, trades=trades)
            elif low <= position['tp']:
                save(index=index, price=position['tp'] ,position=position, trades=trades)

    # Open new positions based on Harmonia signal
    if harmonia_signal >= config["TRIGGER"] and harmonia_signal < config["OVER_TREND"]:
        sl = (1 - config["SL"]) * price
        tp = (1 + config["TP"]) * price
        positions.append({
            'type': 'long',
            'entry_time': index,
            'entry_price': price,
            'sl': sl,
            'tp': tp,
            'status': 'open'
        })
    elif harmonia_signal <= -config["TRIGGER"] and harmonia_signal > -config["OVER_TREND"]:
        sl = (1 + config["SL"]) * price
        tp = (1 - config["TP"]) * price
        positions.append({
            'type': 'short',
            'entry_time': index,
            'entry_price': price,
            'sl': sl,
            'tp': tp,
            'status': 'open'
        })
trade_data = []
cumulative_percentage = 0
# Print trade results
for trade in trades:
    if trade['type'] == 'long':
        if (trade['exit_price'] - trade['entry_price'])>=0:
            trade_return = (trade['exit_price'] - trade['entry_price']) / trade['entry_price'] * 100 * (1-commission)
        else:
            trade_return = (trade['exit_price'] - trade['entry_price']) / trade['entry_price'] * 100 * (1+commission)
    else:  # short
        if (trade['entry_price'] - trade['exit_price'])>=0:
            trade_return = (trade['entry_price'] - trade['exit_price']) / trade['entry_price'] * 100 * (1-commission)
        else:
            trade_return = (trade['entry_price'] - trade['exit_price']) / trade['entry_price'] * 100 * (1+commission)
    cumulative_percentage += trade_return
    trade_data.append(
        {
            'trade_entry_time': trade['entry_time'],
            "trade_return":trade_return,
            "trade_type":trade['type'],
            "trade_exit_time":trade['exit_time'],
            "trade_tp":trade["tp"],
            "trade_sl":trade["sl"],
            "trade_return":trade_return
                }
        )
    print(f"Trade on {trade['entry_time'].date()}: {trade['type']} | Entry: {trade['entry_price']} | Exit: {trade['exit_price']} | Return: {trade_return}%")

print(f"Cumulative Percentage: {cumulative_percentage:.2f}%")
buy_and_hold = (df["Open"].iloc[-1]-df["Open"].iloc[0])/df["Open"].iloc[-1]*100
print(f"Buy and Hold Percentage: {buy_and_hold:.2f}%")

trade_df = pd.DataFrame(trade_data)
trade_df.to_csv(f"trades_harmonia_automatic_tp_{config['TP']}_sl_{config['SL']}_TRIGGER_{config['TRIGGER']}_OVERTREND_{config['OVER_TREND']}_TIMELIMIT_{hours}h_trading_comission_{commission}.csv")
trade_df['date'] = pd.to_datetime(trade_df['trade_entry_time'])
trade_df['month'] = trade_df['date'].dt.to_period('M')

# Calculate cumulative returns per month
monthly_returns = trade_df.groupby('month')['trade_return'].sum().cumsum()
import plotly.graph_objects as go
fig = go.Figure()

fig.add_trace(go.Scatter(x=monthly_returns.index.to_timestamp(), y=monthly_returns.values,
                         mode='lines+markers', name='Cumulative Return'))

fig.update_layout(title='Cumulative Return per Month',
                  xaxis_title='Month',
                  yaxis_title='Cumulative Return (%)',
                  xaxis=dict(type='date'),
                  yaxis=dict(title='Cumulative Return (%)'),
                  template='plotly_dark')

fig.show()