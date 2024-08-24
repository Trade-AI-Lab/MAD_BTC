
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
commission = 0.01/100


def _backtesting_process(config, df, saved=False):
    # Initialize variables
    positions = []
    trades = []
    percentage_returns = [0]  # Start with 0% return
    date_history = []

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

    for trade in trades:
        if trade['type'] == 'long':
            trade_return = ((1-commission)*trade['exit_price'] - (1+commission)*trade['entry_price']) / trade['entry_price'] * 100
        else:  # short
            trade_return = ((1-commission)*trade['entry_price'] - (1+commission)*trade['exit_price']) / trade['entry_price'] * 100
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
        

    trade_df = pd.DataFrame(trade_data)
    if saved:
        trade_df.to_csv(f"trades_harmonia_automatic_tp_{config['TP']}_sl_{config['SL']}_TRIGGER_{config['TRIGGER']}_OVERTREND_{config['OVER_TREND']}_TIMELIMIT_{hours}h_trading_comission_{commission}.csv")
    trade_df['date'] = pd.to_datetime(trade_df['trade_entry_time'])
    trade_df['month'] = trade_df['date'].dt.to_period('M')

    # Calculate cumulative returns per month
    monthly_returns = trade_df.groupby('month')['trade_return'].sum().cumsum()
    return trade_df, monthly_returns, cumulative_percentage

import plotly.graph_objects as go
import multiprocessing as mp
import datetime

def process_config(args):
    tp_value, sl_value, df, hours = args
    # Example configuration
    config = {
        "TP": tp_value,   # Take Profit (1%)
        "SL": sl_value,  # Stop Loss (0.5%)
        "TRIGGER": tp_value * 10,
        "OVER_TREND": sl_value * 100,
        "TIME_LIMIT": datetime.timedelta(hours=hours).total_seconds()  # Time limit of 8 hours in seconds
    }
    trade_df, monthly_returns, cumulative_percentage = _backtesting_process(config, df)
    if cumulative_percentage > 0:
        return monthly_returns.index.to_timestamp(), monthly_returns.values, config
    return None

def main():
    hours = 8  # example time limit in hours

    # Create a list of all combinations of tp_value and sl_value
    args_list = [(tp_value / 100, sl_value / 1000, df, hours) for tp_value in range(1, 35) for sl_value in range(1, 35)]

    # Create a pool of processes
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(process_config, args_list)
        print("Done")
    fig = go.Figure()
    # Process the results
    for result in results:
        if result is not None:
            x_values, y_values, config = result
            fig.add_trace(go.Scatter(
                x=x_values, 
                y=y_values, 
                mode='lines+markers', 
                name=f"TP_{config['TP']} ,SL_{config['SL']} ,Trigger_{config['TRIGGER']} ,OverTrend_{config['OVER_TREND']} Cumulative Return"
            ))

    fig.update_layout(title='Cumulative Return per Month',
                        xaxis_title='Month',
                        yaxis_title='Cumulative Return (%)',
                        xaxis=dict(type='date'),
                        yaxis=dict(title='Cumulative Return (%)'),
                        template='plotly_dark')
    fig.show("browser")
    
if __name__ == '__main__':
    main()