import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
df = pd.read_csv("https://github.com/blockhain-elite-labs/harmonia-v20.1-82-backtests/raw/main/harmonia-20.1-82-2024-market-9months.zip")

print(df.columns)
df["open_date"] = pd.to_datetime(df["open_date"])
# Améliorer la stratégie par exemple en phase haussière éviter de prendre des shorts (Utiliser des EMA)
# Faire belle présentation, améliorer Notebook (sexy) rendre ergonomique qu'ils trouvent eux-mêmes!

def build_ema(df:pd.DataFrame, column_name:str, period:int)->None:
    ema_name = f"ema_{column_name}_{period}"
    df[ema_name] = df[column_name].ewm(span=period, adjust=False).mean()
    return ema_name

one_min_delay = False

if one_min_delay:
    offset_time = '1min'
    df_1min = pd.read_csv(r"D:\ressources_v3\2_READY_candles\BINANCE-ONLY\1m\ETHUSDT-1m-without-aggr.zip")
    df_1min["open_date"] = pd.to_datetime(df_1min["open_date"])
    max_date = "2024-06-28 00:00:00"
    min_date = df["open_date"].iloc[0]
    df_1min = df_1min[(df_1min["open_date"]<=max_date) & (df_1min["open_date"]>=min_date)]
    df_1min = df_1min.reset_index(drop=True)
    df_1min.set_index('open_date', inplace=True)
    # Resample and aggregate the required columns with an offset
    df_5min = df_1min.resample('5T', offset=offset_time).agg({
        'open_price': 'first',  # First value in the 5-minute window
        'high': 'max',          # Maximum value in the 5-minute window
        'low': 'min',           # Minimum value in the 5-minute window
        'close': 'last',        # Last value in the 5-minute window
        # Add more columns and aggregation methods if necessary
    })
    del df_1min
    df_5min.reset_index(inplace=True)
    df_5min = df_5min[(df_5min["open_date"]<=max_date) & (df_5min["open_date"]>min_date)]
    df_5min = df_5min[['open_date', 'open_price', 'high', 'low', 'close']]
    df = df[(df["open_date"]<max_date) & (df["open_date"]>=min_date)]
    df_5min.reset_index(inplace=True)
    df.reset_index(inplace=True)
    df_5min["harmonia_prediction"] = df["harmonia_prediction"]
    
if one_min_delay:
    df = df_5min.copy() 
    del df_5min 
df.rename(columns={"harmonia_prediction":"harmonia_prediction_2024"}, inplace=True)
df = df.set_index("open_date")
df.columns = [col.capitalize() for col in df.columns]
df.rename(columns={
    "Open_price": "Open"
}, inplace=True)




def _backtesting_process(config, df):
    # Initialize variables
    positions = []
    trades = []
    saved = config["SAVED"] if "SAVED" in config.keys() else False
    ema_column = build_ema(df, "Open", config["EMA"]["period"]) if config["EMA"]["period"]>0 else False
    commission = config["comission"]
        
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
        if ema_column:
            ema_value = row[ema_column]

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
            if not ema_column:
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
            elif price>ema_value: # si on a une colonne ema et si le prix est supérieur à l'EMA <=> phase haussière on obéit
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
            if not ema_column:
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
            elif price<ema_value:
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
            else:
                pass
    if not trades:
        # If no trades were made, return empty results
        return pd.Series(dtype='float64'), 0, pd.Series(dtype='float64'), config
    
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
                "entry_price": trade['entry_price'],
                "exit_price":trade['exit_price'],
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
    trade_df['month'] = trade_df['date'].dt.to_period('W')
    # Calculate cumulative returns per month
    monthly_returns = trade_df.groupby('month')['trade_return'].sum().cumsum()
    return monthly_returns, cumulative_percentage, trade_df, config

def process_config(args):
    tp_value, sl_value, trigger, overtrend, hours, period = args
    # Example configuration
    config = {
        "TP": tp_value,   # Take Profit (1%)
        "SL": sl_value,  # Stop Loss (0.5%)
        "TRIGGER": trigger,
        "OVER_TREND": overtrend,
        "TIME_LIMIT": datetime.timedelta(hours=hours).total_seconds(),
        "EMA":{
            "period":period
            },# Time limit of 8 hours in seconds
        "SAVED":False,
        "comission":0.01/100
    }
    return _backtesting_process(config, df)
    

def plot_one_backtest(hours:int, commission:float, config:dict):
    """
    plot_one_backtest _summary_
    
    Example:
    hours  = 8
    commission = 0.01/100
    config = {
        "TP": 0.01,   # Take Profit (1%)
        "SL": 0.002,  # Stop Loss (0.5%)
        "TRIGGER": 0.1,
        "OVER_TREND": 0.3,
        "TIME_LIMIT": datetime.timedelta(hours=hours).total_seconds()  # Time limit of 8 hours in seconds
    }
    plot_one_backtest(hours, commission, config)

    Args:
        hours (int): window size before we end the trade
        commission (float): comission is broker dependant
        config (dict): dictionnary containing the configuration of the strategy
    """
    import plotly.graph_objects as go
    fig = go.Figure()
    monthly_returns, cumulative_percentage, trade_df, config = _backtesting_process(config, df)
    print(f"Cumulative Percentage Returns:{cumulative_percentage}")
    print(f"Nombre de Trades:{len(trade_df)}")
    fig.add_trace(go.Scatter(
        x=monthly_returns.index.to_timestamp(), 
        y=monthly_returns.values, 
        mode='lines+markers', 
        name=f"TP_{config['TP']} ,SL_{config['SL']} ,Trigger_{config['TRIGGER']} ,OverTrend_{config['OVER_TREND']} Cumulative Return"
    ))
    fig.update_layout(title=f"Cumulative Return per Week, TP_{config['TP']} ,SL_{config['SL']} ,Trigger_{config['TRIGGER']} ,OverTrend_{config['OVER_TREND']}, Commission_{commission}, Time duration max_{hours}h",
                        xaxis_title='Month',
                        yaxis_title='Cumulative Return (%)',
                        xaxis=dict(type='date'),
                        yaxis=dict(title='Cumulative Return (%)'),
                        template='plotly_dark')
    fig.show("browser")

def main():
    import multiprocessing as mp
    """
    tp_value = [0.001,0.0025,0.005,0.01,0.015,0.02,0.03,0.04]
    sl_value = [0.001,0.002,0.005,0.0075, 0.01, 0.015]
    trigger = [0.05, 0.08, 0.1, 0.11, 0.12, 0.13, 0.14]
    overtrend_value = [0.15,0.16,0.17,0.18,0.19,0.20,0.22,0.24,0.26,0.28,0.3]
    hours_value = [2,4,6,8]
    ema_period_value = [0,1,5,30,50]
    """
    tp_value = [0.01]
    sl_value = [0.001]
    trigger = [0.05,0.08]
    overtrend_value = [0.15,0.16]
    hours_value = [2]
    ema_period_value = [0]
    
    # Create a list of all combinations of tp_value and sl_value
    args_list = [ (tp,sl,tr,overtrend, hours,period_value) for tp in tp_value for sl in sl_value for tr in trigger for overtrend in overtrend_value for hours in hours_value for period_value in ema_period_value]

    # Create a pool of processes
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(process_config, args_list)
    #results = process_config(args_list[0])
    list_config = []
    for result in results:
        if result is not None:
            _, cumulative_percentage, _, config = result
            config["cumulative_percentage"] = cumulative_percentage
            list_config.append(config)
    df = pd.DataFrame(list_config)
    df.to_csv("list_configs_cumulative_percentage.csv")
    
if __name__=="__main__":
    hours  = 8
    commission = 0.01/100
    config = {
        "TP": 0.01,   # Take Profit (1%)
        "SL": 0.002,  # Stop Loss (0.5%)
        "TRIGGER": 0.1,
        "OVER_TREND": 0.3,
        "TIME_LIMIT": datetime.timedelta(hours=hours).total_seconds(),  # Time limit of 8 hours in seconds
        "EMA":{
            "period":0,
        },
        "SAVED":False,
        "comission":commission
    }
    #plot_one_backtest(hours, commission, config)
    main()