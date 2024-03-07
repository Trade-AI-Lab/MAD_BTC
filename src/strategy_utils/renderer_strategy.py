import matplotlib.pyplot as  plt
import pandas as pd
def core_img(evolution_df:dict):
    plt.figure(figsize=(10, 6))
    plt.plot(evolution_df["Time"], evolution_df['Portfolio'], label='Portfolio', marker='o')
    plt.plot(evolution_df["Time"], evolution_df['Buy_and_Hold'], label='Buy and Hold', marker='x')
    plt.xlabel('Timesteps')
    plt.ylabel('Value')
    plt.title('Portfolio vs Buy and Hold Comparison')
    plt.legend()
    plt.grid(True)

def save(title_image:str,strategy_folder_path:str,evolution_df:dict):
    core_img(evolution_df)
    evolution_df.to_csv(strategy_folder_path+"/"+'output_file.csv', index=False) # Can  cause an error
    plt.savefig(strategy_folder_path+"/"+title_image+".png")
    plt.close()

def renderer(evolution_df:dict,verbose:bool=False):
    core_img(evolution_df)
    if verbose:
        plt.show()
        plt.pause(20)
    plt.close()

def monthly_return(df_evolution:pd.DataFrame):
    import plotly.graph_objects as go
    df_evolution.set_index('Time', inplace=True)
    df_monthly = df_evolution.resample('M')
    df_monthly = df_monthly.ffill()
    df_monthly['MonthlyPctChange'] = df_monthly['Portfolio'].pct_change() * 100
    # Create a new column for color based on monthly percentage change
    df_monthly['Color'] = ['green' if x > 0 else 'red' for x in df_monthly['MonthlyPctChange']]
    # Plotting with Plotly
    fig = go.Figure()

    # Add bar trace
    fig.add_trace(go.Bar(
        x=df_monthly.index,
        y=df_monthly['MonthlyPctChange'],
        marker_color=df_monthly['Color'],
    ))

    # Customize layout
    fig.update_layout(
        title='Monthly Percentage Change of Portfolio',
        xaxis_title='Time',
        yaxis_title='Monthly Percentage Change (%)',
        showlegend=False,
    )

    # Show the plot
    fig.show()

if __name__=="__main__":
    config = {
                    "crypto_currency":"ETHUSDT",
                    "time":"4h",
                    "exchange":"binance",
                    "x":12,
                    "y":90,
                    "ratio_method":"ema_ratio",
                    "initial_capital":1000
                    
    }
    a = pd.read_csv(r"C:\Users\alexw\Documents\Git\MAD_BTC\experience\BTCUSDT\binance\1d\momentum_strategy\1a5a3cdad501941507dbdbbf6b9cc6aaf22482e21a2c7f08f95e35b129ff1cf7\output_file.csv")
    a["Time"] =  pd.to_datetime(a["Time"])
    monthly_return(a)