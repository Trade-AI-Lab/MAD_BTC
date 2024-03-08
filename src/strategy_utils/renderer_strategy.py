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

def monthly_return(df_evolution:pd.DataFrame,verbose=False,saving=False,path="./"):
    import plotly.graph_objects as go
    import plotly.subplots as sp
    df_evolution.set_index('Time', inplace=True)
    df_monthly = df_evolution.resample('w')
    df_monthly = df_monthly.ffill()
    df_monthly['MonthlyPctChange'] = df_monthly['Portfolio'].pct_change() * 100 # (df_monthly['Portfolio'].pct_change() * 100 - df_monthly['Buy_and_Hold'].pct_change() * 100) if df_monthly['Buy_and_Hold'].pct_change() * 100>0 else   
    # Create a new column for color based on monthly percentage change
    df_monthly['Color'] = ['green' if x > 0 else 'red' for x in df_monthly['MonthlyPctChange']]
    # Plotting with Plotly
    fig = sp.make_subplots(rows=1, cols=2, column_widths=[0.7, 0.3], shared_yaxes=True)

    # Add bar trace to the first subplot
    fig.add_trace(go.Bar(
        x=df_monthly.index,
        y=df_monthly['MonthlyPctChange'],
        marker_color=df_monthly['Color'],
        text=df_monthly['MonthlyPctChange'],
        textposition='outside',  # Display text outside the bar
        textfont=dict(size=10),  # Adjust text font size
    ), row=1, col=1)
    
    # Add histogram trace to the second subplot
    fig.add_trace(go.Box(
        y=df_monthly['MonthlyPctChange'],
        marker_color='blue',
        boxpoints='all',  # Display all data points
        jitter=0.3,  # Add jitter for better visualization
        pointpos=-1.8,  # Adjust the position of the data points
    ), row=1, col=2)

    # Customize layout
    fig.update_layout(
        title='Monthly Percentage Change and Distribution of Portfolio',
        xaxis_title='Time',
        yaxis_title='Monthly Percentage Change (%)',
        showlegend=False,
    )


    # Show the plot
    if verbose:
        fig.show()
    if saving:
        fig.write_image(path+".png")

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
    PATH = r"C:\Users\alexw\Documents\Git\MAD_BTC\experience\BNBUSDT\binance\4h\momentum_strategy\2a90a504e84d572a958dff33f59f8f26ba7cdb598f4c23d476439ecbd7fa1522\output_file.csv"
    a = pd.read_csv(PATH)
    a["Time"] =  pd.to_datetime(a["Time"])
    monthly_return(a,verbose=True)