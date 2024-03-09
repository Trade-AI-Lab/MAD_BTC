import matplotlib.pyplot as  plt
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp

def core_img(evolution_df:dict):
    """
    Core Image Plotting Function

    This function generates a core image plot comparing the portfolio value and Buy and Hold strategy over time.

    :param evolution_df: A dictionary containing time and portfolio values.
    :type evolution_df: dict
    """
    plt.figure(figsize=(10, 6))
    plt.plot(evolution_df["Time"], evolution_df['Portfolio'], label='Portfolio', marker='o')
    plt.plot(evolution_df["Time"], evolution_df['Buy_and_Hold'], label='Buy and Hold', marker='x')
    plt.xlabel('Timesteps')
    plt.ylabel('Value')
    plt.title('Portfolio vs Buy and Hold Comparison')
    plt.legend()
    plt.grid(True)

def save(title_image:str,strategy_folder_path:str,evolution_df:dict):
    """
    Save Core Image Plot and DataFrame to Files

    This function saves the core image plot and the DataFrame to CSV and PNG files.

    :param title_image: Title for the saved image file.
    :type title_image: str
    :param strategy_folder_path: Path to the folder where files will be saved.
    :type strategy_folder_path: str
    :param evolution_df: A dictionary containing time and portfolio values.
    :type evolution_df: dict
    """
    core_img(evolution_df)
    evolution_df.to_csv(strategy_folder_path+"/"+'output_file.csv', index=False) # Can  cause an error
    plt.savefig(strategy_folder_path+"/"+title_image+".png")
    plt.close()

def renderer(evolution_df:dict,verbose:bool=False):
    """
    Render Core Image Plot

    This function renders the core image plot, optionally displaying it for a brief period.

    :param evolution_df: A dictionary containing time and portfolio values.
    :type evolution_df: dict
    :param verbose: If True, display the plot and pause for 20 seconds. Defaults to False.
    :type verbose: bool, optional
    """
    core_img(evolution_df)
    if verbose:
        plt.show()
        plt.pause(20)
    plt.close()

def monthly_return(df_evolution:pd.DataFrame,verbose=False,saving=False,path="./"):
    """
    Generate Monthly Return Plot

    This function generates a plot showing the monthly percentage change in the portfolio and its distribution.

    :param df_evolution: DataFrame containing time-indexed portfolio values.
    :type df_evolution: pd.DataFrame
    :param verbose: If True, display the plot. Defaults to False.
    :type verbose: bool, optional
    :param saving: If True, save the plot as a PNG file. Defaults to False.
    :type saving: bool, optional
    :param path: Path to save the PNG file. Defaults to "./".
    :type path: str, optional
    """
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