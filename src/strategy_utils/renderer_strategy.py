import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp


def core_img(evolution_df: dict, Title: str):
    """
    Core Image Plotting Function

    This function generates a core image plot comparing the portfolio value and Buy and Hold strategy over time.

    :param evolution_df: A dictionary containing time and portfolio values.
    :type evolution_df: dict
    """
    rsi = evolution_df[f"RSI_Portfolio/Buy_and_Hold"]
    RSI_Portfolio_Buy_and_Hold = round(rsi.mean(), 3)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(30, 6))
    time = evolution_df["Time"]
    portfolio_value = evolution_df["Portfolio"]
    buy_and_hold_value = evolution_df["Buy_and_Hold"]
    relative_portfolio_value = portfolio_value.pct_change()
    relative_buy_and_hold_value = buy_and_hold_value.pct_change()
    diff = relative_portfolio_value - relative_buy_and_hold_value
    value = round((diff).mean() * 100, 4)
    # Plot absolute values on the left subplot
    ax1.plot(time, portfolio_value, label="Portfolio", marker="o", color="b")
    ax1.plot(time, buy_and_hold_value, label="Buy and Hold", marker="x", color="g")
    ax1.set_xlabel("Timesteps")
    ax1.set_ylabel("Value")
    ax1.set_title("Portfolio vs Buy and Hold Comparison (Absolute Values)")
    ax1.legend()
    ax1.grid(True)

    # Plot percentage change values on the right subplot

    ax2.plot(
        time,
        relative_portfolio_value,
        label="Portfolio % Change",
        marker="o",
        color="r",
    )
    ax2.plot(
        time,
        relative_buy_and_hold_value,
        label="Buy and Hold % Change",
        marker="x",
        color="m",
    )
    ax2.set_xlabel("Timesteps")
    ax2.set_ylabel("Percentage Change")
    ax2.set_title(f"Portfolio vs Buy and Hold Comparison (% Change) {value}")
    ax2.legend()
    ax2.grid(True)

    # Plot percentage change values on the right subplot
    ax3.plot(time, diff, label="Difference", marker="o", color="r")
    ax3.set_xlabel("Timesteps")
    ax3.set_ylabel("Difference")
    ax3.set_title(f"Porfolio minus Hold Comparison")
    ax3.legend()
    ax3.grid(True)

    ax4.plot(
        time, rsi, label="RSI Portfolio vs Buy and Hold", marker="o", color="purple"
    )
    ax4.set_xlabel("Timesteps")
    ax4.set_ylabel("RSI value")
    ax4.set_title(f"RSI: {RSI_Portfolio_Buy_and_Hold}")
    ax4.legend()
    ax4.grid(True)
    fig.suptitle(Title)
    # Adjust layout
    plt.tight_layout()


def save(title_image: str, strategy_folder_path: str, evolution_df: dict):
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

    core_img(evolution_df, Title=title_image)
    evolution_df.to_csv(
        strategy_folder_path + "/" + "output_file.csv", index=False
    )  # Can  cause an error
    plt.savefig(strategy_folder_path + "/" + title_image + ".png")
    plt.close()


def renderer(evolution_df: dict, verbose: bool = False):
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
    plt.close()


def monthly_return(
    df_evolution: pd.DataFrame,
    verbose=False,
    saving=False,
    path="./",
    title_image="blank",
):
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
    df_evolution.set_index("Time", inplace=True)
    df_monthly = df_evolution.resample("W")
    df_monthly = df_monthly.ffill()
    df_monthly["MonthlyPctChange"] = (
        df_monthly["Portfolio"].pct_change() * 100
    )  # (df_monthly['Portfolio'].pct_change() * 100 - df_monthly['Buy_and_Hold'].pct_change() * 100) if df_monthly['Buy_and_Hold'].pct_change() * 100>0 else
    # Create a new column for color based on monthly percentage change
    df_monthly["Color"] = [
        "green" if x > 0 else "red" for x in df_monthly["MonthlyPctChange"]
    ]
    # Plotting with Plotly
    fig = sp.make_subplots(rows=1, cols=2, column_widths=[0.7, 0.3], shared_yaxes=True)

    # Add bar trace to the first subplot
    fig.add_trace(
        go.Bar(
            x=df_monthly.index,
            y=df_monthly["MonthlyPctChange"],
            marker_color=df_monthly["Color"],
            text=df_monthly["MonthlyPctChange"],
            textposition="outside",  # Display text outside the bar
            textfont=dict(size=10),  # Adjust text font size
        ),
        row=1,
        col=1,
    )

    # Add histogram trace to the second subplot
    fig.add_trace(
        go.Box(
            y=df_monthly["MonthlyPctChange"],
            marker_color="blue",
            boxpoints="all",  # Display all data points
            jitter=0.3,  # Add jitter for better visualization
            pointpos=-1.8,  # Adjust the position of the data points
        ),
        row=1,
        col=2,
    )

    # Customize layout
    fig.update_layout(
        title="Monthly Percentage Change and Distribution of Portfolio",
        xaxis_title="Time",
        yaxis_title="Monthly Percentage Change (%)",
        showlegend=False,
    )

    # Show the plot
    if verbose:
        fig.show()
    if saving:
        fig.write_image(path + "/" + title_image + ".png")


if __name__ == "__main__":
    PATH = r"C:\Users\alexw\Documents\Git\MAD_BTC\experience\BNBUSDT\binance\4h\momentum_strategy\2a90a504e84d572a958dff33f59f8f26ba7cdb598f4c23d476439ecbd7fa1522\output_file.csv"
    a = pd.read_csv(PATH)
    a["Time"] = pd.to_datetime(a["Time"])
    monthly_return(a, verbose=True)
