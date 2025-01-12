import pandas as pd
import plotly.graph_objects as go

# Parameters
monthly_investment = 500  # Monthly amount to invest
investment_interval = 'M'  # Resampling frequency: 'M' for monthly, 'W' for weekly, etc.

# Example DataFrame (Replace this with your actual DataFrame)
# The DataFrame should have columns: 'Open', 'High', 'Low', 'Close', and a time index
# Example:
# df = pd.DataFrame({
#     'Open': [...],
#     'High': [...],
#     'Low': [...],
#     'Close': [...]
# }, index=pd.date_range(start='2020-01-01', periods=..., freq='D'))

def calculate_dca_from_dataframe(df, monthly_investment, interval):
    """
    Calculate DCA portfolio value given a DataFrame with 'Close' prices.

    Args:
        df (pd.DataFrame): DataFrame containing at least a 'Close' column and a datetime index.
        monthly_investment (float): Fixed amount invested per interval.
        interval (str): Resampling frequency (e.g., 'M' for monthly, 'W' for weekly).

    Returns:
        portfolio_values (list): Portfolio value over time.
        total_invested (float): Total amount invested.
    """
    # Resample the DataFrame to the specified interval
    resampled_df = df['Close'].resample(interval).first()  # Use the first close price of the period

    total_invested = 0
    total_shares = 0
    portfolio_values = []

    for price in resampled_df.dropna():
        shares_bought = monthly_investment / price
        total_shares += shares_bought
        total_invested += monthly_investment
        portfolio_values.append(total_shares * price)

    return portfolio_values, total_invested, resampled_df.index

# Replace this with your actual DataFrame
data = {
    'Open': [100, 102, 104, 103, 105, 108, 107, 109, 111, 113],
    'High': [102, 105, 107, 106, 108, 110, 109, 112, 114, 115],
    'Low': [99, 101, 103, 102, 104, 107, 106, 108, 110, 112],
    'Close': [101, 103, 106, 104, 107, 109, 108, 110, 112, 114]
}
df = pd.DataFrame(data, index=pd.date_range(start='2023-01-01', periods=10, freq='D'))

# Calculate DCA
portfolio_values, total_invested, resampled_dates = calculate_dca_from_dataframe(df, monthly_investment, investment_interval)

# Final portfolio value and return
final_portfolio_value = portfolio_values[-1]
return_percentage = ((final_portfolio_value - total_invested) / total_invested) * 100

# Print results
print(f"Total Invested: ${total_invested:.2f}")
print(f"Final Portfolio Value: ${final_portfolio_value:.2f}")
print(f"Return on Investment: {return_percentage:.2f}%")

# Plot results with Plotly
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Asset Price', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=resampled_dates, y=portfolio_values, mode='lines', name='Portfolio Value', line=dict(color='green')))

fig.update_layout(
    title='Dollar-Cost Averaging (DCA) Simulation',
    xaxis_title='Time',
    yaxis_title='Value ($)',
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    template='plotly_white'
)

fig.show()
