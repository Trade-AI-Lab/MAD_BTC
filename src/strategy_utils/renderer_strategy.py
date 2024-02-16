import matplotlib.pyplot as  plt
import pandas as pd
def renderer(title_image:str,strategy_folder_path:str,evolution_df:dict):
    plt.figure(figsize=(10, 6))
    plt.plot(evolution_df["Time"], evolution_df['Portfolio'], label='Portfolio', marker='o')
    plt.plot(evolution_df["Time"], evolution_df['Buy_and_Hold'], label='Buy and Hold', marker='x')
    plt.xlabel('Timesteps')
    plt.ylabel('Value')
    plt.title('Portfolio vs Buy and Hold Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig(strategy_folder_path+"/"+title_image+".png")
    plt.close()
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
    title_image = f"{config['exchange']}-{config['crypto_currency']}-{config['time']}_portfolio_vs_buy_hold_{config['ratio_method']}_{config['x']}_{config['y']}_{config["initial_capital"]}$"
