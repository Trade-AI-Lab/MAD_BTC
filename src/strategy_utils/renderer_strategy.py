import matplotlib.pyplot as  plt

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
    evolution_df.to_csv(strategy_folder_path+'output_file.csv', index=False) # Can  cause an error
    plt.savefig(strategy_folder_path+"/"+title_image+".png")
    plt.close()

def renderer(evolution_df:dict,verbose:bool=False):
    core_img(evolution_df)
    if verbose:
        plt.show()
        plt.pause(20)
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
