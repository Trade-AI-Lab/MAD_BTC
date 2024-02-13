import pandas as pd
from src.utils.utils_df import open_dataframe
from src.utils.utils_signature import create_signature
from src.utils.utils_json import save_dict_to_json
from src.strategy.momentum import indicator_momentum
import matplotlib.pyplot as plt
import os

absolute_path = os.path.abspath(__file__)[:os.path.abspath(__file__).find("MAD_BTC")+len("MAD_BTC")]

def computing_saving_strategy(config,dataframe):

    signature_config = create_signature(config)

    df = dataframe.copy(deep=True)

    df[f'{config["ratio_method"]}_{config["x"]}_{config["y"]}'] = getattr(indicator_momentum,config["ratio_method"])(df, config["x"],config["y"])

    df = df.dropna().reset_index(drop=True)


    initial_capital = config["initial_capital"]

    df['buy_and_hold'] = df['open'] / df['open'][0] * initial_capital

    # Your trading strategy
    df['buy_signal'] = (df[f'{config["ratio_method"]}_{config["x"]}_{config["y"]}'] > 1) & (df[f'{config["ratio_method"]}_{config["x"]}_{config["y"]}'].shift(1) <= 1)
    df['sell_signal'] = (df[f'{config["ratio_method"]}_{config["x"]}_{config["y"]}'] < 1) & (df[f'{config["ratio_method"]}_{config["x"]}_{config["y"]}'].shift(1) >= 1)
    df_filtered = df[df['buy_signal'] | df['sell_signal']].reset_index(drop=True)

    position = 0
    portfolio = initial_capital
    crypto = 0
    list_portfolio = []
    list_buy_hold = []
    fee_percentage = 0.1 / 100  # 0.1%
    # Iterate through the DataFrame
    for i in range(len(df_filtered)):
        if df_filtered.iloc[i]['buy_signal'] and position != 1 and portfolio > 0:
            position = 1
            crypto = portfolio / (df_filtered.iloc[i]['open']*(1 + fee_percentage))
            portfolio = 0

        elif df_filtered.iloc[i]['sell_signal'] and position != -1 and crypto > 0:
            position = -1
            portfolio = crypto * df_filtered.iloc[i]['open'] * (1-fee_percentage)
            crypto = 0

            # Store portfolio values for comparison
            list_portfolio.append(portfolio)
            list_buy_hold.append(df_filtered['buy_and_hold'].iloc[i])

        elif crypto == 0 and portfolio == 0:
            break 
    
    list_portfolio.append(list_portfolio[-1])
    list_buy_hold.append(df['open'].iloc[-1] / df['open'][0] * initial_capital)
    
    # Create a DataFrame with the evolution data
    evolution_df = pd.DataFrame({
        'Portfolio': list_portfolio,
        'Buy_and_Hold': list_buy_hold
    })


    experience_folder_path = absolute_path + "/experience"
    os.makedirs(experience_folder_path,exist_ok=True)

    crypto_currency_efp = experience_folder_path +f"/{config['crypto_currency']}"
    os.makedirs(crypto_currency_efp,exist_ok=True)

    exchange_ccefp = crypto_currency_efp +f"/{config['exchange']}"
    os.makedirs(exchange_ccefp,exist_ok=True)

    time_eccefp = exchange_ccefp +f"/{config['time']}"
    os.makedirs(time_eccefp,exist_ok=True)
    
    strategy_folder_path = time_eccefp+f"/{signature_config}"
    os.makedirs(strategy_folder_path,exist_ok=True)
    save_dict_to_json(config, strategy_folder_path, 'strategy.json')

    final_value = evolution_df['Portfolio'].iloc[-1]


    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(evolution_df.index, evolution_df['Portfolio'], label='Portfolio', marker='o')
    plt.plot(evolution_df.index, evolution_df['Buy_and_Hold'], label='Buy and Hold', marker='x')
    plt.xlabel('Timesteps')
    plt.ylabel('Value')
    plt.title('Portfolio vs Buy and Hold Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig(strategy_folder_path+"/"+f"{config['exchange']}-{config['crypto_currency']}-{config['time']}_portfolio_vs_buy_hold_{config['ratio_method']}_{config['x']}_{config['y']}_{initial_capital}$.png")
    # plt.show()
    plt.close()


    return final_value


if __name__=="__main__":
    import optuna
    
    config = {
            "crypto_currency":"ETHUSDT",
            "time":"4h",
            "exchange":"binance",
            "x":12,
            "y":90,
            "ratio_method":"ema_ratio",
            "initial_capital":1000
            
        }
    PATH = absolute_path + rf"/data/{config['exchange']}-{config['crypto_currency']}-{config['time']}.pkl"
    df = open_dataframe(PATH)

    def objective(trial):
        X = trial.suggest_int('x', 1, 30)
        Y = trial.suggest_int('y', 60 ,90)
        ratio_method = trial.suggest_categorical('ratio_method',["ma_ratio","ema_ratio"])
        config["x"] = X
        config["y"] = Y
        config["ratio_method"] = ratio_method
        return computing_saving_strategy(config,df)
    
    study_name = "study"+f"_{config['crypto_currency']}_{config['exchange']}_{config['time']}"

    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(storage_name, direction="maximize")
    study.optimize(objective, n_trials=200)

    study.best_params 
        