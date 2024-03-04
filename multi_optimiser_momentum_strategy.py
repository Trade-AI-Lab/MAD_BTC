import pandas as pd
from src.utils.utils_df import open_dataframe
from src.strategy_utils.launcher_strategy import trading_strategy,momentum_strategy
from src.strategy_utils.strategy_comparator import evalutation
import os
import optuna

def objective(trial:optuna.trial,strategy:trading_strategy):
    """_summary_

    Args:
        trial (optuna.trial): _description_
        strategy (trading_strategy): _description_

    Returns:
        _type_: _description_
    """
    
    result_cache = {}
    
    X = trial.suggest_int('x', 1, 34)
    Y = trial.suggest_int('y', 35 ,120)
    ratio_method = trial.suggest_categorical('ratio_method',["ma_ratio","ema_ratio"])

    if (X, Y, ratio_method) in result_cache:
        return result_cache[(X, Y, ratio_method)]
    config["strategy"]["x"] = X
    config["strategy"]["y"] = Y
    config["strategy"]["ratio_method"] = ratio_method
    strategy.change_config(config)
    final_df = strategy.compute_simulation()
    
    col_1,col_2 = "Portfolio","Buy_and_Hold"
    final_df = evalutation(final_df,col_1,col_2)
    result_cache[(X, Y, ratio_method)] = final_df["Portfolio"].iloc[-1]*(final_df[f"RSI_{col_1}/{col_2}"]/100).mean()
    return result_cache[(X, Y, ratio_method)]
    
def loop_objective(config:dict,storage_name:str,strategy:trading_strategy,study_name:str)->None:
    """_summary_

    Args:
        config (dict): _description_
        storage_name (str): _description_
        strategy (trading_strategy): _description_
        study_name (str): _description_
    """
    
    PATH = absolute_path + rf"/data/{config['exchange']}-{config['crypto_currency']}-{config['time']}.pkl"
    df = open_dataframe(PATH)
            
    study = optuna.create_study(storage_name, study_name=study_name,direction="maximize",load_if_exists="True")
    study.optimize(lambda trial: objective(trial=trial,strategy=strategy), n_trials=200)

def optimiser_multiple_currency(config:dict,strategy:trading_strategy,list_crypto_currency:str=["BTCUSDT","ETHUSDT","ADAUSDT","BNBUSDT"],list_time:str=["1h","4h","1d"]):
    """_summary_

    Args:
        config (dict): _description_
        strategy (trading_strategy): _description_
        list_crypto_currency (str, optional): _description_. Defaults to ["BTCUSDT","ETHUSDT","ADAUSDT","BNBUSDT"].
        list_time (str, optional): _description_. Defaults to ["1h","4h","1d"].
    """
    for crypto_currency in list_crypto_currency:
         for time in list_time:
            # Modifying the config file
            config["time"] = time 
            config["crypto_currency"] = crypto_currency

            os.makedirs(absolute_path+"/study",exist_ok=True)
            study_name = "study"+f"_{config['crypto_currency']}_{config['exchange']}_{config['time']}"
            storage_name = f"sqlite:///{os.path.join(absolute_path, 'study', study_name)}.db"
            loop_objective(config,storage_name=storage_name,strategy=strategy,study_name=study_name) # an optuna is launched for the new config
            
if __name__=="__main__":
    absolute_path = os.path.abspath(__file__)[:os.path.abspath(__file__).find("MAD_BTC")+len("MAD_BTC")]
    config = {
            "crypto_currency":"BTCUSDT",
            "time":"4h",
            "exchange":"binance",
            "strategy":{
            "name":"momentum_strategy",
            "x":12,
            "y":90,
            "ratio_method":"ema_ratio",
            },
            "initial_capital":1000
            
        }
    strategy = momentum_strategy(config)
    optimiser_multiple_currency(config=config,strategy=strategy,list_crypto_currency=["BTCUSDT","ETHUSDT","ADAUSDT","BNBUSDT"],list_time=["1d","4h","1h"])
    

    
        