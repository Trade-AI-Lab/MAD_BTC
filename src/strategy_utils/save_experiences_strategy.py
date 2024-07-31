import os
import sys
absolute_path = os.path.abspath(__file__)[:os.path.abspath(__file__).find("MAD_BTC")+len("MAD_BTC")]
sys.path.append(absolute_path)
from src.utils.utils_signature import create_signature
from src.utils.utils_json import save_dict_to_json
from typing import Tuple

def create_save_one_experience(config:dict,time_eccefpsn:str)->str:
    """
    Create and Save One Experience

    This function creates and saves an experience based on the provided configuration.

    :param config: Configuration parameters for the experience.
    :type config: dict
    :param time_eccefpsn: Time information for creating the experience folder.
    :type time_eccefpsn: str
    :return: Path to the created strategy folder.
    :rtype: str
    """
    signature_config = create_signature(config)
    strategy_folder_path = time_eccefpsn+f"/{signature_config}"
    os.makedirs(strategy_folder_path,exist_ok=True)
    save_dict_to_json(config, strategy_folder_path, 'strategy.json')
    return strategy_folder_path

def create_save_experiences(config:dict)->str:
    """
    Create and Save Experiences

    This function creates and saves experiences based on the provided configuration.

    :param config: Configuration parameters for the experiences.
    :type config: dict
    :return: Path to the created experience folder.
    :rtype: str
    """
    experience_folder_path = absolute_path + "/experience"
    os.makedirs(experience_folder_path,exist_ok=True)

    list_keys = config.keys()
    if "crypto_currency" in list_keys:
        crypto_currency_efp = experience_folder_path +f"/{config['crypto_currency']}"
    elif "list_crypto_currencies" in list_keys:
        if isinstance(config['list_crypto_currencies'],list):
           crypto = from_list_to_str(config['list_crypto_currencies'])
        crypto_currency_efp = experience_folder_path +f"/{crypto}"
    os.makedirs(crypto_currency_efp,exist_ok=True)

    exchange_ccefp = crypto_currency_efp +f"/{config['exchange']}"
    os.makedirs(exchange_ccefp,exist_ok=True)

    time_eccefp = exchange_ccefp +f"/{config['time']}"
    os.makedirs(time_eccefp,exist_ok=True)

    time_eccefpsn = time_eccefp +f"/{config['strategy']['name']}"
    os.makedirs(time_eccefpsn,exist_ok=True)

    return time_eccefpsn

def from_list_to_str(list_symbols:str)->str:
    symbols_sorted = sorted(list_symbols)
    str_symbols = "_".join(symbols_sorted)
    # Join the sorted elements with underscores
    return str_symbols

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
    a = create_save_experiences(config)
    print(a)
    b = create_save_one_experience(config,a)
    print(b)
    