import json
import os

def save_dict_to_json(data:dict, folder_path:str, file_name:str):
    """_summary_

    Args:
        data (dict): _description_
        folder_path (str): _description_
        file_name (str): _description_
    """
    # Create the folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)

    # Construct the full file path
    file_path = os.path.join(folder_path, file_name)

    # Save the dictionary as JSON
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=2)

def load_json_to_dict(folder_path:str, file_name:str):
    """_summary_

    Args:
        folder_path (str): _description_
        file_name (str): _description_

    Returns:
        _type_: _description_
    """
    
    # Construct the full file path
    file_path = os.path.join(folder_path, file_name)

    # Load the JSON file into a dictionary
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)

    return data

def config_data_available(PATH:str):
    """
    Building the Configuration Dictionary:

    After collecting all unique values, convert the sets to 
    lists and create a dictionary with keys "exchange", "crypto", and "timeframe".
    Args:
        PATH (str): _description_
    """
    import os


    # Initialize sets to hold unique values
    exchanges = set()
    cryptos = set()
    timeframes = set()

    # Iterate over all files in the directory
    for filename in os.listdir(PATH):
        if filename.endswith('.pkl'):
            # Extract parts from filename
            basename = os.path.splitext(filename)[0]
            parts = basename.split('-')
            
            if len(parts) == 3:
                exchange, crypto, timeframe = parts
                exchanges.add(exchange)
                cryptos.add(crypto)
                timeframes.add(timeframe)

    # Convert sets to lists for JSON serialization
    config = {
        'exchange': list(exchanges),
        'crypto_currency': list(cryptos),
        'time': list(timeframes)
    }
    return config

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

    save_dict_to_json(config, os.getcwd(), 'output.json')

    loaded_data = load_json_to_dict( os.getcwd(), 'output.json')
    # Print the loaded data
    print(loaded_data)
    config =  {
    "crypto_currency":"ETHUSDT",
    "time":"4h",
    "exchange":"binance",
    "strategy":{
        "x":12,
        "y":90,
        "ratio_method":"ema_ratio"
    },
    "initial_capital":1000
    
    }
    save_dict_to_json(config, os.getcwd(), 'output.json')
    loaded_data = load_json_to_dict( os.getcwd(), 'output.json')
    # Print the loaded data
    print(loaded_data)
    config = config_data_available("data")
    print(config)