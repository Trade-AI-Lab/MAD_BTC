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