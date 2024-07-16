import pandas as pd
import os
import sys
import numpy as np
import json

__file__ = os.getcwd()
absolute_path = os.path.abspath(__file__)[
    : os.path.abspath(__file__).find("MAD_BTC") + len("MAD_BTC")
]
sys.path.append(absolute_path)
from src.utils.utils_df import open_dataframe
from src.strategy.momentum import indicator_momentum
from torch.utils.data import Dataset, DataLoader

def load_json(file_path: str) -> dict:
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def convert_time_iloc(dt, first, end, window_size):
    """
    Args:
        dt ():
        first ():
        end ():
        window_size ():

    Returns:
    """
    start = end - dt * window_size
    duration = start - first
    start_index = duration // dt + 1  # le +1 car on démarre à iloc[0]
    end_index = start_index + window_size
    return start_index, end_index


class dataset_creator(Dataset):
    def __init__(self, config: dict):
        self.config = config
        path_momentum_strategy = (
            absolute_path
            + rf"\experience\{self.config['crypto_currency']}\{self.config['exchange']}\{self.config['time']}\momentum_strategy"
        )
        path_data = (
            absolute_path
            + rf"\data\{self.config['exchange']}-{self.config['crypto_currency']}-{self.config['time']}.pkl"
        )
        self.window_size = config["window_size"]
        self.data_df = open_dataframe(path_data)
        self.list_strategies = os.listdir(path_momentum_strategy)
        self.dt = self.data_df["open_date"].diff().iloc[1]
        self.first = self.data_df["open_date"].iloc[0]
        self.dict_df_train = {}
        self.dict_df_validation = {}

    def _load_csv_on_fly(self, path: str):
        df = pd.read_csv(path)
        df["binary_classification"] = np.where(
            df["Relative_Portfolio_VS_Relative_Buy&Hold"] > 0, 1, 0
        )
        df["Time"] = pd.to_datetime(df["Time"])
        return df[["Time", "binary_classification"]]

    def add_df(self, dictionnary: dict, path: str):
        strategy_config = load_json(path)
        key = f"{strategy_config["x"]}_{strategy_config["y"]}_{strategy_config["ratio_method"]}"
        if key not in dictionnary.keys():
            dictionnary[key] = {}
            dictionnary[key]["df"] = self._load_csv_on_fly(path)
            dictionnary[key]["ratio_method"] = getattr(indicator_momentum,strategy_config["ratio_method"])(self.data_df,strategy_config["x"],strategy_config["y"])
    
    def get_data(self, dictionnary: dict):
        


def dataset_creator(path: str):
    df = pd.read_csv(path)
    df["binary_classification"] = np.where(
        df["Relative_Portfolio_VS_Relative_Buy&Hold"] > 0, 1, 0
    )
    data_df = open_dataframe(absolute_path + r"\data\binance-BNBUSDT-1D.pkl")

    df["Time"] = pd.to_datetime(df["Time"])
    dt = data_df["open_date"].diff().iloc[1]
    window_size = 20
    i = 30
    end = df["Time"].iloc[i]
    first = data_df["open_date"].iloc[0]

    print(end, "Time for the data")
    print(dt)
    start_index, end_index = convert_time_iloc(dt, first, end, window_size)
    print(
        data_df[["high", "low", "open", "open_date"]].iloc[
            start_index - 1 : end_index - 1
        ]
    )


if __name__ == "__main__":
    config = {
        "crypto_currency": "BNBUSDT",
        "exchange": "binance",
        "time": "1d",
    }
    PATH = (
        absolute_path
        + rf"\experience\{crypto_currency}\{exchange}\{time}\momentum_strategy\0ab83010636d975d48474b5abee4903145a93cf60c8953c46bd62ceb65317f51\output_file.csv"
    )
    dataset_creator(PATH)
