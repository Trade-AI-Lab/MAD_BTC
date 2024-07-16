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
from torch.utils.data import Dataset

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
        self.path_momentum_strategy = (
            absolute_path
            + rf"\experience\{self.config['crypto_currency']}\{self.config['exchange']}\{self.config['time']}\momentum_strategy"
        )
        self.path_data = (
            absolute_path
            + rf"\data\{self.config['exchange']}-{self.config['crypto_currency']}-{self.config['time']}.pkl"
        )
        self.window_size = config["window_size"]
        self.data_df = open_dataframe(self.path_data)
        self.list_strategies = os.listdir(self.path_momentum_strategy)
        self.dt = self.data_df["open_date"].diff().iloc[1]
        self.first = self.data_df["open_date"].iloc[0]
        self.nb_strats = self.config["percentage_strats"]*len(self.list_strategies)
        self.features = self.config["features"]

    def _load_csv_on_fly(self, path: str):
        df = pd.read_csv(path)
        df["binary_classification"] = np.where(
            df["Relative_Portfolio_VS_Relative_Buy&Hold"] > 0, 1, 0
        )
        df["Time"] = pd.to_datetime(df["Time"])
        return df[["Time", "binary_classification"]][1:]

    def add_df(self, dictionnary: dict, path: str):
        strategy_config = load_json(path)
        key = f"{strategy_config["x"]}_{strategy_config["y"]}_{strategy_config["ratio_method"]}"
        if key not in dictionnary.keys():
            dictionnary[key] = {}
            dictionnary[key]["df"] = self._load_csv_on_fly(path)
            self.data_df["ratio_method"] = getattr(indicator_momentum,strategy_config["ratio_method"])(self.data_df,strategy_config["x"],strategy_config["y"])
            dictionnary[key]["ratio_method"] = self.data_df[["open_date","ratio_method"]]
    
    def __len__(self):
        return self.total_length
    
    def create_dict_df(self):
        self.dict_df = {}
        for name_strategy in self.list_strategies[:self.nb_strats]:
            self.add_df(self,os.path.join(self.path_momentum_strategy, name_strategy, "output.csv"))
        # Calculate the number of elements in each dataframe
        lengths = [len(self.dict_df[key]["df"]) for key in self.dict_df.keys()]
        self.total_length = sum(lengths)
        
        # Determine which dataframe to sample from based on cumulative sum
        self.cum_lengths = np.cumsum(lengths)
        
    def __getitem__(self, idx):
        selected_df_idx = np.searchsorted(self.cum_lengths, idx, side='right')
        selected_df_key = list(self.dict_df.keys())[selected_df_idx]
        if selected_df_idx == 0:
            row_idx = idx
        else:
            row_idx = idx - self.cum_lengths[selected_df_idx - 1]
        
        selected_df = self.dict_df[selected_df_key]["df"]
        end = selected_df["Time"].iloc[row_idx]
        start_index, end_index = convert_time_iloc(self.dt, self.first, end, self.window_size)
        X = pd.concat(self.data_df,self.dict_df[selected_df_key]["ratio_method"])[self.features].iloc[start_index - 1 : end_index - 1]
        Y = selected_df["binary_classification"]
        return X,Y

if __name__ == "__main__":
    config = {
        "crypto_currency": "BNBUSDT",
        "exchange": "binance",
        "time": "1d",
    }
    dataset_creator(config)
