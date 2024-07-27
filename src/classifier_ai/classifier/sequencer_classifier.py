import pandas as pd
import os
import sys
import numpy as np
import json
import torch

__file__ = os.getcwd()
absolute_path = os.path.abspath(__file__)[
    : os.path.abspath(__file__).find("MAD_BTC") + len("MAD_BTC")
]
sys.path.append(absolute_path)
from src.utils.utils_df import open_dataframe
from src.strategy.momentum import indicator_momentum
from torch.utils.data import Dataset, DataLoader
from src.classifier_ai.classifier.normaliser.soft_normaliser import (
    complex_fct_normalizer,
)


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


class DatasetCreator(Dataset):
    def __init__(self, config: dict):
        self.config = config
        self.path_momentum_strategy = os.path.join(
            absolute_path,
            f"experience/{self.config['crypto_currency']}/{self.config['exchange']}/{self.config['time']}/momentum_strategy",
        )
        self.path_data = os.path.join(
            absolute_path,
            f"data/{self.config['exchange']}-{self.config['crypto_currency']}-{self.config['time']}.pkl",
        )
        self.window_size = config["window_size"]
        self.data_df = open_dataframe(self.path_data)
        self.list_strategies = [
            strategy
            for strategy in os.listdir(self.path_momentum_strategy)
            if "." not in strategy
        ]
        self.dt = self.data_df["open_date"].diff().iloc[1]
        self.nb_first_stats = int(
            self.config["nb_first_strats"] * len(self.list_strategies)
        )
        self.nb_last_strats = int(
            self.config["nb_last_strats"] * len(self.list_strategies)
        )
        self.features = self.config["features"]
        self.create_dict_df()
        self.normalizer_tool = complex_fct_normalizer()
        self.data_df.set_index('open_date', inplace=True)

    def _load_csv_on_fly(self, path: str):
        df = pd.read_csv(path)
        df["binary_classification"] = np.where(
            df["Relative_Portfolio_VS_Relative_Buy&Hold"] > 0, 1, 0
        )
        df["Time"] = pd.to_datetime(df["Time"])
        return df[["Time", "binary_classification"]][1:]

    def _add_df(self, path: str):
        path_csv = os.path.join(path, "output_file.csv")
        path_json = os.path.join(path, "strategy.json")
        strategy_config = load_json(path_json)
        key = f"{strategy_config['strategy']['x']}_{strategy_config['strategy']['y']}_{strategy_config['strategy']['ratio_method']}"
        if key not in self.dict_df.keys():
            self.dict_df[key] = {}
            self.data_df["ratio_method"] = -1 + getattr(
                indicator_momentum, strategy_config["strategy"]["ratio_method"]
            )(
                self.data_df,
                strategy_config["strategy"]["x"],
                strategy_config["strategy"]["y"],
            )
            last_nan_index = self.data_df["ratio_method"].isna()[::-1].idxmax() if self.data_df["ratio_method"].isna().any() else None
            no_nan_window = last_nan_index + self.window_size + 1
            no_nan_ratio_method = self.data_df["open_date"].iloc[last_nan_index]
            no_nan_window_date = self.data_df["open_date"].iloc[no_nan_window]
            df = self._load_csv_on_fly(path_csv)
            self.dict_df[key]["df"] = df[(df["Time"] > no_nan_window_date)]
            self.dict_df[key]["df"].set_index('Time', inplace=True)
            del df
            df = self.data_df[
                ["open_date", "ratio_method"]
            ]
            self.dict_df[key]["ratio_method"] = df[(df["open_date"] >= no_nan_ratio_method)]
            self.dict_df[key]["ratio_method"].set_index('open_date', inplace=True)
            self.dict_df[key]["last_time"] = [no_nan_window_date, no_nan_ratio_method]
            del df


    def __len__(self):
        return self.total_length

    def create_dict_df(self):
        self.dict_df = {}
        for name_strategy in self.list_strategies[
            self.nb_first_stats : self.nb_last_strats
        ]:
            self._add_df(
                os.path.join(self.path_momentum_strategy, name_strategy),
            )
        # Calculate the number of elements in each dataframe
        lengths = [len(self.dict_df[key]["df"]) for key in self.dict_df.keys()]
        self.total_length = sum(lengths)

        # Determine which dataframe to sample from based on cumulative sum
        self.cum_lengths = np.cumsum(lengths)

    def __getitem__(self, idx):
        selected_df_idx = np.searchsorted(self.cum_lengths, idx, side="right")
        selected_df_key = list(self.dict_df.keys())[selected_df_idx]
        if selected_df_idx == 0:
            row_idx = idx
        else:
            row_idx = idx - self.cum_lengths[selected_df_idx - 1]

        selected_df = self.dict_df[selected_df_key]["df"]
        end = selected_df.index[row_idx]
        start = end - self.dt * self.window_size
        ratio_method = self.dict_df[selected_df_key]["ratio_method"].loc[
            start - self.dt : end - self.dt
        ]["ratio_method"]
            
        _x = self.data_df[self.features].loc[start - self.dt : end - self.dt]
        _x[["open", "low", "high", "close"]] = (
            self.normalizer_tool.log_division_normalization(
                df=_x[["open", "low", "high", "close"]], row=0
            )
        )
        _x["ratio_method"] = ratio_method
        if _x.isna().any().sum():
            print("Problem Nan")
        X = torch.tensor(_x.values, dtype=torch.float32)
        Y = torch.tensor(
            [selected_df["binary_classification"].iloc[row_idx]],
            dtype=torch.float32,
        )
        return X, Y


if __name__ == "__main__":
    config = {
        "crypto_currency": "BNBUSDT",
        "exchange": "binance",
        "time": "1d",
        "window_size": 50,
        "nb_first_strats": 0.0,
        "nb_last_strats": 0.5,
        "features": [
            "open",
            "high",
            "low",
            "close",
            "ratio_method",
            "sin_weekly",
            "sin_daily",
        ],
    }
    dataset = DatasetCreator(config)
    A, B = dataset.__getitem__(3)
    print(A, B)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    # Get one batch of data
    for batch in dataloader:
        # Assuming the batch is a tuple (inputs, labels)
        inputs, labels = batch

        # Print the first item
        print("First input:", inputs.size())
        print("First label:", labels.size())

        # Break after the first batch
        break
        