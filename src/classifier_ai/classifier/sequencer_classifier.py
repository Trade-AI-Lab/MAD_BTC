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


class DatasetCreator(Dataset):
    def __init__(self, config: dict):
        self.config = config
        self.path_momentum_strategy = os.path.join(
            absolute_path,
            f"experience/{config['crypto_currency']}/{config['exchange']}/{config['time']}/momentum_strategy",
        )
        self.path_data = os.path.join(
            absolute_path,
            f"data/{config['exchange']}-{config['crypto_currency']}-{config['time']}.pkl",
        )
        self.window_size = config["window_size"]
        # n_forward: how many bars ahead defines the label horizon
        self.n_forward = config.get("n_forward", 1)
        self.begin_date = pd.Timestamp(config["begin_date"])
        self.end_date = pd.Timestamp(config["end_date"])
        # ratio_method is added per-strategy in __getitem__; strip it from the base feature list
        self.features = [f for f in config["features"] if f != "ratio_method"]

        # Load and regularise the OHLC grid (forward-fill gaps)
        raw_df = open_dataframe(self.path_data)
        self.dt = raw_df["open_date"].diff().iloc[1]
        full_dates = pd.date_range(
            start=raw_df["open_date"].min(),
            end=raw_df["open_date"].max(),
            freq=self.dt,
        )
        self.data_df = (
            pd.DataFrame({"open_date": full_dates})
            .merge(raw_df, on="open_date", how="left")
            .ffill()
        )

        # Cyclical time features (computed before set_index to keep open_date as column)
        ts = self.data_df["open_date"].map(pd.Timestamp.timestamp)
        day, week = 86_400, 604_800
        self.data_df["sin_weekly"] = np.sin(ts * (2 * np.pi / week))
        self.data_df["sin_daily"] = np.sin(ts * (2 * np.pi / day))

        # Use DatetimeIndex for loc-based slicing throughout
        self.data_df.set_index("open_date", inplace=True)
        self.normalizer_tool = complex_fct_normalizer()
        self._price_cols = [c for c in ["open", "high", "low", "close"] if c in self.features]

        # Optional computational subset of strategies (does NOT affect temporal split)
        all_strats = sorted(s for s in os.listdir(self.path_momentum_strategy) if "." not in s)
        nb_first = int(config.get("nb_first_strats", 0.0) * len(all_strats))
        nb_last = int(config.get("nb_last_strats", 1.0) * len(all_strats))
        self.list_strategies = all_strats[nb_first:nb_last]

        self.create_dict_df()

    # ------------------------------------------------------------------
    # Label construction
    # ------------------------------------------------------------------

    def _build_labels(self, signal_path: str) -> pd.DataFrame:
        """
        Forward-looking, causal label for every bar T where the strategy is long.

        label[T] = 1  if close[T + n_forward] > close[T]   (following the signal earns)
                 = 0  otherwise

        Only bars with Position == 1 produce a sample; Position == -1 bars are
        skipped (nothing to decide when already out of the market).
        """
        sig = pd.read_csv(os.path.join(signal_path, "output_file.csv"))
        sig["Time"] = pd.to_datetime(sig["Time"])
        sig = sig.set_index("Time")[["Position"]].sort_index()

        close = self.data_df["close"]

        # Reconstruct position at every OHLC bar by forward-filling the sparse signal CSV
        pos = sig["Position"].reindex(close.index).ffill().dropna()

        long_idx = np.where(pos.values == 1)[0]
        # Drop bars that are too close to the end of the series
        long_idx = long_idx[long_idx + self.n_forward < len(close)]
        if len(long_idx) == 0:
            return pd.DataFrame(columns=["binary_classification"])

        c = close.values
        labels = pd.DataFrame(
            {"binary_classification": (c[long_idx + self.n_forward] > c[long_idx]).astype(np.int8)},
            index=close.index[long_idx],
        )
        labels.index.name = "Time"
        return labels

    # ------------------------------------------------------------------
    # Strategy ingestion
    # ------------------------------------------------------------------

    def _add_df(self, strat_dir: str):
        cfg = load_json(os.path.join(strat_dir, "strategy.json"))
        x = cfg["strategy"]["x"]
        y = cfg["strategy"]["y"]
        rm = cfg["strategy"]["ratio_method"]
        key = f"{x}_{y}_{rm}"
        if key in self.dict_df:
            return

        # Indicator computed on the full OHLC series — causal (MA uses only past bars)
        # reset_index() so indicator functions can access data["open"] as a column
        raw_ind = getattr(indicator_momentum, rm)(self.data_df.reset_index(), x, y)
        ratio_series = pd.Series(
            (-1 + raw_ind).values, index=self.data_df.index, name="ratio_method"
        )

        # Warmup: skip the first NaN region of the indicator + window_size bars
        nan_positions = np.where(ratio_series.isna().values)[0]
        last_nan_pos = int(nan_positions[-1]) if len(nan_positions) > 0 else -1
        min_valid_date = self.data_df.index[last_nan_pos + self.window_size + 1]

        labels = self._build_labels(strat_dir)
        if labels.empty:
            return

        # Temporal split: only keep bars inside [begin_date, end_date)
        lo = max(self.begin_date, min_valid_date)
        labels = labels[(labels.index >= lo) & (labels.index < self.end_date)]
        if len(labels) == 0:
            return

        self.dict_df[key] = {"labels": labels, "ratio": ratio_series}

    def create_dict_df(self):
        self.dict_df = {}
        for name in self.list_strategies:
            self._add_df(os.path.join(self.path_momentum_strategy, name))

        lengths = [len(v["labels"]) for v in self.dict_df.values()]
        self.total_length = sum(lengths)
        self.cum_lengths = np.cumsum(lengths)
        self._keys = list(self.dict_df.keys())

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        si = np.searchsorted(self.cum_lengths, idx, side="right")
        key = self._keys[si]
        ri = idx if si == 0 else idx - self.cum_lengths[si - 1]

        entry = self.dict_df[key]
        end = entry["labels"].index[ri]          # label bar T
        start = end - self.dt * self.window_size  # window covers [start, end)

        # OHLC + temporal features for the window ending just before T
        _x = self.data_df[self.features].loc[start : end - self.dt].copy()

        # Normalize ALL price columns relative to the first bar of the window
        # (including open — previously missing from normalization)
        if self._price_cols:
            _x[self._price_cols] = self.normalizer_tool.log_division_normalization(
                df=_x[self._price_cols], row=0
            )

        # Append strategy indicator (already centered: MA_fast/MA_slow - 1)
        _x["ratio_method"] = entry["ratio"].loc[start : end - self.dt].values

        X = torch.tensor(_x.values, dtype=torch.float32)
        Y = torch.tensor(
            [entry["labels"]["binary_classification"].iloc[ri]], dtype=torch.float32
        )
        return X, Y


if __name__ == "__main__":
    config = {
        "crypto_currency": "BNBUSDT",
        "exchange": "binance",
        "time": "4h",
        "window_size": 50,
        "n_forward": 12,
        "begin_date": "2018-01-01",
        "end_date": "2023-01-01",
        "nb_first_strats": 0.0,
        "nb_last_strats": 0.5,
        "features": ["open", "high", "low", "close", "ratio_method", "sin_weekly", "sin_daily"],
    }
    dataset = DatasetCreator(config)
    A, B = dataset[3]
    print(A, B)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    for batch in loader:
        X, Y = batch
        print("X shape:", X.size())
        print("Y shape:", Y.size())
        break
