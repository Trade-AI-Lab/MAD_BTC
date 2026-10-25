"""Rebuild 4h OHLCV from the 1h pickles (bins aligned on 00/04/08/..., incomplete bins dropped)."""
import os, sys
import pandas as pd
ROOT = os.path.abspath(__file__)[: os.path.abspath(__file__).find("MAD_BTC") + len("MAD_BTC")]
for a in sys.argv[1:]:
    d = pd.read_pickle(f"{ROOT}/data/binance-{a}-1h.pkl").sort_values("open_date")
    g = d.set_index("open_date").resample("4h")
    o = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                      "close": g["close"].last(), "volume": g["volume"].sum(), "n": g["open"].count()})
    o = o[o.n == 4].drop(columns="n").reset_index()
    o["timestamp_open"] = (o["open_date"].astype("int64") // 10**6)
    o = o[["timestamp_open", "open", "high", "low", "close", "volume", "open_date"]]
    o.to_pickle(f"{ROOT}/data/binance-{a}-4h.pkl")
    print(a, o.shape, o.open_date.min(), o.open_date.max())
