"""Append missing 4h bars (public Binance spot klines, closed candles only) to data/binance-<SYM>-4h.pkl.
Originals are copied once to data/backup_4h/. Usage: python fetch_klines.py BTCUSDT ETHUSDT ..."""
import os, sys, time, shutil, json
import urllib.request
import pandas as pd
ROOT = os.path.abspath(__file__)[: os.path.abspath(__file__).find("MAD_BTC") + len("MAD_BTC")]
URL = "https://data-api.binance.vision/api/v3/klines?symbol={s}&interval=4h&startTime={t}&limit=1000"
os.makedirs(f"{ROOT}/data/backup_4h", exist_ok=True)
now_ms = int(time.time() * 1000)
for sym in sys.argv[1:]:
    path = f"{ROOT}/data/binance-{sym}-4h.pkl"
    bak = f"{ROOT}/data/backup_4h/binance-{sym}-4h.pkl"
    if not os.path.exists(bak):
        shutil.copy(path, bak)
    old = pd.read_pickle(path).sort_values("open_date")
    t = int(old["timestamp_open"].iloc[-1]) + 4 * 3600 * 1000
    rows = []
    while t < now_ms:
        with urllib.request.urlopen(URL.format(s=sym, t=t), timeout=30) as r:
            k = json.load(r)
        if not k:
            break
        rows += [x for x in k if x[6] < now_ms]  # closed candles only
        t = k[-1][0] + 4 * 3600 * 1000
        time.sleep(0.2)
    if not rows:
        print(sym, "no new bars"); continue
    new = pd.DataFrame(rows).iloc[:, :6]
    new.columns = ["timestamp_open", "open", "high", "low", "close", "volume"]
    new = new.astype({"timestamp_open": "int64", "open": float, "high": float, "low": float, "close": float, "volume": float})
    new["open_date"] = pd.to_datetime(new["timestamp_open"], unit="ms")
    out = pd.concat([old[old.timestamp_open < new.timestamp_open.min()], new]).reset_index(drop=True)
    out.to_pickle(path)
    print(sym, "+%d bars" % len(new), out.open_date.max(), "| dtypes ok:", list(out.columns) == list(old.columns))
