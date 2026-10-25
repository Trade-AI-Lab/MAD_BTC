"""
Rule `dz top third + volume` (4h) evaluated per asset vs buy & hold, lookbacks 5 and 10.
The dz threshold is the 0.67-quantile of dz over purged, earlier trades pooled over all 8 assets
(expanding walk-forward, same folds as run_regime_study); the rule is applied to each asset's own trades.
"""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.abspath(__file__)[: os.path.abspath(__file__).find("MAD_BTC") + len("MAD_BTC")]
sys.path.extend([ROOT, ROOT + "/src"])
from src.meta_labeling.run_meta import load, events, FEE

ASSETS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "DOTUSDT", "SOLUSDT", "XRPUSDT", "LINKUSDT"]
TF, BPY = "4h", 6 * 365
EMBARGO = np.timedelta64(2, "D")
dfs = {a: load(a, TF) for a in ASSETS}


def study(lb):
    parts = []
    for a in ASSETS:
        ev = events(dfs[a], lb)
        ev["t_out"] = dfs[a]["open_date"].values[ev["i_out"].values]; ev["asset"] = a
        parts.append(ev)
    ev = pd.concat(parts).sort_values("t_in").reset_index(drop=True)
    n = len(ev); thr = np.full(n, np.nan)
    edges = np.linspace(int(n * 0.4), n, 7).astype(int)
    for a, b in zip(edges[:-1], edges[1:]):
        t0 = ev["t_in"].iloc[a].to_datetime64()
        tr = np.flatnonzero((ev["t_out"].values + EMBARGO) < t0)
        thr[a:b] = ev["dz"].iloc[tr].quantile(0.67)
    ev["thr"] = thr
    return ev[ev.thr.notna()].copy()


def curve(a, sel, t_start):
    df = dfs[a]; df = df[df.open_date >= t_start]
    d, o = dfs[a]["open_date"].values, dfs[a]["open"].values
    r = pd.Series(0.0, index=df["open_date"].values)
    if sel is None:
        return pd.Series((df["open"].shift(-1) / df["open"] - 1).values, index=df["open_date"].values).fillna(0)
    for row in sel.itertuples():
        i, j = row.i_in, row.i_out
        r.loc[d[i:j]] = o[i + 1:j + 1] / o[i:j] - 1
        r.loc[d[i]] -= FEE; r.loc[d[j - 1]] -= FEE
    return r


def stats(r):
    eq = (1 + r).cumprod(); dd = (eq / eq.cummax() - 1).min()
    return dict(final=eq.iloc[-1], ann=eq.iloc[-1] ** (BPY / len(r)) - 1,
                sharpe=r.mean() / r.std() * np.sqrt(BPY), maxdd=dd, in_mkt=(r != 0).mean())


curves, rows = {}, []
for lb in (5, 10):
    ev = study(lb); t_start = ev.t_in.min()
    ev_rule = ev[(ev.dz > ev.thr) & (ev.vol_ratio > 1)]
    for a in ASSETS:
        for name, sel in (("buy&hold", None), ("all MAD trades", ev[ev.asset == a]), (f"rule lb{lb}", ev_rule[ev_rule.asset == a])):
            if name == "buy&hold" and lb == 10:
                continue
            r = curve(a, sel, t_start)
            curves[(a, name)] = r
            n_tr = len(sel) if sel is not None else np.nan
            rows.append(dict(asset=a.replace("USDT", ""), strategy=name, lb=lb if name != "buy&hold" else "-", trades=n_tr, **stats(r)))
    print(f"lb{lb}: OOS from {t_start.date()}")
tab = pd.DataFrame(rows)
pd.set_option("display.width", 200)
for a in ASSETS:
    t = tab[tab.asset == a.replace("USDT", "")].drop(columns="asset")
    print("\n" + a.replace("USDT", "")); print(t.round(3).to_string(index=False))
tab.to_csv(f"{ROOT}/src/meta_labeling/results/per_asset.csv", index=False)
pd.to_pickle(curves, f"{ROOT}/src/meta_labeling/results/per_asset_curves.pkl")
