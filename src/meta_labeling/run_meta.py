"""
Meta-labeling on top of vol-normalised MAD (Lopez de Prado, AFML ch.3).

Primary rule: long when vol-normalised MAD z crosses above 0, exit when it crosses
below 0 (trade at the open of the bar, fee per side). Secondary model: predicts
whether the trade ends net-profitable, from features known at entry only, and
vetoes trades it does not like. Evaluated with an expanding walk-forward in which
training trades overlapping the test window are purged and an embargo is applied.

Every (lookback, model, threshold) combination counts toward the DSR trial count.
"""

import os
import sys
import itertools
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from ta.trend import ADXIndicator

ROOT = os.path.abspath(__file__)[: os.path.abspath(__file__).find("MAD_BTC") + len("MAD_BTC")]
sys.path.append(ROOT)
sys.path.append(ROOT + "/src")
from src.strategy.momentum import indicator_momentum
from src.multi_optimiser_momentum_strategy import deflated_sharpe_ratio

FEE = 0.001  # per side
LOOKBACKS = [5, 10, 20, 40]
THRESHOLDS = [0.5, 0.55, 0.6]
EMBARGO_BARS = 10


def load(symbol: str, tf: str) -> pd.DataFrame:
    df = pd.read_pickle(f"{ROOT}/data/binance-{symbol}-{tf}.pkl")
    return df.sort_values("open_date").reset_index(drop=True)


def features(df: pd.DataFrame, z: pd.Series) -> pd.DataFrame:
    """All features use bars < t plus open[t]; no information from bar t's high/low/close."""
    c1 = df["close"].shift(1)
    lr = np.log(df["close"]).diff().shift(1)
    f = pd.DataFrame(index=df.index)
    f["z"] = z
    f["dz"] = z - z.shift(1)
    f["z_max_20"] = z.rolling(20).max()
    adx = ADXIndicator(df["high"], df["low"], df["close"], window=14).adx().shift(1)
    f["adx"] = adx
    f["rv_20"] = lr.rolling(20).std()
    f["rv_ratio"] = lr.rolling(20).std() / lr.rolling(100).std()
    f["rv_rel"] = lr.rolling(20).std() / lr.rolling(20).std().rolling(500, min_periods=100).median()
    for n in (1, 6, 24):
        f[f"ret_{n}"] = np.log(c1 / c1.shift(n))
    f["vol_ratio"] = df["volume"].shift(1) / df["volume"].shift(1).rolling(50).mean()
    h = df["open_date"].dt.hour
    f["hr_sin"] = np.sin(2 * np.pi * h / 24)
    f["hr_cos"] = np.cos(2 * np.pi * h / 24)
    f["gap_open"] = np.log(df["open"] / c1)
    return f


def events(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    z = indicator_momentum.vol_normalized_mad(df, lookback)
    f = features(df, z)
    up = (z > 0) & (z.shift(1) <= 0)
    dn = (z < 0) & (z.shift(1) >= 0)
    ent = np.flatnonzero(up.values)
    dn_idx = np.flatnonzero(dn.values)
    o = df["open"].values
    rows = []
    for e in ent:
        xs = dn_idx[dn_idx > e]
        if len(xs) == 0:
            continue
        x = xs[0]
        ret = (o[x] / o[e]) * (1 - FEE) ** 2 - 1
        rows.append((e, x, ret))
    ev = pd.DataFrame(rows, columns=["i_in", "i_out", "ret"])
    ev["t_in"] = df["open_date"].values[ev["i_in"].values]
    X = f.iloc[ev["i_in"].values].reset_index(drop=True)
    ev = pd.concat([ev, X], axis=1).dropna().reset_index(drop=True)
    ev["y"] = (ev["ret"] > 0).astype(int)
    return ev


FEATS = ["z", "dz", "z_max_20", "adx", "rv_20", "rv_ratio", "ret_1", "ret_6", "ret_24",
         "vol_ratio", "hr_sin", "hr_cos", "gap_open"]


def make_model(name: str):
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=500))
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150,
                                          min_samples_leaf=20, l2_regularization=1.0, random_state=0)


def walk_forward(ev: pd.DataFrame, model_name: str, n_splits: int = 5, min_train: int = 40):
    """Expanding window over trade index; returns OOS probability per trade (NaN where no model)."""
    n = len(ev)
    p = np.full(n, np.nan)
    start = int(n * 0.4)
    edges = np.linspace(start, n, n_splits + 1).astype(int)
    for a, b in zip(edges[:-1], edges[1:]):
        test = np.arange(a, b)
        t0 = ev["i_in"].iloc[a]
        # purge: train trades must be fully closed (plus embargo) before the first test entry
        tr = np.flatnonzero((ev["i_out"].values + EMBARGO_BARS) < t0)
        if len(tr) < min_train or ev["y"].iloc[tr].nunique() < 2:
            continue
        m = make_model(model_name).fit(ev[FEATS].iloc[tr], ev["y"].iloc[tr])
        p[test] = m.predict_proba(ev[FEATS].iloc[test])[:, 1]
    return p


def sharpe(r: np.ndarray) -> float:
    return r.mean() / r.std(ddof=1) if len(r) > 2 and r.std(ddof=1) > 0 else np.nan


def run():
    assets = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    tfs = ["1h", "4h"]
    n_trials = len(LOOKBACKS) * 2 * len(THRESHOLDS)  # per timeframe, pooled over assets
    out = []
    for tf in tfs:
        for model_name in ("logreg", "gboost"):
            for lb in LOOKBACKS:
                base, filt = {th: [] for th in THRESHOLDS}, {th: [] for th in THRESHOLDS}
                allr = []
                for a in assets:
                    df = load(a, tf)
                    ev = events(df, lb)
                    p = walk_forward(ev, model_name)
                    ok = ~np.isnan(p)
                    allr.append(ev["ret"].values[ok])
                    for th in THRESHOLDS:
                        filt[th].append(ev["ret"].values[ok & (p >= th)])
                r_all = np.concatenate(allr)
                for th in THRESHOLDS:
                    r_f = np.concatenate(filt[th])
                    out.append(dict(tf=tf, model=model_name, lookback=lb, thr=th,
                                    n_all=len(r_all), n_kept=len(r_f),
                                    mean_all=r_all.mean(), mean_kept=r_f.mean() if len(r_f) else np.nan,
                                    sr_all=sharpe(r_all), sr_kept=sharpe(r_f),
                                    skew=skew(r_f) if len(r_f) > 3 else 0.0,
                                    kurt=kurtosis(r_f, fisher=False) if len(r_f) > 3 else 3.0))
    res = pd.DataFrame(out)
    res["dsr"] = [
        deflated_sharpe_ratio(r.sr_kept, n_trials, r.n_kept, r.skew, r.kurt)
        if r.n_kept > 5 and not np.isnan(r.sr_kept) else np.nan
        for r in res.itertuples()
    ]
    pd.set_option("display.width", 200)
    print(res.round(4).to_string(index=False))
    res.to_csv(f"{ROOT}/src/meta_labeling/results/results.csv", index=False)
    for tf in tfs:
        s = res[res.tf == tf].sort_values("sr_kept", ascending=False).head(1)
        print(f"\nBest {tf} (n_trials={n_trials}):\n", s.round(4).to_string(index=False))


if __name__ == "__main__":
    run()
