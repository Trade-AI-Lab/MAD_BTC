"""Shared pieces for the follow-up studies: data, events, walk-forward dz threshold, per-asset curves, stats."""
import os, sys
import numpy as np, pandas as pd
from scipy.stats import skew, kurtosis
ROOT = os.path.abspath(__file__)[: os.path.abspath(__file__).find("MAD_BTC") + len("MAD_BTC")]
sys.path.extend([ROOT, ROOT + "/src"])
from src.meta_labeling.run_meta import load, events, FEE
from src.multi_optimiser_momentum_strategy import deflated_sharpe_ratio

ASSETS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "DOTUSDT", "SOLUSDT", "XRPUSDT", "LINKUSDT"]
TF, BPY = "4h", 6 * 365
EMBARGO = np.timedelta64(2, "D")
FREEZE = pd.Timestamp("2026-02-28")  # data up to 2026-02-27 is "dev"; later bars are the forward test
Q = 0.67
_dfs = {}


def df_of(a):
    if a not in _dfs:
        _dfs[a] = load(a, TF)
    return _dfs[a]


def all_events(lb, assets=ASSETS):
    parts = []
    for a in assets:
        d = df_of(a)
        ev = events(d, lb)
        ev["t_out"] = d["open_date"].values[ev["i_out"].values]; ev["asset"] = a
        parts.append(ev)
    return pd.concat(parts).sort_values("t_in").reset_index(drop=True)


def walk_forward_thr(ev, q=Q, n_splits=6):
    """dz threshold per trade = q-quantile of dz over earlier, closed (+embargo) trades; NaN before the first fold."""
    n = len(ev); thr = np.full(n, np.nan)
    edges = np.linspace(int(n * 0.4), n, n_splits + 1).astype(int)
    for a, b in zip(edges[:-1], edges[1:]):
        t0 = ev["t_in"].iloc[a].to_datetime64()
        tr = np.flatnonzero((ev["t_out"].values + EMBARGO) < t0)
        thr[a:b] = ev["dz"].iloc[tr].quantile(q)
    return thr


def rule_mask(ev, thr):
    return ((ev["dz"] > thr) & (ev["vol_ratio"] > 1)).values


def curve(a, sel):
    """Per-bar net return and position (0/1) for asset a given its selected trades; index = open_date."""
    d = df_of(a); o = d["open"].values
    r = np.zeros(len(d)); pos = np.zeros(len(d))
    for i, j in zip(sel["i_in"].values, sel["i_out"].values):
        r[i:j] = o[i + 1:j + 1] / o[i:j] - 1
        r[i] -= FEE; r[j - 1] -= FEE
        pos[i:j] = 1
    idx = d["open_date"].values
    return pd.DataFrame({"ret": r, "pos": pos}, index=idx)


def buyhold(a):
    d = df_of(a)
    return pd.Series((d["open"].shift(-1) / d["open"] - 1).fillna(0).values, index=d["open_date"].values)


def stats(r, pos=None):
    r = pd.Series(r)
    eq = (1 + r).cumprod(); dd = (eq / eq.cummax() - 1).min()
    out = dict(final=eq.iloc[-1], ann=eq.iloc[-1] ** (BPY / len(r)) - 1,
               sharpe=r.mean() / r.std() * np.sqrt(BPY) if r.std() > 0 else np.nan, maxdd=dd)
    if pos is not None:
        out["expo"] = float(np.mean(pos))
    return out


def weekly_dsr(r, n_trials):
    r = pd.Series(r); r.index = pd.to_datetime(r.index)
    w = (1 + r).groupby(r.index.to_period("W")).prod() - 1
    if w.std() == 0 or len(w) < 20:
        return np.nan
    return float(deflated_sharpe_ratio(w.mean() / w.std(ddof=1), n_trials, len(w), skew(w), kurtosis(w, fisher=False)))
