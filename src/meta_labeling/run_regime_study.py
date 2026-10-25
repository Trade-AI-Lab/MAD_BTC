"""
Regime filter vs ML filter, judged on a portfolio equity curve (return, Sharpe, max drawdown).

Strategies (long-only MAD z>0 entries, equal-weight over assets, 0.1%/side, same OOS window):
  base      : every MAD trade
  adx       : only if ADX>25                         (fixed, 0 fitted parameters)
  rvrel     : only if 20-bar vol > its 500-bar median (fixed)
  gboost    : walk-forward gradient boosting, p>=0.5 (pooled, purged by timestamp)
  gb&adx    : gboost AND adx
Reference: equal-weight buy & hold over the same window.
Uncertainty: week-block bootstrap of the Sharpe / max-drawdown difference vs base.
"""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.abspath(__file__)[: os.path.abspath(__file__).find("MAD_BTC") + len("MAD_BTC")]
sys.path.extend([ROOT, ROOT + "/src"])
from src.meta_labeling.run_meta import load, events, make_model, FEATS, FEE

TAG = os.environ.get("TAG", "dev")
ASSETS = os.environ.get("ASSETS", "BTCUSDT,ETHUSDT,BNBUSDT,ADAUSDT,DOTUSDT").split(",")
TF = sys.argv[1] if len(sys.argv) > 1 else "4h"
LB = int(sys.argv[2]) if len(sys.argv) > 2 else 5
BPY = {"1h": 24 * 365, "4h": 6 * 365}[TF]
EMBARGO = np.timedelta64(2, "D")
rng = np.random.default_rng(0)

dfs, parts = {}, []
for a in ASSETS:
    df = load(a, TF); dfs[a] = df
    ev = events(df, LB)
    ev["t_out"] = df["open_date"].values[ev["i_out"].values]; ev["asset"] = a
    parts.append(ev)
ev = pd.concat(parts).sort_values("t_in").reset_index(drop=True)

# walk-forward probabilities
n = len(ev); p = np.full(n, np.nan)
QS = (0.5, 0.67, 0.8); thr = {q: np.full(n, np.nan) for q in QS}
edges = np.linspace(int(n * 0.4), n, 7).astype(int)
for a, b in zip(edges[:-1], edges[1:]):
    t0 = ev["t_in"].iloc[a].to_datetime64()
    tr = np.flatnonzero((ev["t_out"].values + EMBARGO) < t0)
    m = make_model("gboost").fit(ev[FEATS].iloc[tr], ev["y"].iloc[tr])
    p[a:b] = m.predict_proba(ev[FEATS].iloc[a:b])[:, 1]
    for q in QS:  # dz threshold = q-quantile of dz over the purged training trades only
        thr[q][a:b] = ev["dz"].iloc[tr].quantile(q)
ev["p"] = p
for q in QS: ev[f"thr{q}"] = thr[q]
ev = ev[ev.p.notna()].copy()
t_start = ev.t_in.min()
print(f"{TF} lb{LB}: OOS from {t_start.date()}, {len(ev)} trades, {ev.asset.nunique()} assets")

masks = {
    "base": np.ones(len(ev), bool),
    "adx": (ev.adx > 25).values,
    "rvrel": (ev.rv_rel > 1).values,
    "gboost": (ev.p >= 0.5).values,
    "gb&adx": ((ev.p >= 0.5) & (ev.adx > 25)).values,
    **{f"dz>q{int(q*100)}": (ev.dz > ev[f"thr{q}"]).values for q in QS},
    "dz67&vol": ((ev.dz > ev["thr0.67"]) & (ev.vol_ratio > 1)).values,
}


def bar_returns(sel):
    """Per-bar return series (index = open_date) averaged over assets; flat = 0."""
    cols = {}
    for a in ASSETS:
        df = dfs[a]; df = df[df.open_date >= t_start]
        r = pd.Series(0.0, index=df["open_date"].values)
        if sel is None:  # buy & hold
            cols[a] = pd.Series((df["open"].shift(-1) / df["open"] - 1).values, index=df["open_date"].values).fillna(0)
            continue
        o = dfs[a]["open"].values; d = dfs[a]["open_date"].values
        for row in ev[(ev.asset == a) & sel].itertuples():
            i, j = row.i_in, row.i_out
            seg = o[i + 1:j + 1] / o[i:j] - 1
            idx = d[i:j]
            r.loc[idx] = seg
            r.loc[d[i]] -= FEE; r.loc[d[j - 1]] -= FEE
        cols[a] = r
    return pd.concat(cols, axis=1).mean(axis=1)


def stats(r):
    eq = (1 + r).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    ann = eq.iloc[-1] ** (BPY / len(r)) - 1
    sr = r.mean() / r.std() * np.sqrt(BPY) if r.std() > 0 else np.nan
    return dict(ann=ann, sharpe=sr, maxdd=dd, calmar=ann / abs(dd) if dd else np.nan, in_mkt=(r != 0).mean())


series = {"buy&hold": bar_returns(None)}
for k, m in masks.items():
    series[k] = bar_returns(m)
tab = pd.DataFrame({k: stats(v) for k, v in series.items()}).T
tab["n_trades"] = [np.nan] + [int(m.sum()) for m in masks.values()]
pd.set_option("display.width", 200)
print(tab.round(3).to_string())
pd.DataFrame(series).to_csv(f"{ROOT}/src/meta_labeling/series_{TAG}_{TF}_lb{LB}.csv")

# trade-level DSR (n_trials = 2 timeframes x 3 lookbacks x 6 rules = 36; pooled trades overlap in time, so optimistic)
from scipy.stats import skew, kurtosis
from src.multi_optimiser_momentum_strategy import deflated_sharpe_ratio
print("trade-level DSR (36 trials):", {k: round(float(deflated_sharpe_ratio(
    (lambda r: r.mean() / r.std(ddof=1))(ev.ret[m].values), 36, int(m.sum()), skew(ev.ret[m]), kurtosis(ev.ret[m], fisher=False))), 2)
    for k, m in masks.items() if m.sum() > 10})

# cluster-aware DSR: weekly portfolio returns (assets already averaged -> no cross-asset overlap)
wk = {k: (1 + v).groupby(pd.to_datetime(v.index).to_period("W")).prod() - 1 for k, v in series.items()}
print("weekly-return DSR (n=%d weeks):" % len(wk["base"]))
for nt in (36, 100, 300):
    print(f"  trials={nt}:", {k: round(float(deflated_sharpe_ratio(w.mean() / w.std(ddof=1), nt, len(w), skew(w), kurtosis(w, fisher=False))), 2)
                            for k, w in wk.items() if k in ("base", "gboost", "dz>q67", "dz67&vol")})
if len(sys.argv) > 3: sys.exit()

# week-block bootstrap vs base
blk = {"1h": 168, "4h": 42}[TF]
R = pd.DataFrame(series).fillna(0).values; names = list(series)
nb = len(R) // blk
res = {k: {"sr": 0, "dd": 0} for k in names}
B = 1000
for _ in range(B):
    ix = np.concatenate([np.arange(s, s + blk) for s in rng.integers(0, len(R) - blk, nb)])
    S = R[ix]
    eq = np.cumprod(1 + S, axis=0); dd = (eq / np.maximum.accumulate(eq) - 1).min(axis=0)
    sr = S.mean(0) / np.where(S.std(0) > 0, S.std(0), np.nan)
    b = names.index("base")
    for j, k in enumerate(names):
        res[k]["sr"] += sr[j] > sr[b]; res[k]["dd"] += dd[j] > dd[b]
print("\nP(strategy beats base) over week-block bootstrap:")
print(pd.DataFrame(res).T.div(B).rename(columns={"sr": "P(Sharpe>base)", "dd": "P(smaller maxDD)"}).round(2).to_string())
tab.to_csv(f"{ROOT}/src/meta_labeling/regime_dz_{TAG}_{TF}_lb{LB}.csv")

# Exposure-matched control: random veto keeping the same number of trades as each filter.
# Answers "is the lower drawdown just lower exposure?"
NP = 300
print(f"\nRandom-veto control ({NP} draws, same trade count): percentile of the filter among random vetoes")
for k in ("gboost", "dz>q50", "dz>q67", "dz>q80", "dz67&vol"):
    m = masks[k]; kk = int(m.sum()); sr_l, dd_l = [], []
    for _ in range(NP):
        rm = np.zeros(len(ev), bool); rm[rng.choice(len(ev), kk, replace=False)] = True
        s = stats(bar_returns(rm)); sr_l.append(s["sharpe"]); dd_l.append(s["maxdd"])
    o = stats(series[k])
    print(f"  {k:7s} n={kk:4d}  Sharpe {o['sharpe']:.2f} beats {np.mean(np.array(sr_l) < o['sharpe']):.0%} of random "
          f"(random median {np.nanmedian(sr_l):.2f}) | maxDD {o['maxdd']:.2f} beats {np.mean(np.array(dd_l) < o['maxdd']):.0%} "
          f"(random median {np.median(dd_l):.2f})")
