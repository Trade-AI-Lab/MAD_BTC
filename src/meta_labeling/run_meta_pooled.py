"""
Iteration 2: one model per timeframe trained on events pooled across assets (time-ordered,
purged by timestamp), plus a permutation control: how often does a random veto with the same
keep-count reach the observed mean return? (p-value for "the filter adds selection skill").
"""
import os, sys
import numpy as np, pandas as pd
from scipy.stats import skew, kurtosis
ROOT = os.path.abspath(__file__)[: os.path.abspath(__file__).find("MAD_BTC") + len("MAD_BTC")]
sys.path.extend([ROOT, ROOT + "/src"])
from src.meta_labeling.run_meta import load, events, make_model, sharpe, FEATS, THRESHOLDS
from src.multi_optimiser_momentum_strategy import deflated_sharpe_ratio

ASSETS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "DOTUSDT"]
LOOKBACKS = [5, 10, 20]
EMBARGO = pd.Timedelta(days=2)
rng = np.random.default_rng(0)


def pooled(tf, lb):
    parts = []
    for a in ASSETS:
        p = f"{ROOT}/data/binance-{a}-{tf}.pkl"
        if not os.path.exists(p):
            continue
        df = load(a, tf)
        ev = events(df, lb)
        ev["t_out"] = df["open_date"].values[ev["i_out"].values]
        ev["asset"] = a
        parts.append(ev)
    return pd.concat(parts).sort_values("t_in").reset_index(drop=True)


def wf(ev, model_name, n_splits=6):
    n = len(ev); p = np.full(n, np.nan)
    edges = np.linspace(int(n * 0.4), n, n_splits + 1).astype(int)
    for a, b in zip(edges[:-1], edges[1:]):
        t0 = ev["t_in"].iloc[a]
        tr = np.flatnonzero((ev["t_out"].values + EMBARGO.to_timedelta64()) < t0.to_datetime64())
        if len(tr) < 60 or ev["y"].iloc[tr].nunique() < 2:
            continue
        m = make_model(model_name).fit(ev[FEATS].iloc[tr], ev["y"].iloc[tr])
        p[a:b] = m.predict_proba(ev[FEATS].iloc[a:b])[:, 1]
    return p


rows = []
for tf in ("1h", "4h"):
    n_trials = len(LOOKBACKS) * 2 * len(THRESHOLDS)
    for model_name in ("logreg", "gboost"):
        for lb in LOOKBACKS:
            ev = pooled(tf, lb); p = wf(ev, model_name); ok = ~np.isnan(p)
            r_all = ev["ret"].values[ok]
            for th in THRESHOLDS:
                keep = ok & (p >= th); r = ev["ret"].values[keep]
                if len(r) < 6:
                    continue
                perm = np.array([rng.choice(r_all, len(r), replace=False).mean() for _ in range(2000)])
                rows.append(dict(tf=tf, model=model_name, lb=lb, thr=th, n_all=len(r_all), n_kept=len(r),
                                 mean_all=r_all.mean(), mean_kept=r.mean(), sr_kept=sharpe(r),
                                 p_perm=(perm >= r.mean()).mean(),
                                 dsr=deflated_sharpe_ratio(sharpe(r), n_trials, len(r), skew(r), kurtosis(r, fisher=False))))
res = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(res.round(4).to_string(index=False))
res.to_csv(f"{ROOT}/src/meta_labeling/results_pooled.csv", index=False)
print("\nmax DSR", res.dsr.max().round(3), "| min perm-p", res.p_perm.min(), "| cells p<0.05:", (res.p_perm < .05).sum(), "of", len(res))
