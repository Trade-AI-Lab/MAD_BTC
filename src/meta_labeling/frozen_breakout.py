"""
Frozen breakout rule: k=12, H=24, zb>2.0 first cross, volume[t-1]/mean50 > 1.5, long, 0.1%/side, no overlap per asset.
`python frozen_breakout.py --freeze`  writes frozen_breakout.json once (parameters, code hash, freeze timestamp, baselines).
`python frozen_breakout.py`           fetches new bars, scores ONLY entries at/after the freeze timestamp, appends to
                                      frozen_breakout_history.csv, and prints the pre-registered verdict.
"""
import sys, json, hashlib, subprocess, datetime
from core import *
from run_breakout import breakout_events, random_mean
D = f"{ROOT}/src/meta_labeling"; J = f"{D}/results/frozen_breakout.json"; H_CSV = f"{D}/results/frozen_breakout_history.csv"
K, H, VOL = 12, 24, True
rng = np.random.default_rng(7)
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
CODE = {f: sha(f"{D}/{f}") for f in ("run_breakout.py", "core.py", "run_meta.py")}


def entry_dates(a, e):
    return pd.to_datetime(df_of(a)["open_date"].values[e["i_in"].values])


def score(ev, since, until=None):
    rets, pos_assets, wk = [], 0, []
    for a, e in ev.items():
        t = entry_dates(a, e); m = t >= since
        if until is not None: m &= t < until
        r = e.ret[m]; rets.append(r)
        pos_assets += int(len(r) > 0 and r.mean() > 0)
        wk.append(pd.DataFrame({"ret": r.values, "wk": t[m].to_period("W")}))
    r = pd.concat(rets); w = pd.concat(wk)
    out = dict(n=len(r), mean_net=r.mean() if len(r) else np.nan, win=(r > 0).mean() if len(r) else np.nan, assets_pos=pos_assets)
    if len(r) >= 10:
        g = w.groupby("wk").ret.agg(["sum", "count"]); idx = np.arange(len(g))
        bs = [g["sum"].values[s].sum() / g["count"].values[s].sum() for s in (rng.choice(idx, len(idx)) for _ in range(2000))]
        out["boot5"] = float(np.percentile(bs, 5))
        perm = []
        for a, e in ev.items():
            t = entry_dates(a, e); m = t >= since
            if until is not None: m &= t < until
            if m.sum(): perm.append(random_mean(a, int(m.sum()), H, 500) * int(m.sum()))
        out["p_random"] = float(((sum(perm) / len(r)) >= r.mean()).mean())
    return out


ev = {a: breakout_events(a, K, H, VOL) for a in ASSETS}
last = max(df_of(a)["open_date"].max() for a in ASSETS)
if "--freeze" in sys.argv:
    assert not os.path.exists(J), "already frozen; delete deliberately to refreeze"
    fz = pd.Timestamp(last) + pd.Timedelta(hours=4)
    spec = dict(rule=dict(k=K, H=H, zb=2.0, vol_spike=1.5, vol_window=50, sigma_window=100, fee_per_side=FEE, side="long", overlap="none per asset"),
                assets=ASSETS, freeze_ts=str(fz), frozen_at=datetime.datetime.now().isoformat(timespec="seconds"),
                git_commit=subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
                code_sha=CODE,
                baselines={"dev_before_2026-02-28": score(ev, pd.Timestamp("2018-01-01"), pd.Timestamp("2026-02-28")),
                           "seen_2026-02-28_to_freeze": score(ev, pd.Timestamp("2026-02-28"), fz)},
                preregistered_verdict={"min_forward_trades": 100,
                                       "accept_if": "mean_net > 0 AND week-block bootstrap 5th pct > 0 AND p_random < 0.05",
                                       "reject_if": "n >= 100 AND mean_net <= 0",
                                       "otherwise": "inconclusive, keep collecting"})
    json.dump(spec, open(J, "w"), indent=1, default=float)
    print("FROZEN at", fz); print(json.dumps(spec["baselines"], indent=1, default=float)); sys.exit()

fr = json.load(open(J)); fz = pd.Timestamp(fr["freeze_ts"])
if fr["code_sha"] != CODE:
    print("WARNING: code changed since freeze:", {k: (fr["code_sha"][k], CODE[k]) for k in CODE if fr["code_sha"][k] != CODE[k]})
sc = score(ev, fz)
sc.update(asof=str(pd.Timestamp(last).date()), freeze=str(fz))
print("forward score:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in sc.items()})
pd.DataFrame([sc]).to_csv(H_CSV, mode="a", header=not os.path.exists(H_CSV), index=False)
need = fr["preregistered_verdict"]["min_forward_trades"]
if sc["n"] < need:
    print(f"verdict: inconclusive ({sc['n']}/{need} forward trades). Baselines were n=%d mean %.4f (dev) and n=%d mean %.4f (seen window)." % (
        fr["baselines"]["dev_before_2026-02-28"]["n"], fr["baselines"]["dev_before_2026-02-28"]["mean_net"],
        fr["baselines"]["seen_2026-02-28_to_freeze"]["n"], fr["baselines"]["seen_2026-02-28_to_freeze"]["mean_net"]))
else:
    ok = sc["mean_net"] > 0 and sc.get("boot5", -1) > 0 and sc.get("p_random", 1) < 0.05
    print("verdict:", "ACCEPT (evidence holds)" if ok else ("REJECT" if sc["mean_net"] <= 0 else "inconclusive"))
