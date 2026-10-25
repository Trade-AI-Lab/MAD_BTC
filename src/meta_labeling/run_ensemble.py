"""
(1) Lookback ensemble: rule `dz>q67 & volume>mean50` at lookbacks 5/10/20 as equal-capital sleeves, per asset and pooled.
(3) Exposure-matched drawdown: is the drawdown cut more than 'being invested less'? Benchmarks: all-trades ensemble and
    buy & hold scaled to the rule's average exposure, plus a vol-managed buy & hold at the same average exposure.
Walk-forward OOS (dz threshold from earlier closed trades only), data through the end of the file.
"""
from core import *
LBS = (5, 10, 20)
curves = {}
starts = []
for lb in LBS:
    ev = all_events(lb); ev["thr"] = walk_forward_thr(ev); ev = ev[ev.thr.notna()]
    starts.append(ev.t_in.min())
    rule = ev[rule_mask(ev, ev.thr)]
    for a in ASSETS:
        curves[(lb, "rule", a)] = curve(a, rule[rule.asset == a])
        curves[(lb, "all", a)] = curve(a, ev[ev.asset == a])
t0 = max(starts)
print("common OOS start", t0.date())


def cut(x):
    return x[x.index >= t0.to_datetime64()]


def sleeve(kind, a, lbs=LBS):
    rs = [cut(curves[(lb, kind, a)]) for lb in lbs]
    return pd.DataFrame({"ret": sum(x.ret for x in rs) / len(rs), "pos": sum(x.pos for x in rs) / len(rs)})


rows = []
for a in ASSETS:
    bh = cut(buyhold(a).to_frame("ret"))
    cand = {"buy&hold": (bh.ret, 1.0),
            **{f"rule lb{lb}": (lambda s: (s.ret, s.pos.mean()))(sleeve("rule", a, (lb,))) for lb in LBS},
            "rule ensemble": (lambda s: (s.ret, s.pos.mean()))(sleeve("rule", a)),
            "all-trades ensemble": (lambda s: (s.ret, s.pos.mean()))(sleeve("all", a))}
    for k, (r, e) in cand.items():
        rows.append(dict(asset=a.replace("USDT", ""), strat=k, expo=e, **{m: v for m, v in stats(r).items()}))
tab = pd.DataFrame(rows)
pd.set_option("display.width", 220)
piv = lambda m: tab.pivot(index="asset", columns="strat", values=m)[["buy&hold", "rule lb5", "rule lb10", "rule lb20", "rule ensemble", "all-trades ensemble"]].reindex([a.replace("USDT", "") for a in ASSETS])
print("\nFinal growth of $1"); print(piv("final").round(2).to_string())
print("\nSharpe"); print(piv("sharpe").round(2).to_string())
print("\nMax drawdown"); print(piv("maxdd").round(2).to_string())
sh = piv("sharpe")
print("\nSharpe dispersion across assets (std): ", sh.std().round(2).to_dict())
print("Assets where Sharpe beats buy&hold:", {c: int((sh[c] > sh["buy&hold"]).sum()) for c in sh.columns[1:]})
fin = piv("final")
print("Assets where final beats buy&hold:  ", {c: int((fin[c] > fin["buy&hold"]).sum()) for c in fin.columns[1:]})

# ---- pooled portfolio (equal weight over the 8 assets) + exposure matching
def port(f):
    return pd.concat({a: f(a) for a in ASSETS}, axis=1).T.groupby(level=1).mean().T if False else pd.concat([f(a) for a in ASSETS], axis=1)
R = {k: pd.concat([v[0](a) for a in ASSETS], axis=1).mean(axis=1) for k, v in {
    "buy&hold": (lambda a: cut(buyhold(a).to_frame("ret")).ret, None),
    "rule ensemble": (lambda a: sleeve("rule", a).ret, None),
    "all-trades ensemble": (lambda a: sleeve("all", a).ret, None)}.items()}
E = {k: pd.concat([f(a) for a in ASSETS], axis=1).mean(axis=1) for k, f in {
    "rule ensemble": lambda a: sleeve("rule", a).pos, "all-trades ensemble": lambda a: sleeve("all", a).pos}.items()}
e_rule, e_all = E["rule ensemble"].mean(), E["all-trades ensemble"].mean()
print(f"\nAverage exposure: rule ensemble {e_rule:.2f}, all-trades ensemble {e_all:.2f}")


def vol_managed(a, target_expo):
    """Buy & hold with weight min(1, tv/sigma_ann), sigma from trailing 100 bars (lagged); tv bisected to hit target_expo."""
    r = cut(buyhold(a).to_frame("ret")).ret
    full = buyhold(a); sig = full.rolling(100).std().shift(1) * np.sqrt(BPY)
    sig = sig.reindex(r.index).bfill()
    lo, hi = 0.01, 5.0
    for _ in range(40):
        tv = (lo + hi) / 2; w = np.minimum(1, tv / sig)
        lo, hi = (tv, hi) if w.mean() < target_expo else (lo, tv)
    w = np.minimum(1, tv / sig)
    return (r * w) - FEE * 2 * w.diff().abs().fillna(0) / 2  # rebalancing cost on weight changes
vm = pd.concat([vol_managed(a, e_rule) for a in ASSETS], axis=1).mean(axis=1)

cmp = {
    "buy&hold (100%)": R["buy&hold"],
    f"buy&hold x{e_rule:.2f} (const)": R["buy&hold"] * e_rule,
    f"vol-managed b&h (avg {e_rule:.2f})": vm,
    "all-trades ensemble": R["all-trades ensemble"],
    f"all-trades x{e_rule/e_all:.2f} (matched)": R["all-trades ensemble"] * (e_rule / e_all),
    "rule ensemble": R["rule ensemble"],
}
print("\nPooled 8-asset portfolio, equal weight")
out = pd.DataFrame({k: stats(v) for k, v in cmp.items()}).T
out["weekly DSR (12 trials)"] = [weekly_dsr(v, 12) for v in cmp.values()]
print(out.round(3).to_string())
tab.to_csv(f"{ROOT}/src/meta_labeling/ensemble_per_asset.csv", index=False)
out.to_csv(f"{ROOT}/src/meta_labeling/ensemble_pooled.csv")
pd.DataFrame(cmp).to_csv(f"{ROOT}/src/meta_labeling/ensemble_series.csv")
