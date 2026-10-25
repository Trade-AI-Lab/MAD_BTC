"""
Frozen forward test. Rule: long at MAD(vol-normalised, lookback L) z-cross above 0 when dz > THR_L and volume_ratio > 1.
THR_L = 0.67-quantile of dz over all trades (8 assets pooled) closed before FREEZE, written once to frozen_rule.json
and never refit. Scored only on trades entered on/after FREEZE (bars the rule-selection work never saw).
Re-run after fetch_klines.py appends new bars: the forward window just grows.
"""
import json
from core import *
LBS = (5, 10, 20)
path = f"{ROOT}/src/meta_labeling/frozen_rule.json"
if os.path.exists(path):
    fr = json.load(open(path))
else:
    fr = {"freeze": str(FREEZE.date()), "quantile": Q, "volume_rule": "vol[t-1]/mean50 > 1", "assets": ASSETS, "dz_threshold": {}}
    for lb in LBS:
        ev = all_events(lb)
        tr = ev[ev.t_out < FREEZE.to_datetime64()]
        fr["dz_threshold"][str(lb)] = float(tr["dz"].quantile(Q))
    json.dump(fr, open(path, "w"), indent=1)
print("frozen thresholds:", fr["dz_threshold"])
t0 = pd.Timestamp(fr["freeze"]); rows = []; curves = {}
for lb in LBS:
    ev = all_events(lb); ev = ev[ev.t_in >= t0.to_datetime64()]
    sel = ev[(ev.dz > fr["dz_threshold"][str(lb)]) & (ev.vol_ratio > 1)]
    print(f"lb{lb}: {len(ev)} MAD trades since {t0.date()}, rule keeps {len(sel)}; mean net/trade all {ev.ret.mean():+.3%} rule {sel.ret.mean():+.3%}, win {(sel.ret>0).mean():.0%}")
    for a in ASSETS:
        curves[(lb, "rule", a)] = curve(a, sel[sel.asset == a]); curves[(lb, "all", a)] = curve(a, ev[ev.asset == a])
cut = lambda x: x[x.index >= t0.to_datetime64()]
for a in ASSETS:
    bh = cut(buyhold(a).to_frame("ret")).ret
    ens = sum(cut(curves[(lb, "rule", a)]).ret for lb in LBS) / len(LBS)
    ens_all = sum(cut(curves[(lb, "all", a)]).ret for lb in LBS) / len(LBS)
    pos = sum(cut(curves[(lb, "rule", a)]).pos for lb in LBS) / len(LBS)
    for k, r in (("buy&hold", bh), ("all-trades ensemble", ens_all), ("rule ensemble", ens)):
        rows.append(dict(asset=a.replace("USDT", ""), strat=k, **stats(r), expo=(1.0 if k == "buy&hold" else pos.mean())))
tab = pd.DataFrame(rows)
pd.set_option("display.width", 200)
for m in ("final", "maxdd"):
    print(f"\n{m}"); print(tab.pivot(index="asset", columns="strat", values=m).reindex([a.replace('USDT','') for a in ASSETS]).round(3).to_string())
pooled = {k: pd.concat([ (cut(buyhold(a).to_frame("ret")).ret if k == "buy&hold" else sum(cut(curves[(lb, "rule" if k == "rule ensemble" else "all", a)]).ret for lb in LBS) / len(LBS)) for a in ASSETS], axis=1).mean(axis=1)
          for k in ("buy&hold", "all-trades ensemble", "rule ensemble")}
print(f"\nPooled 8 assets, {t0.date()} -> {pd.Series(pooled['buy&hold']).index[-1].date()}")
print(pd.DataFrame({k: stats(v) for k, v in pooled.items()}).T.round(3).to_string())
tab.to_csv(f"{ROOT}/src/meta_labeling/frozen_per_asset.csv", index=False)
