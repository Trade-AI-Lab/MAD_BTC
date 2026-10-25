from core import *
from run_breakout import breakout_events
ev = {a: breakout_events(a, 12, 24, True) for a in ASSETS}
rows = []
for a in ASSETS:
    r = curve(a, ev[a]); b = buyhold(a)
    for y in sorted(set(pd.to_datetime(b.index).year)):
        m = pd.to_datetime(b.index).year == y
        if m.sum() < 1000: continue  # skip partial years
        rows.append(dict(asset=a[:-4], year=y, bh=(1 + b[m]).prod() - 1, rule=(1 + r.ret[m]).prod() - 1, expo=r.pos[m].mean()))
t = pd.DataFrame(rows)
lost = t[t.bh < 0]
print(f"asset-years where buy&hold lost: {len(lost)} of {len(t)}")
print(f"  mean B&H {lost.bh.mean():+.1%}, mean rule {lost.rule.mean():+.1%}, rule > B&H in {(lost.rule > lost.bh).mean():.0%}, rule positive in {(lost.rule > 0).mean():.0%}, median rule {lost.rule.median():+.1%}")
won = t[t.bh >= 0]
print(f"asset-years where buy&hold gained: {len(won)} | mean B&H {won.bh.mean():+.1%}, mean rule {won.rule.mean():+.1%}, rule > B&H in {(won.rule > won.bh).mean():.0%}")
print("\nby year (asset-years where B&H lost):")
print(lost.groupby("year").agg(n=("bh", "size"), bh=("bh", "mean"), rule=("rule", "mean")).round(3).to_string())
print("\nworst 12 B&H asset-years:"); print(t.sort_values("bh").head(12).round(3).to_string(index=False))
# pooled equal-weight, years where pooled B&H lost
P = pd.concat([buyhold(a) for a in ASSETS], axis=1).mean(axis=1); R = pd.concat([curve(a, ev[a]).ret for a in ASSETS], axis=1).mean(axis=1)
print("\npooled by year:"); 
for y in sorted(set(pd.to_datetime(P.index).year)):
    m = pd.to_datetime(P.index).year == y
    print(y, f"B&H {(1+P[m]).prod()-1:+.1%}  rule {(1+R[m]).prod()-1:+.1%}")
# drawdown windows: bars where the asset is >40% below its running peak
d = []
for a in ASSETS:
    b = buyhold(a); eq = (1 + b).cumprod(); under = (eq / eq.cummax() - 1) < -0.4
    r = curve(a, ev[a]).ret
    d.append(pd.DataFrame({"bh": b[under], "rule": r[under.values]}).assign(asset=a[:-4]))
d = pd.concat(d)
print(f"\nbars while asset is >40% below its peak: n={len(d)} ({len(d)/sum(len(buyhold(a)) for a in ASSETS):.0%} of all bars); cumulative-equivalent per-bar mean (ann.): B&H {d.bh.mean()*BPY:+.1%}, rule {d.rule.mean()*BPY:+.1%}")
