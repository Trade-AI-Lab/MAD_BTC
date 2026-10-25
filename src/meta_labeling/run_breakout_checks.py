"""Robustness of the MAD-free breakout: forward window (>= FREEZE), per asset, per year, fee x2, extra asset (MATIC), week-block bootstrap, DSR at 12/100 trials."""
import core
from core import *
from run_breakout import breakout_events, random_mean, KS, HS

FWD = FREEZE
rng = np.random.default_rng(1)
dates = lambda a, ev: df_of(a)["open_date"].values[ev["i_in"].values]
print("== forward window (entries on/after %s), all 12 variants" % FWD.date())
rows = []
for k in KS:
    for H in HS:
        for vol in (False, True):
            ev = {a: breakout_events(a, k, H, vol) for a in ASSETS}
            fw = {a: e[dates(a, e) >= FWD.to_datetime64()] for a, e in ev.items()}
            r = pd.concat([e.ret for e in fw.values()])
            rows.append(dict(k=k, H=H, volume=vol, n_fwd=len(r), mean_net=r.mean(), win=(r > 0).mean(), n_pos_assets=sum(e.ret.mean() > 0 for e in fw.values() if len(e))))
print(pd.DataFrame(rows).round(4).to_string(index=False))

for (k, H, vol) in ((6, 12, True), (12, 24, True)):
    print(f"\n== variant k={k} H={H} volume={vol}")
    ev = {a: breakout_events(a, k, H, vol) for a in ASSETS}
    t = pd.DataFrame([dict(asset=a.replace("USDT", ""), n=len(e), mean_net=e.ret.mean(), win=(e.ret > 0).mean()) for a, e in ev.items()])
    print(t.round(4).to_string(index=False))
    allr = pd.concat([e.assign(year=pd.to_datetime(dates(a, e)).year) for a, e in ev.items()])
    print("per year mean net / n:", {int(y): (round(g.ret.mean(), 4), len(g)) for y, g in allr.groupby("year")})
    print("fees x2 (0.2%/side): mean net", round((((1 + allr.ret) / (1 - FEE) ** 2) * (1 - 2 * FEE) ** 2 - 1).mean(), 4))
    # week-block bootstrap of the pooled mean (events clustered in time)
    allr["wk"] = pd.to_datetime(np.concatenate([dates(a, e) for a, e in ev.items()])).to_period("W")
    g = allr.groupby("wk").ret.agg(["sum", "count"]); idx = np.arange(len(g))
    bs = [g["sum"].values[s].sum() / g["count"].values[s].sum() for s in (rng.choice(idx, len(idx)) for _ in range(2000))]
    print("week-block bootstrap of mean net/trade: 5%%=%.4f 50%%=%.4f 95%%=%.4f" % tuple(np.percentile(bs, [5, 50, 95])))
    port = pd.concat([curve(a, e).ret for a, e in ev.items()], axis=1).mean(axis=1)
    print("portfolio:", {m: round(v, 3) for m, v in stats(port).items()}, "| weekly DSR 12 / 100 / 300 trials:", [round(weekly_dsr(port, n), 3) for n in (12, 100, 300)])

# extra asset not in the survivor list (renamed/delisted: MATIC, 4h data ends 2024)
a = "MATICUSDT"
if os.path.exists(f"{ROOT}/data/binance-{a}-4h.pkl"):
    core._dfs.pop(a, None)
    for (k, H, vol) in ((6, 12, True), (12, 24, True)):
        e = breakout_events(a, k, H, vol)
        print(f"MATIC k={k} H={H}: n={len(e)} mean net {e.ret.mean():.4f} win {(e.ret>0).mean():.2f} | random-entry p={(random_mean(a, len(e), H) >= e.ret.mean()).mean():.3f}")
