"""Builds the LaTeX table fragments for report.tex from the study outputs (run from anywhere). Needs data/ and src/meta_labeling/*.csv."""
import os, sys
ROOT = os.path.abspath(__file__)[: os.path.abspath(__file__).find("MAD_BTC") + len("MAD_BTC")]
sys.path.insert(0, ROOT + "/src/meta_labeling")
from core import *
from run_breakout import breakout_events
T = f"{ROOT}/report/2026-10-25/tables"; S = f"{ROOT}/src/meta_labeling"
pct = lambda x, d=1: f"{100 * x:.{d}f}\\,\\%".replace("-", "$-$")
num = lambda x, d=2: f"{x:.{d}f}".replace("-", "$-$")


def write(name, rows):
    open(f"{T}/{name}.tex", "w").write("\n".join(" & ".join(map(str, r)) + r" \\" for r in rows) + "\n")


# ---- 1. breakout grid (12 variants)
b = pd.read_csv(f"{S}/breakout.csv")
write("breakout_grid", [[int(r.k), int(r.H), "yes" if r.volume else "no", int(r.n), pct(r.mean_net, 2), pct(r.win, 0), num(r.sharpe), pct(r.maxdd, 0),
                         pct(r.expo, 0), num(r.sharpe_1st_half, 2) + " / " + num(r.sharpe_2nd_half, 2), num(r.wdsr, 2)] for r in b.itertuples()])

# ---- 2. CAGR / Sharpe / drawdown table (full span and sub-spans)
ev = {v: {a: breakout_events(a, 12, 24, v) for a in ASSETS} for v in (False, True)}
series = {"Breakout k=12, H=24, no volume (green)": pd.concat([curve(a, e).ret for a, e in ev[False].items()], axis=1).mean(axis=1),
          "Breakout k=12, H=24, volume (frozen, orange)": pd.concat([curve(a, e).ret for a, e in ev[True].items()], axis=1).mean(axis=1),
          "Green rule on BTC only": curve("BTCUSDT", ev[False]["BTCUSDT"]).ret,
          "BTC buy \\& hold": buyhold("BTCUSDT"),
          "Equal-weight buy \\& hold, 8 coins": pd.concat([buyhold(a) for a in ASSETS], axis=1).mean(axis=1)}
def cagr(r):
    r = r.dropna(); yrs = (pd.to_datetime(r.index[-1]) - pd.to_datetime(r.index[0])).days / 365.25; eq = (1 + r).cumprod()
    return eq.iloc[-1] ** (1 / yrs) - 1, eq.iloc[-1], r.mean() / r.std() * np.sqrt(BPY), (eq / eq.cummax() - 1).min()
rows = []
for k, r in series.items():
    c, f, sh, dd = cagr(r); rows.append([k, pct(c), num(f, 1), num(sh), pct(dd, 0)])
write("cagr_full", rows)
rows = []
for start in ("2020-01-01", "2022-01-01"):
    for k in list(series)[:2] + ["BTC buy \\& hold"]:
        r = series[k]; r = r[pd.to_datetime(r.index) >= start]; c, f, sh, dd = cagr(r)
        rows.append([start[:4] + " on", k, pct(c), num(sh), pct(dd, 0)])
write("cagr_sub", rows)

# ---- 3. calendar years: green, frozen, pooled buy & hold, BTC buy & hold
yrs = {}
for k, r in {"Green": series["Breakout k=12, H=24, no volume (green)"], "Frozen": series["Breakout k=12, H=24, volume (frozen, orange)"],
             "BTC b\\&h": series["BTC buy \\& hold"], "8-coin b\\&h": series["Equal-weight buy \\& hold, 8 coins"]}.items():
    y = pd.to_datetime(r.index).year; yrs[k] = {int(yy): (1 + r[y == yy]).prod() - 1 for yy in sorted(set(y))}
Y = pd.DataFrame(yrs)
write("years", [[str(yy)] + [pct(Y.loc[yy, c], 0) for c in Y.columns] for yy in Y.index])

# ---- 4. buy & hold lost years (frozen rule, per asset-year)
rows = []
for a in ASSETS:
    r = curve(a, ev[True][a]); bh = buyhold(a); y = pd.to_datetime(bh.index).year
    for yy in sorted(set(y)):
        m = y == yy
        if m.sum() >= 1000: rows.append(dict(asset=a[:-4], year=yy, bh=(1 + bh[m]).prod() - 1, rule=(1 + r.ret[m]).prod() - 1))
t = pd.DataFrame(rows); lost = t[t.bh < 0]; won = t[t.bh >= 0]
write("bh_lost", [["Years buy \\& hold lost", len(lost), pct(lost.bh.mean(), 1), pct(lost.rule.mean(), 1), pct((lost.rule > 0).mean(), 0), pct((lost.rule > lost.bh).mean(), 0)],
                  ["Years buy \\& hold gained", len(won), pct(won.bh.mean(), 0), pct(won.rule.mean(), 0), pct((won.rule > 0).mean(), 0), pct((won.rule > won.bh).mean(), 0)]])
byy = lost.groupby("year").agg(n=("bh", "size"), bh=("bh", "mean"), rule=("rule", "mean"))
write("bh_lost_by_year", [[str(yy), int(r.n), pct(r.bh, 1), pct(r.rule, 1)] for yy, r in byy.iterrows()])

# ---- 5. exposure-matched drawdown (pooled) and ensemble
e = pd.read_csv(f"{S}/ensemble_pooled.csv", index_col=0)
write("exposure_matched", [[i.replace("&", "\\&").replace("x", "$\\times$"), num(r.final), pct(r.ann), num(r.sharpe), pct(r.maxdd, 0), num(r["weekly DSR (12 trials)"], 2)] for i, r in e.iterrows()])
pa = pd.read_csv(f"{S}/ensemble_per_asset.csv")
order = [a[:-4] for a in ASSETS]
fin = pa.pivot(index="asset", columns="strat", values="final").reindex(order); sh = pa.pivot(index="asset", columns="strat", values="sharpe").reindex(order)
write("ensemble_assets", [[a, num(fin.loc[a, "buy&hold"]), num(fin.loc[a, "rule ensemble"]), num(sh.loc[a, "buy&hold"]), num(sh.loc[a, "rule ensemble"])] for a in order])

# ---- 6. frozen forward test of the dz/MAD rule (pooled) and ML pooled best cells
fz = pd.read_csv(f"{S}/frozen_per_asset.csv")
write("mad_forward", [[a, num(fz[(fz.asset == a) & (fz.strat == "buy&hold")].final.iloc[0], 2), num(fz[(fz.asset == a) & (fz.strat == "rule ensemble")].final.iloc[0], 2),
                       pct(fz[(fz.asset == a) & (fz.strat == "buy&hold")].maxdd.iloc[0], 0), pct(fz[(fz.asset == a) & (fz.strat == "rule ensemble")].maxdd.iloc[0], 0)] for a in order])
mp = pd.read_csv(f"{S}/results_pooled.csv"); mp = mp[(mp.tf == "4h") & (mp.model == "gboost")]
write("ml_pooled_4h", [[int(r.lb), num(r.thr, 2), int(r.n_kept), pct(r.mean_all, 2), pct(r.mean_kept, 2), num(r.sr_kept), num(r.p_perm, 3), num(r.dsr, 2)] for r in mp.itertuples()])
print("tables written:", sorted(os.listdir(T)))

# ---- 7. per-asset metrics: buy & hold vs green vs frozen (CAGR, Sharpe, max drawdown, exposure, trades, mean net per trade)
def full_stats(r):
    c, f, sh, dd = cagr(r); return c, f, sh, dd
rows_bh, rows_g, rows_f = [], [], []
for a in ASSETS:
    nm = a[:-4]; bh = buyhold(a); d = df_of(a)
    yrs = (d["open_date"].iloc[-1] - d["open_date"].iloc[0]).days / 365.25
    c, f, sh, dd = full_stats(bh); rows_bh.append([nm, d["open_date"].iloc[0].strftime("%Y-%m"), num(yrs, 1), pct(c), num(f, 1), num(sh), pct(dd, 0)])
    for rows, vol in ((rows_g, False), (rows_f, True)):
        e = ev[vol][a]; cv = curve(a, e); c, f, sh, dd = full_stats(cv.ret)
        rows.append([nm, pct(c), num(f, 1), num(sh), pct(dd, 0), pct(cv.pos.mean(), 0), len(e), pct(e.ret.mean(), 2), pct((e.ret > 0).mean(), 0)])
write("asset_bh", rows_bh); write("asset_green", rows_g); write("asset_frozen", rows_f)
# side-by-side Sharpe / CAGR / maxDD for the three
rows = []
for a, b_, g, fz_ in zip(ASSETS, rows_bh, rows_g, rows_f):
    rows.append([a[:-4], b_[3], g[1], fz_[1], b_[5], g[3], fz_[3], b_[6], g[4], fz_[4]])
write("asset_compare", rows)
print("per-asset tables written")

# ---- 8. per-coin per-calendar-year returns: frozen rule and buy & hold
def per_year(f):
    out = {}
    for a in ASSETS:
        r = f(a); y = pd.to_datetime(r.index).year
        out[a[:-4]] = {int(yy): (1 + r[y == yy]).prod() - 1 for yy in sorted(set(y)) if (y == yy).sum() > 300}
    return pd.DataFrame(out).T
Yf = per_year(lambda a: curve(a, ev[True][a]).ret); Yb = per_year(lambda a: buyhold(a))
cols = list(range(2018, 2027))
fmt = lambda v: "--" if pd.isna(v) else pct(v, 0)
write("asset_year_frozen", [[a] + [fmt(Yf.loc[a].get(c, np.nan)) for c in cols] for a in Yf.index])
write("asset_year_bh", [[a] + [fmt(Yb.loc[a].get(c, np.nan)) for c in cols] for a in Yb.index])

# ---- 9. worked example: one frozen-rule BTC trade, recomputed step by step
d = df_of("BTCUSDT"); e = ev[True]["BTCUSDT"]
dt = pd.to_datetime(d["open_date"].values[e.i_in.values]); row = e[(dt >= "2024-02-01")].iloc[0]; i, j = int(row.i_in), int(row.i_out)
c = np.log(d["close"].values); sig = pd.Series(np.diff(c, prepend=np.nan)).rolling(100).std().values
k = 12; zb = (c[i - 1] - c[i - 1 - k]) / (sig[i - 1] * np.sqrt(k)); zb_prev = (c[i - 2] - c[i - 2 - k]) / (sig[i - 2] * np.sqrt(k))
vr = d["volume"].values[i - 1] / d["volume"].rolling(50).mean().values[i - 1]
o_in, o_out = d["open"].values[i], d["open"].values[j]
ex = dict(decision=str(d["open_date"].iloc[i - 1]), entry=str(d["open_date"].iloc[i]), exit=str(d["open_date"].iloc[j]), ret12=np.exp(c[i - 1] - c[i - 1 - k]) - 1,
          sig=sig[i - 1], zb=zb, zb_prev=zb_prev, vr=vr, o_in=o_in, o_out=o_out, gross=o_out / o_in - 1, net=row.ret)
open(f"{T}/example.tex", "w").write(
    "\\begin{tabular}{ll}\\toprule Step & Value\\\\\\midrule\n"
    f"Last closed bar before entry (decision) & {ex['decision']} UTC\\\\\n"
    f"12-bar return ending at that close & {pct(ex['ret12'], 2)}\\\\\n"
    f"Standard deviation of 1-bar log returns, last 100 bars & {pct(ex['sig'], 3)}\\\\\n"
    f"$zb$ on that bar (needs $>2.0$) / on the bar before & {ex['zb']:.2f} / {ex['zb_prev']:.2f}\\\\\n"
    f"Volume of that bar over its 50-bar mean (needs $>1.5$) & {ex['vr']:.2f}\\\\\n"
    f"Entry: open of next bar, {ex['entry']} & {ex['o_in']:,.2f} USDT\\\\\n"
    f"Exit: open 24 bars (4 days) later, {ex['exit']} & {ex['o_out']:,.2f} USDT\\\\\n"
    f"Gross return & {pct(ex['gross'], 2)}\\\\\n"
    f"Net return after 0.1\\,\\% on each side & {pct(ex['net'], 2)}\\\\\\bottomrule\\end{{tabular}}\n")
print("extra tables written")
