"""
Breakout without MAD. Event on bar t (entered at open[t], using bars < t only):
  zb = (log close[t-1] - log close[t-1-k]) / (sigma1 * sqrt(k)),  sigma1 = std of 1-bar log returns over 100 bars (lagged)
  first cross of zb above 2.0, optionally with volume[t-1] / mean50 > 1.5 ("spike"). Hold H bars, no overlap per asset.
Variants (all fixed, no fitting): k in {6,12} x H in {6,12,24} x volume filter in {no, yes} = 12 trials.
Controls: (a) random entries per asset, same trade count and H, 1000 draws -> percentile of mean net return per trade;
          (b) the breakout without the volume spike (same k, H) -> does volume add anything?
"""
from core import *
rng = np.random.default_rng(0)
ZB, VOLSPIKE = 2.0, 1.5
KS, HS = (6, 12), (6, 12, 24)


def breakout_events(a, k, H, vol):
    d = df_of(a); c = np.log(d["close"].values); n = len(d)
    r1 = pd.Series(np.diff(c, prepend=np.nan))
    sig = r1.rolling(100).std().values
    zb = np.full(n, np.nan)
    zb[k:] = (c[k:] - c[:-k]) / (sig[k:] * np.sqrt(k))
    vr = (d["volume"] / d["volume"].rolling(50).mean()).values
    z1 = np.r_[np.nan, zb[:-1]]; z2 = np.r_[np.nan, np.nan, zb[:-2]]; v1 = np.r_[np.nan, vr[:-1]]  # signal known at open t
    sig_bar = (z1 > ZB) & ~(z2 > ZB)
    if vol: sig_bar &= v1 > VOLSPIKE
    o = d["open"].values; out = []; busy = -1
    for t in np.flatnonzero(sig_bar):
        if t <= busy or t + H >= n: continue
        out.append((t, t + H, (o[t + H] / o[t]) * (1 - FEE) ** 2 - 1)); busy = t + H
    return pd.DataFrame(out, columns=["i_in", "i_out", "ret"])


def random_mean(a, ntr, H, draws=1000):
    d = df_of(a); o = d["open"].values; n = len(d)
    ret = (o[H + 20:] / o[20:n - H]) * (1 - FEE) ** 2 - 1
    return np.array([rng.choice(ret, ntr, replace=False).mean() for _ in range(draws)])


if __name__ == "__main__":
    rows = []
    for k in KS:
        for H in HS:
            for vol in (False, True):
                ev = {a: breakout_events(a, k, H, vol) for a in ASSETS}
                allr = pd.concat([e.ret for e in ev.values()])
                # per-asset random control, combined by pooling asset-level means weighted by counts
                perm = sum(random_mean(a, len(e), H) * len(e) for a, e in ev.items() if len(e) > 0) / sum(len(e) for e in ev.values())
                curves = [curve(a, e) for a, e in ev.items()]
                port = pd.concat([c.ret for c in curves], axis=1).mean(axis=1)
                half = len(port) // 2
                rows.append(dict(k=k, H=H, volume=vol, n=len(allr), mean_net=allr.mean(), win=(allr > 0).mean(),
                                 p_random=(perm >= allr.mean()).mean(), sharpe=stats(port)["sharpe"], maxdd=stats(port)["maxdd"],
                                 expo=float(np.mean([c.pos.mean() for c in curves])), sharpe_1st_half=stats(port.iloc[:half])["sharpe"],
                                 sharpe_2nd_half=stats(port.iloc[half:])["sharpe"], wdsr=weekly_dsr(port, 12)))
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(res.round(4).to_string(index=False))
    bh = pd.concat([buyhold(a) for a in ASSETS], axis=1).mean(axis=1)
    print("\nEqual-weight buy & hold, same full span:", {k: round(v, 3) for k, v in stats(bh).items()})
    res.to_csv(f"{ROOT}/src/meta_labeling/results/breakout.csv", index=False)
