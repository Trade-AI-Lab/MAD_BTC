"""Growth of $1: MAD-free breakout variants vs buy & hold (pooled equal weight), and the frozen rule per asset."""
from core import *
from run_breakout import breakout_events
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
VARS = [((12, 24, True), "#eb6834", "-", "FROZEN k=12, H=24, volume"), ((12, 24, False), "#1baf7a", "-", "k=12, H=24, no volume"),
        ((6, 12, True), "#2a78d6", "-", "k=6, H=12, volume (fails forward)")]
ev = {v[0]: {a: breakout_events(a, *v[0]) for a in ASSETS} for v in VARS}
bh = {a: buyhold(a) for a in ASSETS}
eq = lambda r: (1 + r).cumprod()
fig = plt.figure(figsize=(13, 14))
gs = fig.add_gridspec(5, 2, height_ratios=[1.6, 1, 1, 1, 1])
ax = fig.add_subplot(gs[0, :])
pooled_bh = pd.concat(bh.values(), axis=1).mean(axis=1).dropna()
ax.plot(pooled_bh.index, eq(pooled_bh), color="#8a8984", lw=2, ls="--", label="Buy & hold (equal weight)")
for key, col, ls, lab in VARS:
    r = pd.concat([curve(a, e).ret for a, e in ev[key].items()], axis=1).mean(axis=1)
    ax.plot(r.index, eq(r), color=col, lw=2, ls=ls, label=lab)
    ax.annotate(f"{eq(r).iloc[-1]:.1f}", (r.index[-1], eq(r).iloc[-1]), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, color=MUTED)
ax.annotate(f"{eq(pooled_bh).iloc[-1]:.1f}", (pooled_bh.index[-1], eq(pooled_bh).iloc[-1]), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, color=MUTED)
for d, t in ((pd.Timestamp("2026-02-28"), "after: not used to choose rule"), (pd.Timestamp("2026-10-25 20:00"), "freeze")):
    ax.axvline(d, color=MUTED, lw=1, ls=":")
ax.set_title("Pooled 8-asset portfolio (equal weight, 0.1% fee/side). Dotted lines: 2026-02-28 and freeze", loc="left", fontsize=10, color=INK)
ax.legend(frameon=False, fontsize=9, loc="upper left")
axes = [fig.add_subplot(gs[1 + i // 2, i % 2]) for i in range(8)]
key = VARS[0][0]
for a, x in zip(ASSETS, axes):
    r = curve(a, ev[key][a]).ret
    x.plot(bh[a].index, eq(bh[a]), color="#8a8984", lw=1.6, ls="--", label="Buy & hold")
    x.plot(r.index, eq(r), color="#eb6834", lw=2, label="Frozen rule")
    x.annotate(f"{eq(r).iloc[-1]:.1f}", (r.index[-1], eq(r).iloc[-1]), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, color=MUTED)
    x.annotate(f"{eq(bh[a]).iloc[-1]:.1f}", (bh[a].index[-1], eq(bh[a]).iloc[-1]), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, color=MUTED)
    x.axvline(pd.Timestamp("2026-02-28"), color=MUTED, lw=1, ls=":")
    x.set_title(a.replace("USDT", ""), loc="left", fontsize=9, color=INK)
axes[0].legend(frameon=False, fontsize=8, loc="upper left")
for x in [ax] + axes:
    x.set_yscale("log"); x.set_facecolor("#fcfcfb"); x.grid(color=GRID, lw=0.8); x.set_axisbelow(True)
    for sp in ("top", "right"): x.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): x.spines[sp].set_color(GRID)
    x.tick_params(colors=MUTED, labelsize=8); x.margins(x=0.06); x.set_ylabel("growth of $1 (log)", fontsize=8, color=MUTED)
fig.patch.set_facecolor("#fcfcfb"); fig.tight_layout()
fig.savefig(f"{ROOT}/src/meta_labeling/results/breakout_curves.png", dpi=120); print("ok")
