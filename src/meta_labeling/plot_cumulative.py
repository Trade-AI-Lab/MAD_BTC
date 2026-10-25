"""Cumulative growth of $1 for the filters, 4h. Rows = MAD lookback, columns = original assets / SOL-XRP-LINK."""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(__file__)[: os.path.abspath(__file__).find("MAD_BTC") + len("MAD_BTC")]
D = f"{ROOT}/src/meta_labeling"
# fixed order, never cycled (reference categorical palette slots 1,2,3,5; buy & hold = neutral)
STYLE = {
    "buy&hold": ("#8a8984", "Buy & hold", "--"),
    "base": ("#2a78d6", "All MAD trades", "-"),
    "gboost": ("#eb6834", "ML filter (gboost)", "-"),
    "dz>q67": ("#1baf7a", "dz top third", "-"),
    "dz67&vol": ("#e87ba4", "dz top third + volume", "-"),
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
cols = [("dev", "BTC, ETH, BNB, ADA, DOT"), ("frozen", "SOL, XRP, LINK (rebuilt from 1h)")]
fig, axes = plt.subplots(3, 2, figsize=(13, 11), sharex=False)
for r, lb in enumerate((5, 10, 20)):
    for c, (tag, title) in enumerate(cols):
        ax = axes[r, c]
        s = pd.read_csv(f"{D}/series_{tag}_4h_lb{lb}.csv", index_col=0, parse_dates=True)
        eq = (1 + s).cumprod()
        for k, (col, lab, ls) in STYLE.items():
            ax.plot(eq.index, eq[k], color=col, lw=2, ls=ls, label=lab)
            ax.annotate(f"{eq[k].iloc[-1]:.2f}", (eq.index[-1], eq[k].iloc[-1]), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=8, color=MUTED)
        ax.set_yscale("log")
        ax.set_facecolor("#fcfcfb")
        ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"): ax.spines[sp].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_title(f"lookback {lb} | {title}", loc="left", fontsize=10, color=INK)
        ax.set_ylabel("growth of $1 (log)", fontsize=8, color=MUTED)
        ax.margins(x=0.06)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=5, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.995))
fig.suptitle("MAD entry filters, walk-forward out of sample, 0.1% fee per side, equal weight", y=0.97, fontsize=11, color=INK, x=0.01, ha="left")
fig.patch.set_facecolor("#fcfcfb")
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = f"{D}/cumulative_returns.png"
fig.savefig(out, dpi=130)
print(out)
