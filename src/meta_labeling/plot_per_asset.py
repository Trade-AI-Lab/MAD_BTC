"""Per-asset growth of $1: buy & hold vs `dz top third + volume` rule (lookback 5 and 10), 4h, walk-forward OOS."""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.abspath(__file__)[: os.path.abspath(__file__).find("MAD_BTC") + len("MAD_BTC")]
D = f"{ROOT}/src/meta_labeling/results"
curves = pd.read_pickle(f"{D}/per_asset_curves.pkl")
assets = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "DOTUSDT", "SOLUSDT", "XRPUSDT", "LINKUSDT"]
STYLE = [("buy&hold", "#8a8984", "--", "Buy & hold"), ("rule lb5", "#2a78d6", "-", "Rule, lookback 5"), ("rule lb10", "#eb6834", "-", "Rule, lookback 10")]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
fig, axes = plt.subplots(4, 2, figsize=(12, 13))
for ax, a in zip(axes.flat, assets):
    for k, col, ls, lab in STYLE:
        eq = (1 + curves[(a, k)]).cumprod()
        ax.plot(eq.index, eq, color=col, lw=2, ls=ls, label=lab)
        ax.annotate(f"{eq.iloc[-1]:.2f}", (eq.index[-1], eq.iloc[-1]), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, color=MUTED)
    ax.set_yscale("log"); ax.set_facecolor("#fcfcfb"); ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8); ax.margins(x=0.08)
    ax.set_title(a.replace("USDT", ""), loc="left", fontsize=10, color=INK)
    ax.set_ylabel("growth of $1 (log)", fontsize=8, color=MUTED)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.995))
fig.suptitle("dz top third + volume rule vs buy & hold, per asset (4h, walk-forward OOS from Sep 2021, 0.1% fee/side)", y=0.97, x=0.01, ha="left", fontsize=11, color=INK)
fig.patch.set_facecolor("#fcfcfb"); fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(f"{D}/per_asset.png", dpi=125); print("ok")
