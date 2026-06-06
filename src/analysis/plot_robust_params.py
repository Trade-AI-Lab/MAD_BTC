"""
Visualise the robust best parameter set per study.

For each (coin, timeframe) regime study this script:
  1. Re-evaluates the saved trial configs under the *current* consistency-
     penalised WFO objective (mean_OOS_IR * fraction-of-folds-beating-B&H) and
     picks the most robust one — NOT the old spiky IR-cap "best".
  2. Renders two figures on the FULL price history for that robust config:
       - a trading chart: price + MAD ratio + ADX regime, with buy/sell markers
       - the PnL evolution: portfolio vs buy-and-hold (+ % change, RSI)
Figures are written to study/robust_plots/.

Usage:
    PYTHONPATH=src python src/analysis/plot_robust_params.py [plain|regime]
"""

import os
import sys
import ast

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt

absolute_path = os.path.abspath(__file__)[
    : os.path.abspath(__file__).find("MAD_BTC") + len("MAD_BTC")
]
sys.path.insert(0, os.path.join(absolute_path, "src"))

from strategy_utils.launcher_strategy import (  # noqa: E402
    momentum_strategy,
    regime_momentum_strategy,
)
from multi_optimiser_momentum_strategy import (  # noqa: E402
    optimiser_momentum_strategy,
    optimiser_regime_momentum_strategy,
    compute_sharpe_ratio,
    compute_maximum_drawdown,
)

COINS = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
TIMEFRAMES = ["1h", "4h"]
WFO = {"1h": (12, 3), "4h": (18, 6)}  # (train_months, val_months)

VARIANTS = {
    "plain": ("momentum_strategy", momentum_strategy, optimiser_momentum_strategy),
    "regime": (
        "regime_momentum_strategy",
        regime_momentum_strategy,
        optimiser_regime_momentum_strategy,
    ),
}


def find_robust_config(opt, saved_csv, top_n=25):
    """
    Re-score the top saved configs (by their old objective) under the current
    consistency-penalised objective and return the strategy-config dict of the
    most robust one, plus its recomputed score.
    """
    df = pd.read_csv(saved_csv)
    df = df[df["result"].apply(lambda v: str(v) not in ("-inf", "inf", "nan"))]
    if len(df) == 0:
        return None, None
    df["result"] = df["result"].astype(float)
    # Re-score only the strongest candidates to keep this cheap.
    df = df.sort_values("result", ascending=False).head(top_n)

    best_cfg, best_score = None, -np.inf
    for _, row in df.iterrows():
        cfg = ast.literal_eval(row["config"])
        opt.config["strategy"] = cfg["strategy"]
        opt.strategy.change_config(opt.config)
        score, _, _ = opt._score_on_windows()
        if score > best_score:
            best_score, best_cfg = score, cfg["strategy"]
    return best_cfg, best_score


def plot_trading_chart(strat, title, out_path):
    """Price + MAD ratio + ADX regime, with buy/sell entry markers."""
    df = strat.child_df  # open_date is a column here (rules computed, sim on copy)
    t = pd.to_datetime(df["open_date"])
    buys = df[df["buy_signal"] > 0]
    sells = df[df["sell_signal"] > 0]
    has_adx = "adx" in df.columns

    # Reconstruct the held (long) intervals so we can shade "buy-and-hold"
    # periods. A buy opens a long, the next sell closes it: between them the
    # strategy is simply holding, which is the classic MAD behaviour and makes
    # the chart readable (vs. a confusing scatter of individual markers).
    events = df[(df["buy_signal"] > 0) | (df["sell_signal"] > 0)].copy()
    events["t"] = pd.to_datetime(events["open_date"])
    held_intervals = []
    open_t = None
    for _, r in events.iterrows():
        if r["buy_signal"] > 0 and open_t is None:
            open_t = r["t"]
        elif r["sell_signal"] > 0 and open_t is not None:
            held_intervals.append((open_t, r["t"]))
            open_t = None
    if open_t is not None:
        held_intervals.append((open_t, t.iloc[-1]))

    n_buys, n_sells = int(df["buy_signal"].sum()), int(df["sell_signal"].sum())

    def draw_price(ax, sub_df):
        st = pd.to_datetime(sub_df["open_date"])
        ax.plot(st, sub_df["open"], color="black", lw=0.8, label="Open price")
        for a, b in held_intervals:
            ax.axvspan(a, b, color="green", alpha=0.10)
        sb = sub_df[sub_df["buy_signal"] > 0]
        ss = sub_df[sub_df["sell_signal"] > 0]
        ax.scatter(pd.to_datetime(sb["open_date"]), sb["open"], marker="^",
                   color="green", s=55, zorder=3, label="Buy")
        ax.scatter(pd.to_datetime(ss["open_date"]), ss["open"], marker="v",
                   color="red", s=55, zorder=3, label="Sell")
        ax.grid(True, alpha=0.3)

    nrows = 4 if has_adx else 3  # +1 row for the zoomed panel
    fig, axes = plt.subplots(nrows, 1, figsize=(16, 4 * nrows))
    ax_price, ax_zoom, ax_ratio = axes[0], axes[1], axes[2]

    # Full-history price with shaded long-held periods + markers.
    draw_price(ax_price, df)
    ax_price.set_ylabel("Price")
    ax_price.set_title(f"{title}\n{n_buys} buys / {n_sells} sells over {len(df)} bars "
                       f"(green = held long)")
    ax_price.legend(loc="upper left")

    # Zoom into the single year with the most signals so flips are readable.
    yrs = t.dt.year
    busiest = (events["t"].dt.year.value_counts().idxmax()
               if len(events) else int(yrs.iloc[0]))
    zmask = yrs == busiest
    draw_price(ax_zoom, df[zmask.values])
    ax_zoom.set_xlim(pd.Timestamp(f"{busiest}-01-01"), pd.Timestamp(f"{busiest}-12-31"))
    ax_zoom.set_ylabel("Price")
    ax_zoom.set_title(f"Zoom: {busiest} (busiest year — see entry/exit timing)")
    ax_zoom.legend(loc="upper left")

    ax_ratio.plot(t, df[strat.strategy_name], color="blue", lw=0.8, label="MAD ratio")
    ax_ratio.axhline(1.0, color="grey", ls="--", lw=0.8)
    ax_ratio.set_ylabel("MAD ratio")
    ax_ratio.legend(loc="upper left")
    ax_ratio.grid(True, alpha=0.3)

    if has_adx:
        ax_adx = axes[3]
        thr = strat.config["strategy"].get("adx_threshold", 25)
        ax_adx.plot(t, df["adx"], color="purple", lw=0.8, label="ADX")
        ax_adx.axhline(thr, color="orange", ls="--", lw=0.8, label=f"threshold={thr}")
        ax_adx.fill_between(t, 0, df["adx"], where=df["adx"] > thr,
                            color="purple", alpha=0.1, label="trending")
        ax_adx.set_ylabel("ADX")
        ax_adx.legend(loc="upper left")
        ax_adx.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close()


def plot_pnl(evolution_df, title, out_path):
    """Portfolio vs buy-and-hold equity, % change, and drawdown."""
    t = pd.to_datetime(evolution_df["Time"])
    pf = evolution_df["Portfolio"]
    bh = evolution_df["Buy_and_Hold"]
    run_max = pf.cummax()
    dd = (pf - run_max) / run_max * 100

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    ax1.plot(t, pf, color="b", marker="o", ms=3, label="Portfolio")
    ax1.plot(t, bh, color="g", marker="x", ms=3, label="Buy & Hold")
    ax1.set_ylabel("Equity")
    ax1.set_title(title)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2.fill_between(t, dd, 0, color="red", alpha=0.4)
    ax2.set_ylabel("Drawdown (%)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close()


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else "regime"
    strat_name, strat_cls, opt_cls = VARIANTS[variant]

    out_dir = os.path.join(absolute_path, "study", "robust_plots")
    os.makedirs(out_dir, exist_ok=True)

    summary = []
    for coin in COINS:
        for tf in TIMEFRAMES:
            saved = os.path.join(
                absolute_path,
                f"experience/{coin}/binance/{tf}/{strat_name}/{coin}_{tf}_{strat_name}.csv",
            )
            if not os.path.exists(saved):
                print(f"skip {coin}_{tf}: no saved CSV")
                continue

            base_cfg = dict(
                crypto_currency=coin, time=tf, exchange="binance",
                broker="binance", initial_capital=1000,
                strategy=dict(name=strat_name),
            )
            opt = opt_cls(config=base_cfg)
            tr, va = WFO[tf]
            opt.generate_wfo_windows(train_months=tr, val_months=va, n_splits=None)

            robust_strat, score = find_robust_config(opt, saved)
            if robust_strat is None:
                print(f"skip {coin}_{tf}: no valid config")
                continue
            print(f"{coin}_{tf} robust: {robust_strat} (consistency-obj={score:.3f})")

            # Render on the FULL history (no date filtering) for the chart + PnL.
            cfg = dict(base_cfg)
            cfg["strategy"] = robust_strat
            s = strat_cls(cfg)
            s._suppress_save = True
            s.begin_date = None
            s.end_date = None

            # child_df for the trading chart (open_date stays a column).
            s.compute_needed_features()
            s.strategy_rules()
            tag = f"{coin}_{tf}_{robust_strat['ratio_method']}_x{robust_strat['x']}_y{robust_strat['y']}"
            plot_trading_chart(
                s, f"{coin} {tf} — {strat_name} trades\n{robust_strat}",
                os.path.join(out_dir, f"trades_{variant}_{tag}.png"),
            )

            # Fresh instance for the simulation (loop_simulation mutates child_df).
            s2 = strat_cls(cfg)
            s2._suppress_save = True
            s2.begin_date = None
            s2.end_date = None
            evo = s2.compute_simulation()
            sharpe = compute_sharpe_ratio(evo)
            mdd = compute_maximum_drawdown(evo, "Portfolio")
            final = evo["Portfolio"].iloc[-1]
            bh_final = evo["Buy_and_Hold"].iloc[-1]
            plot_pnl(
                evo,
                f"{coin} {tf} — PnL  final={final:.0f} vs B&H={bh_final:.0f}  "
                f"Sharpe={sharpe:.2f}  MaxDD={mdd}%",
                os.path.join(out_dir, f"pnl_{variant}_{tag}.png"),
            )

            summary.append(
                dict(study=f"{coin}_{tf}", **robust_strat,
                     consistency_obj=round(score, 3), sharpe=round(sharpe, 2),
                     max_dd_pct=mdd, final=round(final, 0), bh_final=round(bh_final, 0))
            )

    if summary:
        sdf = pd.DataFrame(summary)
        pd.set_option("display.width", 200)
        print("\n=== Robust params summary ===")
        print(sdf.to_string(index=False))
        print(f"\nPlots written to {out_dir}")


if __name__ == "__main__":
    main()
