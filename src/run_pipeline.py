#!/usr/bin/env python3
"""
WFO + Merton stress-test pipeline.

Run from the MAD_BTC root:
    python src/run_pipeline.py
"""
import os
import sys
import asyncio
import toml
import optuna

absolute_path = os.path.abspath(__file__)[
    : os.path.abspath(__file__).find("MAD_BTC") + len("MAD_BTC")
]
sys.path.append(absolute_path)
sys.path.append(os.path.join(absolute_path, "src"))

from multi_optimiser_momentum_strategy import optimiser_momentum_strategy
from strategy_utils.launcher_strategy import momentum_strategy
from strategy_utils.merton_stress_test import MertonStressTest
from utils.utils_df import open_dataframe
from utils.data_checks import check_data_availability_in_folder
from utils.downloader import downloader

# ── Settings ───────────────────────────────────────────────────────────────────
ASSETS    = ["BTCUSDT", "ETHUSDT"]
TIMEFRAME = "1d"
EXCHANGE  = "binance"
TOP_K     = 10       # top WFO trials passed to Merton
N_PATHS   = 50       # synthetic paths per scenario per trial
N_BARS    = 500      # bars per synthetic path (~83 days at 4h)

# ── Config ─────────────────────────────────────────────────────────────────────
config_path = os.path.join(absolute_path, "src", "configs", "optimiser_momentum_config.toml")
with open(config_path, "r") as f:
    config = toml.load(f)

# ── Step 1: Data availability ──────────────────────────────────────────────────
print("=" * 60)
print("Step 1: Data availability")
print("=" * 60)

missing = check_data_availability_in_folder(
    os.path.join(absolute_path, "data"), [EXCHANGE], ASSETS, [TIMEFRAME]
)
if missing[0]:  # non-empty currencies list → some files missing
    print(f"Downloading: symbols={missing[0]}  tf={missing[1]}  broker={missing[2]}")
    dl_cfg_path = os.path.join(absolute_path, "src", "configs", "download_config.toml")
    with open(dl_cfg_path, "r") as f:
        dl_cfg = toml.load(f)
    dl_cfg["symbols"]    = missing[0]
    dl_cfg["time_frame"] = missing[1]
    dl_cfg["broker"]     = missing[2]
    asyncio.get_event_loop().run_until_complete(downloader(dl_cfg))
    print("Download complete.")
else:
    print("All data present.")

# ── Step 2: Walk-Forward Optimization ─────────────────────────────────────────
print()
print("=" * 60)
print("Step 2: Walk-Forward Optimization (Optuna)")
print(f"  Assets    : {ASSETS}")
print(f"  Timeframe : {TIMEFRAME}")
print(f"  WFO cfg   : {config.get('wfo', {})}")
print("=" * 60)

optimiser_momentum_strategy(config=config).optimiser_multiple_currency(
    list_crypto_currency=ASSETS,
    list_time=[TIMEFRAME],
    list_brokers=[EXCHANGE],
)

# ── Step 3: Merton stress test ─────────────────────────────────────────────────
print()
print("=" * 60)
print("Step 3: Merton Stress Test")
print("=" * 60)

strategy_name = config["strategy"]["name"]

for crypto in ASSETS:
    print(f"\n── {crypto} {TIMEFRAME} ──")

    study_name = f"study_{strategy_name}_{crypto}_{EXCHANGE}_{TIMEFRAME}"
    storage    = f"sqlite:///{os.path.join(absolute_path, 'study', study_name)}.db"
    study      = optuna.load_study(study_name=study_name, storage=storage)

    data_path = os.path.join(absolute_path, "data", f"{EXCHANGE}-{crypto}-{TIMEFRAME}.pkl")
    hist_df   = open_dataframe(data_path)

    # Per-asset config so each Merton worker loads the right pkl file.
    asset_cfg = {
        **config,
        "crypto_currency": crypto,
        "time": TIMEFRAME,
        "exchange": EXCHANGE,
    }

    stress = MertonStressTest(
        strategy_class=momentum_strategy,
        base_config=asset_cfg,
        study=study,
        freq=TIMEFRAME,
    )
    stress.fit_merton_params(hist_df)
    results = stress.run(top_k=TOP_K, n_paths_per_scenario=N_PATHS, n_bars=N_BARS)

    print(f"\n=== Merton Results: {crypto} {TIMEFRAME} ===")
    print(
        results[["rank", "x", "y", "ratio_method", "wfo_score", "robustness"]]
        .to_string(index=False)
    )

    output_path = os.path.join(absolute_path, "study", f"merton_stress_{study_name}.csv")
    results.drop(columns=["scenario_breakdown"]).to_csv(output_path, index=False)
    print(f"Saved → {output_path}")

print("\nAll done.")
