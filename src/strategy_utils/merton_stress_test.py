import os
import sys
import copy
import multiprocessing
import numpy as np
import pandas as pd
import optuna

absolute_path = os.path.abspath(__file__)[
    : os.path.abspath(__file__).find("MAD_BTC") + len("MAD_BTC")
]
sys.path.append(absolute_path)


# ---------------------------------------------------------------------------
# Merton jump-diffusion path generation
# ---------------------------------------------------------------------------

def generate_merton_path(
    S0: float,
    mu: float,
    sigma: float,
    lam: float,
    mu_j: float,
    sigma_j: float,
    n_steps: int,
    dt: float,
    seed: int = None,
) -> np.ndarray:
    """
    Generate a Merton jump-diffusion price path.

    dS = (mu - lambda*kappa)*S dt + sigma*S dW + S dJ

    :param S0: Initial price.
    :param mu: Annualised drift.
    :param sigma: Annualised diffusion volatility.
    :param lam: Jump intensity (expected jumps per year).
    :param mu_j: Mean log-jump size.
    :param sigma_j: Std of log-jump size.
    :param n_steps: Number of time steps (bars).
    :param dt: Time step length in years (e.g. 4h → 4/8760).
    :param seed: Optional RNG seed for reproducibility.
    :return: Price array of length n_steps + 1 (includes S0).
    """
    rng = np.random.default_rng(seed)

    kappa = np.exp(mu_j + 0.5 * sigma_j**2) - 1  # E[e^J - 1]
    drift = (mu - 0.5 * sigma**2 - lam * kappa) * dt

    dW = rng.normal(0.0, np.sqrt(dt), n_steps)
    n_jumps = rng.poisson(lam * dt, n_steps)

    # Vectorised jump component: generate all jump sizes at once, then
    # accumulate into each time step using np.add.at (no Python loop).
    total_jumps = int(n_jumps.sum())
    jump_component = np.zeros(n_steps)
    if total_jumps > 0:
        all_jumps = rng.normal(mu_j, sigma_j, total_jumps)
        step_indices = np.repeat(np.arange(n_steps), n_jumps)
        np.add.at(jump_component, step_indices, all_jumps)

    log_returns = drift + sigma * dW + jump_component
    prices = S0 * np.exp(np.concatenate([[0.0], np.cumsum(log_returns)]))
    return prices  # length n_steps + 1


def path_to_ohlc(
    prices: np.ndarray,
    start_date: pd.Timestamp,
    freq: str = "4h",
    intrabar_noise: float = 0.005,
    seed: int = None,
) -> pd.DataFrame:
    """
    Convert a Merton price path to an OHLC DataFrame compatible with the
    trading_strategy base class (requires open_date, open, high, low, close).

    :param prices: Price array of length n_steps + 1 from generate_merton_path.
    :param start_date: First bar's open_date.
    :param freq: Pandas frequency string matching the strategy timeframe.
    :param intrabar_noise: Fraction of bar range added to high / subtracted from low.
    :param seed: Optional RNG seed.
    :return: DataFrame with columns [open_date, open, high, low, close].
    """
    rng = np.random.default_rng(seed)
    n = len(prices) - 1  # number of complete bars

    opens = prices[:-1]
    closes = prices[1:]
    noise = rng.uniform(0.0, intrabar_noise, n)

    highs = np.maximum(opens, closes) * (1.0 + noise)
    lows = np.minimum(opens, closes) * (1.0 - noise)

    dates = pd.date_range(start=start_date, periods=n, freq=freq)

    return pd.DataFrame({
        "open_date": dates,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
    })


# ---------------------------------------------------------------------------
# Scenario library
# ---------------------------------------------------------------------------

def build_scenarios(base: dict) -> list[dict]:
    """
    Build a set of stress-test scenarios by perturbing the base Merton params.
    Each scenario is a dict of kwargs for generate_merton_path (minus S0/n_steps/dt).

    :param base: Fitted base params (mu, sigma, lam, mu_j, sigma_j).
    :return: List of scenario dicts.
    """
    return [
        {"name": "historical",   **base},
        {"name": "bull",         **{**base, "mu": abs(base["mu"]) * 2.0}},
        {"name": "bear",         **{**base, "mu": -abs(base["mu"])}},
        {"name": "crash",        **{**base, "lam": base["lam"] * 4, "mu_j": -0.15, "sigma_j": 0.08}},
        {"name": "high_vol",     **{**base, "sigma": base["sigma"] * 2.0}},
        {"name": "low_vol",      **{**base, "mu": 0.05, "sigma": base["sigma"] * 0.4, "lam": 0.5}},
        {"name": "sideways",     **{**base, "mu": 0.0, "sigma": base["sigma"] * 0.6}},
    ]


# ---------------------------------------------------------------------------
# Multiprocessing worker (module-level — required for pickling on all platforms)
# ---------------------------------------------------------------------------

# Each worker process caches its own strategy instance to avoid re-reading the
# pkl file for every task. The initializer sets this global once per process.
_worker_strategy = None


def _init_merton_worker(strategy_class, config: dict) -> None:
    """Called once per worker process. Reads the pkl file exactly once."""
    global _worker_strategy
    cfg = copy.deepcopy(config)
    cfg.pop("begin_date", None)
    cfg.pop("end_date", None)
    _worker_strategy = strategy_class(cfg)
    _worker_strategy._suppress_save = True


def _merton_path_worker(args: tuple) -> float:
    """
    Score one (trial_params, Merton scenario, path seed) task.
    Reuses the per-process cached strategy; only swaps parent_df.
    """
    global _worker_strategy
    trial_params, S0, n_bars, dt, merton_params, path_seed, start_date, freq = args

    prices = generate_merton_path(
        S0=S0, n_steps=n_bars, dt=dt, seed=path_seed, **merton_params
    )
    synthetic_df = path_to_ohlc(prices, start_date=start_date, freq=freq, seed=path_seed)

    # Swap strategy state for this task — no pkl re-read needed.
    # Apply every Optuna param generically so this worker handles both plain
    # MAD and regime-gated MAD (which adds adx_window / adx_threshold). The only
    # special case is y, which the objective stores as an offset from x.
    for key, value in trial_params.items():
        _worker_strategy.config["strategy"][key] = value
    if "y" in trial_params:
        _worker_strategy.config["strategy"]["y"] = (
            trial_params["x"] + trial_params["y"]
        )
    _worker_strategy.parent_df = synthetic_df
    _worker_strategy.child_df = None
    _worker_strategy.evolution_df = None
    _worker_strategy.begin_date = None
    _worker_strategy.end_date = None

    try:
        sim_df = _worker_strategy.compute_simulation()
    except Exception:
        return -np.inf

    if sim_df is None or len(sim_df) < 3:
        return -np.inf

    portfolio_returns = sim_df["Portfolio"].pct_change().dropna()
    bh_returns = sim_df["Buy_and_Hold"].pct_change().dropna()
    excess = portfolio_returns - bh_returns

    if len(excess) < 2 or excess.std() == 0 or np.isnan(excess.std()):
        return -np.inf

    return (excess.mean() / excess.std()) * np.sqrt(365)


# ---------------------------------------------------------------------------
# MertonStressTest
# ---------------------------------------------------------------------------

class MertonStressTest:
    """
    Validate the best Optuna params against synthetic Merton price paths.

    Usage:
        stress = MertonStressTest(strategy_class, base_config, study, freq="4h")
        stress.fit_merton_params(historical_df)
        results = stress.run(top_k=10, n_paths_per_scenario=50)
        print(results)
    """

    PERIODS_PER_YEAR = {"1h": 8760, "4h": 2190, "1d": 365}

    def __init__(self, strategy_class, base_config: dict, study: optuna.Study, freq: str = "4h"):
        """
        :param strategy_class: The strategy class to instantiate (e.g. momentum_strategy).
        :param base_config: Full config dict used during WFO (dates will be overridden).
        :param study: Completed Optuna study from WFO.
        :param freq: Pandas/bar frequency string matching the strategy timeframe.
        """
        self.strategy_class = strategy_class
        self.base_config = base_config
        self.study = study
        self.freq = freq
        self.merton_base_params = None

    # ------------------------------------------------------------------
    # Param fitting
    # ------------------------------------------------------------------

    def fit_merton_params(self, df: pd.DataFrame) -> dict:
        """
        Estimate Merton params from a historical OHLC DataFrame.
        Jumps are identified as log-returns beyond 3 standard deviations.

        :param df: DataFrame with at least an 'open' column and 'open_date'.
        :return: Dict with keys mu, sigma, lam, mu_j, sigma_j.
        """
        ppy = self.PERIODS_PER_YEAR.get(self.freq, 365)
        log_ret = np.log(df["open"] / df["open"].shift(1)).dropna()

        mu = log_ret.mean() * ppy
        sigma_raw = log_ret.std() * np.sqrt(ppy)
        threshold = 3.0 * log_ret.std()

        jumps = log_ret[np.abs(log_ret) > threshold]
        lam = len(jumps) / (len(log_ret) / ppy)
        mu_j = float(jumps.mean()) if len(jumps) > 0 else 0.0
        sigma_j = float(jumps.std()) if len(jumps) > 1 else sigma_raw * 0.1

        # Adjust diffusion vol: var(total) = var(diffusion) + lam*(mu_j^2 + sigma_j^2)
        jump_var = lam * (mu_j**2 + sigma_j**2) / ppy
        sigma = np.sqrt(max(sigma_raw**2 - jump_var, sigma_raw**2 * 0.1))

        params = dict(mu=mu, sigma=sigma, lam=lam, mu_j=mu_j, sigma_j=sigma_j)
        self.merton_base_params = params
        print(f"Fitted Merton params: {params}")
        return params

    # ------------------------------------------------------------------
    # Core runner — parallelised with multiprocessing.Pool
    # ------------------------------------------------------------------

    def run(
        self,
        top_k: int = 10,
        n_paths_per_scenario: int = 50,
        n_bars: int = 500,
        S0: float = 10_000.0,
        start_date: pd.Timestamp = None,
        n_workers: int = None,
    ) -> pd.DataFrame:
        """
        Stress-test the top-K Optuna trials against synthetic Merton paths.

        All (trial × scenario × path) evaluations are dispatched to a
        multiprocessing.Pool. Each worker process reads the pkl file once via
        _init_merton_worker, then reuses the cached strategy for every task.

        :param top_k: Number of best trials (by WFO score) to test.
        :param n_paths_per_scenario: Synthetic paths generated per scenario per param set.
        :param n_bars: Number of bars per synthetic path.
        :param S0: Initial synthetic price.
        :param start_date: Start date for synthetic OHLC timestamps.
        :param n_workers: Worker processes (defaults to cpu_count).
        :return: DataFrame ranked by robustness, with columns:
                 [rank, x, y, ratio_method, wfo_score, robustness,
                  mean_score, scenario_breakdown].
        """
        if self.merton_base_params is None:
            raise RuntimeError("Call fit_merton_params() before run().")

        if start_date is None:
            start_date = pd.Timestamp("2020-01-01")

        ppy = self.PERIODS_PER_YEAR.get(self.freq, 365)
        dt = 1.0 / ppy
        scenarios = build_scenarios(self.merton_base_params)
        n_scenarios = len(scenarios)

        best_trials = sorted(
            [t for t in self.study.trials if t.value is not None and not np.isinf(t.value)],
            key=lambda t: t.value,
            reverse=True,
        )[:top_k]

        # Build a flat task list ordered as [trial][scenario][path] so we can
        # reassemble by slicing without tracking individual indices.
        tasks = []
        for trial in best_trials:
            for scenario in scenarios:
                s_params = {k: v for k, v in scenario.items() if k != "name"}
                for path_idx in range(n_paths_per_scenario):
                    seed = trial.number * 10_000 + hash(scenario["name"]) % 10_000 + path_idx
                    tasks.append((
                        trial.params, S0, n_bars, dt,
                        s_params, seed, start_date, self.freq,
                    ))

        n_workers = n_workers or multiprocessing.cpu_count()
        chunksize = max(1, len(tasks) // (n_workers * 4))
        print(f"Merton stress test: {len(tasks)} tasks across {n_workers} workers "
              f"(chunksize={chunksize})")

        with multiprocessing.Pool(
            processes=n_workers,
            initializer=_init_merton_worker,
            initargs=(self.strategy_class, self.base_config),
        ) as pool:
            all_scores = pool.map(_merton_path_worker, tasks, chunksize=chunksize)

        # Reassemble flat scores → nested (trial, scenario) structure.
        records = []
        for trial_idx, trial in enumerate(best_trials):
            params = trial.params
            scenario_scores = {}

            for scenario_idx, scenario in enumerate(scenarios):
                s_name = scenario["name"]
                start_idx = (
                    trial_idx * n_scenarios * n_paths_per_scenario
                    + scenario_idx * n_paths_per_scenario
                )
                fold_scores = all_scores[start_idx : start_idx + n_paths_per_scenario]
                valid = [s for s in fold_scores if not np.isinf(s)]
                scenario_scores[s_name] = {
                    "robustness": float(np.mean([s > 0 for s in fold_scores])),
                    "mean_score": float(np.mean(valid)) if valid else -np.inf,
                }

            robustness_per_scenario = [sc["robustness"] for sc in scenario_scores.values()]
            records.append({
                "trial_number": trial.number,
                "x": params.get("x"),
                "y": params.get("x", 0) + params.get("y", 0),
                "ratio_method": params.get("ratio_method"),
                "wfo_score": trial.value,
                "robustness": float(np.mean(robustness_per_scenario)),
                "scenario_breakdown": scenario_scores,
            })

        results = pd.DataFrame(records).sort_values("robustness", ascending=False)
        results.insert(0, "rank", range(1, len(results) + 1))
        return results.reset_index(drop=True)


# ---------------------------------------------------------------------------
# CLI entry point — run after WFO study is complete
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import toml
    from strategy_utils.launcher_strategy import (
        momentum_strategy,
        regime_momentum_strategy,
    )

    # Optional CLI arg selects which strategy/study to stress-test:
    #   python merton_stress_test.py [plain|regime]   (default: plain)
    variant = sys.argv[1] if len(sys.argv) > 1 else "plain"
    strategy_class = {
        "plain": momentum_strategy,
        "regime": regime_momentum_strategy,
    }[variant]

    with open("src/configs/optimiser_momentum_config.toml", "r") as f:
        config = toml.load(f)

    crypto = config["crypto_currency"]
    timeframe = config["time"]
    exchange = config["exchange"]
    # The regime variant runs under its own study name (see the optimiser).
    strategy_name = (
        "regime_momentum_strategy" if variant == "regime" else config["strategy"]["name"]
    )

    study_name = f"study_{strategy_name}_{crypto}_{exchange}_{timeframe}"
    storage = f"sqlite:///{os.path.join(absolute_path, 'study', study_name)}.db"
    study = optuna.load_study(study_name=study_name, storage=storage)

    from utils.utils_df import open_dataframe
    data_path = os.path.join(absolute_path, f"data/{exchange}-{crypto}-{timeframe}.pkl")
    hist_df = open_dataframe(data_path)

    stress = MertonStressTest(
        strategy_class=strategy_class,
        base_config=config,
        study=study,
        freq=timeframe,
    )
    stress.fit_merton_params(hist_df)
    results = stress.run(top_k=10, n_paths_per_scenario=50)

    print("\n=== Merton Stress Test Results ===")
    print(results[["rank", "x", "y", "ratio_method", "wfo_score", "robustness"]].to_string(index=False))

    output_path = os.path.join(absolute_path, f"study/merton_stress_{study_name}.csv")
    results.drop(columns=["scenario_breakdown"]).to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
