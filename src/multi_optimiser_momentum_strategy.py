import os
import optuna
import copy
import toml
import asyncio
import numpy as np
import pandas as pd
import imageio
import multiprocessing
from itertools import product
from strategy_utils.launcher_strategy import (
    momentum_strategy,
    band_momentum_strategy,
    regime_momentum_strategy,
    envelop_strategy,
    signal_momentum_strategy,
)
from strategy_utils.strategy_comparator import evalutation
from utils.utils_signature import create_signature
from utils.data_checks import check_data_availability_in_folder
from utils.downloader import downloader

absolute_path = os.path.abspath(__file__)[
    : os.path.abspath(__file__).find("MAD_BTC") + len("MAD_BTC")
]


def compute_sharpe_ratio(
    df, column: str = "Portfolio", risk_free_rate=0.0, periods_per_year=None
):
    """
    Annualised Sharpe ratio.

    IMPORTANT: evolution_df has one row *per trade*, not per day, and trades are
    irregularly spaced (often ~weeks apart). The annualisation factor must
    therefore be derived from the actual elapsed time between rows, not a
    hardcoded sqrt(365) — otherwise the Sharpe is distorted by trade frequency
    and can land at absurd values like -16 or +14.

    When a 'Time' column is present, periods_per_year is inferred from the mean
    spacing between rows: periods_per_year = 365 / mean_days_between_trades.
    Pass an explicit periods_per_year to override (e.g. for genuinely periodic
    series). Falls back to 365 if the spacing cannot be determined.
    """
    returns = df[column].pct_change().dropna()
    if len(returns) < 2:
        return np.nan

    if periods_per_year is None:
        if "Time" in df.columns:
            t = pd.to_datetime(df["Time"])
            mean_gap_days = t.diff().dropna().dt.total_seconds().mean() / 86400.0
            periods_per_year = (
                365.0 / mean_gap_days if mean_gap_days and mean_gap_days > 0 else 365.0
            )
        else:
            periods_per_year = 365.0

    excess_returns = returns - risk_free_rate / periods_per_year

    std_excess_return = excess_returns.std()
    if std_excess_return == 0 or np.isnan(std_excess_return):
        return np.nan

    sharpe_ratio = (
        np.sqrt(periods_per_year) * excess_returns.mean() / std_excess_return
    )
    return sharpe_ratio


def compute_maximum_drawdown(df, column: str):
    series = df[column]

    # Running maximum
    running_max = series.cummax()

    # Drawdown series (in percentage)
    drawdown = (series - running_max) / running_max

    # Maximum drawdown (positive percentage)
    max_dd = drawdown.min()

    return round(abs(max_dd) * 100, 3)


class trading_optimisation:
    """
    Trading Optimization

    This class provides a framework for optimizing trading strategies using Optuna.

    """

    # A fold with too few trades has near-zero return variance, which inflates
    # the information ratio to absurd values (e.g. "IR=26946" on daily folds
    # with 2-3 trades). Such folds are degenerate and rejected. The IR is also
    # hard-capped so a single lucky fold cannot dominate the mean.
    MIN_TRADES_PER_FOLD = 5
    IR_CAP = 3.0
    # A config evaluated on fewer than this many valid folds cannot be judged
    # for cross-fold consistency, so it is rejected (overfitting guard).
    MIN_VALID_FOLDS = 3

    def __init__(self, config: dict) -> None:
        """
        Initialize Trading Optimization

        This method initializes the trading optimization.

        :param config: Configuration parameters for optimization.
        :type config: dict
        """
        self.strategy = None
        self.config = config
        os.makedirs(absolute_path + "/study", exist_ok=True)  # create the file study
        self.liste = []
        self.wfo_windows = []  # list of (val_start, val_end) tuples

    def generate_wfo_windows(
        self, train_months: int = 12, val_months: int = 3, n_splits: int = None
    ) -> None:
        """
        Generate rolling walk-forward windows spanning the loaded data.

        Each fold covers val_months of out-of-sample data. The first val window
        starts after train_months from the dataset start, and slides forward by
        val_months each step until the data is exhausted. With ~8 years of data
        this yields ~25+ folds (2019 -> 2026) rather than only the first few,
        so the OOS evaluation is representative of the whole history.

        :param train_months: Minimum history required before the first val window.
        :param val_months: Length of each OOS validation window.
        :param n_splits: Optional cap on the number of folds. None (default)
            generates as many folds as fit the dataset.
        """
        dates = self.strategy.parent_df["open_date"]
        data_start = dates.min()
        data_end = dates.max()

        train_delta = pd.DateOffset(months=train_months)
        val_delta = pd.DateOffset(months=val_months)

        val_start = data_start + train_delta
        windows = []
        while True:
            if n_splits is not None and len(windows) >= n_splits:
                break
            val_end = val_start + val_delta
            if val_end > data_end:
                break
            windows.append((val_start, val_end))
            val_start += val_delta

        self.wfo_windows = windows
        print(f"WFO: {len(windows)} folds generated")
        for i, (s, e) in enumerate(windows):
            print(f"  Fold {i + 1}: val=[{s.date()}, {e.date()}]")

    def _score_simulation(self, simulation_df: pd.DataFrame) -> float:
        """
        Information ratio of the strategy vs buy-and-hold.

        IR = mean(excess_returns) / std(excess_returns) * sqrt(365)

        Excess returns are computed trade-to-trade (the natural frequency of
        evolution_df). Annualisation uses 365 consistently with compute_sharpe_ratio.
        Range is roughly [-3, 3]; IR > 0.5 is good, > 1 is exceptional.

        Folds with fewer than MIN_TRADES_PER_FOLD return rows are rejected
        (-inf): with so few trades the excess-return std collapses toward 0 and
        the IR explodes to meaningless values. The final IR is clipped to
        +/- IR_CAP so no single fold can dominate the cross-fold mean.
        """
        if simulation_df is None or len(simulation_df) < self.MIN_TRADES_PER_FOLD:
            return -np.inf

        portfolio_returns = simulation_df["Portfolio"].pct_change().dropna()
        bh_returns = simulation_df["Buy_and_Hold"].pct_change().dropna()
        excess = portfolio_returns - bh_returns

        if len(excess) < 2 or excess.std() == 0 or np.isnan(excess.std()):
            return -np.inf

        ir = (excess.mean() / excess.std()) * np.sqrt(365)
        return float(np.clip(ir, -self.IR_CAP, self.IR_CAP))

    def _make_fold_strategy(self, val_start, val_end):
        """
        Return a lightweight, isolated copy of self.strategy for one WFO fold.

        Uses a shallow copy so the large parent_df is shared by reference
        (it is never mutated — compute_needed_features always calls
        parent_df.copy(deep=True) internally). Each copy gets fresh None
        values for all mutable per-run attributes so threads don't interfere.
        """
        import copy as _copy
        s = _copy.copy(self.strategy)
        s.child_df = None
        s.evolution_df = None
        s.df_filtered = None
        s.begin_date = val_start
        s.end_date = val_end
        s._suppress_save = True
        return s

    def _score_on_windows(self) -> tuple:
        """
        Evaluate the current strategy config across all WFO val windows in parallel.

        Each fold runs in its own thread with an isolated strategy copy that
        shares the read-only parent_df. ThreadPoolExecutor is chosen over
        ProcessPoolExecutor to avoid serialising the (potentially large)
        parent_df across process boundaries.

        Overfitting guard (replaces the Merton stress test as the param-selection
        gate — see RESEARCH_LOG): a robust config is *consistently* positive
        across folds, not carried by one lucky fold. We therefore return a
        consistency-penalised score:

            objective = mean_OOS_IR * consistency

        where consistency = fraction of valid folds that beat buy-and-hold
        (IR > 0). This is the cheap, real-data analogue of Merton robustness and
        directly discourages the "one huge fold, rest near zero" overfit pattern.

        :return: (objective, mean_sharpe, mean_final_portfolio). The objective is
            maximised; the other two are recorded for reporting. NaN/inf folds
            are excluded.
        """
        from concurrent.futures import ThreadPoolExecutor

        def _score_fold(window):
            val_start, val_end = window
            s = self._make_fold_strategy(val_start, val_end)
            try:
                sim_df = s.compute_simulation()
            except Exception:
                return (-np.inf, np.nan, np.nan)
            if sim_df is None or len(sim_df) < self.MIN_TRADES_PER_FOLD:
                return (-np.inf, np.nan, np.nan)
            return (
                self._score_simulation(sim_df),
                compute_sharpe_ratio(sim_df),
                sim_df["Portfolio"].iloc[-1],
            )

        n_workers = min(len(self.wfo_windows), os.cpu_count() or 4)
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            fold_results = list(executor.map(_score_fold, self.wfo_windows))

        valid = [r for r in fold_results if not np.isinf(r[0])]
        # Require a minimum number of valid folds: a config that only trades in
        # one or two folds cannot be judged for consistency.
        if len(valid) < self.MIN_VALID_FOLDS:
            return (-np.inf, np.nan, np.nan)

        irs, sharpes, portfolios = zip(*valid)
        mean_ir = float(np.mean(irs))
        consistency = float(np.mean([ir > 0 for ir in irs]))
        # Penalise spiky configs: only a config that is positive in most folds
        # keeps most of its mean IR. A negative mean IR is left unscaled so it
        # stays clearly bad (scaling a negative by <1 would flatter it).
        objective = mean_ir * consistency if mean_ir > 0 else mean_ir

        mean_sharpe = np.nanmean(sharpes)
        mean_portfolio = np.nanmean(portfolios)
        return (
            objective,
            float(mean_sharpe) if not np.isnan(mean_sharpe) else np.nan,
            float(mean_portfolio) if not np.isnan(mean_portfolio) else np.nan,
        )

    def optimiser_multiple_currency(
        self,
        list_crypto_currency: str = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "BNBUSDT"],
        list_time: str = ["1d", "4h", "1h"],
        list_brokers=["binance"],
    ):
        """
        Optimize Multiple Currencies

        This method performs optimization for multiple cryptocurrencies, time intervals, and brokers.

        :param list_crypto_currency: List of cryptocurrency symbols, defaults to ["BTCUSDT", "ETHUSDT", "ADAUSDT", "BNBUSDT"]
        :type list_crypto_currency: str, optional
        :param list_time: List of time intervals, defaults to ["1h", "4h", "1d"]
        :type list_time: str, optional
        :param list_brokers: List of broker names, defaults to ["binance"]
        :type list_brokers: list, optional
        """
        nested_list = list(product(list_crypto_currency, list_time, list_brokers))
        # Use multiprocessing Pool to parallelize

        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            pool.starmap(self.loop_optimiser_multiple_currency, nested_list)
        """
        for element in nested_list:
            crypto_currency, time, broker = element
            self.loop_optimiser_multiple_currency(crypto_currency, time, broker)
        """

    def loop_optimiser_multiple_currency(self, crypto_currency, time, broker):
        """
        Loop Optimize Multiple Currencies

        This method allows optimizing a trading strategy for a specific cryptocurrency, time interval, and broker.

        :param crypto_currency: Cryptocurrency symbol.
        :type crypto_currency: str
        :param time: Time interval.
        :type time: str
        :param broker: Broker name.
        :type broker: str
        """
        self.config["crypto_currency"], self.config["time"], self.config["broker"] = (
            crypto_currency,
            time,
            broker,
        )  # because we are looped on them
        study_name = (
            "study"
            + f"_{self.config['strategy']['name']}_{self.config['crypto_currency']}_{self.config['exchange']}_{self.config['time']}"
        )  # the study_name depending on the strategy name, crypto,exchange and time
        storage_name = f"sqlite:///{os.path.join(absolute_path, 'study', study_name)}.db"  # storage name
        study = optuna.create_study(
            storage_name,
            sampler=optuna.samplers.TPESampler(
                multivariate=True, group=True, constant_liar=True, n_startup_trials=300
            ),
            study_name=study_name,
            direction="maximize",
            load_if_exists="True",
        )  # study created else we reload the past strategy
        wfo_cfg = self.config.get("wfo", {})
        # Per-timeframe overrides take precedence over the base wfo settings:
        # higher timeframes need longer val windows to contain enough trades.
        tf_cfg = wfo_cfg.get("per_timeframe", {}).get(self.config["time"], {})
        train_months = tf_cfg.get("train_months", wfo_cfg.get("train_months", 12))
        val_months = tf_cfg.get("val_months", wfo_cfg.get("val_months", 3))
        # n_splits of 0/None spans the entire dataset (~25+ folds over
        # 2019-2026) instead of only the first few windows.
        n_splits = tf_cfg.get("n_splits", wfo_cfg.get("n_splits", None)) or None
        self.generate_wfo_windows(
            train_months=train_months,
            val_months=val_months,
            n_splits=n_splits,
        )

        # n_trials is configurable via [optuna] n_trials in the config; 100 is
        # plenty for validating direction (300 was wasteful for these runs).
        n_trials = self.config.get("optuna", {}).get("n_trials", 100)
        study.optimize(
            lambda trial: self.objective(trial=trial), n_trials=n_trials
        )  # we call optuna to optimize it
        self.save_df()
        self.save_mp4()

    def save_df(self):
        df = pd.DataFrame(
            self.liste,
            columns=["config", "result", "result_portfolio", "sharpe", "signature"],
        )
        self.dataframe_path = (
            self.strategy.time_eccefpsn
            + f"/{self.config['crypto_currency']}_{self.config['time']}_{self.config['strategy']['name']}.csv"
        )
        if os.path.exists(self.dataframe_path):
            existing_df = pd.read_csv(self.dataframe_path)
            df = pd.concat([df, existing_df], ignore_index=True)
            del existing_df
        df = df.drop_duplicates(subset=["result"])
        self.df = df.drop_duplicates(subset=["signature"])
        self.df.to_csv(self.dataframe_path, index=False)
        self.liste = []

    def save_mp4(self):
        mp4_file = (
            self.strategy.time_eccefpsn
            + f"_{self.config['crypto_currency']}_{self.config['time']}_output_video.mp4"
        )

        # Create writer object to write the video
        writer = imageio.get_writer(mp4_file, fps=2)  # Adjust the fps as needed
        # Loop through each directory in self.strategy.time_eccefpsn
        for dir_name in os.listdir(self.strategy.time_eccefpsn):
            if not dir_name.endswith(".csv"):
                dir_path = os.path.join(self.strategy.time_eccefpsn, dir_name)

                # List files in the directory
                files = os.listdir(dir_path)

                # Filter out PNG files
                png_files = [file for file in files if file.endswith(".png")]

                # Loop through PNG files in the directory
                for png_file in png_files:
                    png_full_path = os.path.join(dir_path, png_file)

                    # Read the PNG image
                    image = imageio.imread(png_full_path)
                    try:
                        # Append the image to the video
                        writer.append_data(image)
                    except:
                        print("Image pb")

        # Close the writer
        writer.close()
        print("MP4 Done")

    def objective(self, trial: optuna.trial):
        """
        Objective

        This method defines the objective function for optimization.

        :param trial: Optuna trial object.
        :type trial: optuna.trial
        """
        raise "To be implemented"


class optimiser_momentum_strategy(trading_optimisation):
    """
    Optimizer for Momentum Strategy

    This class specializes in optimizing the Momentum strategy.

    """

    def __init__(self, config: dict) -> None:
        """
        Initialize Optimizer for Momentum Strategy

        This method initializes the optimizer for the Momentum strategy.

        :param config: Configuration parameters for optimization.
        :type config: dict
        """
        super().__init__(config)
        self.strategy = momentum_strategy(config)  # initialize the momentum strategy

    def objective(self, trial: optuna.trial) -> float:
        """
        Objective for Momentum Strategy Optimization.

        When WFO windows are available, scores each trial on the mean OOS
        performance across all val folds (cross-fold objective). Falls back to
        full-dataset scoring when no WFO windows are configured.

        :param trial: Optuna trial object.
        :type trial: optuna.trial
        :return: Objective score to maximise.
        :rtype: float
        """
        self.config["strategy"]["x"] = trial.suggest_int("x", 1, 35)
        self.config["strategy"]["y"] = self.config["strategy"]["x"] + trial.suggest_int(
            "y", 1, 150
        )
        self.config["strategy"]["ratio_method"] = trial.suggest_categorical(
            "ratio_method",
            ["ma_ratio", "ema_ratio", "zlema_ratio", "dema_ratio", "tema_ratio"],
        )
        self.strategy.change_config(self.config)

        if self.wfo_windows:
            # Score purely on OOS val windows — no in-sample data seen by Optuna.
            # The objective is the mean OOS information ratio; mean OOS Sharpe and
            # final portfolio are recorded alongside it for reporting.
            result, sharpe_ratio, result_portfolio = self._score_on_windows()
        else:
            simulation_df = self.strategy.compute_simulation()
            result_portfolio = simulation_df["Portfolio"].iloc[-1]
            result = self._score_simulation(simulation_df)
            sharpe_ratio = compute_sharpe_ratio(simulation_df)

        config = copy.deepcopy(self.config)
        self.liste.append(
            [config, result, result_portfolio, sharpe_ratio, create_signature(self.config)]
        )
        return result


class optimiser_regime_momentum_strategy(trading_optimisation):
    """
    Optimizer for the Regime-Gated Momentum (MAD + ADX) Strategy.

    Searches the same MAD hyperparameters as optimiser_momentum_strategy plus
    the two ADX regime-filter parameters (window and threshold). Scoring reuses
    the WFO / information-ratio machinery of the base class unchanged, so the
    regime gate is judged purely on out-of-sample performance.
    """

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.strategy = regime_momentum_strategy(config)

    def objective(self, trial: optuna.trial) -> float:
        self.config["strategy"]["x"] = trial.suggest_int("x", 1, 35)
        self.config["strategy"]["y"] = self.config["strategy"]["x"] + trial.suggest_int(
            "y", 1, 150
        )
        self.config["strategy"]["ratio_method"] = trial.suggest_categorical(
            "ratio_method",
            ["ma_ratio", "ema_ratio", "zlema_ratio", "dema_ratio", "tema_ratio"],
        )
        # Regime-filter parameters — the only additions over plain MAD.
        self.config["strategy"]["adx_window"] = trial.suggest_int("adx_window", 7, 30)
        self.config["strategy"]["adx_threshold"] = trial.suggest_int(
            "adx_threshold", 15, 40
        )
        self.strategy.change_config(self.config)

        if self.wfo_windows:
            result, sharpe_ratio, result_portfolio = self._score_on_windows()
        else:
            simulation_df = self.strategy.compute_simulation()
            result_portfolio = simulation_df["Portfolio"].iloc[-1]
            result = self._score_simulation(simulation_df)
            sharpe_ratio = compute_sharpe_ratio(simulation_df)

        config = copy.deepcopy(self.config)
        self.liste.append(
            [config, result, result_portfolio, sharpe_ratio, create_signature(self.config)]
        )
        return result


class optimiser_band_momentum_strategy(trading_optimisation):
    """
    Optimizer for the Neutral-Band (hysteresis) MAD Strategy.

    Same MAD hyperparameters as optimiser_momentum_strategy plus a single
    dead-band width `band`. This is the minimal anti-whipsaw addition (one
    parameter, vs the two ADX params of the regime variant). Scored through the
    same WFO / consistency machinery, so band=0 (plain MAD) is always a
    candidate and the optimiser only adopts a band if it genuinely helps OOS.
    """

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.strategy = band_momentum_strategy(config)

    def objective(self, trial: optuna.trial) -> float:
        self.config["strategy"]["x"] = trial.suggest_int("x", 1, 35)
        self.config["strategy"]["y"] = self.config["strategy"]["x"] + trial.suggest_int(
            "y", 1, 150
        )
        self.config["strategy"]["ratio_method"] = trial.suggest_categorical(
            "ratio_method",
            ["ma_ratio", "ema_ratio", "zlema_ratio", "dema_ratio", "tema_ratio"],
        )
        # Neutral-band half-width: 0 reduces to plain MAD, up to +/-5% around 1.
        self.config["strategy"]["band"] = trial.suggest_float("band", 0.0, 0.05)
        self.strategy.change_config(self.config)

        if self.wfo_windows:
            result, sharpe_ratio, result_portfolio = self._score_on_windows()
        else:
            simulation_df = self.strategy.compute_simulation()
            result_portfolio = simulation_df["Portfolio"].iloc[-1]
            result = self._score_simulation(simulation_df)
            sharpe_ratio = compute_sharpe_ratio(simulation_df)

        config = copy.deepcopy(self.config)
        self.liste.append(
            [config, result, result_portfolio, sharpe_ratio, create_signature(self.config)]
        )
        return result


class optimiser_signal_momentum_strategy(trading_optimisation):
    """
    Optimizer for Momentum Strategy

    This class specializes in optimizing the Momentum strategy.

    """

    def __init__(self, config: dict) -> None:
        """
        Initialize Optimizer for Momentum Strategy

        This method initializes the optimizer for the Momentum strategy.

        :param config: Configuration parameters for optimization.
        :type config: dict
        """
        super().__init__(config)
        self.strategy = signal_momentum_strategy(
            config
        )  # initialize the momentum strategy

    def objective(self, trial: optuna.trial) -> None:
        """
        Objective for Momentum Strategy Optimization

        This method defines the objective function for optimizing the Momentum strategy.

        :param trial: Optuna trial object.
        :type trial: optuna.trial
        :return: None
        :rtype: None
        """

        self.config["strategy"]["x"] = trial.suggest_int("x", 1, 35)  # hyperparameters
        self.config["strategy"]["y"] = self.config["strategy"]["x"] + trial.suggest_int(
            "y", 1, 150
        )  # hyperparameters
        self.config["strategy"]["ratio_method"] = trial.suggest_categorical(
            "ratio_method",
            ["ma_ratio", "ema_ratio", "zlema_ratio", "dema_ratio", "tema_ratio"],
        )  # hyperparameters
        self.config["strategy"]["method"] = trial.suggest_categorical(
            "method",
            ["ma", "ema", "zlema", "dema", "tema"],
        )
        self.config["strategy"]["z"] = trial.suggest_int("z", 1, 100)
        self.strategy.change_config(
            self.config
        )  # essential change config wil reinitialize the object
        simulation_df = self.strategy.compute_simulation()  # compute the simulation
        result_portfolio = simulation_df["Portfolio"].iloc[-1]
        sharpe = compute_sharpe_ratio(simulation_df)
        value = round(
            (simulation_df["Relative_Portfolio_VS_Relative_Buy&Hold"]).mean() * 100,
            4,
        )
        result = sharpe * value  # calculate a result
        # result = rsi * value
        config = copy.deepcopy(self.config)
        self.liste.append(
            [config, result, result_portfolio, sharpe, create_signature(self.config)]
        )

        return result


class optimiser_envelop_strategy(trading_optimisation):
    """
    Optimizer for Envelop Strategy

    This class specializes in optimizing the Momentum strategy.

    """

    def __init__(self, config: dict) -> None:
        """
        Initialize Optimizer for Momentum Strategy

        This method initializes the optimizer for the Momentum strategy.

        :param config: Configuration parameters for optimization.
        :type config: dict
        """
        super().__init__(config)
        self.strategy = envelop_strategy(config)  # initialize the momentum strategy

    def objective(self, trial: optuna.trial) -> None:
        """
        Objective for Momentum Strategy Optimization

        This method defines the objective function for optimizing the Momentum strategy.

        :param trial: Optuna trial object.
        :type trial: optuna.trial
        :return: None
        :rtype: None
        """

        self.config["strategy"]["x"] = trial.suggest_int("x", 1, 30)  # hyperparameters
        self.config["strategy"]["method"] = trial.suggest_categorical(
            "ratio_method", ["ma", "ema"]
        )  # hyperparameters
        nb_envelope = trial.suggest_int(f"nb_enveloppe", 2, 5)
        list_envelope = []
        list_porportion = []
        for i in range(nb_envelope):
            if i == 0:
                list_envelope.append(
                    trial.suggest_int(f"envelope_{i}", 1, 1000) * 0.0001
                )
            else:
                list_envelope.append(
                    trial.suggest_int(f"envelope_{i}", 1, 1000) * 0.0001
                    + list_envelope[-1]
                )
            list_porportion.append(
                trial.suggest_int(f"proportion_envelope_{i}", 0, 100)
            )

        envelope_array = np.round(np.array(list_envelope), 3)
        porportion_array = np.array(list_porportion)
        porportion_array = np.round(porportion_array / np.sum(porportion_array), 2)

        self.config["strategy"]["envelop"] = dict(
            zip(envelope_array.astype(str), porportion_array)
        )
        self.strategy.change_config(
            self.config
        )  # essential change config wil reinitialize the object
        simulation_df = self.strategy.compute_simulation()  # compute the simulation

        col_1, col_2 = "Portfolio", "Buy_and_Hold"  # element for computing the reward
        simulation_df = evalutation(
            simulation_df, col_1, col_2
        )  # evaluate the simulation
        # result = simulation_df["Portfolio"].iloc[-1]*(simulation_df[f"RSI_{col_1}/{col_2}"]/100).mean() # calculate a result
        return simulation_df["Portfolio"].iloc[-1]


if __name__ == "__main__":
    import sys

    absolute_path = os.path.abspath(__file__)[
        : os.path.abspath(__file__).find("MAD_BTC") + len("MAD_BTC")
    ]

    # Optional CLI args:
    #   python multi_optimiser_momentum_strategy.py [plain|regime] [config_path]
    # default: plain, src/configs/optimiser_momentum_config.toml. The regime
    # variant runs under its own study name (strategy.name =
    # "regime_momentum_strategy") so its studies and saved results never collide
    # with the plain-MAD ones.
    variant = sys.argv[1] if len(sys.argv) > 1 else "plain"
    config_path = (
        sys.argv[2] if len(sys.argv) > 2
        else "src/configs/optimiser_momentum_config.toml"
    )
    optimiser_class = {
        "plain": optimiser_momentum_strategy,
        "band": optimiser_band_momentum_strategy,
        "regime": optimiser_regime_momentum_strategy,
    }[variant]
    # Each non-plain variant runs under its own study name so studies/results
    # never collide.
    variant_strategy_name = {
        "band": "band_momentum_strategy",
        "regime": "regime_momentum_strategy",
    }

    # read the config file:
    with open(config_path, "r") as file:
        config = toml.load(file)
    if variant in variant_strategy_name:
        config["strategy"]["name"] = variant_strategy_name[variant]

    missing_data = check_data_availability_in_folder(
        "data",
        config["multi"]["list_brokers"],
        config["multi"]["list_crypto_currency"],
        config["multi"]["list_time"],
    )
    if len(missing_data) != 0:
        print("Downloading missing data")
        with open("src/configs/download_config.toml", "r") as file:
            data_config = toml.load(file)
        data_config["symbols"] = missing_data[0]
        data_config["time_frame"] = missing_data[1]
        data_config["broker"] = missing_data[2]
        loop = asyncio.get_event_loop()
        loop.run_until_complete(downloader(data_config))
        print("Downloading finished")

    print(f"Running {variant} optimiser across full grid")
    optimiser_class(config=config).optimiser_multiple_currency(**config["multi"])
