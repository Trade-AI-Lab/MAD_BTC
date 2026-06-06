# Research Log

A running log of changes to the MAD (Moving Average Distance) crypto research
codebase: what was added/fixed, what we learned, and what to try next.

---

## 2026-06-06 — Bug fixes, full WFO relaunch, and regime-gated MAD

### Context

The repository tests whether the Moving Average Distance signal
(`MA_short / MA_long`, crossing 1) from Avramov, Kaplanski & Subrahmanyam (2021)
can produce a wealthy strategy on crypto. The honest standing answer remains
*"it may be possible"* — a single MAD crossover is heavily arbitraged and does
not reliably beat buy-and-hold on its own. The focus of this session was
(1) making the evaluation trustworthy and (2) taking the first low-overfitting
step toward combining MAD with other logic.

### Bugs fixed

1. **WFO Sharpe was hardcoded to `NaN`.**
   In `optimiser_momentum_strategy.objective`, the recorded Sharpe (and final
   portfolio in the WFO branch) were `np.nan`, so the saved CSV had no usable
   risk-adjusted metric. `_score_on_windows` now returns
   `(mean_OOS_IR, mean_OOS_Sharpe, mean_final_portfolio)` across the valid
   out-of-sample folds; the objective still *maximises* the information ratio
   but records real Sharpe / portfolio for reporting. The non-WFO branch now
   computes Sharpe too.
   - Files: `src/multi_optimiser_momentum_strategy.py`

2. **Envelop strategy used the wrong price on the sell side.**
   In `envelop_strategy.loop_simulation`, the entry price ternary used
   `envelop_down_*` for *both* buy and sell. Sell entries now use
   `envelop_up_*`, consistent with `col_condition = "high"` on sells.
   - Files: `src/strategy_utils/launcher_strategy.py`

3. **Envelop strategy crashed on save.**
   `envelop_strategy.loop_simulation` called `save()` directly without first
   building the `RSI_Portfolio/Buy_and_Hold` column (which `core_img` requires),
   raising `KeyError`. It now delegates to the base `save_data()`, which builds
   the column, honours `_suppress_save`, and only writes profitable runs.
   - Files: `src/strategy_utils/launcher_strategy.py`

4. **MAD price column made explicit / configurable.**
   The MAD ratio was silently computed on `open`. This is the safe choice for
   the trade-on-open engine (no lookahead), so the default is preserved, but it
   is now explicit via `config['strategy']['price_column']` (default `"open"`;
   set `"close"` to follow the paper's close-based MAD).
   - Files: `src/strategy_utils/launcher_strategy.py`

### New feature — Regime-gated MAD (MAD + ADX)

Plain MAD crosses 1 constantly in sideways markets and bleeds to fees and
whipsaws. We added a **trend-strength regime filter** so MAD only trades when a
trend is actually present.

- `indicator_momentum.adx()` — Average Directional Index (0–100); ADX > ~25
  conventionally means "trending".
  - File: `src/strategy/momentum.py`
- `regime_momentum_strategy` — subclasses `momentum_strategy`; a MAD crossover
  is acted on only when `ADX > adx_threshold`, otherwise the current position is
  held. Reuses all MAD / simulation / save machinery and adds only **two
  parameters** (`adx_window`, `adx_threshold`) to keep the overfitting surface
  small.
  - File: `src/strategy_utils/launcher_strategy.py`
- `optimiser_regime_momentum_strategy` — searches the MAD params plus the two
  ADX params, scored through the existing WFO / information-ratio machinery.
  - File: `src/multi_optimiser_momentum_strategy.py`

**Sanity check (BTCUSDT 4h, 2022-01 → 2023-06, a hard bear-ish window, same MAD
params `x=10 y=80 tema`):**

| Strategy   | Trades | Final ($) | Max DD |
|------------|-------:|----------:|-------:|
| Plain MAD  |     19 |     867.2 |  32.3% |
| Regime MAD |     15 |    1107.7 |  32.3% |
| Buy & Hold |      — |     588.8 |  60.3% |

The gate removed 4 whipsaw trades and turned a loss into a gain vs plain MAD,
while both far undercut B&H's drawdown. (One window proves the *mechanism*, not
edge — that is what WFO + Merton are for.)

### Results from the full WFO relaunch

Re-ran the full grid — 6 coins × 3 timeframes = 18 Optuna studies, 300 trials
each, 5-fold walk-forward (12mo train / 3mo val). Completed cleanly, no errors.

Key observation, directly relevant to the overfitting question:

- **The daily timeframe produces degenerate, untrustworthy metrics.** E.g.
  `DOTUSDT_1d` reported an "information ratio" of ~26,946 and "Sharpe" ~1,142.
  These are artefacts of folds with only 2–3 trades and near-zero return
  variance (the IR/Sharpe denominator collapses toward 0). They are *not* edge.
- **The trustworthy results are the ones with a sane Sharpe (~1)** — e.g.
  ADAUSDT_1d (1.11), BTCUSDT_1d (1.11), DOTUSDT_4h (1.28), ADAUSDT_4h (1.18),
  ADAUSDT_1h (1.63). Higher-frequency timeframes give enough trades for the
  statistics to mean something.
- **IR and Sharpe disagree by design.** Several configs show positive IR (beats
  buy-and-hold relatively) with strongly negative Sharpe (loses money
  absolutely) — common in bear markets. Selection should weigh both.

---

## Possible improvements / TODO

### Evaluation robustness (highest priority — these guard against overfitting)
- [ ] **Cap the objective / require a minimum trade count per fold.** Reject
      folds with too few trades so degenerate "IR=26946" winners cannot top the
      study. Optionally clip IR to a sane range (e.g. [-3, 3]).
- [ ] **Wire `regime_momentum_strategy` into the Merton stress test**
      (`strategy_class=regime_momentum_strategy`) to check synthetic-path
      robustness, not just historical OOS.
- [ ] **Plateau check on (x, y).** Trust a *region* of nearby parameters that
      all work (the robustness/clustering plots), not a lone spike — a spike is
      almost certainly overfit.
- [ ] **Report drawdown and trade count in the saved CSV**, alongside Sharpe,
      so selection is on risk-adjusted terms, not final value.
- [ ] **Cross-validation / purged k-fold** in addition to walk-forward.

### Strategy combination (the path beyond single-signal MAD)
Ordered by increasing power *and* increasing overfitting risk:
- [ ] **Regime-gated MAD — full-grid WFO run** and compare OOS Sharpe vs plain
      MAD across all 18 studies (done for the mechanism; not yet at full scale).
- [ ] **Equal-weight signal vote** — MAD + time-series momentum (sign of past-N
      return) + a mean-reversion signal (RSI / Bollinger / envelop), trade on
      agreement. Zero fitted weights → lowest overfitting risk.
- [ ] **Portfolio of sleeves** — run MAD, envelop, and a mean-reversion strategy
      as separate capital sleeves and sum equity curves. Generalize
      `launcher_multi_strategy` (which already does multi-*asset*) to
      multi-*strategy*. Decorrelated sleeves → higher combined Sharpe.

### Engineering / hygiene
- [ ] **Envelop trade math produces absurd portfolio values** (~1e22 in tests) —
      a pre-existing numerical bug in `envelop_strategy.loop_simulation`,
      separate from the save fix. Needs a proper review of the buy/sell sizing.
- [ ] **`save_df` merges with stale historical CSVs**, so old pre-fix rows keep
      `NaN` Sharpe. Consider versioning the output or clearing on a fresh run.
- [ ] **Fee model is hardcoded** (0.1% momentum, 0.06% envelop). Move to config.
- [ ] **Finish the TRIX strategy** (noted in README).

### Open research questions
- Does MAD edge survive realistic fees + slippage on the higher-frequency
  timeframes where the statistics are trustworthy?
- The README notes each asset needs its own (x, y) because of differing
  volatility — would **volatility-normalizing** the MAD ratio make a single
  parameter set transfer across coins?
- Is the goal "beat B&H on return" (rarely achievable in a bull run) or "match
  return at much lower drawdown" (achievable, and what the data so far supports)?
