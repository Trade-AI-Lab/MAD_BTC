# Research Log

A running log of changes to the MAD (Moving Average Distance) crypto research
codebase: what was added/fixed, what we learned, and what to try next.

---

## 2026-10-25 (late night) — One-parameter `dz` rule (+ volume) vs ML filter

`run_regime_study.py` extended. `dz` = z[t] - z[t-1] on the entry bar (size of the crossover jump). Rule:
keep the trade if `dz` exceeds the q-quantile of `dz` over the purged training trades only (q in 0.5/0.67/0.8);
`dz67&vol` additionally requires volume[t-1] > its 50-bar mean. Same 5-asset equal-weight walk-forward OOS from 2021,
0.1%/side. DSR below uses 36 trials (2 timeframes x 3 lookbacks x 6 rules, my count) on pooled trades.

4h Sharpe / max DD / DSR:

| lookback | base | gboost | dz>q67 | dz67&vol |
|---------:|------|--------|--------|----------|
| 5  | 0.04 / -57% / 0.04 | 0.94 / -15% / 0.38 | 1.11 / -29% / 0.91 | **1.44 / -21% / 0.98** |
| 10 | 0.24 / -46% / 0.15 | 1.05 / -13% / 0.26 | 0.78 / -29% / 0.59 | 1.07 / -21% / 0.79 |
| 20 | 0.81 / -50% / 0.66 | 0.58 / -23% / 0.14 | 1.05 / -30% / 0.74 | 0.99 / -27% / 0.59 |

- **A one-parameter rule matches or beats the ML model on Sharpe**, with 79-423 trades instead of 53-100 and
  40-47% time in market. Drop-the-model criterion from the previous entry is met: gboost is not needed.
- Beats random vetoes of equal size on Sharpe at 4h: 100% / 98% / 92% (lb5/10/20, q67). Week-block bootstrap
  P(Sharpe > base) 1.00 / 0.97 / 0.85.
- **Drawdown is the weak spot.** `dz` filters keep more exposure than gboost, and their drawdown (-21% to -30%)
  is no better than random vetoes at lb10/lb20 (beats 54% / 30%); only lb5 clearly beats random on max DD
  (78%; 92% with volume). The ML model's smaller DD (-13% to -15%) came from being invested 16-28% of the time.
- **Only 4h lb5 `dz67&vol` clears the DSR bar (0.98 vs 0.95)**, and DSR is optimistic here: pooled trades
  overlap across coins and the trial count is my estimate. lb10 0.79, lb20 0.59.
- **1h fails**: lb5 Sharpe -1.06 -> -0.17/-0.22, lb10 -0.33 -> -0.22/-0.17, all negative, DSR <= 0.01. At lb10 the
  rule is no better than random vetoes (55th percentile Sharpe; drawdown worse than random). Fees dominate at 1h.
- 4h lb20 volume add-on does not help (0.99 vs 1.05 for `dz` alone): the volume gain is lb5/lb10 only.

Verdict: first filter in this project with a DSR near the bar, and it is a single rule, not a model. Still not
validated: one OOS path, same data used to pick lb5 4h as the headline, overlapping assets.

### Correction (same night): the DSR 0.98 does not survive a cluster-aware check

DSR recomputed on weekly returns of the equal-weight portfolio (n = 248-258 weeks, so no cross-asset overlap):

| 4h lookback | rule | trials 36 | trials 100 | trials 300 |
|---|---|---|---|---|
| 5  | dz67&vol | 0.75 | 0.58 | 0.40 |
| 10 | dz67&vol | 0.55 | 0.38 | 0.24 |
| 20 | dz>q67   | 0.49 | 0.33 | 0.20 |

The trade-level 0.98 was inflated by pooling overlapping trades (it treated ~320 correlated trades as independent).
Honest status: no filter passes the 0.95 gate; best is 0.75 at the 36-trial count, and I have run more than 36
configurations in this thread (more like 100). The Sharpe lift over the base rule and over random vetoes is
real in this sample, but "significant after multiple testing" is not established.

### Next
- [ ] Frozen test: fix q=0.67 + volume rule at 4h lb5/lb10 on data not used so far (post-2024 only for the quantile
      refit; SOL/XRP/LINK 4h; or a different exchange).
- [ ] Cluster-aware DSR/bootstrap (calendar-week blocks across assets) to replace the optimistic pooled DSR.
- [ ] Risk-targeted sizing to compare drawdown at equal exposure.

---

## 2026-10-25 (night) — Regime filter vs ML filter on a portfolio equity curve

`run_regime_study.py` (equity curve, drawdown, week-block bootstrap, random-veto control) and
`feature_importance.py`. Long-only MAD entries on BTC/ETH/BNB/ADA/DOT, equal weight, 0.1%/side, walk-forward
OOS from 2021 (buy & hold: Sharpe ~0.2-0.3, max DD -80%). Compared: all trades, fixed ADX>25 rule, fixed
"vol above its 500-bar median" rule, walk-forward gradient boosting (p>=0.5), gboost AND ADX.

| 4h | Sharpe base | Sharpe gboost | maxDD base | maxDD gboost | time in market gboost |
|----|------------:|--------------:|-----------:|-------------:|----------------------:|
| lb5  | 0.04 | 0.94 | -57% | -15% | 16% |
| lb10 | 0.24 | 1.05 | -46% | -13% | 28% |
| lb20 | 0.81 | 0.58 | -50% | -23% | 24% |

- **Simple regime rules do not reproduce the ML filter.** ADX>25 and the vol rule cut drawdown but do not
  lift Sharpe (4h lb5: -0.41 and -0.42 vs base 0.04). The earlier reading "gboost = a regime filter" is not
  supported by these two rules.
- **Most of the drawdown reduction is just lower exposure.** Random vetoes keeping the same number of trades
  have median max DD -16% (lb5) / -19% (lb10) vs gboost -14% / -13%: gboost beats 68% / 88% of random
  vetoes on drawdown. The Sharpe gain is not exposure: gboost beats 99% of random vetoes on Sharpe at both
  lb5 and lb10 (random median -0.05 / 0.16). Week-block bootstrap: P(Sharpe > base) 0.96 / 0.99.
- **Not robust across settings:** 4h lb20 gboost is worse than base on Sharpe (0.58 vs 0.81); 1h gboost keeps
  2-7% of trades and earns ~0 (Sharpe 0.05 / -0.05, drawdown small only because it is almost never invested).
  gboost AND ADX has the best Sharpe/Calmar in 4h lb10 and lb20 (1.27 / 1.10) but on 16-22 trades.
- **What the model uses** (permutation importance, AUC 0.59-0.63 on the last 30%): `dz` (how far z jumped on the
  entry bar, i.e. breakout speed) first, then `vol_ratio` (volume vs its 50-bar mean) and `z`. Realised-vol and
  ADX features rank low. So it prefers sharp, high-volume crossovers, not a trending/high-vol regime.

Caveats: one OOS path; the 4h lb5/lb10 cells were singled out after the earlier pooled run (selection bias,
DSR was 0.80 and 0.44); assets overlap in time; trade counts are 94-100. Not a validated strategy.

### Next on this thread
- [ ] Rule from the finding: 1-parameter filter on entry-bar `dz` (and volume ratio) instead of ML. If it matches
      gboost, drop the model.
- [ ] Freeze gboost (4h, lb5/lb10) trained on <=2023 and score 2024+ untouched; add SOL/XRP/LINK 4h data.
- [ ] Risk-targeted sizing (vol-scaled, probability-scaled) for the filtered trades; report DD at equal exposure.

---

## 2026-10-25 (later) — Meta-labeling iteration 2: pooled model + permutation control, END OF THREAD

`src/meta_labeling/run_meta_pooled.py`. One model per timeframe trained on events pooled over 5 assets
(BTC/ETH/BNB/ADA/DOT), time-ordered, purged by timestamp (2-day embargo), lookbacks 5/10/20, 33 cells,
DSR with 18 trials/timeframe, plus a permutation control (random veto with the same keep count, 2000 draws).

- 1h: nothing. Best DSR 0.27, permutation p-values scattered around 0.5 (noise).
- 4h gradient boosting is the only hint: lb5 thr0.5 keeps 94 of 1185 trades, mean +2.6% vs +0.07%,
  permutation p<0.001, DSR 0.80; lb10 thr0.5 mean +3.2% vs +0.55%, p=0.015, DSR 0.44. 7 of 33 cells have
  p<0.05 (about 2 expected by chance), all but one in 4h gboost.
- Still fails the 0.95 bar, and the hint looks fragile: kept-trade median is negative (-0.9%/-1.8%), win rate
  30-43%, mean falls to +1.0-1.5% without the top 3 trades, 2021 contributes most of lb5's mean, and the
  5 coins' events overlap in time so the 94 kept trades are not independent (DSR is optimistic).
  Mean is positive in every asset, which is the only robust-looking part.
- Reading: the model plausibly selects high-volatility trend regimes where the rare large winners occur,
  not better entries. That is a regime filter, not an entry-timing skill.

Verdict: the ML filter does not rescue MAD under the DSR gate. Thread ended.

### What could still be studied
- [ ] Confirm or kill the 4h gboost hint with data it has not seen: refit on 2018-2023 only, freeze, and
      score the untouched post-2024 period; or fetch more 4h assets (SOL, XRP, LINK exist at 1h only).
- [ ] Cluster-aware inference: block-bootstrap by calendar week across assets, since pooled events overlap.
- [ ] Combinatorial purged CV (CPCV) to get a distribution of OOS Sharpe rather than one path.
- [ ] Test the "regime filter" reading directly: replace the model with a rule on the top features
      (realised-vol percentile, ADX) and compare; if a 1-parameter rule matches gboost, drop ML.
- [ ] Different label: triple-barrier (profit/stop/time) and a regression on trade return instead of
      win/lose, since MAD's payoff is a few big winners (median negative, mean positive).
- [ ] Position sizing from the predicted probability (bet sizing, AFML ch.10) instead of a binary veto.
- [ ] Cross-asset/macro features (BTC dominance, funding rate, open interest, BTC regime for alts);
      none of these are in the OHLCV files.
- [ ] Fix `src/classifier_ai` (argmax on sigmoid, shuffled split) only if a sequence model is wanted;
      with 100-1000 trades per timeframe a TSMixer is not supported by the sample size.

---

## 2026-10-25 — Meta-labeling on vol-normalised MAD (ML filter): negative

`src/meta_labeling/run_meta.py`. Primary rule: long on z-cross above 0, exit on cross below, 0.1%/side.
Secondary model (logreg, HistGradientBoosting) predicts net-profitable trade from entry-time features
(z, dz, ADX, realised vol, returns, volume, hour, gap). Expanding walk-forward on trades, training trades
purged if not closed (+10-bar embargo) before the first test entry. BTC/ETH/BNB pooled, 1h and 4h,
lookbacks 5/10/20/40, thresholds 0.5/0.55/0.6; DSR with 24 trials per timeframe.

Result: best DSR 0.20 (bar 0.95). OOS kept-trade counts are 3-140 per cell. No cell is consistently better
than the unfiltered rule; the few with higher mean (4h gboost lb10: +7.9% vs +1.5%) have kurtosis 40-60,
i.e. a handful of outlier trades. TSMixer was not run: it only earns a place if gboost beats the rule.

Note on `src/classifier_ai`: it does not find entry points. It labels every bar where the rule is already
long by whether close rises over n_forward bars, with a random (shuffled) train/val split, and
`validation_step` takes argmax over a single sigmoid output, so val predictions are always 0.

---

## PROJECT CLOSED — 2026-06-06

**Status: closed. Negative result, fully documented.**

**Question:** can MAD (Moving Average Distance) be a wealthy/robust standalone
crypto trading strategy?

**Answer: No.** Every variant tried — plain MAD, neutral-band (hysteresis),
ADX-regime-gated, and a volatility-normalised single-parameter version designed
specifically to minimise overfitting — failed proper out-of-sample validation.
The Deflated Sharpe Ratio (Bailey & López de Prado) of the best configuration in
each study was < 0.36 everywhere (significance bar 0.95): after correcting for
the number of parameter searches, the returns are indistinguishable from noise.

**Why it's a *successful* project despite the negative result:**
- The research question is now answered rigorously, not abandoned ambiguously.
- It produced a reusable, honest validation stack: full-span walk-forward
  (2019–2026), consistency-penalised objective, trade-spacing-corrected Sharpe,
  and a Deflated Sharpe gate. (`deflated_sharpe_ratio` in
  `multi_optimiser_momentum_strategy.py` is the one piece worth porting
  elsewhere.)
- It confirmed the right *form* for any future momentum signal
  (volatility-normalisation), even though MAD itself is too weak to use it.

**What NOT to do next:** do not fold MAD into the `quant_crypto` HRP+PAMR
pipeline as a sleeve — the DSR evidence says it adds noise, not decorrelated
signal. The robust-portfolio direction (HRP + mean-reversion ensemble) already
exists in `quant_crypto` and should be pursued there, not rebuilt here.

This repository is archived. The dated entries below are the chronology of how
the verdict was reached.

---

## 2026-06-06 (late) — Vol-normalised 1-param MAD + Deflated Sharpe verdict

### Motivation (literature)

The recurring failure mode is that classic MAD needs a *different* (x, y,
ratio_method) per asset — the raw ratio is not scale-invariant, so the optimiser
refits noise per market. Two well-known remedies (López de Prado; Bailey;
Moskowitz/Ooi/Pedersen 2012; Harvey et al. 2018):

1. **Volatility-normalise the signal** so one rule transfers across assets.
2. **Reduce free parameters** — the False Strategy Theorem says the expected max
   Sharpe of N noise strategies grows like sqrt(2 ln N); fewer params/trials =
   smaller selection bias.

### What was built

- `indicator_momentum.vol_normalized_mad`: `z = (EMA_x/EMA_y - 1)/sigma`, a
  dimensionless MAD in volatility units.
- `vol_momentum_strategy` (+ optimiser, `vol` CLI variant): collapses 3
  hyperparameters to **ONE** (`lookback`; long window fixed at `4*lookback`,
  EMA fixed, threshold theta=1.0 fixed). Buy when z crosses +theta, sell at
  -theta — a built-in neutral zone.
- `deflated_sharpe_ratio` (Bailey & López de Prado 2014): converts an observed
  Sharpe + trial count into the probability the true Sharpe > 0. The honest
  "is this multiple-testing luck?" gate.

### Promising in-sample, but it does NOT survive proper validation

A single full-history backtest looked good (same lookback=15 on all coins gave
consistent Sharpe 0.60–0.78) — the scale-invariance *mechanically* works. But on
the proper walk-forward distribution it is middle-of-the-pack, not better:

| variant | studies with positive median OOS Sharpe | best median |
|---------|----------------------------------------:|------------:|
| plain   | 1/6 | 0.39 |
| band    | 2/6 | 0.48 |
| **vol** | **2/6** | **0.32** |
| regime  | 4/6 | 0.33 |

**Deflated Sharpe Ratio (best vol config per study, n_trials=100):**

| study      | lookback | trades | SR/trade | DSR |
|------------|---------:|-------:|---------:|----:|
| BTCUSDT_1h |       59 |    223 |    0.134 | 0.247 |
| BTCUSDT_4h |       19 |    139 |    0.172 | 0.238 |
| ETHUSDT_1h |       40 |    304 |    0.129 | 0.359 |
| ETHUSDT_4h |       45 |     63 |    0.229 | 0.084 |
| BNBUSDT_1h |       25 |    434 |    0.078 | 0.013 |
| BNBUSDT_4h |       18 |    110 |    0.142 | 0.007 |

**Every DSR is far below the 0.95 bar** (max 0.36). And the winning lookback
still varies 18–59 across studies — vol-normalisation made the *signal* more
comparable but did not produce a stable optimal *lookback*.

### Definitive verdict: MAD has no standalone edge in crypto

This is the rigorous confirmation of the earlier intuition. Across plain, band,
regime, AND the vol-normalised 1-parameter version — the variant specifically
designed to minimise overfitting — **no configuration passes the Deflated Sharpe
test.** The observed Sharpes are within what pure noise would produce given the
search. MAD is not a tradeable standalone alpha in this universe.

What the exercise *did* establish (worth keeping):
- The methodology now catches overfitting honestly (full-span WFO +
  consistency-penalised objective + trade-spacing Sharpe + DSR). Any future
  signal can be judged on the same bar.
- Vol-normalisation is the right *form* for any future momentum component, even
  though MAD itself is too weak.

### Next — stop tuning MAD; change the question
- [ ] Move from "find alpha in one signal" to **portfolio construction over many
      weak, decorrelated signals** — where the robustness lives.
- [ ] Hierarchical Risk Parity (de Prado 2016): no return forecasts, no
      per-asset tuning, allocates on correlation structure alone.
- [ ] Add a structurally different signal family (mean-reversion: OLMAR / PAMR,
      Li & Hoi) so the ensemble is genuinely decorrelated.
- [ ] Keep DSR + (eventually) CPCV as the standing acceptance gate.

---

## 2026-06-06 (evening) — Neutral-band MAD + verdict on MAD as a strategy

### Neutral-band (hysteresis) variant

Added `band_momentum_strategy` (+ `optimiser_band_momentum_strategy`, `band`
CLI variant) to attack plain MAD's one clear flaw — threshold whipsaw, where the
ratio wobbling around 1.0 triggers rapid buy/sell flips (visible in the plain
ETH 4h 2025 zoom: ~93 signals vs ~39 for the ADX variant). The band adds a
single dead-band δ: buy at ratio > 1+δ, sell at ratio < 1−δ, hold in between.
One extra parameter vs the regime variant's two. `band=0` reduces to plain MAD,
so the optimiser only adopts a band if it helps out-of-sample.

Mechanically it does what it should (full-history ETH 4h, fixed MAD params):
trades 93 → 51–65, max drawdown 41.5% → ~36%, but final return *falls*
(32k → 23–32k). It trades return for stability — no free lunch.

### Three-way verdict — plain vs band vs regime (BTC/ETH/BNB, full-span WFO)

Distributional OOS Sharpe (median / 90th-pct across all finite trials):

| study      | plain med/p90 | band med/p90 | regime med/p90 |
|------------|--------------:|-------------:|---------------:|
| BTCUSDT_1h |  −0.50 / −0.13 | −0.33 / +0.11 |  −0.26 / +0.47 |
| BTCUSDT_4h |  −0.01 / +0.54 | −0.14 / +0.39 |  +0.13 / +0.58 |
| ETHUSDT_1h |  −0.25 / +0.17 | −0.13 / +0.38 |  +0.00 / +0.49 |
| ETHUSDT_4h |  +0.39 / +0.79 | +0.37 / +0.60 |  +0.33 / +0.92 |
| BNBUSDT_1h |  −0.28 / +0.12 | −0.17 / +0.37 |  −0.01 / +0.52 |
| BNBUSDT_4h |  −0.08 / +0.55 | +0.48 / +0.93 |  +0.30 / +1.10 |

Tally vs plain MAD:
- median-Sharpe wins: band 4/6, regime 5/6
- p90-Sharpe wins:    band 4/6, regime 6/6
- studies with *positive* median Sharpe: plain 1/6, band 2/6, regime 4/6
- best median Sharpe overall: plain 0.39, band 0.48, regime 0.33

### Verdict: MAD is (nearly) dead as a standalone crypto strategy

The evidence across this whole session points one way:

1. **Plain MAD has positive median OOS Sharpe in only 1 of 6 studies.** On its
   own it is essentially a coin flip vs buy-and-hold — not a money-maker. This
   matches the user's intuition that "MAD is a dead strategy."
2. **Both filters (band, regime) help — but marginally, and at a cost.** They
   lift more studies into positive territory (band 2/6, regime 4/6) and improve
   the tails, but the median Sharpes remain small (≤ ~0.5) and the filters
   roughly *halve* the number of evaluable configs (more fragile selection).
3. **Even the best configs do not survive Merton stress testing** (robustness
   0.17–0.49, earlier section) and the top-WFO trials are the least robust.

So: MAD + a filter is a mild *risk-reducer* (smaller drawdowns, slightly better
risk-adjusted tails), not an alpha source. As a standalone edge it is dead. Its
only credible future is as **one weak, decorrelated component inside an ensemble**
(equal-weight signal vote / portfolio of sleeves), where its trend-following
behaviour can complement mean-reversion and other signals — never as the
strategy itself.

### Next
- [ ] Stop tuning MAD variants. Treat MAD as a fixed, low-weight component.
- [ ] Build the equal-weight signal vote (MAD + TSMOM + mean-reversion), the
      first genuine *combination* rather than another MAD filter.
- [ ] Bring in a structurally different signal family (mean-reversion) — the
      diversification, not the MAD tuning, is where any real edge will come from.

---

## 2026-06-06 (pm) — Full-span walk-forward, objective hardening, regime vs plain

### Walk-forward now spans the entire dataset (2019–2026)

The previous WFO was badly under-using the data: it started the first
validation window at `data_start + 12 months` and then took only `n_splits=5`
windows of `val_months=3`, covering just **15 months** (2019-04 → 2020-07) of an
**~8-year** dataset (2018-04 → 2026-02). All evaluation happened in the early,
thin-data period.

Fix: `generate_wfo_windows` no longer hardcodes a fold count. `n_splits=0/None`
(now the default) generates as many rolling folds as fit, so the WFO runs from
~2019 right up to 2026 (**27 folds** on 1h).
- Files: `src/multi_optimiser_momentum_strategy.py`,
  `src/configs/optimiser_momentum_config.toml`

### Per-timeframe windows + 1d dropped

Spanning all data exposed a deeper issue: **a 3-month validation window is too
short on higher timeframes** — the MAD long-MA (`y≈80`) needs ~80 bars just to
warm up, so short folds get consumed by NaN dropping and produce too few trades.
Measured fraction of folds with ≥5 trades (BTCUSDT, `x=10 y=80 tema`):

| Timeframe | Windows                 | Folds | Folds ≥5 trades | Median trades |
|-----------|-------------------------|------:|----------------:|--------------:|
| 1h        | train 12 / val 3        |    27 |          26 ✅  |            16 |
| 4h        | train 18 / val 6        |    12 |           8 ✅  |             5 |
| 1d        | any window size         |   ≤27 |           0 ❌  |             0 |

- **1h** is the only timeframe with robust statistics out of the box.
- **4h** became evaluable only after widening to train 18 / val 6 (was 3/27
  folds, now 8/12).
- **1d is structurally hopeless**: daily MAD generates ~0–1 crossovers per fold
  *regardless of window size* (even a 12-month daily window gave ~1 trade) — the
  long MA is too smooth to cross often. **Daily was dropped from the grid.**

Added `[wfo.per_timeframe.<tf>]` config overrides (1h: 12/3, 4h: 18/6) wired
into the optimiser so each timeframe gets an appropriately sized window.

### Objective hardening (kills the degenerate "winners")

The morning relaunch produced absurd top scores (e.g. DOTUSDT_1d "IR≈26,946")
from folds with 2–3 trades whose excess-return std collapses toward zero. Added
to `trading_optimisation`:
- `MIN_TRADES_PER_FOLD = 5` — folds with fewer return rows are rejected (`-inf`).
- `IR_CAP = 3.0` — the information ratio is clipped to ±3 so a single lucky fold
  cannot dominate the cross-fold mean.

Verified: capped values now sit at exactly 3.0 on 4h; the degenerate daily
"winners" are gone.
- Files: `src/multi_optimiser_momentum_strategy.py`

### Merton stress test wired for the regime strategy

`_merton_path_worker` now applies *all* Optuna params generically (handling the
`y = x + offset`), so it works for both plain and regime strategies (the latter
adds `adx_window` / `adx_threshold`). The Merton CLI takes a `plain|regime` arg.
- Files: `src/strategy_utils/merton_stress_test.py`

### Result — Regime-gated MAD vs plain MAD (fresh full-span WFO)

> **RETRACTED — the Sharpe numbers in the first version of this table were
> wrong.** `compute_sharpe_ratio` annualised trade-to-trade returns with a
> hardcoded `sqrt(365)`, but `evolution_df` has one row *per trade* (irregularly
> spaced, often ~3 weeks apart), not one row per day. The annualisation factor
> was therefore off by a trade-frequency-dependent amount, producing absurd
> Sharpe values (−16 to +14). See the bug fix below. A corrected comparison on
> BTC/ETH/BNB is being regenerated and will replace this section.

**Sharpe bug fix.** `compute_sharpe_ratio` (in both
`multi_optimiser_momentum_strategy.py` and `renderer_strategy.py`) now infers
the annualisation factor from the actual mean spacing of the `Time` column
(`periods_per_year = 365 / mean_days_between_trades`) instead of assuming daily
rows. Verified on a BTCUSDT 4h fold that previously read −13.5: corrected values
now fall in the sane −3 to +1 range. The IR objective was *not* affected (it
compares portfolio vs buy-and-hold at the same row spacing, so the bad factor
largely cancels); only the recorded/plotted Sharpe was distorted.

What still holds from the (invalidated) table: the *IR objective* and the WFO
mechanics are sound, and the **direction** of the regime effect (it removes
whipsaw trades and tightens drawdowns) was independently confirmed on a
single-window backtest earlier. The magnitude/Sharpe claims await the corrected
re-run.

### Corrected comparison — Regime vs plain MAD (BTC/ETH/BNB, fixed Sharpe)

Both variants re-run on BTC/ETH/BNB × {1h, 4h} with the corrected Sharpe.
Distributional comparison (median / 90th-pct OOS Sharpe across all finite
trials — resistant to single-trial cherry-picking):

| study      | plain ShMed | regime ShMed | plain ShP90 | regime ShP90 |
|------------|------------:|-------------:|------------:|-------------:|
| BTCUSDT_1h |       −0.50 |       −0.26  |       −0.13 |        0.47  |
| BTCUSDT_4h |       −0.01 |        0.13  |        0.54 |        0.58  |
| ETHUSDT_1h |       −0.25 |        0.00  |        0.17 |        0.49  |
| ETHUSDT_4h |        0.39 |        0.33  |        0.79 |        0.92  |
| BNBUSDT_1h |       −0.28 |       −0.01  |        0.12 |        0.52  |
| BNBUSDT_4h |       −0.08 |        0.30  |        0.55 |        1.10  |

- **Median OOS Sharpe: regime > plain in 5/6 studies.**
- **Top-decile OOS Sharpe: regime > plain in 6/6 studies.**
- The values are now in a believable range (≈ −0.5 to +1.1), unlike the
  retracted table. The regime edge is **real but modest** — most median Sharpes
  sit near zero. Plain MAD alone is not a money-maker; the gate consistently
  nudges the distribution up, especially on 4h.

### Merton stress test — regime strategy does NOT survive synthetic paths

Ran the Merton jump-diffusion stress test on the top-10 WFO trials of each
regime 4h study (`robustness` = fraction of synthetic paths beating
buy-and-hold; 3,500 path-evaluations per coin):

| Coin (4h) | Best robustness | Observation |
|-----------|----------------:|-------------|
| BNBUSDT   |            0.49 | best trial ≈ coin flip vs B&H |
| BTCUSDT   |            0.23 | top-WFO trials rank low |
| ETHUSDT   |            0.17 | all top trials WFO=3.0, robustness ≤0.17 |

Two clear conclusions:
- **Robustness is low (0.17–0.49) everywhere** — regime-gated MAD beats B&H on
  synthetic paths well under half the time. As a *standalone* strategy it does
  not pass stress testing.
- **The highest-WFO-score trials are NOT the most robust** in any coin — the
  classic overfitting signature, which Merton successfully exposed. Selecting
  params by historical OOS score alone would pick overfit configs.

**Net:** the regime gate is a genuine *improvement component* over plain MAD,
but neither is a finished edge. This motivates the next combination steps
(equal-weight signal vote, portfolio of sleeves) rather than shipping MAD+ADX.

### Merton demoted from selection gate → optional tail-risk tool

On reflection, Merton is the wrong tool for *param selection* of a momentum
strategy, for two reasons:

1. **Model mismatch.** Merton paths are jump-diffusion: increments are
   essentially independent (no trend persistence, no regime structure). MAD is a
   *trend* strategy that exploits serial correlation — which Merton paths do not
   have by construction. A momentum strategy on a near-random-walk *should* score
   ~0.5, so the low robustness (0.17–0.49) is partly a model artifact, not pure
   overfitting. This weakens Merton as selection evidence here.
2. **Cost vs. value.** The one genuine signal it gave (best-WFO ≠ most-robust)
   can be obtained more cheaply and more credibly from the WFO folds themselves,
   on *real* data.

**Replacement — consistency-penalised WFO objective.** `_score_on_windows` now
returns `mean_OOS_IR * consistency`, where `consistency` is the fraction of
valid folds that beat buy-and-hold. A spiky "one lucky fold" config is scaled
down by its low consistency; a config positive across most folds keeps its
score. This bakes the overfitting guard directly into the objective, on real
data, for free. Added `MIN_VALID_FOLDS = 3` so a config that only trades in one
or two folds is rejected outright. Verified: best objective dropped from a spiky
3.0 (IR cap) to a modest ~0.0–0.5, reflecting MAD's true (weak) consistency.

**Merton's remaining role:** optional crash / tail-risk scenario tool (the
`crash`, `high_vol` scenarios answer "does this blow up in a jump?", which WFO
cannot if the history lacks a big crash). It is no longer part of the
param-selection loop.

### Next
- [ ] (done) Replace Merton-as-gate with the consistency-penalised WFO
      objective; keep Merton optional for tail-risk only.
- [ ] Equal-weight signal vote: MAD + time-series momentum + a mean-reversion
      signal (zero fitted weights).
- [ ] Portfolio of sleeves (generalize `launcher_multi_strategy`).
- [ ] Apply the trade-spacing annualisation fix anywhere else Sharpe-like
      metrics are computed on `evolution_df`.

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
