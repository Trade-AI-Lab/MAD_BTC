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

### Replication on SOL / XRP / LINK (4h rebuilt from 1h) and cumulative returns

`resample_4h.py` rebuilds 4h from the 1h pickles (checked on BTC: open exact, close within 0.36%, volume up to
18% off in the worst bar). Same rules and settings, models refit walk-forward on the new assets only. This is a
new-asset test, not a new-time test: the coins are correlated with the original five and the period overlaps.
Chart: `src/meta_labeling/cumulative_returns.png` (`plot_cumulative.py`, series in `series_*.csv`).

Sharpe, 4h, new assets (base / dz>q67 / dz67&vol / gboost):

| lookback | base | dz>q67 | dz67&vol | gboost | weekly DSR dz67&vol (36 / 100 trials) |
|---------:|-----:|-------:|---------:|-------:|------:|
| 5  | 0.43 | 0.23 | **-0.07** | -0.37 | 0.01 / 0.00 |
| 10 | 0.64 | 0.94 | 1.04 | 0.76 | 0.38 / 0.23 |
| 20 | 0.75 | 1.35 | 1.39 | 1.06 | 0.65 / 0.47 |

- **The headline did not replicate.** lb5 + volume (Sharpe 1.44, DSR 0.98 in-sample) is *worse than the plain
  rule* on the new assets (-0.07 vs 0.43), and gboost is negative there too. lb10 and lb20 do replicate in
  direction (dz rules beat base), with lb20 the strongest, the opposite of the original ordering.
- Reading: a `dz` filter helps in 4 of 6 (lookback x asset-set) cells and hurts in 1 clearly (lb5 new assets),
  and lb20 on the original assets was flat. The effect size is not stable across lookbacks. No weekly DSR is
  above 0.65.
- Cumulative growth of $1 (original five, lb5): all trades 0.77, gboost 1.47, dz 3.11, dz+volume 3.89,
  buy & hold 0.73. New assets, lb20: all trades 2.59, dz 5.20, dz+volume 5.20, buy & hold 1.42.
- Drawdown on the new assets is not reliably lower than the plain rule for `dz` filters (lb10 -41% vs -44%,
  lb20 -32% vs -44%, lb5 -30% vs -58%), but those are exposure effects as before.

Status: the filter is a plausible idea with mixed replication, not a validated strategy. Stop here unless a
genuinely new-time test (data after 2026-02) becomes available.

### Per-asset comparison vs buy & hold (`run_per_asset.py`, `per_asset.png`, `per_asset.csv`)

Rule = `dz` above the 0.67-quantile of earlier pooled trades AND volume above its 50-bar mean, 4h, lookback 5 and 10,
8 assets, walk-forward OOS from Sep 2021, 0.1%/side. **Data caveat: the 4h files for BNB, ADA and DOT end
2024-05-27 (others 2026-02-27), so their windows are shorter and not comparable with the other five.** This also
means the earlier 5-asset equal-weight results stopped including ADA/DOT after May 2024.

Growth of $1 (Sharpe, max DD) per asset:

| asset | buy & hold | rule lb5 | rule lb10 |
|-------|-----------|----------|-----------|
| BTC  | 1.52 (0.44, -77%) | **2.14** (0.87, -28%) | 1.19 (0.29, -38%) |
| ETH  | 0.65 (0.20, -81%) | **3.92** (1.20, -32%) | 1.60 (0.54, -43%) |
| BNB* | 1.76 (0.65, -72%) | 2.77 (1.42, -27%) | **3.13** (1.51, -27%) |
| ADA* | 0.21 (-0.30, -90%) | **1.13** (0.31, -41%) | 0.99 (0.18, -44%) |
| DOT* | 0.26 (-0.17, -93%) | 0.93 (0.10, -55%) | **2.30** (0.93, -33%) |
| SOL  | 0.61 (0.38, -97%) | 0.89 (0.10, -51%) | **6.56** (1.12, -53%) |
| XRP  | 1.44 (0.51, -78%) | 0.64 (-0.15, -56%) | **1.57** (0.44, -55%) |
| LINK | 0.36 (0.20, -87%) | 0.39 (-0.49, -67%) | **0.71** (0.00, -51%) |

- Final value beats buy & hold in 7/8 assets at lb5 (not XRP) and 7/8 at lb10 (not BTC).
- Sharpe beats buy & hold in only 5/8 at each lookback: lb5 fails on SOL, XRP, LINK; lb10 fails on BTC, XRP, LINK.
  Where it wins on final value but not Sharpe (SOL lb5, LINK lb5) it is mostly avoiding a crash, not earning.
- Drawdown is smaller than buy & hold on all 8 (28-67% vs 72-97%), but the rule is invested only 13-24% of the
  time, so this is mostly exposure.
- The lookbacks disagree per asset: lb5 is best on BTC/ETH/ADA, lb10 on BNB/DOT/SOL/XRP/LINK (SOL lb5 0.89 vs lb10
  6.56 on one asset, and lb10 SOL is a single-regime run-up from 2023). No lookback wins everywhere, consistent
  with the earlier instability finding. LINK is negative or flat in both.

### Next
- [x] (done above, new assets) Frozen test: fix q=0.67 + volume rule at 4h lb5/lb10 on data not used so far (post-2024 only for the quantile
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

### Frozen breakout vs buy & hold in the years buy & hold lost (`run_bh_lost.py`)
Full-history asset-years (65 full calendar years across 8 assets): buy & hold lost in 32. In those 32 the frozen rule made
+7.6% on average (median +8.3%, positive in 62%) against -42.0% for buy & hold, and beat it in all 32. By year: 2018 rule -8.6% vs
-54.5%, 2019 +20.5% vs -21.8%, 2022 -1.1% vs -71.7%, 2025 +20.2% vs -33.7%, 2026 +9.2% vs -22.2%. Pooled equal-weight: 2022 rule
+0.6% vs -73.4%, 2025 +15.8% vs -28.1%, 2026 (to Oct) +9.8% vs -20.4%. In the 33 asset-years where buy & hold gained, the rule beat it
in only 9% (mean +48% vs +572%: it keeps ~10% exposure). Bars where an asset sits >40% below its peak are 70% of all bars: annualised
mean return there is -11.7% for buy & hold and +14.8% for the rule. Weak spots: 2018 and the worst single asset-years
(ADA 2018 -34%, DOT 2025 -20%, SOL 2022 -12%) still lose, just far less than buy & hold. Summary: the rule is a bear-market and
sideways-market earner and a bull-market laggard; this is what a ~10%-exposure breakout overlay should look like.

---

## 2026-10-25 (report) — Full written report

`report/2026-10-25/mad_btc_report.pdf` (source `mad_btc_report.tex`, tables built by `build_tables.py` from the study outputs, figures in
`figures/`). It documents the data and conventions, every metric (CAGR, Sharpe, max drawdown, exposure, Deflated Sharpe, random-entry p, week-block
bootstrap), how each strategy is computed (MAD, meta-labeling, dz rule, ensemble, breakout, with a worked trade), the results per strategy and
per coin (CAGR, Sharpe, drawdown, exposure, trades, year by year), the frozen rule and its pre-registered verdict, the corrections made along the
way, and consistency with herc_llm's KEPT.md / CLOSED #24, #26 (recommended status: watch-only).

---

## 2026-10-25 (freeze) — Breakout rule k=12, H=24, volume spike FROZEN

`src/meta_labeling/frozen_breakout.py`, spec in `frozen_breakout.json`, scores appended to
`frozen_breakout_history.csv`. Rule: first cross of zb>2.0 (12-bar return / (sigma_1bar over 100 bars * sqrt 12)), volume[t-1]
> 1.5x its 50-bar mean, long at the next open, 24-bar hold, 0.1%/side, no overlap per asset, same 8 assets. Code hashes
(run_breakout.py, core.py, run_meta.py) are stored; the script warns if they change. Forward scoring starts at entries on/after
**2026-10-25 20:00** (first bar after the last downloaded one). Nothing before that counts as forward.

Baselines stored at freeze: dev (entries before 2026-02-28) n=529, +2.75%/trade, win 58%, 8/8 assets positive, bootstrap
5th pct +1.6%, p_random 0.00. Seen window 2026-02-28 -> freeze: n=41, +3.17%, win 59%, 7/8 assets, bootstrap 5th pct
-1.6%, p_random 0.04 (this window was looked at while choosing the rule, so it is NOT out of sample).

Pre-registered verdict: wait for >= 100 forward trades (about 20 months at ~5 trades/month: the rule is selective).
ACCEPT if mean net > 0 AND week-block bootstrap 5th percentile > 0 AND p_random < 0.05; REJECT if n >= 100 and mean net <= 0;
otherwise inconclusive. First re-score (0 forward trades, nothing to score yet): inconclusive.

To re-score: `python src/meta_labeling/frozen_breakout.py` (fetches via `fetch_klines.py` first when run by hand:
`python src/meta_labeling/fetch_klines.py BTCUSDT ETHUSDT BNBUSDT ADAUSDT DOTUSDT SOLUSDT XRPUSDT LINKUSDT`).

---

## 2026-10-25 (follow-up round) — lookback ensemble, exposure-matched drawdown, MAD-free breakout, frozen forward test

New code: `core.py` (shared), `fetch_klines.py`, `run_ensemble.py`, `run_frozen.py`, `run_breakout.py`,
`run_breakout_checks.py`. **Data fixed:** `fetch_klines.py` appended closed 4h bars from the public Binance klines
endpoint to all 8 assets through 2026-10-25 (BNB/ADA/DOT gap 2024-05 -> now filled; originals backed up in
`data/backup_4h/`, data/ is gitignored). Everything below uses data through 2026-10-25.

### 1. Lookback ensemble (rule = dz>q67 & volume>mean50, lookbacks 5/10/20 as equal-capital sleeves)
Walk-forward OOS from 2022-06, 8 assets. Sharpe beats buy & hold on 6/8 assets for the ensemble vs 5/4/5 for lb5/10/20
alone; the across-asset spread of Sharpe falls from 0.67 (lb5) to 0.42 (ensemble). Pooled portfolio: ensemble Sharpe 1.16,
DD -27%, $1 -> 2.34 vs buy & hold 0.59, -69%, 2.13. Weekly DSR (12 trials) 0.70. So averaging does smooth the per-asset
swings, but the result is still below the 0.95 bar.

### 2. Exposure-matched drawdown: the drawdown cut is NOT edge
Rule ensemble is invested 19% of the time. Buy & hold scaled to 19% has DD -19% (rule: -27%); all-trades ensemble scaled to
the same exposure has DD -26%. A vol-managed buy & hold at 19% average exposure has Sharpe 0.45, DD -19%.
The rule's value is return per unit of exposure (ann 21.5% vs 6.6% for scaled buy & hold, Sharpe 1.16 vs 0.59),
not lower drawdown. Earlier "lower drawdown" claims should be read as exposure.

### 3. Frozen forward test (rule fixed on trades closed before 2026-02-28; thresholds in `frozen_rule.json`)
Entries 2026-02-28 -> 2026-10-25, 8 assets: all MAD trades have mean net -0.23% to -0.36% per trade; the rule keeps 106/64/33
trades (lb5/10/20) at +0.19% / +0.04% / -0.15%. Pooled ensemble: Sharpe 0.13, DD -9.7%, $1 -> 1.006, vs buy & hold
0.45, -33.5%, 1.068 and all-trades ensemble -0.39, -25%, 0.90. The filter improves on the unfiltered MAD rule (in 2 of 3
lookbacks) but does not beat buy & hold on return in this window. Eight months, ~200 trades: weak evidence either way.

### 4. MAD-free breakout (the new finding)
Event: first bar where a volatility-scaled k-bar return exceeds 2.0 (zb = log-return over k bars / (sigma_1bar * sqrt(k)),
sigma over 100 bars), entered at the next open, optional volume spike (volume[t-1] > 1.5x its 50-bar mean), held H bars, long
only, 0.1%/side, no overlap. Grid fixed in advance: k in {6,12} x H in {6,12,24} x volume {no,yes} = 12 variants, 8 assets,
2018-2026, no fitting. Control: random entries with the same count and hold per asset.

- Mean net return per trade is positive in all 12 variants, 0.4% to 2.8%, and beats 1000 random-entry draws in every
  case (p ~ 0). Volume spike adds a little (e.g. k6 H12: 1.15% -> 1.45%; k12 H24: 2.67% -> 2.78%).
- Best two: k6/H12/volume (n=924, +1.45%/trade, portfolio Sharpe 1.33, DD -22%, 8% exposure) and k12/H24/volume
  (n=570, +2.78%, Sharpe 1.41, DD -21%, 10% exposure). Equal-weight buy & hold over the same span: Sharpe 0.88, DD -83%.
- Robustness (both variants): positive on all 8 assets; positive in 7-8 of 9 calendar years (weak: 2018, 2022, and k6 in
  2025-26); still positive with 0.2%/side fees (+1.25% / +2.57%); week-block bootstrap 95% interval of mean net per
  trade [+0.9%, +2.0%] and [+1.7%, +3.8%]; MATIC (renamed/delisted, not a survivor) +3.8% / +7.7%, p 0.02-0.03.
- Weekly-portfolio DSR (12 / 100 / 300 trials): k6/H12/vol 0.99 / 0.91 / 0.82; k12/H24/vol 0.997 / 0.956 / 0.90.
  First signal in this project that clears the gate at its own grid size and nearly so at 100 trials.
- Forward window (entries after 2026-02-28, never used): k=12 variants positive (+1.1% to +3.2%/trade, 7 of 8 assets
  positive, n 41-47 for the volume versions); k=6 variants are flat to negative (-0.3% to +0.1%, 3-5 of 8 assets).
  So the slower (k=12, H=24) version holds up forward, the faster one does not.

Caveats: long-only crypto 2018-2026 is bull-dominated (the random-entry control removes drift per asset but not regime);
survivorship (7 of 8 assets survived; MATIC helps but is one asset); the forward window is 8 months; events cluster across
coins in time (the weekly-portfolio DSR and the week-block bootstrap account for that); the 12-variant grid was chosen by
me, so the 12-trial count is a lower bound for the whole project (100+ trials is the honest sensitivity).

Verdict: the breakout, not the MAD crossover, is where the signal was. MAD itself remains a weak base (log above); the dz
filter was partly picking out fast, high-volume moves. This is a hypothesis worth a clean out-of-sample test, not a validated
strategy.

### Next
- [ ] Keep `frozen_rule.json` and a frozen version of k12/H24/volume; re-run `fetch_klines.py` + `run_breakout_checks.py`
      every few weeks and score only newer entries.
- [ ] Add short side / market-state gate (BTC trend) and a volatility-based exit instead of fixed H; compare at equal exposure.
- [ ] Test on other venues/assets not in this set (more delisted coins; futures) and on 1h with higher fees.
- [ ] Portfolio sizing: fixed-risk per trade (vol-scaled) across overlapping signals.

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
