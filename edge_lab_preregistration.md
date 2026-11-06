# Pre-registration: `edge_lab.py` exploration and frozen holdout spec

Status: exploration protocol and decision rules below were fixed **before**
looking at any exploration result; the "Exploration log" table and "Frozen
finalists" section were appended after the bounded exploration run, and the
holdout has not been requested or touched at the time of freezing.  The dev
database only (`../crypto-data/crypto_market.db`) was used.  No file of the
registered pipeline (`features.py`, `backtest.py`, `train.py`, results,
artifacts) was modified.

This lab exists to honestly push the leads that looked mildly alive in
`results_physics.json` (positive but non-significant direction deltas for the
physics-analogue arm, largest at 60 minutes; a mixed volatility result).  The
prior expectation remains a null: the physics features are dimensionless
transformations of the same candles, and a 50–52% minute-level hit rate is
close to what serial dependence and micro-trends can produce without a
cost-viable edge.

## Pre-holdout amendment (2026-10-02; holdout still unseen)

This amendment was written **before** any holdout value was evaluated: the
holdout was only inspected for schema and window bounds (read-only), and no
model has been run on it.  The frozen features, models, horizons, thetas, and
fee scenarios are unchanged.  Three corrections/clarifications follow.

**1. Basis-point unit correction (doc-only).** 1 bp = 0.0001 in return units.
The earlier draft converted some figures as if 1 bp = 0.00001.  Corrected
figures: finalist 1 validate gross +73.1 bp/trade, net (10 bp/side) +53.1
bp/trade; finalist 2 +44.8 / +24.8 bp; finalist 3 +19.8 / −0.2 bp; the frozen
thetas' EXPLORE trade means are −5.9 bp (F1), −14.0 bp (F2), −17.1 bp (F3); the
logistic+physics h15 EXPLORE θ=0 minute-level gross is −0.06 bp/trade.  The
tables in this document already use raw return units; the corrected bp numbers
appear in "What the leads actually showed" and "Honest reading of the freeze".

**2. Power rule for the holdout.**  The holdout
(`../crypto-data/holdout_klines_1m.db`, identical `klines_1m` schema) has
26,584 contiguous minutes per symbol, 2026-09-14T05:09Z → 2026-10-02T16:12Z;
the dev database ends at 2026-09-14T05:08Z, so the two are contiguous.  After
the 240-minute embargo the evaluated window is 26,343 minutes per symbol
(~105k pooled direction rows).  Expected trade counts, written down before
running as (validate trade rate × evaluated holdout window):

| finalist | validate trades | validate window (min) | rate /min | evaluated window (min) | expected holdout trades |
|---|---:|---:|---:|---:|---:|
| d1_logistic_physics_h15_per | 39 | 400,316 | 9.74e−5 | 26,343 | 2.6 |
| d1_logistic_market_h15_per | 131 | 400,316 | 3.27e−4 | 26,343 | 8.6 |
| d2_hgb_physics_h240_per | 78 | 400,316 | 1.95e−4 | 26,343 | 5.1 |

**Power rule (added now).**  A finalist with fewer than 30 holdout trades is
**"INCONCLUSIVE (underpowered)"** for the net-return criterion — neither pass
nor fail; its hit-rate criterion (full coverage, ~105k rows) is still
evaluated and reported, and an overall "pass" is impossible without the net
criterion passing.  With the expected counts above, all three finalists are
expected to be underpowered on the net criterion; the holdout can be extended
later with `../crypto-data/fetch_holdout.py` and the same frozen test rerun on
the extended window (no parameter, theta, or model changes permitted).

**3. Holdout feature warm-up (code fix).**  Building features on the holdout
database alone would leave the first ~479 minutes of physics features cold
(OU warm-up).  `--stage holdout` now builds features on a scratch database
containing the last 600 minutes of the dev database plus all holdout rows
strictly after the dev database's final candle, so every evaluated row has real
rolling history; evaluated rows remain only the holdout rows at/after
`max(first holdout timestamp, dev end) + 240 min`.  Targets on evaluated rows
are overlaid from a holdout-only build (contiguous inside the holdout
database), so the junction can never distort a target.  Verified before
running: no fit-row target extends past the dev end (asserted in code), and no
dev row enters the evaluated set.

## Data split (dev database only)

- Source: `../crypto-data/crypto_market.db`, `klines_1m`, four assets
  (BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT), 1,335,189 contiguous minutes per asset,
  2024-03-01 00:00 UTC through 2026-09-14 05:08 UTC (epoch 1709251200..1789362480).
- EXPLORE = first 70% of the 1,335,189 unique target timestamps:
  `t_split = 1765329120` (2025-12-10 01:12 UTC).  For horizon `h`, a row at
  target open `t` is an explore row only if `t + 60h <= t_split`, so no explore
  fit can use a target that reaches into the validate period.
- VALIDATE = rows with `t >= t_split + 240*60` (embargo = the longest lab
  horizon).  Used only to screen the explore shortlist and to select the
  frozen finalists.
- Holdout (Sept 2026 -> now) is **not** used here; it is provided by the owner
  after freezing.  The frozen holdout protocol is specified below.

## Features, targets, and models

- Feature sets: `MARKET_FEATURES` (14 columns) and
  `MARKET_FEATURES + PHYSICS_FEATURES` (20 columns), imported unchanged from
  `features.py`.  Pooled models add four asset one-hot columns
  (`asset_BTCUSDT`, …, `asset_BNBUSDT`).
- Targets: the registered `forward_return_h` / `forward_vol_h` columns for
  `h in {1, 5, 15, 60}`.  For `h in {120, 240}` the direction target is rebuilt
  from the registered one-minute forward return as
  `expm1(rolling_sum(log1p(r1), h, min_periods=h).shift(-(h-1)))`; the
  reconstruction was verified against the registered 5/15/60-minute columns to
  a max absolute difference < 1e-9 (observed ~1e-15).
- Models and fixed hyperparameters:
  - `logistic`: per-fold `StandardScaler` + `LogisticRegression(C=1.0,
    solver="lbfgs", max_iter=300, random_state=0)` (identical to the registered
    pipeline), score = `decision_function`, `p_up` = `predict_proba[:, 1]`.
  - `ridge`: per-fold `StandardScaler` + `Ridge(alpha=1.0)` (registered
    volatility model).
  - `hgb`: `HistGradientBoostingClassifier` / `Regressor` with
    `learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=100,
    l2_regularization=1.0, max_bins=255, random_state=0, early_stopping=False,
    warm_start=True`; boosting in chunks of 25 iterations up to 200, with early
    stopping on a **time-ordered inner split** (last 15% of the training rows,
    with an `h`-minute gap after the cut); stop after 3 consecutive chunks
    without validation-loss improvement; the best checkpoint is used for the
    test fold.  Validation loss = log-loss (direction) / MSE (volatility).
  - Row cap: pooled HGB and its logistic comparison pair train on at most
    ~1,500,000 of the most recent training rows per fold (time-based cutoff,
    all assets retained), for tractability under the shared-CPU constraint;
    all other configurations use all rows.
- Walk-forward inside EXPLORE: `backtest.make_walk_forward_splits` with
  `n_folds=3`, `initial_train_fraction=0.50`, `embargo=max(h, 60)` minutes.
  Metrics are computed on the pooled OOS rows of all folds (per-asset results
  are also logged).
- Validate screening: fit once on all EXPLORE rows of that horizon, predict the
  VALIDATE window once.  The screening shortlist is the top 6 direction
  configurations by `max(Δhit vs persistence, Δhit vs momentum)` on EXPLORE
  (pooled, `n >= 50 000`).  The seven volatility configurations were also
  evaluated on VALIDATE for generalization reporting only; no selection
  decision (shortlist or finalists) used any volatility validate number, and no
  other search used validate.

## Confidence filtering and trading simulation

- `p_up` = model P(up).  A signal exists when `|p_up - 0.5| > theta`.
- Theta grid: `{0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20}`.
  Theta is chosen **on EXPLORE predictions only**: maximize the taker
  net-of-fee mean return per trade subject to at least 200 explore trades;
  ties (six decimal places) are broken toward the larger trade count.  The
  full coverage/hit-rate/gross/net curve is logged for every direction
  configuration on both EXPLORE and VALIDATE.
- Trade rule: long if `p_up >= 0.5 + theta`, short if `p_up <= 0.5 - theta`,
  otherwise flat; position held `h` minutes; entries are non-overlapping
  (next entry at `t + h*60` at the earliest), evaluated per asset.
- Fee scenarios per side: 0.10% (taker), 0.075% (VIP taker), 0.02% (maker);
  net per trade = `side * forward_return_h - 2*fee`.
- Reported per finalist: trade count, gross and net mean return per trade, hit
  rate, long share, net sum, and a Sharpe-like ratio (per-trade net mean / net
  standard deviation) with a moving-block bootstrap CI over trades
  (block = 10 trades, `n_boot = 4999`).  One-sided bootstrap p-value for
  `net mean > 0`: `(1 + #(bootstrap mean <= 0)) / (n_boot + 1)`.
- Volatility metrics (separate task, no trading): MSE and QLIKE
  (`y/p - log(y/p) - 1`, all values clipped at 1e-12) against the registered
  EWMA baseline `ewma_vol_60m * sqrt(h)`, explore and validate, per asset.

## Multiple testing and honest reporting

- Every configuration tried is listed in the exploration log with its
  explore/validate numbers; no configuration is silently dropped.
- The lab's p-values are descriptive except for the frozen holdout test; the
  holdout decision uses Bonferroni over the finalists:
  `alpha = 0.05 / 3 = 0.016667` (three finalists or fewer).
- A finalist **passes** on the holdout only if both hold at that alpha:
  1. primary: net-of-fee (0.10% per side) mean return per trade > 0 with
     one-sided block-bootstrap `p < 0.016667` (`n_boot = 4999`, block = 10
     trades), and
  2. robustness: hit rate exceeds **both** the persistence (`sign(ret_1m)`) and
     momentum (`sign(ret_15m)`) baselines with paired 60-minute moving-block
     bootstrap two-sided `p < 0.016667` (`n_boot = 4999`, deterministic prefix
     cap of 500,000 paired rows as in `backtest.paired_bootstrap_delta`; the
     reported deltas always use every row).
- The lab claims a usable edge only if at least one finalist passes both
  criteria on the untouched holdout.  Anything else is reported as a null,
  including "gross positive but net negative" outcomes.

## Frozen holdout protocol (executed only after the owner provides the holdout DB)

1. Build features on the holdout range with `features.build_feature_frame`
   (market features only; GCP excluded) and the frozen long-horizon target
   reconstruction; `assert_causal_features` must pass.
2. Fit each finalist on **all dev rows** (EXPLORE + VALIDATE) whose target is
   complete before the holdout starts; the same scaler/model/early-stopping
   code as in `edge_lab.py`; no holdout rows are used for fitting, early
   stopping, feature construction choices, or theta selection.
3. Evaluate on holdout rows with
   `t >= max(first_holdout_timestamp, dev_end) + 240 minutes`
   (embargo = longest lab horizon), using the frozen theta and the frozen fee
   scenarios; report gross and net metrics, trade count, coverage, and the
   decision-rule p-values above.
4. If the holdout contains a gap after the dev range, that gap is not bridged:
   the frozen spec is fit-on-dev / test-on-holdout only.

## Exploration log

*(Appended after the exploration run: every configuration tried, with its
EXPLORE walk-forward numbers and, where run, its VALIDATE numbers.  Δhit is in
percentage points versus the named baseline on the same rows.  `n` is pooled
OOS direction-row count (exact-zero targets excluded).  Full per-asset numbers,
confidence curves, and bootstrap details are in `results_edge_lab_dev.json`.
Wall-clock: 3820 s for explore+validate+freeze, 524 s for the volatility
validate pass, 51 s for the freeze refresh; the frame build (81 s) is cached
under `~/.cache/edge_lab/`.)*

**What the leads actually showed.**

1. **Nonlinearity (HGB) does not beat logistic at short horizons; it adds a
   little at long horizons with physics features.**  HGB − logistic explore
   hit-rate difference on identical features: −0.01 pp (market h15), −0.09 pp
   (market h60), +0.13 pp (physics h15), +0.18 pp (physics h60).  At h=120/240
   with physics features the gap grows to +0.41 pp (51.634 vs 51.222) and
   +0.38 pp (51.498 vs 51.116), and HGB+physics has the highest explore IC in
   the lab (0.0353 at 240 m).
2. **Longer horizons: the physics delta over market-only grows with h, while
   market-only absolute hit rates fall.**  Explore hit: HGB market 50.916
   (h120) / 50.367 (h240) versus HGB physics 51.634 / 51.498; Δhit vs momentum
   +0.97 pp (market h240) versus +2.10 pp (physics h240).  This is the
   strongest direction lead in the lab, yet still ~51.5% absolute.
3. **Pooling across assets does not help.**  Pooled HGB+physics h60 (capped)
   51.219 versus per-asset 51.489 (−0.27 pp); pooled logistic+physics 51.361
   versus per-asset 51.430 (−0.07 pp), with the pooled capped logistic pair
   (51.226) showing the cap alone explains little of the HGB gap.
4. **Confidence filtering only works in a vanishing tail.**  Logistic+physics
   h15 on EXPLORE: θ=0.02 → 47.4% coverage, 52.29% hit, gross −0.15 bp/trade;
   θ=0.05 → 5.7% coverage, 53.17% hit; θ=0.10 → 0.04% coverage (699 of
   1.86 M rows), 54.79% hit, gross +32.7 bp/trade; θ=0.20 → 46 signals, 80.4%
   hit, gross +427 bp/trade.  On VALIDATE at θ=0.10: 90 signals → 39
   non-overlapping trades, 73.3% hit, gross +73.1 bp/trade, net +53.1 bp/trade
   (10 bp/side).  At full coverage every configuration is net-negative against
   every fee scenario, so tail filtering is the only route to a positive trade
   mean — at the price of tens of trades and a selected threshold.
5. **Volatility: the MSE improvement generalizes across assets and horizons;
   the QLIKE improvement does not.**  HGB (physics or market) beats EWMA on MSE
   in all four assets, both periods, for h ∈ {1, 5, 15, 60}.  Relative to EWMA,
   BNB h1/h5/h15 MSE improves by 17%/13%/12% (explore) and 9%/5%/4%
   (validate); the pooled deltas are −6.0e−8…−2.2e−6.  QLIKE improves
   consistently only at h=1 (pooled −0.031 explore, −0.036 validate, negative
   for all four assets in both periods) and at h=60 on validate (−0.010); at
   h=5/15 it is worse for BTC (+0.014…+0.016) and slightly worse for BNB, but
   better for ETH/SOL.  Ridge QLIKE is not interpretable here: ~0.023% of its
   BTC h5 predictions are negative and get clipped at 1e-12, producing QLIKE in
   the thousands; HGB rarely predicts negative volatility.  Economically this
   is a modest variance-forecast improvement with no trading vehicle tested; it
   is not a cost-viable strategy on its own.
6. **Cost-aware trading: no configuration has a positive net-of-fee mean per
   trade at full coverage, and the frozen θ values are also explore-negative.**
   At θ=0 (trade every row), gross means are a few tenths of a basis point
   (logistic+physics h15 explore: gross −0.06 bp) against 20 bp round-trip taker
   fees.  The frozen θ (chosen on EXPLORE with ≥200 trades by maximizing net)
   yields explore trade means of −5.9 bp (finalist 1), −14.0 bp (finalist 2), and
   −17.1 bp (finalist 3) per trade at 10 bp/side; only the small VALIDATE
   windows of finalists 1–2 turn positive.

**Direction configurations (20).**

| cid | features | model | h | mode | explore n | explore hit% | explore Δhit persist (pp) | explore Δhit momentum (pp) | explore IC | validate hit% | validate Δhit persist (pp) | validate Δhit momentum (pp) | θ (explore) | validate trades | validate gross/trade | validate net/trade 10bp | net/trade 7.5bp | net/trade 2bp |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| d1_hgb_market_h15_per | market | hgb | 15 | per-asset | 1863121 | 51.370 | +2.84 | +2.70 | 0.0284 | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| d1_hgb_market_h60_per | market | hgb | 60 | per-asset | 1866204 | 51.041 | +2.51 | +2.12 | 0.0189 | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| d1_hgb_physics_h15_per | physics | hgb | 15 | per-asset | 1862294 | 51.558 | +3.03 | +2.89 | 0.0309 | 52.268 | +5.08 | +4.70 | 0.100 | 101 | +0.001506 | -0.000494 | +0.000006 | +0.001106 |
| d1_hgb_physics_h60_per | physics | hgb | 60 | per-asset | 1865365 | 51.489 | +2.96 | +2.57 | 0.0255 | 52.155 | +4.83 | +4.00 | 0.100 | 483 | -0.000653 | -0.002653 | -0.002153 | -0.001053 |
| d1_logistic_market_h15_per | market | logistic | 15 | per-asset | 1863121 | 51.377 | +2.85 | +2.71 | 0.0281 | 51.950 | +4.76 | +4.38 | 0.100 | 131 | +0.004476 | +0.002476 | +0.002976 | +0.004076 |
| d1_logistic_market_h60_per | market | logistic | 60 | per-asset | 1866204 | 51.131 | +2.60 | +2.21 | 0.0190 | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| d1_logistic_physics_h15_per | physics | logistic | 15 | per-asset | 1862294 | 51.430 | +2.90 | +2.76 | 0.0273 | 52.398 | +5.21 | +4.83 | 0.100 | 39 | +0.007310 | +0.005310 | +0.005810 | +0.006910 |
| d1_logistic_physics_h60_per | physics | logistic | 60 | per-asset | 1865365 | 51.311 | +2.78 | +2.39 | 0.0188 | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| d2_hgb_market_h120_per | market | hgb | 120 | per-asset | 1868903 | 50.916 | +2.31 | +1.73 | 0.0214 | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| d2_hgb_market_h240_per | market | hgb | 240 | per-asset | 1868664 | 50.367 | +1.75 | +0.97 | 0.0185 | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| d2_hgb_physics_h120_per | physics | hgb | 120 | per-asset | 1868063 | 51.634 | +3.03 | +2.45 | 0.0297 | 51.327 | +3.95 | +3.28 | 0.150 | 27 | +0.001804 | -0.000196 | +0.000304 | +0.001404 |
| d2_hgb_physics_h240_per | physics | hgb | 240 | per-asset | 1867824 | 51.498 | +2.88 | +2.10 | 0.0353 | 50.630 | +3.19 | +1.96 | 0.150 | 78 | +0.001984 | -0.000016 | +0.000484 | +0.001584 |
| d2_logistic_market_h120_per | market | logistic | 120 | per-asset | 1868903 | 51.055 | +2.45 | +1.87 | 0.0188 | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| d2_logistic_market_h240_per | market | logistic | 240 | per-asset | 1868664 | 51.211 | +2.59 | +1.81 | 0.0252 | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| d2_logistic_physics_h120_per | physics | logistic | 120 | per-asset | 1868063 | 51.222 | +2.62 | +2.04 | 0.0184 | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| d2_logistic_physics_h240_per | physics | logistic | 240 | per-asset | 1867824 | 51.116 | +2.49 | +1.72 | 0.0230 | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| d3_hgb_physics_h60_pool_cap | physics | hgb | 60 | pooled+cap | 1865365 | 51.219 | +2.69 | +2.30 | 0.0208 | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| d3_logistic_market_h60_pool | market | logistic | 60 | pooled | 1866204 | 51.175 | +2.65 | +2.26 | 0.0199 | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| d3_logistic_physics_h60_pool | physics | logistic | 60 | pooled | 1865365 | 51.361 | +2.83 | +2.44 | 0.0196 | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| d3_logistic_physics_h60_pool_cap | physics | logistic | 60 | pooled+cap | 1865365 | 51.226 | +2.70 | +2.31 | 0.0179 | NA | NA | NA | NA | NA | NA | NA | NA | NA |

**Volatility configurations (7).**

| cid | features | model | h | explore n | explore ΔMSE vs EWMA | explore ΔQLIKE | validate ΔMSE vs EWMA | validate ΔQLIKE | BNB ΔMSE (explore / validate) | BNB ΔQLIKE (explore / validate) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| v_hgb_market_h5_per | market | hgb | 5 | 1869132 | -1.469e-07 | +0.00078 | -1.104e-07 | +0.00130 | -1.350e-07 / -5.196e-08 | +0.00207 / +0.00165 |
| v_hgb_physics_h15_per | physics | hgb | 15 | 1868280 | -3.482e-07 | +0.00300 | -2.225e-07 | +0.00333 | -3.819e-07 / -9.453e-08 | +0.00471 / +0.00425 |
| v_hgb_physics_h1_per | physics | hgb | 1 | 1868304 | -8.273e-08 | -0.03101 | -6.018e-08 | -0.03620 | -6.642e-08 / -3.503e-08 | -0.03021 / -0.03514 |
| v_hgb_physics_h5_per | physics | hgb | 5 | 1868292 | -1.537e-07 | +0.00109 | -1.088e-07 | +0.00144 | -1.328e-07 / -5.328e-08 | +0.00255 / +0.00176 |
| v_hgb_physics_h60_per | physics | hgb | 60 | 1868184 | -2.178e-06 | -0.00427 | -1.138e-06 | -0.01003 | -2.800e-06 / -4.228e-07 | -0.00115 / -0.00675 |
| v_ridge_market_h5_per | market | ridge | 5 | 1869132 | -1.338e-07 | +5980.56233 | -7.620e-08 | +10911.98527 | -1.508e-07 / -3.287e-08 | +0.00495 / +0.00474 |
| v_ridge_physics_h5_per | physics | ridge | 5 | 1868292 | -1.364e-07 | +178.24965 | -7.495e-08 | +1847.56555 | -1.535e-07 / -2.995e-08 | +0.00574 / +0.00528 |

Notes on the tables: configurations without a VALIDATE block were not
shortlisted (the pre-registered rule shortlisted the top 6 direction
configurations; the 7 volatility configurations were additionally validated
for generalization reporting only, and no selection used their validate
numbers).  `validate trades` are non-overlapping (hold h minutes) at the θ
selected on EXPLORE; `net/trade` subtracts 2× the per-side fee from the gross
forward return.

## Frozen finalists

*(Frozen by running the pre-registered selection rule; the holdout has not
been requested or seen.  The rule: among the VALIDATE-screened direction
configurations, take the three with the highest VALIDATE taker net-of-fee mean
return per trade at the EXPLORE-selected θ.)*

| # | cid | exact model | features | h | θ | explore trades at θ (gross / net 10bp) | validate trades | validate gross/trade | validate net/trade 10bp | validate net/trade 7.5bp | validate net/trade 2bp |
|---|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | d1_logistic_physics_h15_per | logistic | physics | 15 | 0.1 | 216 (+0.001409 / -0.000591) | 39 | +0.007310 | +0.005310 | +0.005810 | +0.006910 |
| 2 | d1_logistic_market_h15_per | logistic | market | 15 | 0.1 | 596 (+0.000604 / -0.001396) | 131 | +0.004476 | +0.002476 | +0.002976 | +0.004076 |
| 3 | d2_hgb_physics_h240_per | hgb | physics | 240 | 0.15 | 313 (+0.000289 / -0.001711) | 78 | +0.001984 | -0.000016 | +0.000484 | +0.001584 |

**Validate detail of the three finalists (Bonferroni α = 0.05/3 = 0.016667).**

| # | cid | validate n | validate hit% | Δhit persist (pp) | Δhit momentum (pp) | IC | trades | net mean CI95 | p(net>0) | Sharpe-like [CI95] | passes on validate? |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|---|
| 1 | d1_logistic_physics_h15_per | 1589532 | 52.398 | +5.21 | +4.83 | 0.0537 | 39 | [+0.002195, +0.008688] | 0.0002 | 0.270 [0.133, 0.446] | YES |
| 2 | d1_logistic_market_h15_per | 1589532 | 51.950 | +4.76 | +4.38 | 0.0454 | 131 | [+0.000067, +0.004887] | 0.0232 | 0.125 [0.003, 0.288] | no |
| 3 | d2_hgb_physics_h240_per | 1600312 | 50.630 | +3.19 | +1.96 | 0.0100 | 78 | [-0.002186, +0.002550] | 0.5200 | -0.001 [-0.219, 0.206] | no |

- Finalist 1 hit tests: vs persistence Δ = +5.211 pp (p=0.0004), vs momentum Δ = +4.831 pp (p=0.0004).
- Finalist 2 hit tests: vs persistence Δ = +4.764 pp (p=0.0004), vs momentum Δ = +4.383 pp (p=0.0004).
- Finalist 3 hit tests: vs persistence Δ = +3.194 pp (p=0.0004), vs momentum Δ = +1.959 pp (p=0.1012).

**Honest reading of the freeze (important).**

- On EXPLORE, at the θ the rule selected, the non-overlapping trade mean is
  negative for all three finalists: −5.9 bp (F1, 216 trades), −14.0 bp (F2,
  596 trades), −17.1 bp (F3, 313 trades) per trade at 10 bp/side.  θ was
  chosen because it was the least-negative eligible option (≥200 trades), not
  because it was profitable on explore.
- On VALIDATE, finalist 1 shows +53.1 bp/trade over 39 trades (73.3% hit),
  finalist 2 +24.8 bp over 131 trades, finalist 3 −0.2 bp over 78 trades.  These
  are small, selected samples; the validate window was used only to pick
  finalists, and finalist 1's p-value sits on the bootstrap floor
  (0.0002 = 1/(4999+1)).
- No claim of edge is made now.  The freeze commits three specific
  configurations to one untouched holdout test with the fixed rule below; the
  outcome (including a null) will be reported exactly as computed.

### Exact frozen specification

- Common: per-asset models (four per configuration, one per symbol); direction
  target `sign(close[t+h-1]/close[t-1]-1)` with exact-zero rows excluded;
  prediction timestamp `t` = candle open; features built by
  `features.build_feature_frame` on the database being evaluated (GCP
  excluded) through the same `shift(1)`; `StandardScaler` fit on training rows
  only; trade rule: long if `p_up >= 0.5 + θ`, short if `p_up <= 0.5 - θ`,
  hold `h` minutes, non-overlapping entries; fees 0.10% (primary), 0.075%,
  0.02% per side.
- Finalist 1 `d1_logistic_physics_h15_per`: features = `MARKET_FEATURES +
  PHYSICS_FEATURES` (20 columns, exact order from `features.py`); model =
  `LogisticRegression(C=1.0, solver="lbfgs", max_iter=300, random_state=0)`;
  h = 15; θ = 0.10.
- Finalist 2 `d1_logistic_market_h15_per`: features = `MARKET_FEATURES`
  (14 columns); same model, h = 15; θ = 0.10.
- Finalist 3 `d2_hgb_physics_h240_per`: features = `MARKET_FEATURES +
  PHYSICS_FEATURES`; model = `HistGradientBoostingClassifier` with
  `learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=100,
  l2_regularization=1.0, max_bins=255, random_state=0`, trained with warm-start
  chunks of 25 boosting iterations up to 200 and early stopping on the last 15%
  of training rows (time-ordered, h-minute gap), best checkpoint; h = 240;
  θ = 0.15.
- Holdout execution: fit each finalist on all dev rows whose target completes
  before the holdout begins; evaluate from
  `max(first holdout timestamp, dev end) + 240 min`; command
  `.venv/bin/python edge_lab.py --stage holdout --holdout-db <path>`
  (writes `results_edge_lab_holdout.json`).  Same bootstrap machinery
  (`n_boot = 4999`; block = 10 trades for net means, 60-minute blocks for
  hit-rate deltas) and the same Bonferroni α = 0.016667 apply.
- Decision rule (restated): a finalist passes iff its net-of-fee (10 bp/side)
  mean per trade > 0 with one-sided bootstrap p < 0.016667 **and** its hit
  rate exceeds both persistence and momentum with paired two-sided
  p < 0.016667.  The lab claims a usable edge iff at least one finalist passes
  on the untouched holdout; otherwise the result is a null, including a
  gross-positive/net-negative outcome.

## Holdout results (2026-10-02, single frozen run)

Executed exactly once after this document was frozen:
`.venv/bin/python edge_lab.py --stage holdout --holdout-db ../crypto-data/holdout_klines_1m.db`
→ `results_edge_lab_holdout.json`.  θ, features, models, and fees were not
changed after the freeze.  Fit on 5,303,313–5,337,882 dev rows per finalist
(all four assets, all dev rows with targets complete before the holdout);
holdout test from 2026-09-14T09:09Z (=`max(first holdout, dev end) + 240 min`);
104,420–104,833 evaluated holdout rows per finalist with warm features built on
the combined 600-minute-tail scratch database.

**Hit-rate criterion (full coverage; Bonferroni α = 0.016667).**

| finalist | eval n | hit% | persistence% | momentum% | Δhit persist (pp) | p | Δhit momentum (pp) | p | criterion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| d1_logistic_physics_h15_per | 104,833 | 52.535 | 48.082 | 47.625 | +4.453 | 0.0004 | +4.910 | 0.0004 | PASS |
| d1_logistic_market_h15_per | 104,833 | 52.370 | 48.082 | 47.625 | +4.288 | 0.0004 | +4.745 | 0.0004 | PASS |
| d2_hgb_physics_h240_per | 104,420 | 51.756 | 48.196 | 48.497 | +3.561 | 0.0004 | +3.259 | 0.0016 | PASS |

**Net-return criterion (non-overlapping trades at the frozen θ; power rule < 30 trades = underpowered).**

| finalist | θ | expected trades | actual trades | hit | gross bp | net 10 bp | net 7.5 bp | net 2 bp | p(net>0) 10 bp | net CI95 (10 bp) | criterion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| d1_logistic_physics_h15_per | 0.10 | 2.6 | 0 | — | — | — | — | — | — | — | INCONCLUSIVE (0 trades) |
| d1_logistic_market_h15_per | 0.10 | 8.6 | 3 | 1.000 | +117.46 | +97.46 | +102.46 | +113.46 | null | null | INCONCLUSIVE (<30) |
| d2_hgb_physics_h240_per | 0.15 | 5.1 | 1 | 1.000 | +29.51 | +9.51 | +14.51 | +25.51 | — | — | INCONCLUSIVE (<30) |

*Post-holdout reporting fix (2026-10-02): F2 produced 3 holdout trades, fewer than the 10-trade bootstrap block, so every moving-block resample is identical; its net-return p-value and CI are degenerate and are reported as null (the one-shot run's raw values were p = 0.0002 and CI [+97.46, +97.46] bp). `edge_lab.py` now emits null for these two statistics whenever `n_trades <= 10`, and only these two fields were nulled in `results_edge_lab_holdout.json`; nothing else in the one-shot result is changed.*

**Verdict.**  Per the frozen rule, **no finalist passes and no edge is
claimed**; the net-return criterion is underpowered for all three finalists
(0/3/1 trades versus expected 2.6/8.6/5.1), so the overall result is
**inconclusive**, not a positive claim and not a refutation.  Two observations
must be stated plainly: (i) the direction hit-rate advantage persists
out-of-sample over the 18.3-day holdout at full coverage (52% against ~48%
baselines, p at the bootstrap floor), but the gross edge at full coverage is a
few tenths of a basis point per trade, i.e. two orders of magnitude below the
10 bp/side fee, and the hit-rate baselines are themselves weak in this window;
(ii) finalist 1 — the only validate-window "pass" — produced **zero** trade
signals above θ=0.10 in the holdout, so its validate result did not recur as a
tradable opportunity.  Finalists 2 and 3 produced 3 and 1 trades respectively;
their positive gross/net means have no statistical content at these counts
(p-values sit on the bootstrap floor and the F2 CI is degenerate).  The
holdout can be extended later with `../crypto-data/fetch_holdout.py` and the
same frozen test re-run on the extended window; a decisive cost-aware verdict
would need a pre-registered lower-θ design with far more trades, which this
freeze deliberately does not authorize.

## Post-holdout descriptive check: F1 validate trade concentration (2026-10-02)

How concentrated is finalist F1's validate-window trade mean?  Recomputed from
the cached validate predictions
(`~/.cache/edge_lab/preds/d1_logistic_physics_h15_per_validate_d284a55614cfb68b.pkl`,
dev cache only -- no holdout data, no refit) with `edge_lab.simulate_trades`
at the frozen θ = 0.10, gross of fees.

* 39 trades, mean gross **+73.10 bp/trade** (matches the freeze table).
* Median trade: **+84.80 bp**.
* Mean excluding the single best trade: **+63.00 bp** (removed +456.79 bp).
* Mean excluding the top 3: **+43.49 bp** (removed +456.79, +441.73,
  +386.62 bp).
* Mean excluding the top 5: **+29.37 bp**.

Largest 3 trades (gross):

| # | timestamp (UTC) | symbol | side | return (bp) |
|---|---|---|---:|---:|
| 1 | 2026-01-31 18:44 | SOLUSDT | long | +456.79 |
| 2 | 2026-02-06 00:27 | SOLUSDT | long | +441.73 |
| 3 | 2026-08-22 05:11 | SOLUSDT | long | +386.62 |

Clustering: the three largest trades are all SOLUSDT longs but fall on three
different dates, so they are not one event; the sample does cluster in short
cross-asset bursts -- 6 of the 39 trades (+1522.82 bp of the +2850.78 bp total
gross PnL) sit on 2026-01-31 alone, 26 of 39 trades share their minute with at
least one other asset's trade, and 13 of 39 fall in the 2026-01-30 -> 02-06
window (two market-wide moves).  Most of the mean comes from a few
outlier bars: the top 3 trades are 45% of the total gross PnL, the top 5 are
65%, and removing the top 5 cuts the mean from +73.10 to +29.37 bp/trade,
while a single −571.20 bp SOLUSDT long on 2026-02-06 offsets much of the rest.
The +73.1 bp validate mean is therefore **not a broad-based edge**: it is a few
outlier minutes, mostly SOLUSDT longs inside two bursts, over a 39-trade
sample.
