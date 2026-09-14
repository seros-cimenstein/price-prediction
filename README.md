# Crypto short-horizon prediction: historical evaluation

## Verdict (read this first)

**The market-data models show a small numerical out-of-sample edge, but it is weak and not a practical trading claim. GCP2 does not add convincing incremental information.**

- The pre-registered regularized logistic model beats the persistence and 15-minute momentum baselines in pooled walk-forward hit rate for all 16 asset/horizon rows. Hit rates are only **50.425%--52.462%** and Spearman IC is **0.00435--0.04768**. The Wilson intervals below are narrow because there are hundreds of thousands of minute observations; minute-level serial dependence makes those intervals optimistic as uncertainty intervals, so they are reported as requested but should not be read as independent-trial evidence.
- The pre-registered ridge volatility model has lower MSE than the EWMA baseline in all 16 rows. The absolute errors are small and this does not establish profitable volatility timing; it establishes only this OOS error comparison for the specified target.
- In the paired GCP ablation, hit-rate changes range from **-0.181 to +0.033 percentage points**, IC changes are essentially zero (about -0.00329 to +0.00008), and there is no consistent positive result. The paired 60-minute moving-block bootstrap p-values do not support a GCP claim after considering the 16 pre-registered comparisons. Volatility MSE changes are also tiny and inconsistent. **Do not promote GCP as a predictive signal on this evidence.**
- The shuffled-target controls are at chance (49.939%--50.411% hit rate, IC approximately -0.00168 to -0.00042), which is the required leakage sanity check.

The honest owner-facing summary is therefore: **there is a weak market-data association that clears the named numerical baselines in this historical sample, but no evidence that GCP2 adds skill and no evidence here of a robust, cost-aware trading edge.** Future live use should monitor drift and must not treat the artifacts as a guarantee.

## Reproduce

From the repository root:

```bash
# The environment is already in .venv for this checkout. If recreating it:
python3 -m venv price-prediction/.venv
price-prediction/.venv/bin/pip install numpy pandas scipy scikit-learn joblib

# Default uses crypto-data/gcp2_1m.db when present (the verified minute-mean
# cache); otherwise it falls back to the raw one-second GCP zip directory.
price-prediction/.venv/bin/python price-prediction/train.py
```

The run reads `crypto-data/crypto_market.db`, the GCP minute cache if present, and `gcp2-data/event_hypothesis_archives/events.json`. To force streaming aggregation directly from the raw one-second archives:

```bash
price-prediction/.venv/bin/python price-prediction/train.py \
  --gcp gcp2-data/network_coherence/global_network
```

Useful options are `--n-folds`, `--initial-train-fraction`, `--embargo`, `--bootstrap`, `--start-time-s`, and `--end-time-s`. A smaller date range is useful for a smoke run; do not interpret a smoke run as the reported evaluation. The command prints the table, writes `results.json`, and writes 64 final-history `model_*.joblib` artifacts in this directory.

A live consumer can load an artifact as follows, then construct the exact named feature vector from `features.py` in the same order:

```python
import joblib
artifact = joblib.load("price-prediction/model_BTCUSDT_5m_direction_with_gcp.joblib")
x_scaled = artifact["scaler"].transform(x[artifact["feature_names"]])
score = artifact["model"].decision_function(x_scaled)
```

The artifact contains `feature_names`, the fitted `StandardScaler`, fitted model, fit end timestamp, target definition, and the complete pre-registration manifest. A live process must only use closed candles and must reproduce the trailing GCP minute alignment; never fit a new scaler on live/test data.

## OOS result table

The table is from `results.json` generated on the full source ranges with 3 expanding folds, 50% initial training, and embargo `max(horizon,  horizon option)` minutes. `N dir` excludes exact-zero forward returns. `N vol` is the volatility-target count. GCP deltas are **with-GCP minus market-only** on exactly paired OOS rows. `pp` means percentage points. MSE is in squared return units.

| Asset | h | N dir | hit % [95% Wilson] | IC | Δhit persist (pp) | Δhit momentum (pp) | N vol | Ridge MSE | EWMA MSE | ΔMSE vs EWMA | GCP Δhit (pp; p) | GCP ΔIC (p) | GCP ΔMSE (p) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BTCUSDT | 1 | 616,134 | 50.425 [50.300,50.550] | 0.01496 | 1.744 | 0.798 | 636,657 | 1.68648e-07 | 1.99529e-07 | -3.08809e-08 | 0.006; 0.8704 | -0.00021; 0.5183 | 1.95559e-12; 0.9635 |
| BTCUSDT | 5 | 634,743 | 51.376 [51.253,51.499] | 0.03215 | 3.627 | 2.617 | 636,657 | 4.20692e-07 | 4.63310e-07 | -4.26178e-08 | -0.016; 0.8173 | -0.00022; 0.5316 | 8.44406e-12; 0.6844 |
| BTCUSDT | 15 | 636,252 | 51.807 [51.684,51.930] | 0.03846 | 4.011 | 3.504 | 636,657 | 9.17522e-07 | 9.99206e-07 | -8.16842e-08 | -0.047; 0.2458 | -0.00119; 0.1927 | 6.38160e-12; 0.7575 |
| BTCUSDT | 60 | 636,519 | 51.221 [51.098,51.344] | 0.02839 | 3.455 | 2.703 | 636,657 | 3.16697e-06 | 3.60059e-06 | -4.33617e-07 | -0.181; 0.4983 | -0.00278; 0.6645 | 1.32528e-10; 0.7508 |
| ETHUSDT | 1 | 628,998 | 51.489 [51.365,51.612] | 0.03458 | 2.787 | 2.693 | 636,657 | 3.85714e-07 | 4.63216e-07 | -7.75026e-08 | -0.010; 0.0797 | -0.00013; 0.0465 | 3.08263e-12; 0.0199 |
| ETHUSDT | 5 | 635,310 | 52.130 [52.007,52.253] | 0.04598 | 3.683 | 4.104 | 636,657 | 1.01877e-06 | 1.15130e-06 | -1.32528e-07 | -0.004; 0.1262 | -0.00025; 0.0997 | 2.63726e-11; 0.3522 |
| ETHUSDT | 15 | 635,964 | 52.462 [52.339,52.585] | 0.04768 | 3.752 | 4.860 | 636,657 | 2.27916e-06 | 2.55591e-06 | -2.76752e-07 | -0.031; 0.1794 | -0.00040; 0.0864 | 8.63017e-11; 0.8505 |
| ETHUSDT | 60 | 636,351 | 51.690 [51.567,51.812] | 0.03074 | 2.768 | 3.331 | 636,657 | 7.62770e-06 | 9.08767e-06 | -1.45997e-06 | -0.036; 0.5781 | -0.00122; 0.1993 | 9.19779e-10; 0.4319 |
| SOLUSDT | 1 | 603,840 | 51.083 [50.957,51.209] | 0.02285 | 5.012 | 2.713 | 636,657 | 4.69573e-07 | 5.57257e-07 | -8.76848e-08 | 0.026; 0.1860 | 0.00008; 0.0997 | 5.23850e-12; 0.0199 |
| SOLUSDT | 5 | 623,028 | 51.707 [51.583,51.831] | 0.03519 | 5.963 | 4.106 | 636,657 | 1.17447e-06 | 1.30895e-06 | -1.34474e-07 | -0.013; 0.6445 | -0.00015; 0.2924 | 2.34591e-12; 0.9435 |
| SOLUSDT | 15 | 628,881 | 51.743 [51.620,51.867] | 0.03322 | 5.765 | 4.402 | 636,657 | 2.55847e-06 | 2.82924e-06 | -2.70770e-07 | -0.035; 0.0199 | -0.00049; 0.0266 | 7.18094e-11; 0.9369 |
| SOLUSDT | 60 | 632,865 | 50.658 [50.535,50.781] | 0.01621 | 4.483 | 2.697 | 636,657 | 8.79038e-06 | 1.00866e-05 | -1.29623e-06 | -0.138; 0.1661 | -0.00329; 0.1661 | 1.18386e-09; 0.9967 |
| BNBUSDT | 1 | 597,441 | 50.563 [50.436,50.690] | 0.00435 | 1.868 | 0.876 | 636,657 | 2.48281e-07 | 3.01342e-07 | -5.30611e-08 | -0.023; 0.9701 | -0.00045; 0.7309 | 1.71441e-12; 0.1661 |
| BNBUSDT | 5 | 622,539 | 50.730 [50.606,50.855] | 0.01810 | 2.630 | 1.682 | 636,657 | 6.66057e-07 | 7.78063e-07 | -1.12005e-07 | 0.033; 0.0133 | -0.00023; 0.1262 | 7.90322e-12; 0.1860 |
| BNBUSDT | 15 | 628,737 | 51.145 [51.022,51.269] | 0.02579 | 2.740 | 2.491 | 636,657 | 1.67638e-06 | 1.94863e-06 | -2.72247e-07 | -0.012; 0.7309 | -0.00025; 0.8106 | 7.07246e-12; 0.5449 |
| BNBUSDT | 60 | 632,700 | 51.180 [51.057,51.303] | 0.02368 | 2.927 | 2.412 | 636,657 | 6.77908e-06 | 8.08458e-06 | -1.30550e-06 | -0.065; 0.5847 | -0.00146; 0.2326 | 9.45732e-11; 0.8904 |
The zero-change baseline always scores zero because exact-zero forward returns are removed from the directional denominator; its corresponding hit-rate and IC deltas are recorded for every row in `results.json` (the IC is undefined for a constant score).

## Data and alignment

- `features.py` verifies `PRAGMA table_info(klines_1m)` before reading. The verified columns are `symbol`, `open_time_s`, `open`, `high`, `low`, `close`, `volume`, `quote_volume`, and `trades`. The four assets each have about 1.32 million clean rows beginning at epoch `1709251200` (2024-03-01 00:00 UTC); the exact audit is in `results.json`.
- GCP minute data is the mean of 60 one-second netvar values. The verified cache has 1,273,377 rows from 2024-03-01 through 2026-09-02 23:59 UTC; `features.py` can reproduce it by streaming the raw zips when the cache is not supplied.
- A row's prediction timestamp is candle open `t`. Features are shifted one minute and use only candles with `open_time_s < t`; GCP minute `t-1` is the newest GCP observation used. Calendar features are deterministic and are shifted with the feature matrix (they use no market data).
- Direction target: `close[t+h-1] / close[t-1] - 1`; direction is its sign. Volatility target: `sqrt(sum(return[t:t+h-1]^2))`, where `return[t] = close[t]/close[t-1]-1`. Thus all target inputs are future relative to the feature row.

## Pre-registration and model specification

The feature set was fixed in `features.py` before inspecting results. There was no scan over alternative windows/models. Market features are:

- trailing own-asset returns over 1, 5, 15, and 60 minutes;
- realized volatility over 15 and 60 minutes;
- a 14-minute RSI-like momentum;
- trailing 60-minute volume and trade-count z-scores;
- trailing cross-asset dispersion: cross-sectional standard deviation of the four assets' one-minute returns;
- hour-of-day sine/cosine and day-of-week sine/cosine.

GCP features are netvar level, trailing 60-minute netvar z-score, one-minute velocity, one-minute acceleration, and trailing 60-minute netvar standard deviation. The direction model is L2-regularized logistic regression (`C=1`); the volatility model is ridge regression (`alpha=1`). Every fold fits a new `StandardScaler` on its training rows only, then freezes it for that fold's test rows.

Baselines are persistence (`sign(ret_1m)`), zero-change, and momentum (`sign(ret_15m)`) for direction; volatility uses a 60-minute trailing EWMA of prior squared one-minute returns, scaled by `sqrt(h)`. The ablation uses the same rows, folds, model, scaler policy, and hyperparameters with GCP columns removed.

## Validation and leakage controls

The full run uses expanding walk-forward CV: 50% of unique timestamps initially train, then three chronological test blocks. Each fold's training end is separated from test start by at least the prediction horizon. Fold train/test counts and timestamps are recorded under every result arm in `results.json`.

`assert_causal_features` deterministically spot-checks the recorded latest candle source against every sampled target timestamp and checks that row timestamps agree with `target_time_s`. `shuffled_target_control` permutes targets before fold construction; its BTC controls are recorded for all four horizons under `results.json.leakage_checks`. The controls are not used for feature/model selection.

GCP p-values and confidence intervals use a paired moving-block bootstrap (60-minute blocks), preserving the same OOS rows for both ablation arms. The displayed Wilson intervals are the requested binomial intervals; they do not model long-memory dependence. All horizons/assets and both ablation arms were fixed in advance, and no exploratory result is reported as confirmatory.

## Event study (secondary and descriptive only)
Nineteen of 23 registered events had complete 24-hour pre/post windows in the market range. We report medians of the BTC forward 60-minute realized-volatility target; there is no event-selection test, causal interpretation, or multiple-testing claim. The compact table below shows the five largest observed post/pre ratios; all 19 event rows are retained in `results.json`.

| Event date | Registered event (truncated) | Pre 24h median | Post 24h median | Post/pre |
|---|---|---:|---:|---:|
| 2025-02-03 | Kumbh Mela 2025 | 0.0054525 | 0.0083775 | 1.536 |
| 2024-07-15 | Assassination Attempt on Former President Donald Trump | 0.0038171 | 0.0050717 | 1.329 |
| 2025-05-08 | Pope Leo XIV Elected | 0.0030514 | 0.0039860 | 1.306 |
| 2025-04-09 | World Navkar Mantra Day | 0.0065470 | 0.0085126 | 1.300 |
| 2025-04-02 | 'Liberation Day' Reciprocal Tariffs Announced | 0.0037773 | 0.0047676 | 1.262 |

These are descriptive alignments only; the event archive is not a validated causal news feed.

## Existing legacy model

`crypto-data/model_features_v2.json` and the corresponding `ml_early_lead_model_v2.joblib` / `ml_imminent_model_v2.joblib` are an older 22-feature path. They were not merged into this pre-registered feature set, not used to select configurations, and not included in these metrics.

## Files

- `features.py` -- schema/range audit, raw/cache GCP loading, causal feature/target construction, and leakage self-check.
- `backtest.py` -- expanding folds, frozen scaling, baselines, metrics, block bootstrap, and shuffled-target control.
- `train.py` -- end-to-end build/evaluate/ablate/artifact pipeline.
- `results.json` -- machine-readable full-run numbers, fold counts, source audit, controls, and event study.
- `model_*.joblib` -- final-history artifacts for each asset, horizon, task, and feature arm.
- `research.md` -- time-boxed web research and source links.

## Live inference

`live.py` runs the persisted models on the newest **closed** Binance 1-minute
candles and predicts the next minute open:

```bash
price-prediction/.venv/bin/python price-prediction/live.py --once
price-prediction/.venv/bin/python price-prediction/live.py
```

Each cycle requests `limit=130` candles for BTCUSDT, ETHUSDT, SOLUSDT, and
BNBUSDT, filters by `close_time < now`, and keeps the inner intersection of
common minute opens (at least 61 contiguous minutes are required).  The
`--symbol` option limits emitted records but still fetches all four assets
because cross-asset dispersion is a registered feature.  Feature construction
is imported from `features.py`; no live-only feature formulas are duplicated.

The optional GCP arm calls `https://gcp2.net/api/getcurrentnetvar`.  The bearer
token is read at runtime from `GCP2_TOKEN`, `GCP2_TOKEN_FILE`, `--token-file`,
or the collector/config paths under `gcp_2.0/`, `gcp2-collector/`, and
`crypto-data/`; the token is never hardcoded, logged, or written to JSONL.
Recent trailing GCP history is read read-only from
`crypto-data/gcp2_1m.db`.  Missing credentials, network failures, malformed
responses, or insufficient history produce a warning and market-only output
instead of stopping the cycle.  When the GCP arm is available and passes the
guard below, each record contains both market-only and with-GCP scores.

Before emitting with-GCP scores, `live.py` applies a pre-score distribution
guard to the five registered GCP features.  It uses the frozen scaler in the
with-GCP artifacts and suppresses the with-GCP arms when any feature has
absolute z-score above `--guard-z-threshold` (default `3.0`) or when the
newest cached GCP minute is more than `--max-gcp-staleness-minutes` (default
`120`) behind the fetched NetVar timestamp.  Suppressed records retain
`gcp_available: true`, the fetched `gcp_netvar`, and a `gcp_guard` block with
`in_distribution`, `cache_stale`, `max_abs_z`, `stale_minutes`, and
`suppressed`, while emitting market-only arms only.  This fail-safe is
motivated by the severe live/training shift recorded
in `live_gcp_drift_report.json`; the with-GCP probabilities must not be
treated as trustworthy while that shift persists.

One JSON object per emitted symbol is appended to
`price-prediction/live_predictions.jsonl` (override with `--log-file`).  Each
object contains the UTC `timestamp`, `target_time_s`, `symbol`,
`current_close`, per-horizon direction score/probability and volatility
prediction for each available arm, plus persistence, momentum, and EWMA
baselines.  These are probabilistic estimates, not guarantees.

### Edge significance
`significance.py` applies block-bootstrap tests with Bonferroni correction to accumulated live scores.
Run it with `price-prediction/.venv/bin/python price-prediction/significance.py`.
It writes `live_edge_significance.json` with 16 cells, surviving cells, and per-cell bootstrap CIs.
For `SOLUSDT_15m`, the current p-values are 0.0018 vs chance and 0.0186 vs persistence; it survives chance only (threshold 0.003125); thirds are 62.7%, 67.2%, and 67.2%.
The sample is small and cell membership has flipped between runs (`SOLUSDT_15m` in/out), consistent with a marginal/unstable sample;
the cumulative verdict remains not a confirmed trading edge: 3/16 survive vs chance, 2/16 survive vs persistence and both (`BTCUSDT_60m`, `SOLUSDT_60m`).
`--windowed` adds a `windowed` output key with consecutive UTC six-hour windows, per-window tests, and per-cell survival fractions; the current run is 2/16 consistent, 1/16 marginal (`SOLUSDT_15m`), and 13/16 noise (descriptive overall, with only one n>=50 window per 60m cell).
If a stability contingency table has any expected cell count below 5, its chi-square statistic and p-value are left null with reason `insufficient data for chi-square` rather than forcing a test.

#### Autonomous verification

`orchestrator.py` runs monitor → live outcome scoring → windowed significance in order and records each cycle's parsed verdict.
For continuous use: `nohup price-prediction/.venv/bin/python price-prediction/orchestrator.py >>price-prediction/orchestrator.log 2>&1 &` (Append the log; truncation loses per-tool event lines.)
Each cycle keeps `track_record.jsonl`, `live_eval.json`, and `live_edge_significance.json` fresh, then appends `verdict_history.jsonl`.
The `.orchestrator.lock` file prevents concurrent orchestrators from interleaving tool runs or history writes.

## Live outcome evaluation

`live_eval.py` closes the loop from `live.py`'s append-only prediction log to
realized Binance outcomes.  It reads
`price-prediction/live_predictions.jsonl`, keeps the latest line for each
`(symbol, target_time_s, horizon)` key, and fetches the required 1-minute
klines from Binance's public REST API.  A target is scored only after every
required kline has `close_time < now`; predictions whose final candle is
still open are reported as pending and retained for a later continuous-loop
poll.  HTTP failures are retried once and then reported as skipped rather
than stopping the evaluator.

The semantic mapping is intentionally the same as `features.py`: `t` is the
logged `target_time_s` (the next minute open), `current_close` is
`close[t-1]`, and horizon `h` uses the anchor close at `t-1` plus closes at
`t` through `t+h-1`.  Thus realized direction is
`close[t+h-1] / close[t-1] - 1`, and realized volatility is the square root
of the sum of squared one-minute returns in those `h` future candles.  Exact
zero returns are excluded from directional hit-rate denominators.  The
scorecard reports model and baseline hit rates, Spearman direction IC (only
when at least ten rows are available), and model/EWMA absolute volatility
errors for each available arm.

Run a single evaluation, or keep polling for newly matured targets:

```bash
price-prediction/.venv/bin/python price-prediction/live_eval.py --once
price-prediction/.venv/bin/python price-prediction/live_eval.py
price-prediction/.venv/bin/python price-prediction/live_eval.py \
  --once --predictions /path/to/live_predictions.jsonl \
  --out /path/to/live_eval.json
price-prediction/.venv/bin/python price-prediction/live_eval.py --self-test
```

The JSON output is written to `price-prediction/live_eval.json` by default.
Its top-level `records_read` counts nonblank JSONL lines; top-level
`evaluated`, `pending`, and `skipped` count deduplicated prediction records by
status (with row-level counts under `prediction_rows`);
`scorecard[symbol][horizon].arms[arm]` contains the per-arm metrics.  For
example, a printed row such as
`BTCUSDT 1m market_only 1 100.00% ... - 0.0001 0.0002` means one evaluated
prediction row for that symbol/horizon/arm, a 100% directional hit rate on
nonzero outcomes, an unavailable IC because fewer than ten rows were
available, and the two final columns are model and EWMA mean absolute
volatility errors.  With only the four sample records, these numbers prove
that log parsing, maturity gating, Binance retrieval, target alignment, and
aggregation are wired correctly; **four scored records do not establish
predictive skill**.
