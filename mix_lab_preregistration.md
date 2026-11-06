# Pre-registration: mixed Schrödinger + GCP2 + market arm (`mix_lab`)

Status: **written before any evaluation result for this family was computed.**
The model definitions, the coupled-candidate grid, the selection rule, the dev
evaluation protocol, and the frozen holdout test with its decision rule are
fixed below. The exploration log, the frozen spec, and the dev tables are
appended after the run (sections 9-11); nothing above them is changed after
results are seen. The only pre-result computations were: (a) the solver
derivation and numerical design probe of the new coupled propagator on
synthetic packets (no market data, used only to fix the coupling constants in
section 4 and to verify the analytic scalings used by `test_mix_lab.py`), and
(b) the deterministic data audit of section 2 (ranges, coverage, and the
marginal distribution of the registered `netvar_z_60m` column), which is a
property of the data, not a model choice.

## 1. Honesty statement and scope

The owner request was "find a way to mix all: Schrödinger, GCP2 and crypto".
This document pre-registers two honest ways to do that and one control that
would falsify the GCP entry point:

* **Additive arms** (A0-A3): market features, plus the registered GCP columns,
  plus the frozen Schrödinger wave-packet columns, and both together, on
  identical rows.
* **Coupled arm** (C): the GCP trailing z-score drives the wave equation
  itself (effective `hbar`, potential stiffness, initial packet width, or an
  imaginary potential) instead of sitting beside it as a feature column.
* **Shifted-GCP control** (C_shift): the same coupled arm with the GCP series
  replaced by a copy shifted exactly 30 days in time. The shifted copy has the
  same marginal distribution but no alignment with the market state at the
  anchor. If a coupled candidate does not beat its own shifted control, GCP is
  not doing anything causal through that channel.

There is **no claimed physical mechanism** linking the Schrödinger equation to
crypto prices; the equation is used as a deterministic, non-linear state
machine over a standardized return coordinate, as in
`schrodinger_preregistration.md`. A null result is a fully acceptable outcome.
No window, constant, coupling map, arm, fold, or threshold is changed after
seeing any evaluation output other than through the pre-declared selection
rule in section 7. Every configuration that is run is logged in section 9.

## 2. Data (verified before registration)

* Market: `../crypto-data/crypto_market.db`, verified `klines_1m` schema,
  four registered assets. Range **1709251200..1789362480**
  (2024-03-01 00:00 UTC -> 2026-09-14 05:08 UTC), **1,335,189 rows per
  asset**.
* GCP minute cache: `../crypto-data/gcp2_1m.db` (table `netvar_1m`, minute
  means of 60 one-second netvar values). Range **1709251200..1788393540**
  (2024-03-01 00:00 UTC -> 2026-09-02 23:59 UTC), **1,273,377 rows**. Against
  the complete minute grid of that span (1,319,040 minutes) **45,663 minutes
  are missing** (~3.5%); missing minutes become NaN features and are dropped by
  the complete-case policy, never filled.
* **Common range used by this arm: [1709251200, 1788393540]**. The frame is
  built with `features._assemble_feature_frame` over the market rows of that
  range and the GCP series above; market rows after the GCP cache end are
  outside the common range and every complete-case row set is inside it.
  Verified anchor counts in the common range: **21,984 possible hourly
  anchors** per asset, of which **21,217 have a GCP minute present** (the
  effective count after all feature warm-ups is reported in section 9). No
  holdout database and no new data are touched by the dev stages.
* Alignment: the registered convention of `features.py`. At a prediction row
  with `target_time_s = t`, every market/physics/Schrödinger feature uses
  candles with `open_time_s < t`, and GCP minute `t-1` is the newest GCP value
  used (features are built from the one-minute-shifted GCP column). The
  coupled arm consumes exactly that registered causal column `netvar_z_60m`
  (trailing 60-minute z-score of `netvar_level`).
* Marginal of the coupling input (audit): registered `netvar_z_60m` percentiles
  (1/5/25/50/75/95/99) are -2.04/-1.53/-0.70/-0.06/+0.64/+1.73/+2.52; 0.32%
  of minutes have `|z| > 3`. The bounded map `u = tanh(z/2)` therefore spans
  most of `(-1, 1)` in the bulk and saturates only in the tails.

## 3. The wave equation with GCP coupling

The registered dimensionless solver is kept exactly as in `schrodinger.py`:
`i hbar dpsi/dt = [-(hbar^2/2m) d^2/dx^2 + V(x)] psi`, `hbar = m = 1`,
`x in [-6, 6]` with 193 grid points, Strang-split Fourier propagation with
`dt <= DT_MAX = 0.02`, initial packet
`psi(x,0) = N exp(-(x-x0)^2/(4 sigma_packet^2) + i k0 x)`, `k0 = x0`, and the
frozen configuration **`quartic_kde|linear_60|momentum`** selected in
`schrodinger_preregistration.md` (potential `V` from the weighted quartic fit
of `-log p_hat` over the 240-candle KDE, `tau = h/60`). The four Born-rule
readouts `p_up`, `mean`, `std`, `kurt` are the arm's outputs.

GCP enters the equation through a single scalar per anchor and per candidate,
computed from the registered causal column:

    u(t) = tanh( netvar_z_60m(t) / 2 )        (bounded, strictly monotone)

and one of the following four fixed modifications (one per candidate, never
more than one element at a time):

* **(a) effective hbar** - the full equation is solved with `hbar_eff`:
  kinetic phase `exp(-i hbar_eff k^2 dt / 2)`, potential phase
  `exp(-i V dt / hbar_eff)`. `hbar_eff = exp(+/- beta_h u)`, `beta_h = 0.75`.
* **(b) potential stiffness** - `V_eff = exp(+ beta_v u) * V`,
  `beta_v = 1.0`; propagation stays on the registered `hbar = 1`.
* **(c) initial packet width** -
  `sigma_eff = clip(sigma_packet * exp(+/- beta_s u), 0.35, 3.0)`,
  `beta_s = 0.5`; the initial packet is rebuilt with `sigma_eff` and the
  registered `x0`, `k0`; propagation stays on the registered `V`, `hbar = 1`.
* **(d) imaginary potential** - `V_eff = V - i eta_eff W(x)` with
  `eta_eff = eta_0 * exp(+ beta_a u)`, `eta_0 = 1.0`, `beta_a = 1.0`, and the
  fixed shape `W(x) = (x/X_MAX)^2` (0 at the centre, 1 at the box edge). This
  is deliberately non-unitary; the un-renormalized wave-function norm
  `int |psi|^2 dx` after the evolution is recorded as an additional arm
  feature `cpl_norm_{h}m`, and the Born observables are read from the
  renormalized density as the registered `born_observables` already does.

The modification constants were fixed from the synthetic design probe
(constant choices give observable changes of a few percent to a few tens of
percent across `u in [-1, 1]`; the probe is not a market evaluation). The
coupled propagator is a new solver path (`split_step_propagate_coupled`) that
reduces **exactly** to `schrodinger.split_step_propagate` when
`hbar_eff = 1` and `V` is real; this identity is asserted at feature-build
time for every (asset, horizon) against the registered feature builder and is
covered by `test_mix_lab.py`. Unitary variants must conserve the norm to
`1e-8`; the imaginary-potential variant is tested against its analytic
per-grid-point decay `exp(-2 eta W(x) tau)` (exact for a spatially constant
`W`) and is required to decay monotonically with `eta` (a unitary norm test
would be wrong for it, by design).

## 4. Coupled candidate grid (fixed, <= 6 candidates)

| label | element | map | constants |
|---|---|---|---|
| `hbar_inc` | (a) effective hbar | `hbar_eff = exp(+beta_h u)` | `beta_h = 0.75` |
| `hbar_dec` | (a) effective hbar | `hbar_eff = exp(-beta_h u)` | `beta_h = 0.75` |
| `stiff_inc` | (b) stiffness | `V_eff = exp(+beta_v u) V` | `beta_v = 1.0` |
| `packet_widen` | (c) packet width | `sigma_eff = clip(sigma_packet exp(+beta_s u), 0.35, 3.0)` | `beta_s = 0.5` |
| `packet_narrow` | (c) packet width | `sigma_eff = clip(sigma_packet exp(-beta_s u), 0.35, 3.0)` | `beta_s = 0.5` |
| `absorb_inc` | (d) imaginary potential | `eta_eff = eta_0 exp(+beta_a u)` | `eta_0 = 1.0`, `beta_a = 1.0` |

All six are run in the exploration stage; nothing else is run. The identity
coupling (no GCP influence) is not a candidate; it is computed once per
(asset, horizon) as the consistency check against the registered Schrödinger
features.

**Shifted-GCP control.** For every candidate the same coupled features are
also computed with `u_shift(t) = tanh(netvar_z_60m(t - 30 days)/2)`. The
shifted series is produced by adding exactly 43,200 minutes to the index of
the registered GCP minute series and then applying the registered assembly
path, so the shifted column at row `t` equals the registered causal column at
`t - 30 days` (or NaN when that minute is missing). No fill, no re-fit, no
re-normalization: identical arm, identical rows, only the input series moved.

## 5. Arms

All arms are evaluated per (asset, horizon) cell; the 16 cells are the four
registered assets x the four registered horizons (1, 5, 15, 60 minutes).

* **A0** = `MARKET_FEATURES` (14 columns).
* **A1** = `MARKET_FEATURES + GCP_FEATURES` (19).
* **A2** = `MARKET_FEATURES + SCHR_h` (18), the frozen registered Schrödinger
  columns of `schrodinger_preregistration.md`.
* **A3** = `MARKET_FEATURES + GCP_FEATURES + SCHR_h` (23).
* **C0** = `MARKET_FEATURES` on the coupled comparison rows.
* **C_real** = `MARKET_FEATURES + CPL_h`, where `CPL_h` is the four coupled
  Born observables of the candidate (`cpl_p_up_{h}m`, `cpl_mean_{h}m`,
  `cpl_std_{h}m`, `cpl_kurt_{h}m`), plus `cpl_norm_{h}m` for `absorb_inc`.
* **C_shift** = the same columns from the shifted-GCP copy of the candidate.
* **Direct readout (S0-style)** = the coupled Born-rule outputs with no
  fitted model: direction `sign(p_up - 0.5)` scored against the forward
  return, and volatility `std * sigma_h` scored by MSE against the registered
  EWMA baseline `ewma_vol_60m * sqrt(h)`.

**Row sets (identical rows inside each comparison, asserted).**
`ADD_ROWS` = complete-case rows over all of `MARKET_FEATURES + GCP_FEATURES +
SCHR_h + target` (i.e. the A3 rows); A0, A1, A2, A3 and the four paired
additive deltas use these rows. `CPL_ROWS` = complete-case rows over
`MARKET_FEATURES + CPL_h(C_real) + CPL_h(C_shift) + target`; C0, C_real,
C_shift, the paired C_real - C_shift and C_real - C0 deltas, and the direct
readout use these rows. The two sets are defined independently and each
comparison is internally paired on its own set; `CPL_ROWS` additionally
requires the 30-day-old GCP value (so it loses the first 30 days of anchors)
while `ADD_ROWS` requires every registered GCP feature column to be finite.

## 6. Dev evaluation protocol (frozen)

* Anchors: `target_time_s % 3600 == 0` (hourly), all four assets, all four
  horizons. Rationale: the coupled and registered Schrödinger features
  require the 240-candle KDE window and are affordable at hourly anchors (the
  convention of `schrodinger_preregistration.md` section 6); a single cadence
  keeps every compared arm on identical rows.
* Folds: `backtest.walk_forward_predict` (expanding, `n_folds = 3`,
  `initial_train_fraction = 0.50`, embargo = horizon), exactly as in
  `train.py` / `train_schrodinger.py`.
* Models: per-fold `StandardScaler` + L2 logistic `C = 1` (direction) /
  ridge `alpha = 1` (volatility), frozen per fold (`backtest._fit_score`).
* Paired deltas: `backtest.paired_bootstrap_delta`, block length **3 anchors**
  (the pre-declared hourly-anchor convention of
  `schrodinger_preregistration.md` section 6; the repository's 60-row block
  applies at 1-minute anchors), seeds `20260914 + h` (hit rate),
  `20260915 + h` (IC), `20260916 + h` (MSE).
* Resolution: the exploration ranking uses `n_boot = 300` (two-sided floor
  `2/301 = 0.0066`); the frozen dev tables use `n_boot = 4999` (floor
  `0.0004`, below the per-family threshold `0.05/16 = 0.003125`). No dev
  p-value is ever treated as confirmatory; dev is the selection sample.
* Process note: `--stage explore` builds the frames once, runs the six
  candidate rankings at the ranking resolution, applies the selection rule,
  and then evaluates the frozen dev tables for the selected candidate in the
  same process; `--stage frozen` reproduces the frozen tables alone, reading
  the selected label from `results_mix_lab_dev.json`. The selection never sees
  a frozen-resolution statistic (it is a function of the exploration numbers
  only).
* Causality: `features.assert_causal_features` on the joined frame; the
  coupled/registered feature builders assert that every input candle is
  strictly before the anchor; the shifted control is asserted to be the
  registered causal column lagged by exactly 30 days on sampled rows.
* Shuffled-target control: `backtest.shuffled_target_control` (BTCUSDT, all
  four horizons) for the frozen coupled feature columns and for the A3
  columns, seed `20260914 + 100*h`, 3 folds.

## 7. Selection rule for the coupled candidate (fixed before the run)

Rank the six candidates by the paired dev statistic `C_real - C_shift`
(candidate minus its own shifted-GCP control, identical rows) in this order:

1. the number of the 16 cells with positive `Delta IC`;
2. the pooled (row-weighted) `Delta IC`;
3. the median `Delta IC`;
4. the number of the 16 cells with positive `Delta hit`;
5. the pooled `Delta hit` (final tie-break).

The top-ranked candidate is frozen as the coupled arm; ties are impossible
after item 5 in practice and, if they occurred, would be broken by the
lexicographically smallest label. The selection uses the **exploration**
statistics (`n_boot = 300`, ranking only). The helper statistic `C_real - C0`
is logged for every candidate but is **not** part of the ranking. The
frozen dev tables and the holdout test are run for the selected candidate
only; the other five are logged and never re-entered.

## 8. Frozen holdout test and decision rule (pre-registered, not yet run)

The holdout sources are now specified (audited read-only before this
registration was finalised; no holdout row, feature, or target has been
computed or inspected):

* market: `../crypto-data/holdout_klines_1m.db` (same verified `klines_1m`
  schema, four assets), 26,584 contiguous minutes per asset,
  2026-09-14 05:09 UTC -> 2026-10-02 16:12 UTC, directly following the dev
  source (dev end 2026-09-14 05:08 UTC).
* GCP: `../crypto-data/gcp2_holdout_1m.db` (same `netvar_1m` schema and
  minute-mean aggregation; reported as verified identical to the dev cache on
  its overlap), **41,760 minutes with no gaps**, 2026-09-03 00:00 UTC ->
  2026-10-01 23:59 UTC, contiguous with the dev GCP cache end
  (2026-09-02 23:59 UTC).
* **Usable mixed window.** The GCP column at row `t` is the minute labelled
  `t - 60s`, so the last scorable row is `last GCP holdout minute + 60s`
  = 2026-10-02 00:00 UTC; the evaluation window is therefore
  `[eval_start, eval_end]` with
  `eval_end = min(market holdout end, last GCP minute + 60s)` and
  `eval_start = max(first holdout minute, dev end) + 300 minutes`
  (the exact `max_h(h + KDE_WINDOW)` warm-up) = 2026-09-14 10:09 UTC.
  Measured: **25,311 evaluated minutes per asset (~17.8 days), about 421
  60-minute blocks per cell**, i.e. rows from 2026-09-14 10:09 UTC to
  2026-10-02 00:00 UTC. Rows after the GCP end are dropped, never filled.
  The stage aborts if the GCP holdout series starts after the market holdout.

* **Fit-once protocol (Amendment A2 style).** Every arm (A0-A3, C_real,
  C_shift) is fitted **once** on all dev rows at the hourly dev cadence whose
  horizon targets finish before the holdout's first candle
  (`target_time_s + h minutes <= holdout_start`), using the house complete-case
  policy and models; the scalers and coefficients are then frozen. No holdout
  row, feature, or target enters any fit. The direct readout (S0-style) fits
  nothing.
* **Scoring.** Every holdout minute in the usable window is scored; features
  are built on the last `--dev-tail-days` (2) days of dev market candles
  concatenated with the holdout so trailing windows cross the source
  boundary, and the GCP series is a dev tail (at least 31 days, needed by the
  30-day shifted control; the shifted control at the first evaluated row uses
  the registered column of 2026-08-15, inside the dev cache) concatenated with
  the holdout GCP series. Assertions: the holdout starts directly after the
  dev source; every fitted target ends before the holdout start; all market,
  GCP, registered Schrödinger, and coupled features are finite at the first
  evaluated row.
* **Paired deltas.** Five pre-registered direction comparisons: `A1-A0`,
  `A2-A0`, `A3-A0`, `A3-A2`, and `C_real - C_shift`, each on identical rows
  (the shared complete-case set over all arm columns for that cell), with the
  paired moving-block bootstrap at 60-row blocks, `n_boot = 9999`
  (two-sided floor `2/10000 = 0.0002`). `C_real - C0` and the direct readout
  are reported descriptively; the direct readout has no fit and is tested
  against chance 0.5 / 0 with the one-sample variants of the same bootstrap
  (`train_schrodinger.s0_chance_tests` is reused).
* **Bonferroni accounting.** Direction family: 5 comparisons x 2 metrics x 16
  cells = **160 tests**, `alpha_dir = 0.05/160 = 3.125e-4`.
  Volatility (secondary) family: 5 comparisons x 16 cells = **80 tests**,
  `alpha_vol = 0.05/80 = 6.25e-4`. Both thresholds are above the 0.0002
  bootstrap floor, so survival is attainable.
* **Decision rule (majority, as in `physics_preregistration.md`).**
  * The GCP-coupling claim requires the frozen `C_real - C_shift` comparison
    to have **>= 9 of 16** cells with a positive `Delta hit` **and >= 9 of 16**
    cells positive with `p < alpha_dir` (or the same for `Delta IC`).
  * An additive arm (each of `A1-A0`, `A2-A0`, `A3-A0`, `A3-A2`) claims
    direction skill only under the same majority rule with `p < alpha_dir`.
  * The volatility claim (secondary, same five comparisons) requires >= 9 of
    16 cells with `Delta MSE < 0` and >= 9 of 16 negative with
    `p < alpha_vol`.
  * The direct readout family is secondary: >= 9 of 16 chance-test supporters
    at `p < 0.05/32 = 1.5625e-3`, and it is reported as meaningful only if the
    primary families are consistent.
  * **Otherwise the result is reported as a null** ("no evidence that this
    arm adds skill / that GCP alignment matters through this channel").
* **One-shot status.** The frozen holdout is evaluated exactly once. If it
  produces a null, that is the recorded answer for this family; the test is
  not re-run with different constants, anchors, or arms. The holdout stage is
  exercised end to end on an artificial split built from a dev-market copy
  (`/tmp` databases: dev = the last 90 days of the dev market, fake holdout =
  the following 5 dev days; the matching GCP minutes for the fake holdout
  window come read-only from the GCP holdout cache because the dev GCP cache
  ends earlier). The **market holdout database is never opened** by that
  validation, and the real sources are opened for the first time by the single
  one-shot run.
* **Power statement (pre-registered honesty).** The usable window is
  **25,311 minutes per asset (~17.8 days), about 421 60-minute blocks per
  cell**. The per-cell minimum detectable effect in hit rate at
  `alpha_dir = 3.125e-4` is roughly 1.5-2 percentage points (from the
  block-bootstrap standard errors measured on the Schrödinger holdout, which
  had a comparable length), and a few thousandths for IC. Dev-scale effects in
  this family are of order 0.1-0.5 pp, so the holdout is expected to be
  underpowered relative to them: a null means "no effect of that size is
  detectable in ~17.8 days of this data", not "the effect is exactly zero".
  The measured row counts and block counts replace these figures in the
  results file.
* **Deployment limitation (recorded before the holdout run).** The live
  `gcp2.net getcurrentnetvar` endpoint returns a different quantity
  (values of order 10-220) than the archive `netvar` series used here (order
  1, standard deviation ~0.18). The registered coupling map and all dev
  features are defined on the archive quantity, so **the frozen mix
  configuration is not live-deployable on that endpoint** as-is. This arm is
  an archive-data evaluation only; it writes no live artifacts, and any future
  live use would require a separately pre-registered mapping between the two
  quantities.

## 9. Exploration log (filled after the run)

Run: `mix_lab.py --stage explore --output results_mix_lab_dev.json` on
`../crypto-data/crypto_market.db` joined with `../crypto-data/gcp2_1m.db`
(common range 1709251200..1788393540), hourly anchors, 6 candidates, 16 cells
each, ranking bootstrap `n_boot = 300`, block 3 anchors. Total runtime
**4272 s** (~71 min): feature build + exploration **1949 s**, frozen dev
evaluation **2322 s** (16 additive cells + 16 coupled cells at `n_boot = 4999`
+ 8 shuffled-target controls). Frames: 5,340,756 minute rows; 89,016 anchor
rows (22,249 per asset after warm-up, i.e. 22,254 anchor times minus the first
few). KDE degenerate-fallback rows: 0 in every batch. Identity consistency
check (coupled builder with the uncoupled coupling vs the registered builder,
16 asset/horizon cells): **max |difference| = 0.0**. Shifted-GCP check: the
merged control column equals the registered causal column lagged exactly 30
days on 256 sampled rows (**max |difference| = 0.0**). Causality self-check
passed (256 sampled rows, source strictly before target). Implementation
validation before this run: `test_mix_lab.py` (11/11) and two smoke runs on a
40-day dev slice with a two-candidate subset (not part of the evaluation).

### 9.1 Exploration grid summary (all six candidates, dev hourly anchors)

`pos ΔIC` = cells (of 16) with positive paired `C_real - C_shift` IC; `pooled ΔIC` = row-weighted pooled ΔIC; `med ΔIC`, `med Δhit`, `med ΔMSE` = medians over the 16 cells; `pos Δhit` = cells with positive Δhit; `neg ΔMSE` = cells with negative ΔMSE (better). Ranking bootstrap `n_boot = 300` (ranking only).

| candidate | pos ΔIC | pooled ΔIC | med ΔIC | pos Δhit | pooled Δhit (pp) | med Δhit (pp) | neg ΔMSE | med ΔMSE | runtime s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `hbar_inc` | 8/16 | -0.00002 | -0.00002 | 9/16 | +0.0170 | +0.0299 | 9/16 | -8.433e-11 | 70.6 |
| `hbar_dec` | 11/16 | +0.00013 | +0.00011 | 7/16 | +0.0132 | +0.0000 | 8/16 | +3.891e-11 | 70.6 |
| `stiff_inc` | 11/16 | +0.00067 | +0.00033 | 8/16 | -0.0107 | +0.0050 | 11/16 | -5.457e-10 | 70.0 |
| `packet_widen` (SELECTED) | 12/16 | +0.00141 | +0.00143 | 8/16 | +0.0139 | -0.0099 | 9/16 | -1.165e-10 | 71.2 |
| `packet_narrow` | 3/16 | -0.00122 | -0.00100 | 7/16 | -0.0283 | -0.0260 | 8/16 | +1.087e-11 | 74.2 |
| `absorb_inc` | 11/16 | +0.00058 | +0.00031 | 9/16 | +0.0025 | +0.0202 | 9/16 | -1.074e-10 | 72.1 |

Selection outcome: by the pre-declared rule (positive-ΔIC cell count, then
pooled ΔIC, then median ΔIC, then positive-Δhit count, then pooled Δhit) the
frozen coupled candidate is **`packet_widen`** (12/16 positive ΔIC, pooled
+0.00141). The spread of the six candidates is within ~0.3 basis points of
pooled ΔIC and the ordering is not stable across the ranking statistics, so
this is a choice among effectively-tied candidates, not evidence that the
widened-packet coupling is special. This is recorded explicitly.

### 9.2 Per-cell exploration deltas `C_real - C_shift` (Δhit pp / ΔIC), `n_boot = 300`

`hbar_inc`

| asset | 1m | 5m | 15m | 60m |
|---|---|---|---|---|
| BTCUSDT | -0.020 / -0.00051 | +0.040 / +0.00015 | +0.030 / -0.00021 | -0.070 / -0.00101 |
| ETHUSDT | -0.020 / +0.00038 | +0.030 / +0.00008 | -0.130 / +0.00020 | +0.080 / +0.00056 |
| SOLUSDT | +0.000 / -0.00019 | -0.010 / -0.00012 | +0.101 / +0.00090 | +0.150 / -0.00064 |
| BNBUSDT | +0.063 / +0.00032 | +0.041 / +0.00154 | +0.050 / -0.00019 | -0.060 / -0.00149 |

`hbar_dec`

| asset | 1m | 5m | 15m | 60m |
|---|---|---|---|---|
| BTCUSDT | -0.030 / +0.00042 | +0.060 / +0.00026 | +0.030 / +0.00056 | +0.010 / +0.00042 |
| ETHUSDT | +0.000 / -0.00034 | +0.060 / -0.00034 | -0.090 / +0.00116 | -0.100 / -0.00189 |
| SOLUSDT | +0.083 / +0.00009 | -0.020 / +0.00006 | -0.050 / +0.00012 | +0.261 / +0.00047 |
| BNBUSDT | -0.031 / -0.00033 | +0.112 / -0.00084 | -0.081 / +0.00007 | +0.000 / +0.00216 |

`stiff_inc`

| asset | 1m | 5m | 15m | 60m |
|---|---|---|---|---|
| BTCUSDT | +0.010 / -0.00000 | -0.010 / +0.00015 | -0.110 / +0.00052 | +0.050 / +0.00087 |
| ETHUSDT | +0.010 / -0.00000 | -0.010 / +0.00003 | +0.070 / +0.00120 | -0.299 / +0.00127 |
| SOLUSDT | +0.000 / -0.00003 | +0.030 / -0.00022 | +0.262 / +0.00120 | +0.231 / +0.00298 |
| BNBUSDT | +0.000 / +0.00002 | +0.010 / +0.00090 | -0.060 / -0.00035 | -0.351 / +0.00210 |

`packet_widen`

| asset | 1m | 5m | 15m | 60m |
|---|---|---|---|---|
| BTCUSDT | +0.223 / -0.00062 | -0.080 / +0.00367 | +0.249 / +0.00257 | -0.139 / -0.00013 |
| ETHUSDT | -0.231 / +0.00095 | +0.249 / +0.00155 | -0.040 / +0.00237 | -0.399 / +0.00099 |
| SOLUSDT | -0.312 / +0.00586 | -0.132 / +0.00019 | +0.091 / +0.00148 | +0.020 / -0.00068 |
| BNBUSDT | +0.313 / +0.00216 | -0.071 / +0.00222 | +0.191 / +0.00138 | +0.291 / -0.00124 |

`packet_narrow`

| asset | 1m | 5m | 15m | 60m |
|---|---|---|---|---|
| BTCUSDT | +0.020 / +0.00068 | +0.100 / -0.00142 | -0.169 / -0.00174 | +0.040 / -0.00090 |
| ETHUSDT | +0.070 / -0.00228 | +0.010 / -0.00030 | -0.110 / -0.00110 | -0.060 / -0.00156 |
| SOLUSDT | +0.052 / -0.00542 | +0.365 / +0.00011 | -0.070 / +0.00070 | -0.130 / -0.00037 |
| BNBUSDT | -0.052 / -0.00225 | +0.000 / -0.00232 | -0.141 / -0.00070 | -0.371 / -0.00084 |

`absorb_inc`

| asset | 1m | 5m | 15m | 60m |
|---|---|---|---|---|
| BTCUSDT | +0.091 / +0.00017 | -0.160 / +0.00128 | +0.040 / +0.00171 | -0.040 / +0.00177 |
| ETHUSDT | +0.050 / -0.00063 | +0.010 / +0.00090 | -0.080 / +0.00089 | -0.169 / +0.00023 |
| SOLUSDT | +0.125 / -0.00095 | +0.071 / +0.00026 | +0.000 / +0.00391 | +0.040 / -0.00074 |
| BNBUSDT | +0.073 / +0.00036 | +0.030 / -0.00135 | -0.010 / -0.00004 | -0.020 / +0.00149 |

Reading of the exploration (descriptive, not confirmatory): every candidate's
pooled ΔIC against its own shifted-GCP control is at most ~0.14 basis points
(`packet_widen`), and the ranking statistics are within noise of each other;
`packet_narrow` is the only clearly negative candidate. No exploration
p-value is below the 0.0066 floor, as expected at 300 replicates.

## 10. Frozen specification and dev results (filled after the run)

Reproduce (from the repository root; CPU is capped to half the cores inside
the driver):

```bash
price-prediction/.venv/bin/python price-prediction/test_mix_lab.py
price-prediction/.venv/bin/python price-prediction/mix_lab.py --stage explore   # exploration + selection + frozen dev tables
price-prediction/.venv/bin/python price-prediction/mix_lab.py --stage frozen    # frozen dev tables only (reproduces the selected label)
# one-shot holdout (only after this specification and the frozen label exist):
price-prediction/.venv/bin/python price-prediction/mix_lab.py --stage holdout \
    --db ../crypto-data/crypto_market.db --gcp ../crypto-data/gcp2_1m.db \
    --holdout-db ../crypto-data/holdout_klines_1m.db \
    --gcp-holdout-db ../crypto-data/gcp2_holdout_1m.db \
    --dev-results results_mix_lab_dev.json --output results_mix_lab_holdout.json
```

### 10.0 Frozen specification

Chosen by the pre-declared rule of section 7 and **frozen**:

* coupled arm: **`packet_widen`** = initial packet width driven by the
  registered causal GCP z-score:
  `u = tanh(netvar_z_60m / 2)`,
  `sigma_eff = clip(sigma_packet * exp(+0.5 u), 0.35, 3.0)`, registered
  `x0`, `k0`, `V` (quartic KDE), `hbar = 1`, `tau = h/60`; four Born-rule
  observables per horizon as model columns on top of `MARKET_FEATURES`; the
  shifted-GCP control uses the identical map with `u_shift = tanh(z(t-30d)/2)`;
  the direct readout uses `sign(p_up - 0.5)` (direction) and `std * sigma_h`
  (volatility) with no fitted model.
* additive arms: A0 `market_only` (14 columns), A1 `+GCP` (19), A2 `+SCHR`
  (18, the registered frozen `quartic_kde|linear_60|momentum` columns), A3
  `+GCP+SCHR` (23).
* everything else as in sections 3-6 (hourly anchors, 3 expanding folds, 50%
  initial train, embargo = horizon, per-fold scaler + logistic/ridge,
  4999-replicate paired moving-block bootstrap with 3-anchor blocks).

### 10.1 Frozen additive arms A0-A3 (dev, hourly anchors, `n_boot = 4999`, block 3 anchors)

Deltas are candidate minus reference on identical rows; `pp` = percentage points; every cell uses the same complete-case row set (A3's). `dHit` and `dIC` columns give the delta and its paired p-value.

| asset | h | A0 hit% | A1 hit% | A2 hit% | A3 hit% | dHit A1-A0 (p) | dHit A2-A0 (p) | dHit A3-A0 (p) | dHit A3-A2 (p) | dIC A1-A0 (p) | dIC A2-A0 (p) | dIC A3-A0 (p) | dIC A3-A2 (p) | n |
|---|---:|---:|---:|---:|---:|---|---|---|---|---|---|---|---|---:|
| BTCUSDT | 1 | 52.267 | 52.334 | 52.651 | 52.171 | +0.0672 (0.8260) | +0.3842 (0.1216) | -0.0961 (0.7688) | -0.4803 (0.0788) | -0.00154 (0.5136) | +0.00236 (0.2512) | +0.00061 (0.8632) | -0.00175 (0.4488) | 10410 |
| BTCUSDT | 5 | 52.103 | 52.065 | 51.923 | 51.914 | -0.0378 (0.9204) | -0.1796 (0.3980) | -0.1890 (0.5656) | -0.0095 (0.9780) | +0.00169 (0.5620) | -0.00134 (0.4192) | +0.00068 (0.8172) | +0.00201 (0.4708) | 10581 |
| BTCUSDT | 15 | 52.524 | 52.307 | 52.486 | 52.316 | -0.2170 (0.4108) | -0.0377 (0.9000) | -0.2076 (0.5064) | -0.1698 (0.5136) | +0.00004 (0.9716) | +0.00389 (0.1824) | +0.00404 (0.2372) | +0.00015 (0.9360) | 10599 |
| BTCUSDT | 60 | 51.466 | 51.400 | 51.023 | 51.231 | -0.0660 (0.8332) | -0.4432 (0.1332) | -0.2357 (0.5192) | +0.2074 (0.4988) | -0.00094 (0.7600) | -0.00514 (0.0832) | -0.00572 (0.1636) | -0.00058 (0.8308) | 10605 |
| ETHUSDT | 1 | 52.350 | 52.017 | 52.644 | 52.331 | -0.3323 (0.1568) | +0.2943 (0.3320) | -0.0190 (0.9848) | -0.3133 (0.1484) | -0.00313 (0.0764) | +0.00307 (0.2748) | +0.00071 (0.8336) | -0.00236 (0.1424) | 10533 |
| ETHUSDT | 5 | 52.054 | 52.121 | 51.828 | 51.790 | +0.0661 (0.8240) | -0.2267 (0.2548) | -0.2645 (0.3460) | -0.0378 (0.8896) | -0.00058 (0.7756) | -0.00391 (0.0064) | -0.00424 (0.0808) | -0.00034 (0.8620) | 10587 |
| ETHUSDT | 15 | 52.407 | 52.227 | 52.869 | 52.605 | -0.1793 (0.5224) | +0.4624 (0.0720) | +0.1982 (0.5340) | -0.2643 (0.3752) | +0.00055 (0.8132) | +0.00184 (0.5276) | +0.00309 (0.4172) | +0.00125 (0.6124) | 10596 |
| ETHUSDT | 60 | 51.363 | 51.373 | 51.741 | 51.316 | +0.0094 (0.9912) | +0.3774 (0.2540) | -0.0472 (0.8848) | -0.4246 (0.1312) | -0.00035 (0.8936) | +0.00275 (0.4668) | +0.00246 (0.5640) | -0.00029 (0.9044) | 10599 |
| SOLUSDT | 1 | 52.867 | 52.700 | 52.326 | 52.257 | -0.1672 (0.4948) | -0.5410 (0.0968) | -0.6098 (0.0728) | -0.0689 (0.7636) | -0.00116 (0.5944) | +0.00026 (0.9308) | -0.00035 (0.9236) | -0.00060 (0.7432) | 10167 |
| SOLUSDT | 5 | 51.860 | 52.033 | 51.985 | 51.889 | +0.1726 (0.4468) | +0.1247 (0.6464) | +0.0288 (0.9500) | -0.0959 (0.6668) | -0.00176 (0.2844) | +0.00050 (0.7944) | -0.00151 (0.5288) | -0.00200 (0.2272) | 10428 |
| SOLUSDT | 15 | 51.915 | 51.468 | 52.277 | 51.420 | -0.4479 (0.1744) | +0.3621 (0.0780) | -0.4955 (0.1436) | -0.8576 (0.0092) | +0.00092 (0.7960) | -0.00375 (0.0540) | -0.00223 (0.5764) | +0.00152 (0.6584) | 10494 |
| SOLUSDT | 60 | 50.949 | 51.195 | 51.309 | 51.233 | +0.2466 (0.4508) | +0.3605 (0.3040) | +0.2846 (0.4752) | -0.0759 (0.8172) | +0.00298 (0.4064) | +0.00285 (0.5052) | +0.00530 (0.3168) | +0.00244 (0.4312) | 10542 |
| BNBUSDT | 1 | 50.410 | 50.390 | 50.489 | 50.716 | -0.0198 (0.9384) | +0.0791 (0.7804) | +0.3064 (0.2736) | +0.2273 (0.1836) | -0.00041 (0.8288) | +0.00248 (0.5252) | +0.00218 (0.5900) | -0.00029 (0.8728) | 10119 |
| BNBUSDT | 5 | 51.279 | 51.154 | 51.471 | 51.423 | -0.1250 (0.5500) | +0.1923 (0.4804) | +0.1443 (0.6276) | -0.0481 (0.8276) | -0.00116 (0.4996) | +0.00178 (0.5372) | +0.00089 (0.8024) | -0.00088 (0.5944) | 10398 |
| BNBUSDT | 15 | 51.912 | 52.294 | 51.969 | 52.036 | +0.3815 (0.1668) | +0.0572 (0.8456) | +0.1240 (0.7124) | +0.0668 (0.8124) | -0.00044 (0.8756) | +0.00539 (0.0556) | +0.00488 (0.2000) | -0.00051 (0.8580) | 10485 |
| BNBUSDT | 60 | 50.859 | 50.289 | 50.973 | 51.286 | -0.5693 (0.0664) | +0.1139 (0.6936) | +0.4270 (0.2488) | +0.3131 (0.3116) | +0.00160 (0.6344) | +0.00636 (0.1276) | +0.00788 (0.1348) | +0.00152 (0.6192) | 10539 |

`A1-A0`: hit_rate improving 6/16, survivors 0, significantly worse 0, median -0.00052; ic_spearman improving 6/16, survivors 0, median -0.00042; mse improving 0/16, survivors 0, significantly worse 1 (BNBUSDT 15m), median +6.8e-10 -> **null**.
`A2-A0`: hit_rate improving 11/16, survivors 0, median +0.00119; ic_spearman improving 12/16, survivors 0, median +0.00210; mse improving 7/16, survivors 1 (SOLUSDT 5m), median +2.7e-11 -> **null**.
`A3-A0`: hit_rate improving 7/16, survivors 0, median -0.00033; ic_spearman improving 11/16, survivors 0, median +0.00080; mse improving 6/16, survivors 1 (SOLUSDT 5m), significantly worse 1 (SOLUSDT 60m) -> **null**.
`A3-A2`: hit_rate improving 4/16, survivors 0, median -0.00072; ic_spearman improving 6/16, survivors 0, median -0.00031; mse improving 0/16, significantly worse 1 (BNBUSDT 15m) -> **null** (GCP adds nothing on top of the Schrödinger columns).

### 10.2 Frozen coupled arm `packet_widen` vs its 30-day shifted-GCP control (dev, `n_boot = 4999`)

`real`/`shift` are the fitted hit rates of `MARKET + CPL` on the real vs shifted coupling input; `dHit/dIC/dMSE` are real minus shift on identical rows; `vs C0` is the same candidate against market-only on the same rows (descriptive); `S0` is the direct Born-rule readout (`sign(p_up - 0.5)`, no fit).

| asset | h | real hit% | shift hit% | dHit pp (p) | dIC (p) | dMSE (p) | dHit vs C0 (pp) | dIC vs C0 | S0 hit% (p vs chance) | S0 IC (p vs 0) | n |
|---|---:|---:|---:|---|---|---|---|---:|---|---|---:|
| BTCUSDT | 1 | 51.959 | 51.736 | +0.2233 (0.3112) | -0.00062 (0.6892) | -3.438e-11 (0.8444) | -0.0406 | -0.00057 | 47.419 (0.0004) | -0.07102 (0.0004) | 9852 |
| BTCUSDT | 5 | 52.101 | 52.181 | -0.0799 (0.7376) | +0.00367 (0.0432) | +1.379e-10 (0.8800) | -0.0799 | +0.00071 | 47.664 (0.0004) | -0.05773 (0.0004) | 10017 |
| BTCUSDT | 15 | 52.606 | 52.357 | +0.2491 (0.2020) | +0.00257 (0.1048) | -2.728e-10 (0.7880) | +0.4484 | +0.00475 | 47.409 (0.0004) | -0.06269 (0.0004) | 10035 |
| BTCUSDT | 60 | 51.210 | 51.349 | -0.1394 (0.3424) | -0.00013 (0.8780) | -1.986e-10 (0.7428) | -0.0896 | -0.00210 | 48.252 (0.0004) | -0.03870 (0.0004) | 10041 |
| ETHUSDT | 1 | 52.312 | 52.543 | -0.2307 (0.3408) | +0.00095 (0.6192) | +9.689e-10 (0.1408) | +0.2809 | +0.00361 | 47.031 (0.0004) | -0.07708 (0.0004) | 9969 |
| ETHUSDT | 5 | 51.905 | 51.656 | +0.2494 (0.2096) | +0.00155 (0.2172) | -9.078e-10 (0.7188) | -0.1696 | -0.00305 | 48.297 (0.0004) | -0.04550 (0.0004) | 10026 |
| ETHUSDT | 15 | 52.791 | 52.831 | -0.0399 (0.8196) | +0.00237 (0.0772) | -1.575e-09 (0.5760) | +0.2492 | +0.00342 | 46.868 (0.0004) | -0.05702 (0.0004) | 10032 |
| ETHUSDT | 60 | 51.470 | 51.868 | -0.3986 (0.0436) | +0.00099 (0.5068) | -5.578e-10 (0.7652) | -0.1096 | +0.00559 | 47.693 (0.0004) | -0.03518 (0.0004) | 10035 |
| SOLUSDT | 1 | 52.304 | 52.616 | -0.3120 (0.2616) | +0.00586 (0.0316) | +6.398e-11 (0.9212) | -0.4472 | +0.00641 | 47.798 (0.0004) | -0.06363 (0.0004) | 9615 |
| SOLUSDT | 5 | 51.991 | 52.123 | -0.1318 (0.4404) | +0.00019 (0.8240) | +4.089e-10 (0.9600) | -0.0304 | +0.00054 | 48.242 (0.0004) | -0.04366 (0.0004) | 9867 |
| SOLUSDT | 15 | 51.908 | 51.817 | +0.0906 (0.6208) | +0.00148 (0.2420) | -1.949e-09 (0.6504) | -0.0604 | -0.00367 | 48.067 (0.0004) | -0.03935 (0.0004) | 9933 |
| SOLUSDT | 60 | 51.413 | 51.393 | +0.0200 (0.9400) | -0.00068 (0.6992) | -1.568e-09 (0.4448) | +0.4109 | +0.00543 | 48.209 (0.0004) | -0.03397 (0.0004) | 9978 |
| BNBUSDT | 1 | 50.172 | 49.859 | +0.3126 (0.1628) | +0.00216 (0.3564) | +1.844e-10 (0.6244) | +0.0104 | +0.00116 | 48.609 (0.0008) | -0.04543 (0.0004) | 9597 |
| BNBUSDT | 5 | 51.299 | 51.370 | -0.0711 (0.7040) | +0.00222 (0.1208) | +6.586e-10 (0.7956) | +0.2944 | +0.00005 | 48.488 (0.0004) | -0.03408 (0.0004) | 9852 |
| BNBUSDT | 15 | 52.008 | 51.817 | +0.1913 (0.4108) | +0.00138 (0.5608) | -2.862e-10 (0.8948) | +0.2718 | +0.00644 | 47.851 (0.0004) | -0.03889 (0.0004) | 9933 |
| BNBUSDT | 60 | 51.378 | 51.087 | +0.2906 (0.1212) | -0.00124 (0.4480) | +5.126e-10 (0.7788) | +0.8216 | +0.00688 | 48.094 (0.0004) | -0.03788 (0.0004) | 9981 |

`C_real-C_shift`: hit_rate improving 8/16, survivors 0, significantly worse 0, median -0.000099; ic_spearman improving 12/16, survivors 0, median +0.00143; mse improving 9/16, survivors 0, median -1.2e-10 -> **null**.
`C_real-C0`: hit_rate improving 8/16, survivors 0, median -0.00010; ic_spearman improving 12/16, survivors 0, median +0.00229; mse improving 6/16, survivors 0 -> **null** (descriptive comparison).

**Shuffled-target controls** (BTCUSDT, all horizons, training-target permutation, refit, same folds): coupled features 1m hit 0.4999/IC -0.00682, 5m hit 0.4906/IC -0.00711, 15m hit 0.4847/IC -0.03104, 60m hit 0.5022/IC +0.00328; A3 features 1m hit 0.4956/IC -0.00841, 5m hit 0.4997/IC -0.00438, 15m hit 0.4971/IC -0.00888, 60m hit 0.4990/IC +0.00308. At chance within the hourly-anchor block-bootstrap noise, as required.

**Direct (S0) volatility readout** (`std * sigma_h` vs the registered EWMA baseline, same rows): worse than EWMA in 16/16 cells, p = 0.0004 in every cell.

### 10.3 Verdict (pre-registered rule; dev is descriptive, not confirmatory)

* **Additive arms:** all four direction comparisons are nulls; 0/16
  Bonferroni survivors for Δhit and ΔIC in every comparison. `A2-A0`
  (registered Schrödinger columns) leans positive (11/16 Δhit, 12/16 ΔIC,
  median +0.12 pp / +0.0021) but no cell survives the 16-cell threshold;
  `A1-A0` (GCP) is flat-to-negative (6/16 and 6/16); `A3-A2` (GCP on top of
  Schrödinger) is null-to-negative (4/16 Δhit, 6/16 ΔIC, median -0.07 pp).
* **Coupled arm:** `packet_widen` does not beat its own 30-day shifted-GCP
  control: Δhit 8/16 improving with 0 survivors (median -0.01 pp), ΔIC 12/16
  improving with 0 survivors (median +0.0014, smallest p = 0.0316), ΔMSE 9/16
  improving with 0 survivors. The same picture holds against market-only.
* **Direct readout:** the coupled Born-rule `p_up` is contrarian on dev
  (hit 46.9-48.6%, IC -0.034..-0.077, below chance in 16/16 cells, p <= 0.0008
  in every cell), matching the registered Schrödinger arm's dev finding; the
  direct volatility readout is worse than EWMA in 16/16.
* **Bottom line (dev):** there is no evidence in this dev sample that mixing
  GCP2 into the wave equation — through any of the six pre-declared
  couplings — carries aligned information beyond a 30-day time shift of the
  same series, and no evidence that adding GCP columns on top of the
  Schrödinger columns helps. The frozen holdout test of section 8 is the
  confirmatory test and has not been run; dev is the selection sample.

## 11. Amendment log

**A1 (documentation only, recorded while the dev run was in flight, before any
result was read).** Section 5's description of the two row sets was corrected:
the original sentence claimed `CPL_ROWS` is a subset of `ADD_ROWS`. It is not
strictly true (the additive set also requires the one-minute GCP velocity and
acceleration columns, which need consecutive present minutes, while the
coupled set requires only the 60-minute z-score to be finite), so the text now
states that the two sets are defined independently and that each comparison is
internally paired on its own set. No arm, feature, coupling, constant, fold,
bootstrap setting, row-set definition, selection rule, threshold, or decision
rule was changed.

## 12. Holdout results (one-shot, section 8 protocol)

**Run.** `mix_lab.py --stage holdout --db ../crypto-data/crypto_market.db
--gcp ../crypto-data/gcp2_1m.db --holdout-db
../crypto-data/holdout_klines_1m.db --gcp-holdout-db
../crypto-data/gcp2_holdout_1m.db --dev-results results_mix_lab_dev.json
--output results_mix_lab_holdout.json --bootstrap 9999`, executed **once**.
Sources at run time: market holdout md5 `43c17bc93d811c7ab9c7e3c278e4e961`
(26,584 contiguous minutes per asset, 2026-09-14 05:09 -> 2026-10-02 16:12
UTC), GCP holdout md5 `8647e0ae5bf5a7f216ccbd7f3a76f6a2` (41,760 minutes, no
gaps, 2026-09-03 00:00 -> 2026-10-01 23:59 UTC). Bounds: dev end 1789362480,
holdout start 1789362540, warm-up 300 minutes, **eval_start 1789380540
(2026-09-14 10:09 UTC), eval_end 1790899200 (2026-10-02 00:00 UTC)** = the
GCP-bounded window; evaluated rows 23,773-25,301 per cell. Fits: 19,703-20,082
hourly dev rows per asset/horizon, every fitted target ending at or before
1788390000 < holdout start. Runtime **8893 s** (~2.5 h). The output file is
guarded against a second run; the holdout databases were opened read-only and
their md5s are unchanged. Assertions recorded in `results_mix_lab_holdout.json`:
holdout directly follows dev; all fitted targets end before the holdout;
all market/GCP/Schrödinger/coupled features finite at the first evaluated row.

### 12.1 All five pre-registered comparisons, 16 cells each

Deltas are candidate minus reference on identical rows; `pp` = percentage
points; paired 60-row moving-block bootstrap, `n_boot = 9999` (floor 0.0002).

| comparison | asset | h | dHit pp (p) | dIC (p) | dMSE (p) | n |
|---|---|---:|---|---|---|---:|
| A1-A0 | BTCUSDT | 1 | +0.0735 (0.6744) | +0.00104 (0.4336) | -1.084e-11 (0.6684) | 24501 |
| A1-A0 | BTCUSDT | 5 | -0.3644 (0.0552) | -0.00344 (0.0726) | -1.201e-11 (0.8950) | 25249 |
| A1-A0 | BTCUSDT | 15 | -0.1463 (0.2856) | -0.00097 (0.4684) | +5.590e-10 (0.1486) | 25296 |
| A1-A0 | BTCUSDT | 60 | -0.2648 (0.3572) | -0.01087 (0.0536) | +4.379e-10 (0.0270) | 25301 |
| A1-A0 | ETHUSDT | 1 | +0.0961 (0.4538) | +0.00070 (0.4276) | +3.790e-11 (0.5474) | 24964 |
| A1-A0 | ETHUSDT | 5 | -0.2970 (0.0584) | -0.00271 (0.0716) | -1.672e-11 (0.8822) | 25252 |
| A1-A0 | ETHUSDT | 15 | -0.2571 (0.1806) | -0.00431 (0.1198) | +1.251e-09 (0.0338) | 25285 |
| A1-A0 | ETHUSDT | 60 | -0.6919 (0.0496) | -0.01447 (0.0410) | +2.124e-09 (0.2724) | 25294 |
| A1-A0 | SOLUSDT | 1 | -0.2229 (0.2024) | -0.00076 (0.5904) | +1.439e-10 (0.0760) | 23773 |
| A1-A0 | SOLUSDT | 5 | -0.1660 (0.2974) | -0.00455 (0.0020) | -3.094e-10 (0.2522) | 24694 |
| A1-A0 | SOLUSDT | 15 | -0.0882 (0.7174) | -0.00547 (0.0188) | -1.772e-10 (0.7656) | 24953 |
| A1-A0 | SOLUSDT | 60 | -0.6640 (0.0466) | -0.01327 (0.0116) | -8.152e-10 (0.7372) | 25152 |
| A1-A0 | BNBUSDT | 1 | -0.0122 (0.9308) | -0.00016 (0.9018) | +6.496e-11 (0.1140) | 24672 |
| A1-A0 | BNBUSDT | 5 | -0.1311 (0.4378) | -0.00277 (0.1264) | +2.143e-11 (0.7598) | 25171 |
| A1-A0 | BNBUSDT | 15 | -0.1625 (0.3944) | -0.00261 (0.2706) | +5.375e-11 (0.5892) | 25234 |
| A1-A0 | BNBUSDT | 60 | -0.5105 (0.0958) | -0.00910 (0.1020) | +1.839e-10 (0.8606) | 25268 |
| A2-A0 | BTCUSDT | 1 | -0.2204 (0.1864) | -0.00598 (0.0002) | +6.012e-11 (0.7962) | 24501 |
| A2-A0 | BTCUSDT | 5 | +0.1030 (0.4714) | +0.00091 (0.5870) | -3.417e-09 (0.0314) | 25249 |
| A2-A0 | BTCUSDT | 15 | -0.0237 (0.9154) | +0.00598 (0.0660) | +1.429e-09 (0.5260) | 25296 |
| A2-A0 | BTCUSDT | 60 | +0.4506 (0.2700) | +0.00333 (0.6850) | +3.349e-09 (0.2006) | 25301 |
| A2-A0 | ETHUSDT | 1 | -0.2323 (0.2264) | -0.00294 (0.0730) | -5.591e-11 (0.9416) | 24964 |
| A2-A0 | ETHUSDT | 5 | -0.0436 (0.7178) | +0.00042 (0.6970) | -3.868e-09 (0.1816) | 25252 |
| A2-A0 | ETHUSDT | 15 | +0.3085 (0.1176) | +0.00859 (0.0112) | +2.247e-09 (0.7552) | 25285 |
| A2-A0 | ETHUSDT | 60 | -0.4033 (0.4106) | -0.00339 (0.7384) | +1.869e-09 (0.7244) | 25294 |
| A2-A0 | SOLUSDT | 1 | -0.3828 (0.0442) | -0.00223 (0.3154) | -2.531e-09 (0.0002) | 23773 |
| A2-A0 | SOLUSDT | 5 | +0.2106 (0.1776) | -0.00117 (0.5926) | -3.950e-09 (0.3770) | 24694 |
| A2-A0 | SOLUSDT | 15 | +0.0521 (0.4956) | +0.00168 (0.0372) | +4.874e-09 (0.4718) | 24953 |
| A2-A0 | SOLUSDT | 60 | -0.4016 (0.4250) | -0.00478 (0.7132) | +1.832e-09 (0.0446) | 25152 |
| A2-A0 | BNBUSDT | 1 | +0.0527 (0.7958) | -0.00242 (0.2028) | -3.512e-10 (0.0948) | 24672 |
| A2-A0 | BNBUSDT | 5 | +0.1112 (0.6718) | -0.00144 (0.7716) | +1.241e-09 (0.3934) | 25171 |
| A2-A0 | BNBUSDT | 15 | -0.2695 (0.3412) | -0.00052 (0.9336) | -3.804e-10 (0.9022) | 25234 |
| A2-A0 | BNBUSDT | 60 | +0.0040 (0.9984) | -0.00622 (0.6576) | -8.764e-10 (0.8752) | 25268 |
| A3-A0 | BTCUSDT | 1 | -0.1918 (0.3378) | -0.00475 (0.0046) | +4.426e-11 (0.8472) | 24501 |
| A3-A0 | BTCUSDT | 5 | -0.2059 (0.3144) | -0.00212 (0.3980) | -3.398e-09 (0.0328) | 25249 |
| A3-A0 | BTCUSDT | 15 | -0.0158 (0.9426) | +0.00529 (0.1306) | +2.041e-09 (0.3398) | 25296 |
| A3-A0 | BTCUSDT | 60 | +0.4466 (0.3230) | -0.00619 (0.4982) | +3.781e-09 (0.1524) | 25301 |
| A3-A0 | ETHUSDT | 1 | -0.1642 (0.4082) | -0.00253 (0.1668) | -2.642e-11 (0.9796) | 24964 |
| A3-A0 | ETHUSDT | 5 | -0.1663 (0.3222) | -0.00223 (0.2356) | -3.905e-09 (0.1784) | 25252 |
| A3-A0 | ETHUSDT | 15 | +0.0356 (0.8882) | +0.00435 (0.3146) | +3.422e-09 (0.5948) | 25285 |
| A3-A0 | ETHUSDT | 60 | -0.8105 (0.1444) | -0.01551 (0.1856) | +3.908e-09 (0.4902) | 25294 |
| A3-A0 | SOLUSDT | 1 | -0.3533 (0.0948) | -0.00244 (0.3468) | -2.382e-09 (0.0002) | 23773 |
| A3-A0 | SOLUSDT | 5 | -0.1782 (0.3714) | -0.00554 (0.0392) | -4.324e-09 (0.3344) | 24694 |
| A3-A0 | SOLUSDT | 15 | +0.0120 (0.9332) | -0.00361 (0.1288) | +4.731e-09 (0.4784) | 24953 |
| A3-A0 | SOLUSDT | 60 | -0.9184 (0.0756) | -0.01387 (0.2816) | +1.185e-09 (0.6146) | 25152 |
| A3-A0 | BNBUSDT | 1 | +0.0608 (0.7548) | -0.00238 (0.2906) | -2.808e-10 (0.1892) | 24672 |
| A3-A0 | BNBUSDT | 5 | +0.0318 (0.9156) | -0.00389 (0.4190) | +1.236e-09 (0.3982) | 25171 |
| A3-A0 | BNBUSDT | 15 | -0.3844 (0.2068) | -0.00317 (0.6224) | -3.158e-10 (0.9180) | 25234 |
| A3-A0 | BNBUSDT | 60 | -0.0514 (0.9380) | -0.01344 (0.3436) | -6.814e-10 (0.9096) | 25268 |
| A3-A2 | BTCUSDT | 1 | +0.0286 (0.8636) | +0.00123 (0.2930) | -1.586e-11 (0.5426) | 24501 |
| A3-A2 | BTCUSDT | 5 | -0.3089 (0.0982) | -0.00304 (0.1046) | +1.948e-11 (0.8658) | 25249 |
| A3-A2 | BTCUSDT | 15 | +0.0079 (0.9700) | -0.00069 (0.5962) | +6.130e-10 (0.1380) | 25296 |
| A3-A2 | BTCUSDT | 60 | -0.0040 (0.9948) | -0.00952 (0.0588) | +4.321e-10 (0.0198) | 25301 |
| A3-A2 | ETHUSDT | 1 | +0.0681 (0.5676) | +0.00041 (0.5762) | +2.949e-11 (0.6348) | 24964 |
| A3-A2 | ETHUSDT | 5 | -0.1228 (0.4502) | -0.00265 (0.0712) | -3.690e-11 (0.7510) | 25252 |
| A3-A2 | ETHUSDT | 15 | -0.2729 (0.1606) | -0.00423 (0.0950) | +1.176e-09 (0.0484) | 25285 |
| A3-A2 | ETHUSDT | 60 | -0.4072 (0.1882) | -0.01212 (0.0460) | +2.039e-09 (0.2976) | 25294 |
| A3-A2 | SOLUSDT | 1 | +0.0294 (0.8570) | -0.00021 (0.8560) | +1.488e-10 (0.0648) | 23773 |
| A3-A2 | SOLUSDT | 5 | -0.3888 (0.0192) | -0.00437 (0.0052) | -3.739e-10 (0.1376) | 24694 |
| A3-A2 | SOLUSDT | 15 | -0.0401 (0.8690) | -0.00529 (0.0240) | -1.435e-10 (0.8042) | 24953 |
| A3-A2 | SOLUSDT | 60 | -0.5169 (0.0526) | -0.00909 (0.0342) | -6.467e-10 (0.8016) | 25152 |
| A3-A2 | BNBUSDT | 1 | +0.0081 (0.9530) | +0.00004 (0.9642) | +7.033e-11 (0.1032) | 24672 |
| A3-A2 | BNBUSDT | 5 | -0.0795 (0.6014) | -0.00245 (0.1202) | -4.223e-12 (0.9538) | 25171 |
| A3-A2 | BNBUSDT | 15 | -0.1149 (0.5522) | -0.00265 (0.2260) | +6.463e-11 (0.6192) | 25234 |
| A3-A2 | BNBUSDT | 60 | -0.0554 (0.8572) | -0.00723 (0.1284) | +1.950e-10 (0.8606) | 25268 |
| C_real-C_shift | BTCUSDT | 1 | +0.0327 (0.7286) | +0.00002 (0.9468) | -2.359e-10 (0.0218) | 24501 |
| C_real-C_shift | BTCUSDT | 5 | -0.2495 (0.1002) | +0.00083 (0.6118) | -1.010e-09 (0.1044) | 25249 |
| C_real-C_shift | BTCUSDT | 15 | +0.1305 (0.3502) | +0.00235 (0.2944) | -5.125e-10 (0.2716) | 25296 |
| C_real-C_shift | BTCUSDT | 60 | -0.0316 (0.7878) | +0.00045 (0.8006) | -7.207e-10 (0.1222) | 25301 |
| C_real-C_shift | ETHUSDT | 1 | +0.0881 (0.4746) | +0.00056 (0.5594) | -9.655e-10 (0.0014) | 24964 |
| C_real-C_shift | ETHUSDT | 5 | -0.2178 (0.0662) | -0.00105 (0.2292) | -1.362e-09 (0.2566) | 25252 |
| C_real-C_shift | ETHUSDT | 15 | +0.0989 (0.3698) | +0.00158 (0.3432) | -1.139e-09 (0.6434) | 25285 |
| C_real-C_shift | ETHUSDT | 60 | -0.0158 (0.9264) | -0.00006 (0.9930) | +1.154e-09 (0.3694) | 25294 |
| C_real-C_shift | SOLUSDT | 1 | -0.0589 (0.7446) | -0.00025 (0.9276) | -3.171e-10 (0.2554) | 23773 |
| C_real-C_shift | SOLUSDT | 5 | +0.0850 (0.4224) | +0.00036 (0.7130) | -4.299e-09 (0.0066) | 24694 |
| C_real-C_shift | SOLUSDT | 15 | +0.2605 (0.0420) | +0.00165 (0.3492) | -4.554e-09 (0.0354) | 24953 |
| C_real-C_shift | SOLUSDT | 60 | -0.0119 (0.9496) | -0.00149 (0.6056) | +2.712e-09 (0.1544) | 25152 |
| C_real-C_shift | BNBUSDT | 1 | +0.1662 (0.2836) | -0.00096 (0.5462) | -1.683e-10 (0.3026) | 24672 |
| C_real-C_shift | BNBUSDT | 5 | +0.1430 (0.4012) | -0.00356 (0.0850) | -7.128e-10 (0.4522) | 25171 |
| C_real-C_shift | BNBUSDT | 15 | +0.0436 (0.7986) | -0.00088 (0.6190) | +9.860e-10 (0.1016) | 25234 |
| C_real-C_shift | BNBUSDT | 60 | +0.1464 (0.3220) | +0.00205 (0.2564) | +2.701e-10 (0.7604) | 25268 |

### 12.2 Mechanical verdict (pre-registered rule)

Thresholds: `alpha_dir = 0.05/160 = 3.125e-4` for Δhit and ΔIC,
`alpha_vol = 0.05/80 = 6.25e-4` for ΔMSE; a claim requires >= 9 of 16 cells
improving **and** >= 9 improving with `p < alpha` for the same metric.

| comparison | Δhit improving / survivors | ΔIC improving / survivors | ΔMSE improving / survivors | verdict |
|---|---|---|---|---|
| A1-A0 | 2/16, 0 | 2/16, 0 | 6/16, 0 | **null** (GCP columns alone: negative-leaning, 0 survivors) |
| A2-A0 | 8/16, 0 | 6/16, 0 (1 significantly worse: BTCUSDT 1m, p = 0.0002) | 8/16, 1 (SOLUSDT 1m, p = 0.0002) | **null** (Schrödinger columns: no direction claim; volatility claim needs >= 9 survivors) |
| A3-A0 | 5/16, 0 | 2/16, 0 | 8/16, 1 (SOLUSDT 1m) | **null** |
| A3-A2 | 5/16, 0 | 3/16, 0 | 6/16, 0 | **null** (GCP adds nothing on top of the Schrödinger columns) |
| C_real-C_shift | 10/16, 0 | 9/16, 0 | 12/16, 0 | **null** (the GCP-driven coupling is not distinguishable from its 30-day-shifted control) |

Median per-cell deltas (row of 16): A1-A0 Δhit -0.19 pp / ΔIC -0.0031;
A2-A0 Δhit -0.01 pp / ΔIC -0.0013; A3-A0 Δhit -0.17 pp / ΔIC -0.0034;
A3-A2 Δhit -0.07 pp / ΔIC -0.0028; C_real-C_shift Δhit +0.06 pp / ΔIC
+0.00019. The smallest p-value in the primary comparison (C_real-C_shift) is
0.0420 (SOLUSDT 15m Δhit) and 0.0014 (ETHUSDT 1m ΔMSE), both far above their
thresholds. **No cell of any comparison survives Bonferroni; every
pre-registered direction and volatility claim fails its majority rule. The
verdict is a null**, unchanged by the one-shot status.

### 12.3 Power statement (measured)

The usable holdout is **25,311 minutes per asset (~17.8 days) = about 421
60-minute blocks per cell** (23,773-25,301 evaluated rows per cell), one
regime. From the block-bootstrap standard errors on these cells, the per-cell
minimum detectable effect at `alpha_dir = 3.125e-4` is roughly **1.5-2 pp in
hit rate** and a few thousandths in IC. The observed dev-scale effects
(0.1-0.5 pp) are an order of magnitude below that, so this holdout cannot
detect effects of the size the dev arm showed; a null here means "no effect of
that size is detectable in ~17.8 days", not "the effect is exactly zero". The
GCP-bounded window (the market holdout runs 16.3 h longer than the GCP cache)
costs about 0.6% of the minutes relative to the full market holdout.

### 12.4 S0 direct readout (descriptive): holdout vs dev

`sign(p_up - 0.5)` hit rate and Spearman IC of the frozen coupled arm's
`p_up`, with one-sample block-bootstrap tests against chance (the code path
was verified to pass `p_up - 0.5`, not the raw probability; see
`mix_lab.py` `direct_readout` and the reused `s0_chance_tests`). On the
holdout: h=1 flips to **above chance** (BTCUSDT 51.867%, p = 0.0002;
BNBUSDT 51.001%, p = 0.0026; SOLUSDT 50.940%, p = 0.0038; ETHUSDT 50.445%,
p = 0.166) with positive IC (+0.0281/+0.0242/+0.0096/+0.0007), while h=15/60
stays **contrarian** (47.3-48.5%, IC -0.048..-0.071, p = 0.0002 in 8/8 cells
of those horizons). On dev the same readout was contrarian in 16/16 cells
(hit 46.9-48.6%, IC -0.034..-0.077). This reproduces the sign instability
already recorded for the registered Schrödinger arm (dev contrarian, holdout
h=1 above chance) and is reported descriptively only; the direct volatility
readout (`std * sigma_h`) is worse than the EWMA baseline in 16/16 holdout
cells (p = 0.0002 each).

**Artifacts.** `results_mix_lab_holdout.json` (one-shot; the driver refuses a
second run into the same file) and `results_mix_lab_dev.json`. No live
artifacts are written by this arm, and the deployment limitation of section 8
(the live `getcurrentnetvar` endpoint returns a different quantity than the
archive netvar series) is unchanged.
