# Pre-registration: literal Schrodinger-equation wave-packet arm (`market_only` vs `market_only + SCHR`)

Status: **written before any dev result for this arm was computed.** The model
definition, the exploration grid, the selection rule, the dev evaluation
protocol, and the holdout test with its decision rule are fixed below. After
the exploration run, the log and the frozen spec are appended (sections 9-11);
nothing above them changes. The only pre-result computations were: (a) the
solver verification tests in `test_schrodinger.py` (analytic checks of the
solver, no market data), and (b) the diagnostic measurement of how often a
trailing standardized sample falls outside the KDE domain in the dev source
range (reported as the fallback-trigger count in the dev audit; it is a
property of the data, not a model choice).

## 1. Honesty statement

This arm **literally solves the one-dimensional time-dependent Schrodinger
equation** with a complex wave function, the Born rule `|psi|^2`, and a
potential derived from trailing market data. There is **no claimed physical
mechanism** linking the Schrodinger equation to crypto prices: the equation is
used as a deterministic, non-linear-in-the-potential state machine over a
standardized return coordinate. It is evaluated with the same leakage controls,
folds, models, and paired bootstrap as the rest of this repository, and a null
result is a fully acceptable outcome.

No window, constant, potential, tau map, or k0 map was changed after seeing any
evaluation output other than through the pre-declared selection rule in
section 5. The exploration log (section 9) lists every configuration that was
run with its dev result; there is no silent selection.

## 2. The equation, units, and solver

Dimensionless time-dependent Schrodinger equation

    i hbar dpsi/dt = [ -(hbar^2 / 2m) d^2/dx^2 + V(x) ] psi

with fixed conventions **hbar = m = 1** (so the kinetic phase is
`exp(-i k^2 dt / 2)`). `x` is a standardized forward log-return coordinate: one
unit of `x` is one trailing h-minute log-return standard deviation `sigma_h(t)`
defined below, and `x = 0` is the current price. `tau` is dimensionless.

Grid: `x in [-6, 6]`, `N_GRID = 193` points, `dx = 0.0625`; `x = 0` is exactly a
grid point. The box is large compared with every packet used (width
`sigma_packet <= 2`, `|k0| <= 4`, `tau <= 1`).

**Production solver (split-step Fourier, Strang splitting).** Per time step
`exp(-i V dt/2) exp(-i T dt) exp(-i V dt/2)`, with the kinetic operator exact in
Fourier space and the potential an exact phase multiplication in real space;
unitary for any `dt`. Time step `dt <= DT_MAX = 0.02` with
`n_steps = ceil(tau / DT_MAX)` and uniform `dt = tau / n_steps` (total evolution
time is exact). The kinetic phase is precomputed once per (tau, dt) and the
potential phase once per (potential, dt) per anchor batch.

**Reference solver (Crank-Nicolson).** `(I + i dt/2 H) psi^{n+1} =
(I - i dt/2 H) psi^n` on the same second-order finite-difference Hamiltonian,
Dirichlet boundaries, vectorized Thomas algorithm. Used only in
`test_schrodinger.py`, never for model output.

Verification (all in `test_schrodinger.py`, run before any dev run):

| test | check | result |
|---|---|---|
| norm conservation, split-step | `abs(sum(abs(psi)^2)*dx - 1) < 1e-8` over a smooth potential, random localized state, tau in {0.25, 1.0} | 6.7e-16, 2.0e-15 |
| norm conservation, Crank-Nicolson | same check | 1.6e-09 |
| free packet spreading | `std(tau) = sigma sqrt(1 + tau^2/(4 sigma^4))`, `mean(tau) = x0 + k0 tau` (analytic, hbar=m=1) | match to 2e-3 for tau in {0, 0.5, 1} |
| QHO coherent state | `<x>(tau) = x0 cos(omega tau) + (k0/omega) sin(omega tau)` | match to 2e-3 at tau in {0.7, 2.4} |
| QHO eigen-energies | finite-difference Hamiltonian eigenvalues vs `(n + 1/2) hbar omega` | max err 1.9e-3 (N=513), 1.2e-4 (N=2049), O(dx^2) decrease |
| solver cross-check | split-step vs Crank-Nicolson L2 difference converges with dt to the O(dx^2) floor | 4.3e-4 -> 3.1e-4, Born-rule observables agree to 4e-5 |
| phase effect | `p_up(k0=+0.8) > 0.5 > p_up(k0=-0.8)` for a free packet, antisymmetric | exact to 1e-6 |
| causality guard | non-candle-open anchor rejected; sigma_h equals the trailing rolling std of h-minute log returns ending at candle `t-60s` | pass |

## 3. Causal inputs and the wave packet (frozen model definition)

All inputs at anchor `t` use only candles with `open_time_s < t` (the newest
candle is `t - 60s`, the same convention as `features.py`).

Notation: `L_s = ln close_s` for candles `s < t`; `R_h(s) = L_s - L_{s-h}` is
the trailing h-minute log return ending at candle `s`.

* **scale**: `sigma_h(t) = std(R_h(s), s in [t-240m, t-60s], ddof=1)` over the
  trailing `KDE_WINDOW = 240` candles. Anchors with a non-finite or zero
  `sigma_h` are dropped.
* **z samples**: `z_s = R_h(s) / sigma_h(t)` for the same 240 candles. Because
  the standardization uses the same window, the sample standard deviation is 1
  by construction.
* **current state** `x0 = clip(z_newest, -4, +4)` (the latest trailing h-minute
  log return in sigma_h units).
* **packet width** `sigma_packet = clip(std of the trailing 60 R_h values /
  sigma_h, 0.5, 2.0)`.
* **phase-encoded momentum** `k0 = c_k * x0` with `c_k` from the pre-declared
  k0 map (`zero`: `c_k = 0`; `momentum`: `c_k = +1`).
* **initial packet** `psi(x,0) = N exp(-(x-x0)^2/(4 sigma_packet^2) + i k0 x)`,
  normalized by the trapezoidal integral of `|psi|^2` on the grid.
* **evolution time** `tau = h / TAU_DENOMINATORS[tau_map]` with
  `linear_60: 60`, `slow_240: 240` (h in minutes).
* **potential** `V(x)` from trailing data, one of the pre-declared families
  below (applied to the batch of anchors; the same numerical regularization for
  all of them):
  * `quantum_kde`: `p_hat` = Gaussian KDE of the 240 z samples evaluated on the
    grid (Silverman bandwidth `b = 1.06 * 240^(-1/5) = 0.3542`; histogram +
    truncated Gaussian smoothing on a grid padded by `5b` on each side, then
    truncated to the box and renormalized to unit trapezoidal mass);
    `psi0 = sqrt(max(p_hat, 1e-12 * peak(p_hat)))`;
    `V = E0 + (hbar^2/2m) psi0'' / psi0` with `E0 = 0` (an additive constant
    only adds a global phase) using the same second-difference stencil as the
    solver; clipped to `[-60, +60]`.
  * `quartic_kde`: weighted least squares fit of `-ln p_hat(x)` by
    `V = a x^2 + c x^3 + b x^4` over `|x| <= 3.5` with weights `p_hat`, no
    intercept (additive constant irrelevant), clipped to `[-60, +60]`.
  * `free_particle`: `V = 0` (honesty control; carries no trailing data in V).
  * **Degenerate fallback** (all families that use the KDE): if no sample kernel
    reaches the padded domain, the row's density is replaced by a Gaussian with
    that sample's own mean and ddof=1 standard deviation, truncated and
    renormalized on the box. In the dev source range at hourly anchors this
    fallback triggers for 0/22254 anchors at every horizon (measured before the
    model runs).
* **Born-rule observables** of `|psi(.,tau)|^2` (density renormalized by its
  trapezoidal mass first): `p_up = int_{x>0} |psi|^2 dx`,
  `mean = <x>`, `std = sqrt(<x^2> - <x>^2)`, `kurt = <(x-<x>)^4>/std^4 - 3`.

The four model columns for horizon `h` are `schr_p_up_{h}m`, `schr_mean_{h}m`,
`schr_std_{h}m`, `schr_kurt_{h}m`. No other columns from the solver enter any
model.

## 4. Arms

* **S0 (direct)** - `p_up` used with no fitted model: direction score
  `p_up` (hit rate of `sign(p_up - 0.5)`, Spearman IC against the forward
  return); volatility prediction `std * sigma_h` evaluated by MSE against the
  forward volatility target, with two baselines on the same rows: the
  registered EWMA baseline (`ewma_vol_60m * sqrt(h)`) and the trailing scale
  `sigma_h`.
* **S1 (fitted)** - `market_only` (the 14 registered `MARKET_FEATURES`) vs
  `market_only + SCHR_h` (the four columns above), identical rows, folds,
  scaler policy, and hyperparameters (L2 logistic `C=1`; ridge `alpha=1`).
* **S2 (control)** - `free_particle` wave-packet features in the S1 design.
  Reported for every exploration cell but **excluded from selection**.
* **Shuffled-target control** - `backtest.shuffled_target_control` for the
  selected config's features, BTCUSDT, all four horizons.

## 5. Exploration grid and selection rule (pre-declared)

Grid (the full cross product, 2 x 2 x 2 = 8 candidate configurations, plus one
`free_particle` control config `free_particle|linear_60|momentum`):

* potential in {`quantum_kde`, `quartic_kde`}
* tau map in {`linear_60`, `slow_240`}
* k0 map in {`zero`, `momentum`}

Dev evaluation for the grid: anchors at `target_time_s % 3600 == 0` (top of the
hour), all four assets, all four horizons; expanding walk-forward CV with
`n_folds = 3`, `initial_train_fraction = 0.50`, embargo = horizon (the
`train.py` defaults); paired moving-block bootstrap at `n_boot = 300` and
`block_len = 3` anchors (see section 6). These exploration numbers are
**ranking-only and not confirmatory** (300 replicates have a two-sided p-value
floor of `2/301 = 0.0066`, above the Bonferroni threshold, so they cannot
support a claim either way).

**Selection rule (fixed before the run).** Rank the 8 candidates by, in order:

1. the number of the 16 (asset, horizon) cells with positive S1 `Delta IC`
   (`candidate - market_only`, paired rows);
2. the pooled S1 `Delta IC` (row-weighted over the 16 cells);
3. the median per-cell S1 `Delta IC`.

The highest-ranked candidate is frozen (ties broken by the pooled `Delta IC`,
then by the median). The `free_particle` control is not selectable. All 8
configurations, the control, and their dev numbers are logged in section 9
regardless of the outcome.

## 6. Dev evaluation protocol (frozen)

* Data: `../crypto-data/crypto_market.db` only (the dev source). No Binance
  download, no GCP, no holdout.
* Anchors: `target_time_s % 3600 == 0`; every arm is evaluated on exactly these
  rows (asserted row-key equality between the S0/OOS rows and the S1 arms).
* Features/targets: `features.build_feature_frame(db, None, include_gcp=False)`
  - the registered market features, `ewma_vol_60m`, and the registered targets
  `forward_return_{h}m`, `forward_vol_{h}m` are reused unchanged.
* Folds: `backtest.make_walk_forward_splits` (expanding, `n_folds=3`,
  `initial_train_fraction=0.50`, embargo = horizon).
* Models: `backtest.walk_forward_predict` (per-fold `StandardScaler`,
  L2 logistic `C=1` / ridge `alpha=1`).
* Paired bootstrap: `backtest.paired_bootstrap_delta`, block length **3
  anchors**. The repository convention is 60 *rows* on minute data; with
  hourly anchors a 60-row block spans 60 hours, so the pre-declared block
  length is 3 anchors (3 hours), which preserves short-range dependence
  without windowing across regimes. Seeds `20260914 + h` (hit rate),
  `20260915 + h` (IC), `20260916 + h` (MSE), as in `train.py`.
* Resolution rule: any p-value compared with `0.05/16 = 0.003125` uses
  `n_boot >= 4999` (two-sided floor `2/5000 = 0.0004`). The exploration grid
  uses 300 replicates for ranking only; the frozen dev table (section 11) uses
  `n_boot = 4999` for hit rate, IC, and MSE.
* Causality: `features.assert_causal_features` on the frame, plus the
  solver-side assertions in `schrodinger.build_features_for_asset` (sample
  candles strictly before the anchor, sigma_h window ending at `t - 60s`).
* S0 tests (added by Amendment A1, still before any holdout data): for the
  direction readout `p_up` the frozen table reports one-sample moving-block
  bootstrap tests against chance -- hit rate vs 0.5 via
  `backtest._block_mean_bootstrap` applied to the per-row hit indicator minus
  0.5, and Spearman IC vs 0 via the same block resampling used by the paired
  IC branch (`backtest._block_sample_indices` + `backtest._spearman`) -- and
  paired MSE bootstraps of the S0 volatility prediction `std * sigma_h`
  against the EWMA and the trailing-scale baselines.  These reuse the
  repository's resampling algorithms and the same block length, seeds
  (`20260914+h`, `20260915+h`, `20260916+h`, `20260917+h`), and `n_boot` as the
  paired tests.  No second bootstrap convention is introduced.

## 7. Holdout test and decision rule (pre-registered)

> **Amended by A2 (pre-holdout, 2026-10-02):** the S1 arms are now fitted
> **once on all dev rows** and frozen before scoring the holdout, instead of
> re-running walk-forward folds inside the holdout; the evaluation start,
> warm-up rule, and assertions are specified in Amendment A2 (section 12).
> The decision rule below is unchanged.

The holdout is an untouched source range (Sept 2026 -> now) that will be
provided as a separate SQLite database **after this specification is frozen**.
The database must expose the same `klines_1m` schema and the four registered
assets; `features.market_db_audit` verifies it before any computation, and the
same `features.build_feature_frame` loader and target formulas are used.  The
frozen configuration, constants, arms, models, folds, targets, and metrics
are exactly those of section 6; the only protocol differences are the data
source, the anchor cadence, and the resulting bootstrap resolution:

* Anchors: **every minute** (`--anchor-minutes 1`). Rationale: the holdout is
  about one month long, so minute anchors are computationally affordable for a
  single frozen configuration (roughly 43k anchors per asset, a few minutes of
  PDE time) and give the same per-cell row counts as the main pipeline. The
  model is defined per anchor and does not depend on the anchor cadence.
* Bootstrap: `block_len = 60` rows (the repository convention, now matching the
  minute anchor cadence) and `n_boot = 4999` for hit rate, IC, and MSE.
* Cells: 4 assets x 4 horizons = 16 cells; deltas are candidate minus
  `market_only` on exactly paired OOS rows.
* Shuffled-target control: BTCUSDT, all four horizons, candidate features.
* Bonferroni threshold: `0.05 / 16 = 0.003125`.
* Decision rule (identical to the pre-registered physics arm):
  * the direction claim ("the Schrodinger arm adds skill") requires **at least
    9 of 16 cells** with a positive `Delta hit` or `Delta IC` and **at least
    9 of 16** positive with bootstrap `p < 0.003125` for that metric;
  * the volatility claim is separate and requires at least 9 of 16 cells with
    `Delta MSE < 0` and at least 9 of 16 negative with
    `p < 0.003125`;
  * otherwise the result is reported as a **null** ("no evidence the arm adds
    skill").
* Secondary pre-registered family (S0 direct, Amendment A1): a cell supports
  S0 direction if the chance test (`p_up` hit rate vs 0.5, or IC vs 0) is
  positive in sign with `p < 0.003125`; a cell supports S0 volatility if the
  paired MSE against the EWMA baseline is negative with `p < 0.003125`.  The
  S0 family "claim" requires at least 9 of 16 cells supporting in the same
  direction, and is only reported as meaningful if the primary S1 family is
  consistent (a second family doubles the number of chances, so an isolated
  S0 family survivor with a flat S1 family is reported as ambiguous, not as a
  claim).
* Secondary, descriptive only (no claim): pooled row-weighted `Delta hit` and
  `Delta IC` over all 16 cells with the same bootstrap.
* Power statement (pre-registered honesty): with one month of holdout and
  strongly dependent minute rows, the per-cell effective independent sample is
  roughly the number of 60-minute blocks (about 700), so the holdout can only
  detect effects on the order of a percentage point in hit rate or a few
  thousandths of IC.  Sixty-minute blocks on a one-month sample also understate
  multi-day regime dependence (measured on dev short windows: the fitted arms'
  own fold-to-fold OOS IC swings by tenths on 4-day windows), so the holdout
  p-values are anti-conservative and any survivor must be treated as weak.  A
  null therefore means "not detectable in one month of this data", not "the
  arm is harmless"; a positive isolated cell means little unless the majority
  rule is met.
* Robustness (descriptive only, Amendment A1): the same holdout statistics are also
  reported with 1440-row (one-day) blocks, which brackets the multi-day
  dependence; with about 30 day-blocks per cell the two-sided p-value floor is
  `2/31 ~ 0.065`, i.e. that view can never survive Bonferroni and is reported
  as a sign/robustness check only.

## 8. Reproduce

```bash
price-prediction/.venv/bin/python price-prediction/test_schrodinger.py
price-prediction/.venv/bin/python price-prediction/train_schrodinger.py --stage explore   # dev exploration grid
price-prediction/.venv/bin/python price-prediction/train_schrodinger.py --stage frozen    # frozen config on dev
# pre-registered holdout run, Amendment A2 (fits once on all dev rows; features
# on the last 2 days of dev + holdout; evaluates every holdout minute from
# max(first holdout minute, dev end) + 300 minutes):
price-prediction/.venv/bin/python price-prediction/train_schrodinger.py --stage holdout \
    --db ../crypto-data/crypto_market.db \
    --holdout-db ../crypto-data/holdout_klines_1m.db \
    --output price-prediction/results_schrodinger_holdout.json
```

## 9. Exploration log (filled after the exploration run)

Run: `train_schrodinger.py --stage explore` on `../crypto-data/crypto_market.db`,
anchors `target_time_s % 3600 == 0` (22,249 feature rows per asset per horizon
after warm-up), 9 configurations (8 candidates + the free-particle control),
16 cells each, 300-replicate paired block bootstrap (ranking only; two-sided
p-value floor `2/301 = 0.0066`, above the Bonferroni threshold, so no
exploration p-value can support a claim).  Feature construction (all 9 configs,
one KDE/potential pass per asset/horizon): **1646.8 s**; feature+evaluation
runtime: **2364.7 s** (~39 min, 8 cores, single process).  KDE
degenerate-fallback rows: **0** (measured 0 in every one of the 36
asset/horizon/config batches and 0/22249 anchors per batch, `4 assets x 4
horizons` anchor batches).  Every configuration that was run is listed below;
the full per-cell deltas for every configuration are in
`results_schrodinger_dev.json` under `exploration.configs`.

### 9.1 Grid summary (all 9 configurations, dev, hourly anchors)

`pos ΔIC` = cells (of 16) with positive S1 `Delta IC`; `pooled ΔIC` =
row-weighted pooled S1 `Delta IC`; `med ΔIC`, `med Δhit`, `med ΔMSE` = medians
over the 16 cells; `neg ΔMSE` = cells with negative S1 `Delta MSE` (better).

| config | pos ΔIC | pooled ΔIC | med ΔIC | pos Δhit | neg ΔMSE | med Δhit (pp) | med ΔMSE | runtime s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `quartic_kde|linear_60|momentum` (SELECTED) | 11/16 | +0.00087 | +0.00178 | 9/16 | 8/16 | +0.086 | -1.615e-11 | 102 |
| `quartic_kde|slow_240|momentum` | 11/16 | +0.00035 | +0.00064 | 10/16 | 11/16 | +0.055 | -1.539e-09 | 82 |
| `quartic_kde|linear_60|zero` | 10/16 | +0.00113 | +0.00151 | 10/16 | 9/16 | +0.126 | -2.845e-10 | 88 |
| `quantum_kde|slow_240|momentum` | 10/16 | +0.00036 | +0.00093 | 9/16 | 11/16 | +0.091 | -1.517e-09 | 74 |
| `quartic_kde|slow_240|zero` | 10/16 | +0.00024 | +0.00058 | 9/16 | 12/16 | +0.059 | -3.624e-09 | 85 |
| `quantum_kde|linear_60|zero` | 9/16 | +0.00062 | +0.00051 | 8/16 | 11/16 | +0.041 | -6.167e-10 | 73 |
| `quantum_kde|slow_240|zero` | 9/16 | +0.00050 | +0.00082 | 9/16 | 12/16 | +0.122 | -3.425e-09 | 64 |
| `quantum_kde|linear_60|momentum` | 9/16 | +0.00049 | +0.00099 | 9/16 | 9/16 | +0.108 | -1.747e-10 | 70 |
| `free_particle|linear_60|momentum` (control, not selectable) | 9/16 | +0.00012 | +0.00056 | 8/16 | 10/16 | -0.004 | -7.670e-10 | 78 |

Selection outcome: by the pre-declared rule (positive-ΔIC cell count, then
pooled ΔIC, then median ΔIC) the frozen configuration is
**`quartic_kde|linear_60|momentum`**.  The spread of all eight candidates is
within a few tenths of a basis point of pooled ΔIC and the free-particle
control is indistinguishable from the fitted potentials on this statistic, so
the selection is a choice among effectively-tied configurations, not evidence
that the selected potential is special.  This is recorded here explicitly.

### 9.2 Frozen-config per-cell dev log (300-replicate ranking bootstrap)

Deltas are candidate minus `market_only`; `S0` columns are the direct
Born-rule readout (`S0 hit%` = hit rate of `sign(p_up - 0.5)`, corrected by
Amendment A3); `n` is the paired direction row count.

| cell | Δhit (pp) | p | ΔIC | p | ΔMSE | p | S0 hit% | S0 IC | S0 MSE | EWMA MSE | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BTCUSDT_1m | -0.1740 | 0.4518 | +0.00139 | 0.5449 | -8.206e-11 | 0.8970 | 47.779 | -0.06515 | 2.390e-07 | 2.261e-07 | 10917 |
| BTCUSDT_5m | -0.1351 | 0.5847 | -0.00217 | 0.2724 | -2.966e-09 | 0.0066 | 48.730 | -0.03134 | 5.513e-07 | 4.848e-07 | 11100 |
| BTCUSDT_15m | +0.2428 | 0.2990 | +0.00445 | 0.0864 | +3.174e-10 | 0.7641 | 48.066 | -0.04756 | 1.688e-06 | 9.731e-07 | 11118 |
| BTCUSDT_60m | -0.2697 | 0.3056 | -0.00476 | 0.1395 | +4.155e-09 | 0.0266 | 48.786 | -0.01931 | 1.125e-05 | 3.489e-06 | 11124 |
| ETHUSDT_1m | +0.2082 | 0.5050 | +0.00267 | 0.3522 | +7.802e-10 | 0.4053 | 47.534 | -0.07358 | 5.222e-07 | 4.946e-07 | 11049 |
| ETHUSDT_5m | -0.1711 | 0.4252 | -0.00406 | 0.0465 | -7.173e-09 | 0.0066 | 49.478 | -0.02065 | 1.232e-06 | 1.110e-06 | 11106 |
| ETHUSDT_15m | +0.2609 | 0.2525 | +0.00229 | 0.4784 | -7.261e-09 | 0.2392 | 47.674 | -0.04097 | 3.903e-06 | 2.545e-06 | 11115 |
| ETHUSDT_60m | +0.0540 | 0.9568 | +0.00217 | 0.4718 | +1.452e-09 | 0.6379 | 48.390 | -0.01798 | 2.474e-05 | 9.138e-06 | 11118 |
| SOLUSDT_1m | -0.6843 | 0.0399 | -0.00009 | 0.9635 | +2.335e-09 | 0.2724 | 48.603 | -0.04427 | 6.935e-07 | 6.572e-07 | 10668 |
| SOLUSDT_5m | +0.1188 | 0.5449 | +0.00020 | 0.9834 | -1.266e-08 | 0.0066 | 49.392 | -0.01900 | 1.661e-06 | 1.414e-06 | 10941 |
| SOLUSDT_15m | +0.1907 | 0.3189 | -0.00401 | 0.0266 | +2.029e-09 | 0.7907 | 48.283 | -0.03277 | 4.815e-06 | 3.000e-06 | 11010 |
| SOLUSDT_60m | +0.2080 | 0.5515 | +0.00385 | 0.3056 | +1.081e-08 | 0.0066 | 48.580 | -0.01856 | 3.153e-05 | 9.562e-06 | 11058 |
| BNBUSDT_1m | -0.0189 | 0.8638 | +0.00251 | 0.5515 | +6.402e-12 | 0.8704 | 49.731 | -0.01730 | 3.718e-07 | 3.406e-07 | 10599 |
| BNBUSDT_5m | +0.3853 | 0.1262 | +0.00078 | 0.7375 | -5.107e-10 | 0.8439 | 49.468 | -0.00880 | 1.067e-06 | 7.553e-07 | 10902 |
| BNBUSDT_15m | -0.0182 | 0.9568 | +0.00379 | 0.0598 | -7.428e-09 | 0.0199 | 48.072 | -0.03152 | 2.691e-06 | 1.658e-06 | 10998 |
| BNBUSDT_60m | +0.2443 | 0.3588 | +0.00491 | 0.2525 | -3.871e-11 | 0.9037 | 48.598 | -0.02865 | 1.819e-05 | 7.888e-06 | 11052 |

Reading of the exploration (descriptive, not confirmatory): all deltas are of
order a tenth of a percentage point in hit rate and a few thousandths in IC;
no exploration p-value is below the 0.0066 floor, as expected with 300
replicates; the direct S0 readout is below chance in hit rate in **16/16**
cells (47.5-49.7%) with a **negative** IC in **16/16** cells -- the Born-rule
`p_up` of this configuration behaves like a weak contrarian at hourly anchors
(corrected S0 numbers, Amendment A3).

## 10. Frozen specification (filled after the exploration run)

Chosen by the pre-declared rule and **frozen** (implemented as `FROZEN_CONFIG`
in `schrodinger.py`; the holdout run cannot override it without an explicit
`--config`, which the pre-registered holdout command does not use):

* potential: `quartic_kde` (weighted quartic fit of `-log p_hat` on
  `|x| <= 3.5`, weights `p_hat`, no intercept, clipped to `+-60`);
* tau map: `linear_60` (`tau = h / 60`);
* k0 map: `momentum` (`k0 = +x0`);
* everything else as in sections 2-3: `hbar = m = 1`, grid `[-6, 6]` with 193
  points, split-step Fourier with `dt <= 0.02`, KDE window 240 with Silverman
  bandwidth `0.3542`, `sigma_h` from the trailing 240 h-minute log returns,
  `x0 = clip(z_newest, +-4)`, `sigma_packet = clip(std_60/sigma_h, 0.5, 2)`,
  Born-rule readouts `p_up`, `mean`, `std`, `kurt` per horizon, arms S0
  (direct) and S1 (market_only vs market_only + SCHR_h), folds/models/
  bootstrap as in section 6.

Frozen config label: `quartic_kde|linear_60|momentum`.  The pre-registered
holdout command (section 8) uses exactly this configuration.

## 11. Dev results for the frozen configuration (filled after the run)

Run: `train_schrodinger.py --stage frozen` (frozen `quartic_kde|linear_60|momentum`,
dev anchors hourly, `n_boot = 4999` for hit rate, IC, and MSE, block length 3
anchors, seeds as in section 6), merged into
`results_schrodinger_dev.json`.  Runtime **1506 s** (~25 min); 22,249 feature
rows per asset/horizon; KDE degenerate-fallback rows **0**.

Deltas are candidate minus `market_only` on paired OOS rows; `pp` = percentage
points.  `S0 hit% (p)` is the direct Born-rule `p_up` hit rate with its
one-sample block-bootstrap p-value against chance 0.5; `S0 IC (p)` is the
direct Spearman IC with its p-value against 0; `S0 vol MSE (p)` is the direct
`std * sigma_h` volatility MSE with the paired p-value against the EWMA
baseline (all with 4999 replicates).

| asset | h | cand hit% [95% CI] | cand IC | Δhit pp (p) | ΔIC (p) | vol MSE | EWMA MSE | ΔMSE (p) | S0 hit% (p vs chance) | S0 IC (p vs 0) | S0 vol MSE (p vs EWMA) | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BTCUSDT | 1 | 52.350 [51.412,53.285] | +0.06691 | -0.1740 (0.5156) | +0.00139 (0.5428) | 2.124e-07 | 2.261e-07 | -8.206e-11 (0.9072) | 47.779 (0.0004) | -0.06515 (0.0004) | 2.390e-07 (0.0004) | 10917 |
| BTCUSDT | 5 | 51.838 [50.908,52.767] | +0.05158 | -0.1351 (0.5380) | -0.00217 (0.2232) | 4.660e-07 | 4.848e-07 | -2.966e-09 (0.0072) | 48.730 (0.0080) | -0.03134 (0.0012) | 5.513e-07 (0.0004) | 11100 |
| BTCUSDT | 15 | 52.518 [51.590,53.446] | +0.05599 | +0.2428 (0.3488) | +0.00445 (0.1128) | 9.296e-07 | 9.731e-07 | +3.174e-10 (0.7876) | 48.066 (0.0008) | -0.04756 (0.0004) | 1.688e-06 (0.0004) | 11118 |
| BTCUSDT | 60 | 51.286 [50.356,52.214] | +0.02220 | -0.2697 (0.3372) | -0.00476 (0.0972) | 3.135e-06 | 3.489e-06 | +4.155e-09 (0.0156) | 48.786 (0.0088) | -0.01931 (0.0352) | 1.125e-05 (0.0004) | 11124 |
| ETHUSDT | 1 | 52.448 [51.516,53.378] | +0.07659 | +0.2082 (0.5036) | +0.00267 (0.3568) | 4.533e-07 | 4.946e-07 | +7.802e-10 (0.3500) | 47.534 (0.0004) | -0.07358 (0.0004) | 5.222e-07 (0.0004) | 11049 |
| ETHUSDT | 5 | 51.945 [51.015,52.873] | +0.04730 | -0.1711 (0.4516) | -0.00406 (0.0184) | 1.030e-06 | 1.110e-06 | -7.173e-09 (0.0044) | 49.478 (0.2832) | -0.02065 (0.0308) | 1.232e-06 (0.0004) | 11106 |
| ETHUSDT | 15 | 52.812 [51.883,53.738] | +0.04589 | +0.2609 (0.2900) | +0.00229 (0.3980) | 2.396e-06 | 2.545e-06 | -7.261e-09 (0.2328) | 47.674 (0.0004) | -0.04097 (0.0004) | 3.903e-06 (0.0004) | 11115 |
| ETHUSDT | 60 | 51.610 [50.681,52.538] | +0.02477 | +0.0540 (0.8596) | +0.00217 (0.5548) | 7.683e-06 | 9.138e-06 | +1.452e-09 (0.6628) | 48.390 (0.0008) | -0.01798 (0.0560) | 2.474e-05 (0.0004) | 11118 |
| SOLUSDT | 1 | 52.043 [51.095,52.991] | +0.05775 | -0.6843 (0.0256) | -0.00009 (0.9844) | 6.305e-07 | 6.572e-07 | +2.335e-09 (0.2648) | 48.603 (0.0056) | -0.04427 (0.0004) | 6.935e-07 (0.0004) | 10668 |
| SOLUSDT | 5 | 52.043 [51.106,52.978] | +0.04454 | +0.1188 (0.6208) | +0.00020 (0.9484) | 1.361e-06 | 1.414e-06 | -1.266e-08 (0.0012) | 49.392 (0.2032) | -0.01900 (0.0552) | 1.661e-06 (0.0004) | 10941 |
| SOLUSDT | 15 | 52.234 [51.301,53.166] | +0.02911 | +0.1907 (0.3800) | -0.00401 (0.0248) | 2.819e-06 | 3.000e-06 | +2.029e-09 (0.7072) | 48.283 (0.0004) | -0.03277 (0.0012) | 4.815e-06 (0.0004) | 11010 |
| SOLUSDT | 60 | 51.293 [50.361,52.224] | +0.01953 | +0.2080 (0.5476) | +0.00385 (0.3752) | 8.501e-06 | 9.562e-06 | +1.081e-08 (0.0024) | 48.580 (0.0020) | -0.01856 (0.0552) | 3.153e-05 (0.0004) | 11058 |
| BNBUSDT | 1 | 50.278 [49.327,51.230] | +0.03345 | -0.0189 (0.9472) | +0.00251 (0.4880) | 3.146e-07 | 3.406e-07 | +6.402e-12 (0.9312) | 49.731 (0.5648) | -0.01730 (0.0724) | 3.718e-07 (0.0004) | 10599 |
| BNBUSDT | 5 | 51.559 [50.621,52.497] | +0.04289 | +0.3853 (0.1320) | +0.00078 (0.7688) | 6.762e-07 | 7.553e-07 | -5.107e-10 (0.8544) | 49.468 (0.2600) | -0.00880 (0.3648) | 1.067e-06 (0.0004) | 10902 |
| BNBUSDT | 15 | 51.982 [51.048,52.915] | +0.04147 | -0.0182 (0.9668) | +0.00379 (0.0848) | 1.426e-06 | 1.658e-06 | -7.428e-09 (0.0200) | 48.072 (0.0008) | -0.03152 (0.0008) | 2.691e-06 (0.0004) | 10998 |
| BNBUSDT | 60 | 51.294 [50.362,52.225] | +0.01881 | +0.2443 (0.4108) | +0.00491 (0.2224) | 6.731e-06 | 7.888e-06 | -3.871e-11 (0.9516) | 48.598 (0.0016) | -0.02865 (0.0024) | 1.819e-05 (0.0004) | 11052 |

**S1 (primary) dev verdict (pre-registered rule).** `Delta hit`: 9/16 improving,
0/16 Bonferroni survivors, median +0.086 pp.  `Delta IC`: 11/16 improving, 0/16
survivors, median +0.00178.  `Delta MSE`: 8/16 improving, **1/16 survivor**
(SOLUSDT 5m, -1.266e-08, p = 0.0012) and **1/16 significantly worse**
(SOLUSDT 60m, +1.081e-08, p = 0.0024).  Verdict: **null for direction** ("no
evidence the arm adds skill"), volatility mixed.  These are dev numbers (the
config was selected on dev), so they are descriptive, not confirmatory.

**S0 (direct, secondary) dev reading.**  The direct Born-rule readout is a weak
*contrarian*: the `p_up` hit rate (of `sign(p_up - 0.5)`) is below 50% in
**16/16** cells (47.5-49.7%) and significantly below chance in **9/16**
(p <= 0.0020, several at the 0.0004 floor); the IC is negative in **16/16**
cells (significantly negative in 9/16, e.g. BTC 1m IC -0.0651).  The direct
volatility readout is worse than the EWMA baseline in **16/16** cells
(significantly worse in 16/16) and better than the trailing `sigma_h` scale in
8/16 cells (significantly better in 1).  So the wave function does carry a
small, detectable signal on dev, but with the opposite sign to the market
models and with no incremental value once the market columns are present.
(These are the corrected S0 numbers from Amendment A3; the pre-fix run had
reported the fraction of positive returns in the hit-rate fields because
`backtest.direction_metrics` expects a signed score and `sign(p_up) = +1`.)

**Why the incremental value is ~0 (diagnostic).**  The Spearman correlation of
`schr_p_up` with the existing registered `ret_1m` is **0.965** at h = 1
(0.331 at 5m, 0.170 at 15m, 0.046 at 60m): for the shortest horizon the
evolution time `tau = 1/60` is so short that the Born-rule readout is
essentially a monotone transform of the previous one-minute return (the phase
`k0 = x0` adds only a small translation).  The evolved width correlates only
weakly with the registered realized volatility (0.343 at h = 1, 0.083 at h = 60).
The four Schrodinger columns are therefore close to a re-parameterization of
information the market arm already spans, which is the same conclusion the
prior physics analogue arm reached.

**Shuffled-target control** (BTCUSDT, frozen features, all horizons): hit rate
0.494-0.505, IC -0.0029 to +0.0140 -- at chance, as required.

**Artifacts.**  `results_schrodinger_dev.json` contains, under
`exploration`, the per-cell log for all 9 configurations (300-replicate
ranking bootstrap) and the full 4999-replicate frozen table under `assets`
(+ `ablation`, `ablation_summary`, `diagnostics`, `leakage_checks`,
`protocol`).  No model artifacts (`*.joblib`) are written by this arm, so the
live loop is untouched.

## 12. Amendment log

**A1 (2026-10-02, before the frozen spec was written and before any holdout
data was touched).** Added: (i) the S0 direct-arm chance tests and paired
volatility tests (section 6 bullet, section 7 secondary family), reusing the
repository's resampling primitives with the same block lengths, seeds, and
replicate counts; (ii) the anti-conservatism caveat and the descriptive
1440-row-block robustness view in section 7. Reasons: a dry run of the holdout
protocol on a 4-day dev slice (1-minute anchors, `--stage frozen --config ...`
with 199 replicates) showed that on short windows the fitted arms' own
fold-to-fold OOS IC swings by tenths (ETHUSDT 60m: fold IC +0.51, -0.39,
-0.12, pooled -0.25, both arms at ~38% hit rate), so a 60-minute-block
Bonferroni test on a one-month holdout cannot be read as a clean significance
statement, and the direct S0 readout -- which involves no fitting and is
therefore not exposed to that instability -- needs its own pre-registered
test. No model definition, constant, feature, arm, fold construction,
dev-protocol item, or decision threshold was changed.

**A2 (pre-holdout, 2026-10-02).** The S1 fitting protocol and the evaluation
start are amended, and the holdout source is now specified
(`../crypto-data/holdout_klines_1m.db`, 26,584 contiguous minutes per asset,
2026-09-14T05:09Z -> 2026-10-02T16:12Z, directly following the dev source).
The frozen configuration, features, arms, metrics, Bonferroni threshold, and
majority decision rule are unchanged.

*Reason.* The original section-7 protocol re-ran the walk-forward folds
*inside* the holdout, so each S1 fold would have been trained on roughly nine
days of holdout data.  That wastes the 2.5 years of dev history, produces the
unstable short-window fits measured in A1 (fold-to-fold OOS IC swings by
tenths on 4-day dev windows), and is a weaker test than fitting on all the dev
data.

*Amended holdout protocol* (implemented as `--stage holdout`; see section 8
for the exact command):

* **S1 training.** Both arms (`market_only` and `market_only + SCHR_h`) are
  fitted **once** on all dev rows at the frozen dev anchor cadence (hourly
  anchors) whose horizon-`h` targets finish before the holdout's first candle
  (`target_time_s + h minutes <= holdout_start`), using the house complete-case
  policy and the house models (per-arm `StandardScaler` + L2 logistic `C=1` /
  ridge `alpha=1`).  The scalers and coefficients are then frozen; no holdout
  row, feature, or target enters the fit.
* **S1 scoring.** Every holdout minute from the evaluation start is scored by
  both frozen arms; the paired deltas (`Delta hit`, `Delta IC`, `Delta MSE`)
  use identical rows, the paired 60-row moving-block bootstrap, and
  `n_boot = 4999` (two-sided p-value floor `0.0004`, below `0.05/16`).  The
  decision rule is unchanged.
* **S0.** Nothing is fitted: the direct Born-rule readout is evaluated on the
  same holdout rows (hit rate vs chance 0.5 and IC vs 0 by one-sample block
  bootstrap, paired volatility MSE vs the EWMA baseline), as in A1.
* **Warm-up and boundary.** Features are built on the last `--dev-tail-days`
  (2) days of dev candles concatenated with the holdout so trailing windows
  cross the source boundary.  The exact warm-up is `max_h (h + KDE_WINDOW) =
  300` trailing candles (60 + 240), derived from the builder's rule
  `last >= h + KDE_WINDOW - 1`; evaluation therefore starts at
  `max(first holdout minute, dev end) + 300 minutes`.  Assertions: the holdout
  source directly follows the dev source with no gap; every fitted target ends
  before the holdout start; all market and Schrodinger features are finite at
  the first evaluated row; the holdout database is opened read-only.
* **Shuffled-target control.** Retained in the amended form: the candidate arm
  is refit on a permutation of the *training* targets and scored on the same
  holdout rows (BTCUSDT, all horizons).
* The A1 1440-row-block view remains available as a descriptive robustness
  check and is not part of any claim.

*What did not change.* The frozen configuration (`quartic_kde|linear_60|momentum`),
the feature definitions and constants, the four S1 feature columns, the arms,
the metric definitions, the Bonferroni threshold, the majority decision rule,
the bootstrap module and seeds, the 60-row block convention at 1-minute
anchors, and the dev results.

*Validation before the one-shot run.* The amended stage was exercised end to
end on artificial dev/holdout splits built from dev data only (no holdout
access); the real holdout source was opened for the first time by the
single pre-registered run whose results are in section 13.

## 13. Holdout results (one-shot, Amendment A2)

**Run.** `--stage holdout` (section 8) on
`../crypto-data/holdout_klines_1m.db` (read-only; md5
`43c17bc93d811c7ab9c7e3c278e4e961` at run time) against the dev source
`../crypto-data/crypto_market.db`.  Bounds: dev end `1789362480`
(2026-09-14 05:08Z), holdout `1789362540..1790957520`
(2026-09-14 05:09Z -> 2026-10-02 16:12Z, 26,584 contiguous minutes per asset),
exact warm-up 300 minutes, evaluation start `1789380540`.  Fits: 21,838-22,247
hourly dev rows per asset/horizon (max fitted target end `1789362300` <
holdout start).  Evaluated holdout rows per cell: 24,694-26,254.  Runtime
**2739.5 s** (~46 min).  Assertions recorded in
`results_schrodinger_holdout.json`: holdout directly follows dev; every fitted
target ends before the holdout; all 35 market + Schrodinger features are
finite at the first evaluated row; sample candles strictly before each anchor.

One-shot status: this is the only run that produced holdout results.  A first
launch was terminated by the environment during the **dev** training-frame
build, before any holdout feature, fit, or metric had been computed; it was
restarted unchanged.  The holdout database was never modified.

S1 deltas are candidate minus `market_only` on paired rows (4999-replicate
paired moving-block bootstrap, 60-row blocks; two-sided floor 0.0004);
`S0 hit% (p)` is the direct Born-rule readout's hit rate with its one-sample
chance test, `S0 IC (p)` its Spearman IC vs 0, `S0 MSE p` the paired EWMA
comparison (see Amendment A3 for how the S0 hit rate is obtained).

| asset | h | cand hit% [95% CI] | cand IC | Δhit pp (p) | ΔIC (p) | vol MSE | EWMA MSE | ΔMSE (p) | S0 hit% (p vs chance) | S0 IC (p vs 0) | S0 MSE-vs-EWMA p | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BTCUSDT | 1 | 48.845 [48.23,49.46] | -0.01089 | -0.2474 (0.1400) | -0.00649 (0.0004) | 1.266e-07 | 1.410e-07 | -2.899e-11 (0.9460) | 51.909 (0.0004) | +0.02768 (0.0008) | 0.0004 | 25464 |
| BTCUSDT | 5 | 51.600 [50.99,52.20] | +0.03746 | +0.0191 (0.8928) | -0.00011 (0.9152) | 3.052e-07 | 3.138e-07 | -3.743e-09 (0.0240) | 48.743 (0.0044) | -0.02764 (0.0160) | 0.0004 | 26217 |
| BTCUSDT | 15 | 52.266 [51.66,52.87] | +0.05467 | +0.3352 (0.0868) | +0.00730 (0.0340) | 6.570e-07 | 6.743e-07 | +5.458e-10 (0.8240) | 47.894 (0.0052) | -0.04685 (0.0076) | 0.0004 | 26254 |
| BTCUSDT | 60 | 53.847 [53.24,54.45] | +0.08828 | +0.3242 (0.4884) | +0.00297 (0.7512) | 2.038e-06 | 2.345e-06 | +2.082e-09 (0.3612) | 47.538 (0.0496) | -0.06926 (0.0252) | 0.0004 | 26215 |
| ETHUSDT | 1 | 49.998 [49.39,50.61] | +0.00873 | -0.2391 (0.1664) | -0.00322 (0.0280) | 2.241e-07 | 2.488e-07 | -6.927e-10 (0.1856) | 50.445 (0.1400) | -0.00017 (0.9872) | 0.0004 | 25931 |
| ETHUSDT | 5 | 52.008 [51.40,52.61] | +0.04658 | -0.1487 (0.2020) | +0.00003 (0.9764) | 5.833e-07 | 5.747e-07 | -4.485e-09 (0.1128) | 49.033 (0.0368) | -0.03170 (0.0068) | 0.0004 | 26221 |
| ETHUSDT | 15 | 53.307 [52.70,53.91] | +0.07874 | +0.1677 (0.3808) | +0.00971 (0.0036) | 1.294e-06 | 1.237e-06 | +3.692e-09 (0.5420) | 47.565 (0.0016) | -0.06367 (0.0004) | 0.0004 | 26244 |
| ETHUSDT | 60 | 53.011 [52.41,53.61] | +0.07526 | -0.5037 (0.3148) | -0.00390 (0.7276) | 3.709e-06 | 4.119e-06 | -3.626e-11 (0.9868) | 47.814 (0.0920) | -0.05974 (0.0464) | 0.0004 | 26208 |
| SOLUSDT | 1 | 49.575 [48.95,50.20] | -0.00064 | -0.0688 (0.7232) | -0.00229 (0.2596) | 3.479e-07 | 3.748e-07 | -2.731e-09 (0.0004) | 50.939 (0.0016) | +0.00824 (0.2252) | 0.0004 | 24694 |
| SOLUSDT | 5 | 51.283 [50.67,51.89] | +0.02843 | +0.1560 (0.3200) | -0.00034 (0.8912) | 8.320e-07 | 7.801e-07 | -3.778e-09 (0.4132) | 49.485 (0.2780) | -0.01776 (0.1116) | 0.0004 | 25648 |
| SOLUSDT | 15 | 52.563 [51.95,53.17] | +0.05969 | +0.1544 (0.1060) | +0.00278 (0.0032) | 1.624e-06 | 1.563e-06 | +5.340e-09 (0.3136) | 47.379 (0.0004) | -0.04846 (0.0096) | 0.0004 | 25906 |
| SOLUSDT | 60 | 52.694 [52.09,53.30] | +0.07826 | -0.6179 (0.2712) | -0.00794 (0.5568) | 4.554e-06 | 5.014e-06 | +4.601e-09 (0.1440) | 47.686 (0.0696) | -0.05327 (0.0848) | 0.0004 | 26058 |
| BNBUSDT | 1 | 49.651 [49.04,50.26] | -0.01109 | -0.0741 (0.6600) | -0.00234 (0.1660) | 1.435e-07 | 1.523e-07 | -6.781e-10 (0.0004) | 51.040 (0.0012) | +0.02344 (0.0012) | 0.0004 | 25631 |
| BNBUSDT | 5 | 50.849 [50.24,51.46] | +0.02169 | +0.0191 (0.9396) | -0.00135 (0.7936) | 3.565e-07 | 3.217e-07 | +6.248e-10 (0.6500) | 50.126 (0.8000) | -0.00372 (0.7464) | 0.0004 | 26134 |
| BNBUSDT | 15 | 51.733 [51.13,52.34] | +0.06014 | -0.2787 (0.1876) | +0.00066 (0.8868) | 6.956e-07 | 6.266e-07 | +5.886e-10 (0.8476) | 48.526 (0.0612) | -0.04990 (0.0040) | 0.0004 | 26190 |
| BNBUSDT | 60 | 52.353 [51.75,52.96] | +0.06570 | +0.4240 (0.4508) | +0.00730 (0.5784) | 1.905e-06 | 1.941e-06 | -8.213e-10 (0.8344) | 47.597 (0.0668) | -0.06855 (0.0272) | 0.0004 | 26180 |

**Mechanical verdict (pre-registered rule, Bonferroni `0.05/16 = 0.003125`).**

* `Delta hit`: **8/16** improving, **0/16** Bonferroni survivors (median
  -0.0249 pp) -> direction hit claim not met.
* `Delta IC`: **7/16** improving, **0/16** survivors, 1/16 significantly worse
  (BTCUSDT 1m, -0.00649, p = 0.0004; median -0.000226) -> direction IC claim
  not met.
* `Delta MSE`: **9/16** improving, **2/16** survivors (SOLUSDT 1m
  -2.731e-09 and BNBUSDT 1m -6.781e-10, both p = 0.0004), 0 significantly
  worse (median -1.4e-11).  The pre-registered volatility claim requires at
  least 9 improving **and** at least 9 surviving; only 2 survive -> not met.
* **Verdict: null.** No evidence that the market + Schrodinger arm adds skill
  on the holdout, for direction or volatility.  No cell of the primary family
  is reported as confirmatory.

**S0 (direct Born-rule readout) sign check (descriptive).**  On the holdout,
5/16 cells are above 50% hit rate and 11/16 below; 5 cells deviate from chance
at the 0.003125 threshold -- three *above* at h = 1 (BTCUSDT 51.909%,
p = 0.0004; SOLUSDT 50.939%, p = 0.0016; BNBUSDT 51.040%, p = 0.0012) and two
*below* at h = 15 (ETHUSDT 47.565%, p = 0.0016; SOLUSDT 47.379%, p = 0.0004).
ICs: 3/16 positive, with BTCUSDT 1m (+0.0277, p = 0.0008) and BNBUSDT 1m
(+0.0234, p = 0.0012) significantly positive and ETHUSDT 15m (-0.0637,
p = 0.0004) significantly negative.  On dev (hourly anchors) the readout was
contrarian in 16/16 cells (hit rates 47.5-49.7%, IC < 0 in 16/16).  **The dev
contrarian pattern does not replicate uniformly:** at h = 1 the holdout sign
flips to momentum-like and is significantly above chance in 3/4 assets, while
at h = 15 the contrarian sign persists in 2 cells.  The sign is not stable
across the source boundary; the contrarian direction was never pre-registered
(the A1 test is two-sided), and no S0 family claim is made (3/16 above-chance
supporters, far below the 9/16 rule).

S0 volatility: `std * sigma_h` is worse than the EWMA baseline in 16/16 cells
(p = 0.0004 in each) and better than the trailing `sigma_h` scale in 8/16
(significantly in 4: BTCUSDT 1m/5m, ETHUSDT 1m, SOLUSDT 1m).

**Shuffled-target control** (training-target permutation, refit, same holdout
rows; BTCUSDT): hit rates 0.494-0.514, IC -0.019..+0.013 -- at chance, as
required.

**Power statement.**  The holdout is **~18.5 days** (26,584 minutes per asset;
24,694-26,254 evaluated rows per cell, i.e. about **440 60-minute blocks**) of
one regime, and full-minute anchors are strongly dependent.  The per-cell
minimum detectable effect at the Bonferroni threshold is about **1.5-2 pp in
hit rate**, derived from the holdout's own block-bootstrap standard errors
(the significant S0 1m cells sit at +1.9/+0.9/+1.0 pp with
p = 0.0004/0.0016/0.0012).  The dev-scale S1 effects (0.1-0.3 pp) are
therefore an order of magnitude below what this holdout can detect: the S1 null
means "no effect of that size is detectable in this ~18.5-day window", not
"the arm has no small effect".  Conversely, the S0 sign deviations are at the
edge of detectability and are reported descriptively only.

## 14. Amendment A3 (post-holdout, 2026-10-02): S0 hit-rate reporting fix

**Bug.**  `backtest.direction_metrics(y, score)` evaluates `sign(score)` against
`sign(y)`; it expects a *signed* score.  The S0 arm passed the Born-rule
probability `p_up` directly, and `sign(p_up) = +1` for every row, so the
`hit_rate`/`hits`/`hit_ci95`/`chance_p_normal` fields of the S0 direction block
were in fact the *fraction of positive forward returns*, not the S0 hit rate.
The IC (rank-based), the one-sample chance tests (`s0_chance_tests`, which
centre the probability itself), and all volatility numbers were unaffected, as
were all S1 numbers.

**Fix and scope.**  The driver now passes `p_up - 0.5` to
`direction_metrics` for the S0 arm.  The fix was applied after the one-shot
holdout run; the holdout was **not** re-run (one-shot rule).  In
`results_schrodinger_holdout.json` the affected fields are therefore
`s0.direction.hit_rate`, `.hits`, `.hit_ci95`, `.hit_delta_vs_50`,
`.chance_p_normal`; the corrected S0 hit rate is
`0.5 + s0.direction_chance.hit_rate_vs_chance.delta` with the p-value from the
same chance test, and that is what section 13 reports.  Dev-only stages were
re-run with the fix so `results_schrodinger_dev.json` is internally
consistent.

The holdout numbers, the frozen model, the metrics, the threshold, the
decision rule, and the mechanical verdict are unchanged by this reporting fix.

## 15. Amendment A4 (post-holdout erratum, 2026-10-02): holdout-length and power-record corrections

Section 7 (the pre-registered holdout protocol) and the Amendment A1 record
were edited in place after the holdout ran: the pre-holdout figures believed
when the protocol was frozen (about a one-month holdout, roughly 43k minute
anchors per asset, about 700 60-minute blocks, about 30 day-blocks) were
replaced by the measured ~18.5-day figures, and A1 gained an
"(actual: ~18.5 days)" annotation.  Those inline edits rewrote the
pre-registration record.  The original pre-holdout wording is restored in
section 7 and in the A1 entry of section 12, and the measured facts are
recorded here instead.  §7 wording was restored verbatim from a snapshot of the file read before the post-holdout edits (the file is untracked in git).

**Measured holdout facts (post-holdout).**

* The holdout is **26,584 minutes per asset (~18.5 days)**, not one month:
  about **440 60-minute blocks** per cell, not about 700; and **26,584 minute
  anchors per asset** were evaluated (24,694-26,254 rows per cell after the
  300-minute warm-up), not the ~43k anchors assumed when the protocol was
  frozen.
* The 1440-row (one-day) block robustness view promised in section 7 (and
  restated in A2) **was NOT run** for this holdout -- A2's amended stage does
  not pass a day-block flag, and no day-block numbers are claimed.  Had it
  been run, the sample contains only **about 18 day-blocks per cell**, far too
  few to reach the Bonferroni threshold.
* The A1 record's "(actual: ~18.5 days)" annotation was removed from
  section 12; the corrected length is recorded in this amendment and in the
  section-13 power statement.

**Consequences (verdict unchanged).** The holdout is shorter than
pre-registered, so its power is lower than the section-7 power statement
promised; section 13 reports the measured ~440-block figure.  The decision
rule, the Bonferroni threshold, the frozen configuration, the one-shot status,
and the mechanical null verdict of section 13 are unchanged by this erratum.
