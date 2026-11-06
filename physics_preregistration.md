# Pre-registration: `PHYSICS_FEATURES` ablation (`market_only` vs `with_physics`)

Status: fixed **before** any result for this arm was computed. No window,
constant, formula, or alternative was changed after seeing any evaluation
output; no alternatives were scanned against results. The only pre-result
computations were deterministic properties of the models below: the
steady-state Kalman gain implied by the registered noise ratios (and its
sensitivity across a few ratio candidates, compared only on their implied
smoothing gains, never against predictive results) and the numerical
verification of the Riemannian formula in a throwaway script that was then
deleted.

## Honesty statement: physics/geometry analogues, not physics

These features are **analogues** drawn from mathematics and physics-inspired
signal processing. They carry no claimed physical mechanism linking markets to
those theories:

- **Gauss**: a local-linear-trend Kalman filter (Gauss–Markov state estimation)
  used as a trend/drift estimator; no Gaussian-process claim about prices.
- **Schrödinger**: the ground state of the quantum harmonic oscillator has the
  same stationary density as an Ornstein–Uhlenbeck process (imaginary-time
  Schrödinger = Fokker–Planck), so the displacement z-score and the OU
  mean-reversion rate are used as mean-reversion descriptors. No Schrödinger
  equation is solved for the market.
- **Navier–Stokes**: the Reynolds number analogue is the Kaufman efficiency
  ratio (path length vs net displacement), a laminar/turbulent flow metaphor.
  No Navier–Stokes PDE is solved. An actual NS/PDE solve has no mechanism
  linking it to prices and is out of scope.
- **Riemann**: the Fisher–Rao information metric on the manifold of univariate
  Gaussian distributions gives a closed-form geodesic distance between local
  return distributions. The Riemann hypothesis, the zeta function, and prime
  distributions have no mechanism linking them to prices and are out of scope.

All features are computed only from **closed candles**, on the same candle
frame as the existing features, and are then passed through the exact same
final `shift(1)` as the market features, so no feature used for the row at
target open `t` sees candle `t` or later. Features are dimensionless by
construction. Values that are undefined (insufficient history, zero variance,
undefined log) are `NaN`; they are never filled or clipped to a fabricated
value.

Notation: for asset `a`, `C_s = close` of candle `s` (open time `s`), `L_s =
ln C_s`, and `r_s = C_s / C_{s-1} - 1` is the existing one-minute simple return
series (`returns` in `features.py`). Windows are counted in candle rows of the
asset's own frame, exactly like the existing rolling features. All rolling
quantities use `min_periods = window` and `ddof = 1` standard deviations, like
`realized_vol_*`.

## The six registered features

### 1. `gauss_kalman_drift_z`

Local-linear-trend state-space model on `L_s`, state `x = [level, slope]'`:

```
x_s = F x_{s-1} + w_s,   F = [[1, 1], [0, 1]],  w ~ N(0, Q),  Q = diag(q_L, q_S)
L_s = H x_s + v_s,       H = [1, 0],            v ~ N(0, R)
```

Fixed pre-registered noise **ratios** (the Kalman gain depends on `Q/R` only,
so these ratios are the full specification): `q_L / R = 1e-4`,
`q_S / R = 1e-8` (with `R = 1`). The implied steady-state gains, obtained from
the deterministic discrete Riccati fixed point (`tol = 1e-14`), are
`alpha = 0.01717137036324189` (level gain) and
`beta = 9.913771379428575e-05` (slope gain). The gains were computed once
before any data-based evaluation; they are a deterministic consequence of the
ratios, not a tuned setting.

**Implementation choice.** The full time-varying Kalman recursion converges to
this fixed point long before one million candles, so the filter is applied in
its fixed-gain (steady-state) form, which is exactly Holt's linear-trend
double-exponential smoothing:

```
e_s   = L_s - (level_{s-1} + slope_{s-1})
level_s = level_{s-1} + slope_{s-1} + alpha * e_s
slope_s = slope_{s-1} + beta * e_s
```

initialized at the first candle with a finite close as `level = L_0`,
`slope = 0`. A per-row time-varying Python loop was unnecessary: the
steady-state gain is pre-registered, and the scalar recursion over 1.3M rows
per asset takes ~1.4 s in this environment. Non-finite closes (none exist in
the historical source; a NaN placeholder exists in the live frame) are skipped
without updating the state.

Feature: `gauss_kalman_drift_z_s = slope_s / realized_vol_60m_s`, where
`realized_vol_60m` is the existing registered 60-row rolling standard deviation
(ddof = 1) of one-minute simple returns, and the feature is `NaN` wherever that
denominator is `NaN` or `<= 0`. No additional warm-up mask is applied beyond
the denominator's (the recursion is exactly initialized at the first close).

### 2. Schrödinger / OU features

- `schrodinger_displacement_60m_s = (L_s - mean_60(L)) / std_60(L)`, with
  the 60-row rolling mean and ddof = 1 standard deviation of `L`; `NaN` if the
  standard deviation is `NaN` or `0`.
- `schrodinger_ou_kappa_240m_s = -ln(phi_s)` where `phi_s` is the windowed
  lag-1 autocovariance ratio (Yule–Walker/biased-autocorrelation estimator) of
  the deviations `d_s = L_s - mean_240(L)`:

  ```
  xbar_s = mean of the 240 values d_{s-239..s}
  phi_s  = sum_{i=s-239}^{s-1} (d_i - xbar_s)(d_{i+1} - xbar_s)
           / sum_{i=s-239}^{s}   (d_i - xbar_s)^2
  ```

  `phi_s` is `NaN` when fewer than 479 candles are available (each `d_i` needs
  240 closes, and the window needs `d` back to `s-239`), or when the
  denominator is `<= 0`. `kappa_s` is `NaN` unless `0 < phi_s < 1` (a
  non-positive lag-1 coefficient is oscillation, not OU mean reversion, and
  `phi >= 1` is not a valid stationary AR(1) parameter; nothing is clipped).

Implemented from rolling sums (equivalent to the formula above, verified
against a brute-force per-window computation in a throwaway script).

### 3. Navier–Stokes / Reynolds features

- `ns_reynolds_60m_s = |sum_{i=s-59}^{s} r_i| / sum_{i=s-59}^{s} |r_i|`
  (Kaufman efficiency ratio; 0 = turbulent, 1 = laminar), `NaN` if the
  denominator is `NaN` or `0`.
- `ns_signed_flow_60m_s = sign(net_60m_s) * ns_reynolds_60m_s`, where
  `net_60m_s = C_s / C_{s-60} - 1` is the existing registered `ret_60m`.

### 4. `riemann_fisher_rao_15_240`

Closed-form Fisher–Rao geodesic distance between the univariate Gaussian fits
of trailing 15-row and 240-row one-minute simple returns,
`N(mu15, sigma15)` and `N(mu240, sigma240)` (means and ddof = 1 standard
deviations, `min_periods` equal to the window):

```
Delta_mu = mu15 - mu240
K1 = sigma15 - sigma240
K2 = sigma15 + sigma240
d = 2 * sqrt(2) * artanh( sqrt( (Delta_mu^2 / 2 + K1^2) / (Delta_mu^2 / 2 + K2^2) ) )
```

`NaN` when either standard deviation is `NaN` or non-positive, when the
numerator/denominator ratio is not in `[0, 1)` (e.g. a degenerate zero-variance
window making the ratio equal to 1), or when fewer than 240 rows are available.
The formula was verified numerically against an independent geodesic-distance
computation (Poincaré half-plane isometry and direct minimization of the
discretized Fisher–Rao path length) in a throwaway script before this
registration; the script was deleted. The identity used for the check: the
Fisher–Rao metric `ds^2 = (dmu^2 + 2 dsigma^2)/sigma^2` is isometric to
`sqrt(2)` times the Poincaré half-plane metric, giving
`d = sqrt(2) * arccosh(1 + (Delta_mu^2 + 2 K1^2) / (4 sigma15 sigma240))`,
which is algebraically equal to the `artanh` form above.

`PHYSICS_FEATURES` (this exact order) is:
`gauss_kalman_drift_z`, `schrodinger_displacement_60m`,
`schrodinger_ou_kappa_240m`, `ns_reynolds_60m`, `ns_signed_flow_60m`,
`riemann_fisher_rao_15_240`.

`with_physics = MARKET_FEATURES + PHYSICS_FEATURES` (no GCP columns), built
inside the same shared assembly function as the market features, in the same
frame, through the same `shift(1)`, so historical, smoke, and live paths are
identical.

## Evaluation protocol (identical to the existing pipeline)

- Expanding walk-forward CV, 3 folds, 50% initial train, embargo
  `max(horizon, --embargo)` (README settings).
- Arm A `market_only` = `MARKET_FEATURES` (14 columns); arm B `with_physics` =
  `MARKET_FEATURES + PHYSICS_FEATURES`. Both arms are fit on the identical
  complete-case rows (`pair_valid`), so every delta is paired on the same OOS
  rows, folds, scaler policy, and hyperparameters (L2 logistic `C=1`, ridge
  `alpha=1`).
- Per cell (4 assets x 4 horizons = 16): `Delta` = `with_physics` minus
  `market_only` for direction hit rate (pp), Spearman IC, and volatility MSE
  (negative is better for MSE). Paired 60-minute moving-block bootstrap
  p-values with the existing machinery (`paired_bootstrap_delta`, n_boot = 300,
  block = 60, seeds 20260914+horizon for hit rate, 20260915+horizon for IC,
  20260916+horizon for MSE).
- Shuffled-target control for the `with_physics` direction arm, BTCUSDT, all
  four horizons, using the existing `shuffled_target_control` (same folds, same
  seed formula as the existing run).

## Pre-registered decision rule

- Bonferroni threshold across the 16 cells: `alpha = 0.05 / 16 = 0.003125`.
  A cell "survives" for a metric only if its delta is positive **and** its
  bootstrap p-value is below the threshold.
- The verdict **"physics adds skill"** is declared for a direction metric
  (`Delta hit` or `Delta IC`) only if **at least 9 of 16 cells** have a positive
  delta and **at least 9 of 16 cells** are positive and survive Bonferroni.
  Requiring a majority of cells (rather than any single cell) is the
  "majority-consistent" bar; isolated significant cells are reported but do not
  change the verdict.
- The volatility claim is separate: improvement is claimed only if at least 9
  of 16 cells have `Delta MSE < 0` and at least 9 of 16 are negative and
  survive Bonferroni.
- **Otherwise the result is reported as a null** ("no evidence that the physics
  arm adds skill"). A null is a valid, expected result: the six features are
  dimensionless transformations of the same candle series, so the market-only
  arm already spans much of their information.

The verdict is computed from the numbers, never adjusted by hand. No result may
be reported as confirmatory outside this rule.

## Amendment (post-result, 2026-10-01): bootstrap resolution only

**What changed.** The paired-bootstrap replicate counts used for the
confirmatory p-values. The original run used `n_boot = 300` for all three
metrics (the `--bootstrap` default inherited from the GCP pipeline). The
two-sided p-value produced by `backtest.paired_bootstrap_delta` has a
resolution floor of `2 / (n_boot + 1)`, i.e. `2/301 = 0.00664` at 300
replicates. The pre-registered Bonferroni threshold is `0.05/16 = 0.003125`,
which lies **below that floor**: no cell could ever be recorded as surviving,
so the original "0/16 survivors -> null" was unsatisfiable by construction
rather than evidence. The amendment raises the resolution to:

- `n_boot = 4999` for `Delta hit` and `Delta MSE` (means of per-row paired
  differences, evaluated with the exact block-sum bootstrap
  `backtest._block_mean_bootstrap`; p-value floor `2/5000 = 0.0004`), and
- `n_boot = 1999` for `Delta IC` (Spearman ranks; floor `2/2000 = 0.0010`,
  below `0.003125`).

The 60-minute blocks, per-horizon seeds (`20260914+horizon`,
`20260915+horizon`, `20260916+horizon`), paired rows, cap policy, Bonferroni
threshold, and the majority-consistency decision rule are unchanged. The
300-replicate default is retained for the existing GCP ablation, which is
reported as-is and not re-run.

**When discovered.** After the first physics full run had produced its table
and verdict, while reviewing the reported p-values (every MSE p-value was
exactly `0.0066`, and no p-value could be below `0.00664`).

**What did not change.** The six feature formulas, windows, constants, and
steady-state gains; the model classes and hyperparameters; fold construction,
initial training fraction, and embargo; the paired-row policy; the
shuffled-target control; the Bonferroni threshold `0.003125`; and the
majority-consistency decision rule. Only the Monte-Carlo resolution of the
p-values changed: it lowers the p-value floor so that Bonferroni survivals
become *attainable* (under 300 replicates survival was impossible because the
floor `0.00664` exceeded the threshold), and it changes no delta and no
threshold. Because the amendment decision was made after seeing a null result,
the reader should weigh that fact; the amended run again returned a null for
the direction metrics, which limits the concern.

The amended run is recorded in `results_physics.json`; its
`ablation.bootstrap` block documents the replicate counts and the per-metric
p-value floors.
