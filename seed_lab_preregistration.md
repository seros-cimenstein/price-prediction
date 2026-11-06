# Pre-registration: "find the seed" arm (`seed_lab`)

Status: **written and saved before any evaluation result for this family was
computed** (2026-10-10). Every formula, window, constant, model, anchor scheme,
comparison, threshold, decision rule, and the holdout window are fixed below.
The dev results (section 11) and the one-shot holdout results (section 12) are
appended after the runs; nothing in sections 1-10 is changed after results are
seen except by a dated amendment in section 13. The only computations made
before registration were the deterministic data audit of section 2 (row
counts, ranges, gap checks), which is a property of the data, not a model
choice.

## 1. Honesty statement and scope

The owner's premise is that the crypto market is "a procedural game with a
fixed seed" that can be recovered with Schrödinger, Gauss and "every schizo
formula". **The literal premise is false**: prices are the aggregate of
traders' orders, not the output of a pseudo-random number generator, and seed
recovery (e.g. Mersenne-Twister state inversion from 624 outputs) is only
possible when the generator is known and its raw outputs are observed. There is
no seed to find.

The honest, testable versions registered here are:

* **H_seed** - the 1-minute price path is (close to) a *low-dimensional
  deterministic* system, so that state-space reconstruction (Takens delay
  embedding + Sugihara-May simplex projection / S-map) forecasts it with
  skill well above zero, with the signatures of deterministic chaos (section 7).
* **H_formula** - "esoteric" formula families (Gaussian-process extrapolation,
  /biz/-style folk TA, cycle/fractal indicators, Sornette LPPLS) carry
  out-of-sample skill on top of the registered market features.

Null results are the expected outcome and are reported plainly. No window,
constant, family, arm, fold or threshold is changed after any evaluation
output is seen; there is **no selection step** in this arm (every family and
every constant is fixed here, nothing is ranked or chosen on dev).

## 2. Data (verified before registration)

* Dev: `../crypto-data/crypto_market.db`, table `klines_1m`, verified schema,
  assets BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT. Range **1709251200..1789362480**
  (2024-03-01 00:00 -> 2026-09-14 05:08 UTC), **1,335,189 rows per asset, 0
  gaps** (all consecutive `open_time_s` differences are 60 s).
* Holdout: `../crypto-data/holdout_klines_1m.db` (same schema), range
  **1789362540..1791662940** (2026-09-14 05:09 -> 2026-10-10 20:09 UTC),
  **38,341 contiguous minutes per asset, 0 gaps**, starting exactly one minute
  after the dev end.
  * Minutes 2026-09-14 05:09 -> 2026-10-02 16:12 UTC were already used as a
    holdout by earlier arms (Schrödinger, edge lab, mix lab).
  * **Primary holdout window (never evaluated by any arm): rows with
    `target_time_s` in [1790957580, 1791662940]**, i.e. 2026-10-02 16:13 UTC
    -> 2026-10-10 20:09 UTC, **11,757 minutes per asset**. A direction target
    at row `t` is `close[t+h-1]/close[t-1] - 1`, so every predicted candle
    (opens `t .. t+h-1`) lies inside the window; rows whose target would need
    a candle after 20:09 have NaN targets and are dropped (11,757 - h + 1
    scored rows per asset and horizon before the complete-case filter).
  * **Secondary descriptive holdout window**: every holdout minute
    2026-09-14 05:09 -> 2026-10-10 20:09 UTC (38,341 minutes), already partly
    used; reported descriptively, never used for a claim.
* GCP is **not** used by this arm.

## 3. Alignment and causality (the `features.py` convention)

At a prediction row with `target_time_s = t` (the open of minute `t`), the
newest closed candle is minute `t-1` (index `n = pos(t) - 1`). **Every feature
of this arm is a function of candles with `open_time_s < t` only**, i.e. of
`close[0..n]`, `volume[0..n]`; the calendar/astronomical features are
evaluated at the open time of candle `n` (`t - 60 s`), matching the final
`shift(1)` that `features.py` applies to its calendar columns. One-minute log
returns are `r_k = ln(close_k / close_{k-1})`. Market features, physics
features, targets and the EWMA baseline are the registered ones from
`features._assemble_feature_frame` (imported, unchanged). The frozen
Schrödinger columns come from `schrodinger.build_features_for_asset_multi`
with the frozen config `quartic_kde|linear_60|momentum` (imported,
unchanged). Programmatic guards: `features.assert_causal_features` on the
joined frame; feature builders slice their input arrays so the candle at or
after the newest anchor is not even present; `test_seed_lab.py` asserts that
perturbing candles at/after an anchor leaves every feature at that anchor
unchanged.

## 4. Formula families (all constants fixed)

Notation: `n` newest closed candle, `c_k` close, `v_k` volume, `r_k` 1m log
return, `h` horizon in minutes (1, 5, 15, 60). Per-horizon columns carry the
suffix `_{h}m`. All features are per asset.

### F1 - determinism / "seed" (Sugihara-May simplex + S-map)

* Library segment: the trailing `L = 2000` returns `r_{n-L+1..n}`,
  standardized in-segment: `z = (r - mean)/std(ddof=1)` (NaN if std = 0 or any
  return non-finite).
* Takens delay embedding, `E = 6`, `tau = 1`: for local index `i` (0-based
  within the segment, `i >= E-1`), `x_i = (z_i, z_{i-1}, ..., z_{i-E+1})`.
* h-step target of library point `i`: `T_h(i) = z_{i+1} + ... + z_{i+h}`
  (cumulative standardized return). Library for horizon `h` = points
  `i in [E-1, L-1-h]` (targets known before `t`); query = `x_{L-1}` (the
  embedding ending at candle `n`).
* **Simplex forecast**: the `E+1 = 7` nearest library points to the query
  (Euclidean), weights `w_j = exp(-d_j / d_1)` (`d_1` = smallest distance; if
  `d_1 = 0` the weights are the indicator `d_j = 0`), forecast
  `sum w_j T_h(j) / sum w_j`. Feature `seed_simplex_{h}m` = forecast /
  `sqrt(h)`.
* **In-library leave-one-out skill**: for the most recent `Q = 250` library
  points `q in [L-h-Q, L-1-h]` (each with a known target), the same simplex
  forecast from the library with an **exclusion radius** `|j - q| <= h + E - 1`
  (removes every neighbour whose embedding overlaps the query's target or the
  query itself; radius >= h as required). Feature `seed_simplex_rho_{h}m` =
  Pearson correlation of the 250 LOO forecasts with their realized `T_h(q)`
  (NaN if either is constant).
* **S-map** (Sugihara 1994), `theta = 2`: weights over the whole library
  `w_j = exp(-theta d_j / dbar)`, `dbar` = mean query-library distance;
  weighted least squares of `T_h(j)` on `[1, x_j]`; forecast at `[1, query]`.
  Feature `seed_smap_{h}m` = forecast / `sqrt(h)`.
* F1 columns per cell: `seed_simplex_{h}m`, `seed_simplex_rho_{h}m`,
  `seed_smap_{h}m` (3).

### F2 - Gauss (Gaussian-process extrapolation)

* Window `W = 120` log closes `y_{n-119..n}`, time coordinate `s = -119..0`
  minutes. `sigma_1` = std(ddof=1) of the 119 one-minute log returns inside the
  window. Normalized path `u = (y - y_n) / sigma_1`.
* Mean function: OLS line `a + b s` fitted to `u`. Residual GP prior: kernel
  `k(s,s') = sf2 exp(-(s-s')^2/(2 ell^2)) + sn2 delta(s,s')` with **fixed
  hyperparameters in vol units**: `ell = 30` minutes, `sf2 = 20`, `sn2 = 1`
  (one minute of return variance). No hyperparameter is fitted.
* Posterior mean at `s* = h`: `mu(h) = a + b h + k*^T (K + sn2 I)^-1 (u - a - b s)`;
  predictive std `sd(h) = sqrt(sf2 - k*^T (K + sn2 I)^-1 k* + sn2)`.
* Features: `gp_drift_{h}m = mu(h)/sqrt(h)` (dimensionless; the forecast
  return from `close_n` in units of `sigma_1 sqrt(h)`), `gp_std_{h}m =
  sd(h) sigma_1` (log-return units).
* Disclosure: with a fixed time grid and fixed hyperparameters, `mu(h)` is a
  fixed linear filter of the window and `sd(h)/sigma_1` is a constant, so
  `gp_std_{h}m` is a scaled 120-minute realized volatility. This is what an
  honest fixed-hyperparameter GP is; it is registered as such. Implemented
  with a hand-rolled Cholesky solve (numpy).
* F2 columns per cell: `gp_drift_{h}m`, `gp_std_{h}m` (2).

### F3 - folk TA ("4chan certified"); horizon-independent

* `fib_dist`: `hi`, `lo` = max/min of the trailing 240 closes `c_{n-239..n}`;
  position `p = (c_n - lo)/(hi - lo)` (NaN if `hi = lo`); signed distance
  `p - L*` to the nearest level `L*` in {0.236, 0.382, 0.5, 0.618, 0.786}.
* `gann_sin`, `gann_cos`: Gann Square-of-9 phase
  `phi = 2 pi ((sqrt(c_n) mod 2)/2)`; sin and cos.
* `round_logdist`: step `= 10^(floor(log10 c_n) - 1)`; nearest round number
  `R = step * round(c_n/step)`; feature `ln(c_n / R)` (signed).
* `dr369_price`: digital root of `c_n` in integer cents
  (`dr(m) = 1 + (m-1) mod 9`, `dr(0) = 0`); indicator `dr in {3,6,9}`.
* `dr369_minute`: digital root of the minute-of-day (0..1439) of candle `n`;
  indicator `dr in {3,6,9}`.
* `lunar_sin`, `lunar_cos`: lunar phase fraction
  `f = ((T - T0)/P) mod 1` with `T` = open time of candle `n`, reference new
  moon `T0 = 2000-01-06 18:14 UTC` (947182440 s), synodic month
  `P = 29.530588853 d`; features `sin(2 pi f)`, `cos(2 pi f)` (full moon:
  `f = 0.5`, `cos = -1`).
* `phi_time_frac`: the most recent trailing-240 swing extreme = the later of
  the (most recent) argmax and argmin of `c_{n-239..n}`; age `a` = minutes
  from it to `n` (0..239); feature = fractional part of
  `log_phi(a + 1)`, `phi = (1 + sqrt 5)/2`.
* `benford_chi2`: leading digits of the strictly positive volumes among the
  trailing 60 candles `v_{n-59..n}`; Pearson chi-square against Benford
  `P(d) = log10(1 + 1/d)`, d = 1..9 (NaN if fewer than 30 positive volumes).
* F3 columns: the 10 above.

### F4 - cycles / fractals

* `fft_extrap_{h}m`: trailing 256 returns `r_{n-255..n}`, demeaned, periodic
  Hann window `w_k = 0.5 - 0.5 cos(2 pi k/256)`, real FFT. Harmonics = the 3
  largest **local maxima** of the amplitude spectrum over bins 1..127 (a bin
  is a local maximum if its amplitude is > the lower neighbour and >= the
  upper neighbour; peak picking avoids selecting Hann leakage bins).
  Amplitude `A = 4|X_k|/256` (Hann coherent gain 1/2), phase `arg X_k`.
  Projection `xhat_{256+j} = sum A cos(2 pi k (256+j)/256 + phase)`,
  feature `= sum_{j=0}^{h-1} xhat_{256+j} / (std(window) sqrt(h))`.
* `hurst_rs_512`: rescaled-range Hurst exponent on the trailing 512 returns:
  for chunk sizes `m in {8,16,32,64,128,256,512}`, mean over non-overlapping
  chunks of `R/S` (R = range of the demeaned cumulative sum, S = population
  std; chunks with S = 0 excluded); OLS slope of `ln(R/S)` on `ln m`.
* `perm_entropy_4_240`: Bandt-Pompe permutation entropy of order 4, delay 1,
  over the trailing 240 closes (237 ordinal patterns, ties broken by a stable
  sort), normalized by `ln(24)`.
* F4 columns per cell: `fft_extrap_{h}m`, `hurst_rs_512`,
  `perm_entropy_4_240` (3).

### F5 - LPPLS (Sornette log-periodic power law singularity)

* Window: trailing 6 h = 360 log closes, time `tau = 0..359` minutes (`359` =
  candle `n`); values `y - y_n`.
* Fixed grid (36 points): `tc = 359 + D`, `D in {15, 30, 60, 120}` minutes;
  `m in {0.3, 0.5, 0.7}`; `omega in {6, 9, 12}`. For each grid point, linear
  OLS of `y = A + B f + C1 g + C2 k` with `f = (tc - tau)^m`,
  `g = f cos(omega ln(tc - tau))`, `k = f sin(omega ln(tc - tau))`.
* Best fit = minimum RSS over the grid.
* `lppls_conf` (horizon-independent): fraction of the 36 grid fits with
  `B < 0` **and** damping `D_s = m|B| / (omega sqrt(C1^2 + C2^2)) >= 0.8`
  (positive-bubble confidence, literal Sornette-style filter).
* `lppls_slope_{h}m`: derivative `dy/dtau` of the best fit at
  `tau* = 359 + min(h, D_best - 1)` (the fitted curve is only defined before
  `tc`), divided by the std(ddof=1) of the 359 one-minute log returns of the
  window (vol units per minute).
* F5 columns per cell: `lppls_conf`, `lppls_slope_{h}m` (2).

### Kitchen sink (KS)

`MARKET_FEATURES` (14) + F1 (3) + F2 (2) + F3 (10) + F4 (3) + F5 (2) + the
frozen Schrödinger columns `schr_p_up/mean/std/kurt_{h}m` (4) +
`PHYSICS_FEATURES` (6) = **44 columns**.

### Calendar time-shift controls

Calendar/astronomical columns `CAL = {lunar_sin, lunar_cos, dr369_minute}`.

* **Literal 30-day shift (descriptive only)**: the same columns evaluated at
  `T - 30 d`. Recorded before any result: this shift is **degenerate** -
  `dr369_minute` is identical (a whole-day shift keeps the minute of day) and
  the lunar phase moves by only `(30 - 29.530588853)/29.530588853 = 0.0159`
  cycles - so it cannot serve as a "no alignment" control and is reported
  descriptively only.
* **Registered decorrelating shift (the control used for the claim)**: the
  same columns at `T - Delta*`, `Delta* = 64,462 minutes` (= 30 d + 14 d
  18 h 22 min). It moves the lunar phase by 0.5159 cycles (near-opposite
  phase) and the minute of day by 1,102 minutes, while keeping the marginal
  distributions.

## 5. Arms, comparisons, row sets

Per (asset, horizon) cell (16 cells), for direction and volatility:

| arm | columns |
|---|---|
| `M` | `MARKET_FEATURES` (market_only) |
| `F1`..`F5` | `MARKET_FEATURES` + that family |
| `KS` | kitchen sink (44) |
| `CAL` | `MARKET_FEATURES` + CAL |
| `CAL_S` | `MARKET_FEATURES` + CAL at `T - Delta*` |
| `CAL_30` | `MARKET_FEATURES` + CAL at `T - 30 d` (descriptive) |

**Registered comparisons (7, Bonferroni family):** `F1-M`, `F2-M`, `F3-M`,
`F4-M`, `F5-M`, `KS-M`, `CAL-CAL_S`. Descriptive only: `CAL-CAL_30`.

**Row set**: one complete-case row set per cell over the union of every arm's
columns + target (+ `ewma_vol_60m`), via
`train_schrodinger.complete_case_rows` (asserted identical to
`backtest._valid_frame`); every arm, every paired delta and every direct
readout of the cell uses exactly these rows.

**No-fit direct readouts (5)**: `sign(seed_simplex_{h}m)`,
`sign(seed_smap_{h}m)`, `sign(gp_drift_{h}m)`, `sign(fft_extrap_{h}m)`,
`sign(lppls_slope_{h}m)`, scored against the realized direction (hit rate vs
0.5) and by Spearman IC vs 0, using the one-sample block-bootstrap chance tests
`train_schrodinger.s0_chance_tests` (the score is presented as
`0.5 + score`, which preserves the sign and the ranks).

**Shuffled-target control**: `backtest.shuffled_target_control` for the KS
columns, BTCUSDT, all four horizons, dev, seed `20260914 + 100 h`, 3 folds.

## 6. Dev evaluation protocol

* Source: the full dev DB. Frame = `features._assemble_feature_frame` (market
  + physics + targets); anchors `target_time_s % 3600 == 0` (hourly), **all
  22,254 anchor times per asset, no subsampling** (anchors lacking a full
  trailing window are dropped by the complete-case policy; F1 needs 2,001
  candles, LPPLS 360, Hurst 513).
* Folds: `backtest.walk_forward_predict`, expanding, `n_folds = 3`,
  `initial_train_fraction = 0.50`, embargo = horizon.
* Models: per-fold `StandardScaler` + L2 logistic `C = 1` (direction) / ridge
  `alpha = 1` (volatility) (`backtest._fit_score`).
* Paired deltas: `backtest.paired_bootstrap_delta`, block **3 anchors**
  (hourly convention), seeds `20260914+h` (hit), `20260915+h` (IC),
  `20260916+h` (MSE), **`n_boot = 4999`** (two-sided floor 0.0004).
* Dev is descriptive/consistency evidence; the claims are decided on the
  primary holdout. The same Bonferroni thresholds and majority rule (section
  8) are applied to dev and reported.
* `--stage explore` = determinism diagnostics (section 7) + the dev tables;
  `--stage frozen` = the dev tables only. Nothing is selected by either stage.

## 7. Determinism diagnostics (dev, descriptive) and the H_seed signature

Computed on dev only, on the diagnostic anchors `target_time_s % 43200 == 0`
(00:00 and 12:00 UTC, ~1,855 per asset; 12 h apart so forecast errors are
nearly independent), restricted to anchors with at least 8,001 prior candles so
every panel uses the same anchors. "rho" = Pearson correlation, across
anchors, of the out-of-sample query forecast with the realized standardized
cumulative return `T_h` (standardized with the library's own mean/std); 95% CI
by Fisher z with `n - 3` degrees of freedom. Per asset and pooled.

1. Simplex rho vs `E = 1..10` (h = 1, L = 2000).
2. Simplex rho vs horizon `h in {1, 2, 3, 5, 10, 15, 30, 60}` (E = 6, L = 2000).
3. Simplex rho vs library size `L in {100, 250, 500, 1000, 2000, 4000, 8000}`
   (E = 6, h = 1), the CCM-style convergence check.
4. S-map rho vs `theta in {0, 0.25, 0.5, 1, 2, 4, 8}` (E = 6, h = 1,
   L = 2000); `theta = 0` is a global linear AR(6).
5. Mean of the F1 feature `seed_simplex_rho_{h}m` over all hourly dev anchors.
6. **Compression**: sign sequence (`-1/0/+1` as bytes) of the last 262,144
   dev 1m returns per asset; `lzma.compress` (default preset 6, xz format);
   ratio = compressed size / mean compressed size of 200 random shuffles
   (`numpy` PCG64, seed `20261010 + asset index`), one-sided p =
   `(1 + #{shuffle <= real}) / 201`.
7. **BDS** (Brock-Dechert-Scheinkman, statsmodels-equivalent formula,
   `eps = 1.5 sd`, m = 2..5) on 8 non-overlapping 4,000-minute blocks per asset,
   evenly spaced over dev, for (a) the in-block standardized 1m log returns and
   (b) vol-standardized returns `r_k / sd(r_{k-60..k-1})`. Reported: median
   statistic and the number of blocks with `|W| > 1.96`. Recorded before
   results: BDS rejects i.i.d. for **any** non-linear dependence, including
   volatility clustering, so a rejection on (a) or (b) is **not** evidence of
   low-dimensional determinism on its own.

**Signature that would support H_seed** (all must hold): simplex rho at
`h = 1, E = 6` **>= 0.10** in at least 3 of 4 assets; rho **decays with h**
(rho(h=1) > rho(h=60)); rho **rises with library size** (rho(8000) -
rho(250) > 0.02 in at least 3 of 4 assets); S-map rho at the best
`theta > 0` exceeds `theta = 0` by more than 0.02 (state-dependent,
non-linear predictability); lzma ratio **< 0.90** (a deterministic generator
of signs would be highly compressible). **Refutation**: |rho| < 0.03 at all E
with no L-convergence, a flat or decreasing theta curve, and a compression
ratio within 1% of 1 (a statistically significant ratio like 0.995 is weak
linear/microstructure dependence, not a seed).

**H_seed verdict rule**: H_seed is "supported" only if the dev signature above
holds **and** on the primary holdout either the F1 comparison meets the
majority rule (section 8) or the simplex or S-map direct readout meets the
direct-readout rule. If neither the dev signature nor the holdout tests hold
it is "not supported (refuted at the tested scale)"; mixed outcomes are
reported as mixed, verbatim.

## 8. Frozen holdout test and decision rules (one-shot)

* **Fit once**: every arm is fitted once on all dev hourly rows of the cell's
  complete-case set whose targets finish before the holdout's first candle
  (`target_time_s + 60 h <= 1789362540`), house models/scaler, then frozen.
  The direct readouts fit nothing.
* **Features on the holdout**: built on the last 3 dev days concatenated with
  the full holdout DB (`features._assemble_feature_frame` on the concatenated
  candles, contiguity asserted), so every trailing window is full from the
  first holdout minute; rows from the primary window are the primary scored
  set; all holdout rows form the secondary set.
* **Primary**: every minute of the primary window (section 2), paired
  bootstrap with **60-row blocks** and **`n_boot = 9999`** (floor 0.0002);
  direct readouts with `s0_chance_tests` at 9999.
* **Secondary (descriptive)**: the same frozen models scored on every holdout
  minute, `n_boot = 4999`, reported, never decisive.
* **Bonferroni (per metric family, 16 cells x 7 registered comparisons =
  112 tests)**: `alpha = 0.05/112 = 4.464e-4` separately for Δhit, ΔIC and
  ΔMSE, on dev and on the holdout. Also reported, not decisive: the stricter
  mix-lab pooled direction family (hit + IC, 224 tests, `alpha = 2.232e-4`),
  attainable on the holdout only. Direct readouts: 5 readouts x 16 cells = 80
  tests per metric, `alpha_direct = 0.05/80 = 6.25e-4`.
* **Decision rules (majority, as `physics_preregistration.md` /
  `mix_lab_preregistration.md`)**:
  * A family (each of `F1-M` .. `F5-M`, `KS-M`) **adds direction skill** iff
    for Δhit or for ΔIC: **>= 9/16 cells improve AND >= 9/16 cells improve
    with p < alpha** (`train._ablation_summary`). Volatility skill: the same
    for ΔMSE < 0.
  * Calendar alignment matters iff `CAL-CAL_S` meets the same rule.
  * A direct readout **has skill** iff >= 9/16 cells have hit rate > 0.5 with
    p < alpha_direct (or IC > 0 with p < alpha_direct); >= 9/16 significantly
    *below* chance is reported as "contrarian", not as skill, and the sign is
    never flipped post hoc.
  * Otherwise: **null** ("no evidence the family adds skill").
* **One-shot**: the holdout stage refuses to run if the output already holds
  a holdout run. It is not re-run with other constants, windows or arms.
* **Power (pre-registered honesty)**: ~11,700 scored minutes per asset and
  horizon, ~195 sixty-minute blocks per cell. The Schrödinger holdout
  (~25,000 minutes) had per-cell detectable Δhit of ~1.5-2 pp at a similar
  threshold; with half the length the minimum detectable Δhit here is roughly
  **2-3 pp** and ΔIC roughly 0.03-0.05. Dev-scale effects in previous arms
  were 0.1-0.5 pp, so a null means "no effect of that size is detectable in
  8.2 days", not "exactly zero". A genuine low-dimensional deterministic
  "seed" would produce effects far larger than that.

## 9. Implementation, parallelism, runtime

`seed_lab.py` (CLI mirrors `mix_lab.py`: `--stage explore|frozen|holdout`).
Feature construction and cell evaluation run in a process pool (default 8
workers, BLAS threads capped at 1 per worker); per-anchor F1 loops are
numpy-vectorized within an anchor; F2/F4/F5 are batched over anchors (the GP
and LPPLS designs are fixed matrices, applied as precomputed linear operators).
`test_seed_lab.py` checks the numerics (simplex skill on Lorenz/logistic map
vs i.i.d. noise, GP linear-trend recovery, FFT sinusoid continuation, LPPLS
parameter recovery, lunar phase at the 2024-04-23 23:49 UTC full moon, BDS
on i.i.d. vs logistic map, the chunked Schrödinger builder equals the
registered one, and the causality guard). No dev subsampling is required for
the registered tables; the only reduced-cadence computation is the
descriptive diagnostics on 12-hourly anchors (section 7).

## 10. Amendment policy

Any change after a result has been seen is recorded in section 13 with a
date, the reason, and what was/was not re-run. Silent changes are not allowed.

## 11. Dev results (`--stage explore`, 2026-10-11, 3,418 s)

Numbers: `results_seed_lab_dev.json`; console log `seed_lab_dev.log`.

**Determinism diagnostics (12-hourly anchors, Pearson rho, per asset).**

| panel | BTCUSDT | ETHUSDT | SOLUSDT | BNBUSDT | pooled |
|---|---:|---:|---:|---:|---:|
| simplex rho, h=1, E=6, L=2000 | +0.029 | -0.030 | +0.026 | +0.003 | +0.008 |
| simplex rho range over E=1..10 | -0.073..+0.065 | -0.054..+0.059 | -0.047..+0.127 | -0.024..+0.082 | -0.013..+0.044 |
| simplex rho h=1 / h=60 | +0.029 / +0.012 | -0.030 / +0.030 | +0.026 / -0.069 | +0.003 / +0.013 | +0.008 / -0.003 |
| simplex rho L=250 / L=8000 | -0.032 / +0.026 | -0.001 / -0.027 | +0.010 / +0.005 | +0.001 / -0.024 | -0.006 / -0.005 |
| S-map rho theta=0 (linear AR(6)) | +0.073 | +0.048 | +0.202 | +0.041 | +0.092 |
| S-map best theta>0 minus theta=0 | -0.011 | -0.015 | +0.010 | +0.001 | -0.003 |
| lzma ratio vs 200 shuffles (p) | 0.9845 (0.005) | 0.9907 (0.005) | 0.9975 (0.045) | 1.0007 (0.741) | |
| BDS m=2..5 median, standardized | 15.8..23.6 | 14.4..21.9 | 13.8..19.5 | 15.3..22.2 | |
| BDS m=2..5 median, vol-standardized | 8.5..11.0 | 8.8..11.2 | 6.0..8.2 | 7.2..10.8 | |

H_seed signature (section 7): rho(h=1,E=6) >= 0.10 in 3/4: **no** (max 0.029);
decay with h in 3/4: **no**; library gain > 0.02 in 3/4: **no**; S-map
non-linear gain > 0.02 in 3/4: **no** (S-map skill is largest at theta = 0 for
BTC/ETH and flat for SOL/BNB, i.e. whatever is there is linear AR structure);
lzma ratio < 0.90 in 3/4: **no** (all within 1.6% of 1). **Signature not met.**
BDS rejects i.i.d. everywhere and is roughly halved by vol-standardization, as
expected from volatility clustering (recorded in advance as non-evidence).

**Dev walk-forward tables (n_boot 4999, alpha 4.464e-4).** Improving / surviving
cells out of 16:

| comparison | Δhit | ΔIC | ΔMSE |
|---|---|---|---|
| F1-M (simplex/S-map) | 9 / 0 | 5 / 0 | 1 / 0 |
| F2-M (Gauss GP) | 10 / 0 | 12 / 0 | 15 / 7 |
| F3-M (folk TA) | 10 / 0 | 14 / 1 | 6 / 0 |
| F4-M (FFT/Hurst/PE) | 11 / 0 | 4 / 0 | 13 / 2 |
| F5-M (LPPLS) | 6 / 0 | 7 / 0 | 9 / 0 |
| KS-M (kitchen sink) | 13 / 1 | 13 / 0 | 13 / 4 |
| CAL-CAL_S | 0 / 0 | 7 / 0 | 1 / 0 (1 worse) |

Direct readouts: simplex 49.58-50.40%, S-map 49.51-51.10%, GP drift
48.36-51.90% (4 cells significantly above and 6 significantly below chance),
FFT 49.36-50.32%, LPPLS slope 48.17-51.38%: all **null** under the rule.
Shuffled KS control (BTCUSDT): 49.95-51.31% hit, |IC| <= 0.021.

## 12. One-shot holdout results (`--stage holdout`, 2026-10-11, 15,876 s)

Numbers: `results_seed_lab_holdout.json`; console log `seed_lab_holdout.log`.
Primary window 2026-10-02 16:13 -> 2026-10-10 20:09 UTC (11,757 scored minutes
per cell), n_boot 9999, alpha 4.464e-4.

| comparison | Δhit | ΔIC | ΔMSE | rule verdict |
|---|---|---|---|---|
| F1-M | 6 / 0 | 8 / 0 | 4 / 1 | null |
| F2-M | 7 / 0 | 9 / 0 | 16 / 4 | null |
| F3-M | 10 / 0 | 8 / 0 | 11 / 7 | null |
| F4-M | 11 / 0 | 7 / 0 | 9 / 2 | null |
| F5-M | 7 / 0 | 5 / 0 | 5 / 0 | null |
| KS-M | 10 / 0 | 9 / 0 | 11 / 2 | null |
| CAL-CAL_S | 0 / 0 | 9 / 0 | 13 / 13 (2 worse) | volatility rule met (see note) |

Direct readouts (alpha 6.25e-4): simplex 49.25-51.74% (0 significant), S-map
49.13-52.36% (1 cell above), GP drift 49.02-52.47% (0), FFT 49.26-51.17% (0),
LPPLS slope 47.98-53.61% (0): all **null**. Secondary (full holdout,
descriptive): every family null for direction; CAL-CAL_S ΔMSE 12 improving / 3
surviving.

**H_seed verdict: not supported (refuted at the tested scale)** - the dev
signature fails on all five criteria and neither F1 nor the simplex/S-map
readouts meet their holdout rules.

**Reporting note on CAL-CAL_S (post-result, 2026-10-11; nothing re-run).** The
volatility rule fires mechanically, but every primary-window ΔMSE is between
8e-18 and 8e-16, a relative change of 4e-11 to 1e-9 of the arm's MSE.
Within an 8.2-day window the lunar phase moves less than 0.28
cycles and `dr369_minute` is a fixed daily pattern, so the paired per-row
difference keeps one sign across the window and the block bootstrap returns
its floor (0.0002) for an effect that is numerically negligible. The same
comparison on dev gives 1/16 improving and 1/16 significantly worse, and on
the 26.6-day secondary window 3/16 survive. This is recorded as the literal
rule outcome, **not** as evidence that lunar or 369 calendar features predict
volatility.

## 13. Amendment log

* 2026-10-11 - no protocol amendment. The first dev `--stage explore` process
  was killed (parent shell exit) after the feature build and diagnostics and
  before any walk-forward table was produced; it was restarted unchanged and
  the complete run above is the only reported dev result.
* 2026-10-11 - reporting note on CAL-CAL_S magnitude added to section 12
  after the holdout; nothing was re-run or re-selected.
