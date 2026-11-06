# Crypto short-horizon prediction: historical evaluation

> This is the project's original working README, moved here unchanged. Commands and paths in it
> are relative to the workspace that held this folder as `price-prediction/` next to
> `crypto-data/` and `gcp2-data/`; file names without a folder refer to the repository root.

## Verdict (read this first)

**The market-data models show a small numerical out-of-sample edge, but it is weak and not a practical trading claim. GCP2 does not add convincing incremental information.**

- The pre-registered regularized logistic model beats the persistence and 15-minute momentum baselines in pooled walk-forward hit rate for all 16 asset/horizon rows. Hit rates are only **50.425%--52.462%** and Spearman IC is **0.00435--0.04768**. The Wilson intervals below are narrow because there are hundreds of thousands of minute observations; minute-level serial dependence makes those intervals optimistic as uncertainty intervals, so they are reported as requested but should not be read as independent-trial evidence.
- The pre-registered ridge volatility model has lower MSE than the EWMA baseline in all 16 rows. The absolute errors are small and this does not establish profitable volatility timing; it establishes only this OOS error comparison for the specified target.
- In the paired GCP ablation, hit-rate changes range from **-0.181 to +0.033 percentage points**, IC changes are essentially zero (about -0.00329 to +0.00008), and there is no consistent positive result. The paired 60-minute moving-block bootstrap p-values do not support a GCP claim after considering the 16 pre-registered comparisons. Volatility MSE changes are also tiny and inconsistent. **Do not promote GCP as a predictive signal on this evidence.** Caveat: those p-values used 300 bootstrap replicates, whose two-sided resolution floor (`2/301 ≈ 0.0066`) is above the 16-cell Bonferroni threshold `0.003125`, so no GCP cell could have been recorded as surviving; the table is reported as-is and is not evidence either way (see "Known resolution limit of the GCP table").
- The pre-registered physics/geometry analogue arm (`with_physics`) also adds no skill. With the amended bootstrap resolution (floors 0.0004 for Δhit/ΔMSE and 0.0010 for ΔIC, below the `0.003125` threshold), Δhit is positive in 15/16 cells and ΔIC in 13/16, but **0/16 survive Bonferroni for either direction metric**. Volatility is mixed rather than improved: ΔMSE is significantly better in 3/16 cells (BNBUSDT 1m/5m/15m) and significantly worse in 5/16 (BTCUSDT 1m/5m/15m/60m and BNBUSDT 60m). See the physics section for the table and the amendment note.
- The literal **Schrödinger wave-function arm** (`schrodinger.py`) is also a **null on dev and on the untouched ~18.5-day holdout**: 0/16 Bonferroni survivors for Δhit and ΔIC in both, and a volatility result that does not meet the pre-registered ≥9/16 rule (dev: 1 survivor and 1 significantly worse; holdout: 2 survivors out of the ≥9 required). Its direct Born-rule readout carries a small signal whose **sign is not stable** (contrarian in 16/16 dev cells, but at h=1 significantly *above* chance on the holdout), and at h=1 `p_up` is close to a monotone transform of the registered `ret_1m` (Spearman 0.965). See the Schrödinger section.
- The **edge lab** (`edge_lab.py`) screened 27 cost-aware configurations and froze three finalists; on the untouched 18.3-day holdout their full-coverage hit rates are ~52.5%/~52.4%/~51.8% against ~48% baselines (p at the bootstrap floor), but the gross edge at full coverage is a few tenths of a basis point per trade versus a 20 bp round-trip taker fee, and the θ-filtered trades came in at 0/3/1 (under the pre-registered 30-trade power rule), so the cost-aware criterion is **inconclusive and no edge is claimed**. See the edge-lab section.
- The **mixed arm** (`mix_lab.py`, frozen in `mix_lab_preregistration.md`) couples the registered causal GCP `netvar_z_60m` into the literal Schrödinger equation itself (six pre-declared couplings: effective ħ, potential stiffness, packet width, imaginary potential), each paired against its own 30-day time-shifted GCP control, plus the additive `market+GCP+SCHR` arms. Dev and the one-shot ~17.8-day holdout are both **null**: **0/16 Bonferroni survivors for Δhit and ΔIC in every one of the five pre-registered comparisons** (α = 3.125e-4), the frozen `packet_widen` coupling does not beat its shifted control (holdout Δhit 10/16 and ΔIC 9/16 improving, 0 survivors), and GCP adds nothing on top of the Schrödinger columns (A3−A2 holdout Δhit 5/16, ΔIC 3/16). The coupled Born-rule readout is contrarian on dev (16/16 cells) but flips sign at h=1 on the holdout, like the Schrödinger arm. Deployment note: the live `gcp2.net getcurrentnetvar` endpoint returns a different quantity (O(10–220)) than the archive netvar (O(1), std 0.18), so this arm is **not live-deployable** on that endpoint. See the mixed-arm section.
- The **seed lab** (`seed_lab.py`, frozen in `seed_lab_preregistration.md`) tests the "fixed-seed procedural market" idea as the falsifiable claim that prices are a low-dimensional deterministic system (Takens embedding + Sugihara simplex/S-map, lzma compressibility, BDS), alongside Gaussian-process extrapolation, folk TA (Fibonacci, Gann Square-of-9, round numbers, 369, lunar phase, golden-ratio time, Benford), FFT/Hurst/permutation entropy, LPPLS and a 44-column kitchen sink. **H_seed is not supported**: simplex rho at h=1 is at most 0.029, there's no library-size convergence, the S-map is best at θ = 0 (linear), and sign sequences compress to within 1.6% of shuffled ones. On the never-evaluated 8.2-day holdout (2026-10-02 16:13 → 2026-10-10 20:09 UTC) **every family is null for direction** (0/16 Δhit and ΔIC survivors at α = 4.464e-4) and all five no-fit readouts are at chance. The calendar control mechanically meets the volatility rule, but with relative ΔMSE ≤ 1e-9 and no dev replication. See the seed-lab section.
- The shuffled-target controls are at chance (49.939%--50.411% hit rate, IC approximately -0.00168 to -0.00042), which is the required leakage sanity check.

The honest owner-facing summary is therefore: **there is a weak market-data association that clears the named numerical baselines in this historical sample, but no evidence that GCP2, the physics analogue family, the Schrödinger wave-function arm, or any "seed"/folk-TA/LPPLS formula adds skill, and no evidence of a robust, cost-aware trading edge.** Future live use should monitor drift and must not treat the artifacts as a guarantee.

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

Useful options are `--n-folds`, `--initial-train-fraction`, `--embargo`, `--bootstrap`, `--start-time-s`, `--end-time-s`, and `--ablation {gcp,physics}` (default `gcp`; see the physics section below). A smaller date range is useful for a smoke run; do not interpret a smoke run as the reported evaluation. The command prints the table, writes `results.json`, and writes 64 final-history `model_*.joblib` artifacts in this directory.

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

**Known resolution limit of the GCP table (not re-run).** The GCP Δ p-values above were produced with 300 bootstrap replicates. The two-sided paired-bootstrap p-value has a resolution floor of `2 / (n_boot + 1) ≈ 0.0066`, which is **above** the 16-cell Bonferroni threshold `0.003125`, so no GCP cell could ever have been recorded as surviving and the "no consistent positive result" statement is not evidence either way. The table is left exactly as reported and the GCP ablation was not re-run; the physics arm below was re-run at higher bootstrap resolution (see its amendment note).

## Physics feature arm (`with_physics`)

A separate pre-registered family of physics/geometry **analogues** is evaluated in the same pipeline: Gauss–Markov (local-linear-trend Kalman drift on log close, normalised by 60-minute realized volatility), Schrödinger/Ornstein–Uhlenbeck (60-minute displacement z-score and 240-minute mean-reversion rate `kappa`), Navier–Stokes (Reynolds analogue / 60-minute Kaufman efficiency ratio and its signed form), and Riemannian geometry (closed-form Fisher–Rao geodesic distance between the trailing 15-minute and 240-minute return Gaussians). The exact formulas, windows, constants, and the ablation decision rule were frozen in `physics_preregistration.md` **before any result for this arm was computed**; no window or constant was tuned afterwards and no alternatives were scanned. These are analogues only: the Riemann hypothesis/zeta and an actual Navier–Stokes PDE solve have no mechanism linking them to prices and are out of scope.

`PHYSICS_FEATURES` is exactly six columns: `gauss_kalman_drift_z`, `schrodinger_displacement_60m`, `schrodinger_ou_kappa_240m`, `ns_reynolds_60m`, `ns_signed_flow_60m`, `riemann_fisher_rao_15_240`. They are computed from closed candles and pass through the same final one-minute shift as the market features, so the historical, smoke, and live paths are identical. The Kalman drift uses the pre-registered ratios `q_level/R = 1e-4`, `q_slope/R = 1e-8`; the deterministic steady-state (Holt-equivalent) gains are `alpha = 0.01717137036324189` and `beta = 9.913771379428575e-05`. `MARKET_FEATURES` and `GCP_FEATURES` are unchanged.

Reproduce the amended run (writes `results_physics.json` and 32 `model_*_with_physics.joblib` artifacts; the default `--ablation gcp` path, `results.json`, and the existing `market_only`/`with_gcp` artifacts keep working). Add `--output-dir /tmp/physics_rerun` when a live process is loading the deployed artifacts, so a re-run does not overwrite them:

```bash
price-prediction/.venv/bin/python price-prediction/train.py --ablation physics \
  --bootstrap-hit-mse 4999 --bootstrap-ic 1999
```

**Result** (full source range, 3 expanding folds, 50% initial train, default embargo, paired 60-minute moving-block bootstrap at the amended resolution: `n_boot = 4999` for Δhit/ΔMSE and `1999` for ΔIC, i.e. two-sided p-value floors of `2/5000 = 0.0004` and `2/2000 = 0.0010`, both below the pre-registered threshold; see the amendment note below). Deltas are `with_physics` minus `market_only` on identical OOS rows; `pp` = percentage points; MSE is in squared return units; the pre-registered Bonferroni threshold over the 16 cells is `0.05/16 = 0.003125`.

| Asset | h | N dir | market hit % | with_physics hit % | with_physics IC | Δhit (pp; p) | ΔIC (p) | ΔMSE (p) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BTCUSDT | 1 | 645780 | 50.383 | 50.439 | 0.01512 | +0.0564; 0.0316 | +0.00016; 0.1160 | +9.030e-11; 0.0004 |
| BTCUSDT | 5 | 665349 | 51.396 | 51.437 | 0.03453 | +0.0404; 0.1068 | +0.00195; 0.1330 | +7.906e-10; 0.0004 |
| BTCUSDT | 15 | 666924 | 51.843 | 51.954 | 0.04340 | +0.1113; 0.7760 | +0.00379; 0.5860 | +3.376e-09; 0.0004 |
| BTCUSDT | 60 | 667176 | 51.487 | 51.949 | 0.04144 | +0.4616; 0.0888 | +0.00985; 0.3470 | +1.051e-08; 0.0004 |
| ETHUSDT | 1 | 659274 | 51.458 | 51.551 | 0.03520 | +0.0927; 0.5924 | +0.00068; 0.5230 | -3.099e-10; 0.1440 |
| ETHUSDT | 5 | 665943 | 52.097 | 52.244 | 0.04741 | +0.1472; 0.3024 | +0.00195; 0.3010 | -1.321e-09; 0.3404 |
| ETHUSDT | 15 | 666627 | 52.425 | 52.639 | 0.05053 | +0.2144; 0.3464 | +0.00406; 0.4670 | +1.571e-09; 0.8128 |
| ETHUSDT | 60 | 667005 | 51.673 | 52.208 | 0.04010 | +0.5352; 0.6592 | +0.00910; 0.9090 | +2.683e-08; 0.4744 |
| SOLUSDT | 1 | 633141 | 51.021 | 51.079 | 0.02190 | +0.0580; 0.7408 | -0.00029; 0.4210 | -7.812e-11; 0.7520 |
| SOLUSDT | 5 | 653124 | 51.689 | 51.735 | 0.03493 | +0.0459; 0.2628 | -0.00007; 0.2680 | -3.395e-10; 0.7232 |
| SOLUSDT | 15 | 659241 | 51.748 | 51.905 | 0.03558 | +0.1570; 0.7644 | +0.00260; 0.7620 | +2.331e-09; 0.1608 |
| SOLUSDT | 60 | 663360 | 50.726 | 51.578 | 0.02805 | +0.8516; 0.2760 | +0.01029; 0.9580 | +1.915e-08; 0.0600 |
| BNBUSDT | 1 | 625866 | 50.615 | 50.585 | 0.00484 | -0.0300; 0.9380 | -0.00077; 0.2010 | -8.092e-11; 0.0004 |
| BNBUSDT | 5 | 652299 | 50.665 | 50.986 | 0.02047 | +0.3218; 0.0920 | +0.00273; 0.1250 | -5.044e-10; 0.0004 |
| BNBUSDT | 15 | 658863 | 51.070 | 51.650 | 0.03123 | +0.5807; 0.0556 | +0.00648; 0.1130 | -8.601e-10; 0.0004 |
| BNBUSDT | 60 | 663117 | 51.277 | 51.567 | 0.03194 | +0.2901; 0.1084 | +0.00612; 0.8030 | +9.359e-09; 0.0004 |

Bonferroni survivors: **0/16 for Δhit and 0/16 for ΔIC**; for ΔMSE, **3/16 cells are significantly better** (`BNBUSDT` 1/5/15m, ΔMSE −8.1e-11…−8.6e-10) and **5/16 cells are significantly worse** (`BTCUSDT` 1/5/15/60m and `BNBUSDT` 60m, ΔMSE +9.0e-11…+1.1e-08), all at the 0.0004 floor. The shuffled-target control (BTCUSDT direction, `with_physics` features) is at chance: hit rate 49.981%–50.620% and IC −0.00113 to −0.00053 across the four horizons.

**Verdict: null.** There is no evidence that the physics arm adds skill under the pre-registered rule. The direction deltas lean positive (Δhit positive in 15/16 cells, median +0.152 pp; ΔIC positive in 13/16, median +0.00266) but the smallest hit-rate p-value is 0.0316 and the smallest IC p-value is 0.1130, above the 16-cell threshold, so neither direction metric reaches a majority of Bonferroni survivors. Volatility MSE is mixed rather than improved: 7/16 cells improve (median +4.4e-10) but 5 cells are significantly *worse* against only 3 significantly better, so the pre-registered volatility claim (≥9 improving and ≥9 surviving) fails and the arm cannot be called skill-enhancing for either task. This is unsurprising: the six features are dimensionless transforms of the same candle series, so the market-only arm already spans most of their information. `results_physics.json` retains every per-cell delta, p-value, bootstrap CI, fold layout, and control.

**Amendment (post-result, 2026-10-01).** The first physics run used the pipeline default `n_boot = 300`, whose two-sided bootstrap p-value floor `2/301 ≈ 0.00664` is *above* the pre-registered `0.003125` threshold, so no cell could ever have been recorded as surviving and the first "0/16 survivors" verdict was unsatisfiable by construction. Only the bootstrap resolution was amended (`n_boot = 4999` for Δhit/ΔMSE using the exact block-sum bootstrap, `1999` for ΔIC; floors 0.0004/0.0004/0.0010). Features, windows, constants, models, folds, the paired-row policy, the shuffled control, the Bonferroni threshold, and the decision rule are unchanged. The amended run reproduces the first run's deltas, folds, and controls bit-for-bit; only p-values changed (verified per cell). The change is recorded in `physics_preregistration.md` and in `results_physics.json` under `ablation.bootstrap`.

## Schrödinger wave-function arm

`schrodinger.py` literally solves the one-dimensional time-dependent
Schrödinger equation `i hbar dpsi/dt = [-(hbar^2/2m) d^2/dx^2 + V(x)] psi` on a
finite grid (193 points, `x in [-6, 6]`, `hbar = m = 1`) with a complex wave
function and reads the Born rule `|psi(x, tau)|^2`.  `x` is the forward log
return in units of the trailing h-minute log-return standard deviation; `V(x)`
is derived from trailing data by a weighted quartic fit of `-log p_hat` (the
Gaussian KDE of the trailing 240 standardized h-minute returns), and the
initial Gaussian packet carries a phase-encoded momentum `k0 = x0` (the current
displacement).  Production propagation is split-step Fourier (Strang
splitting); Crank-Nicolson on the same finite-difference Hamiltonian is the
reference solver.  The four Born-rule features per horizon (`p_up`, `<x>`,
`std`, excess kurtosis) enter the registered evaluation: arm S0 is the direct
readout with no fitted model, arm S1 is `market_only` vs `market_only` + the
four columns on identical rows, and a free-particle control is reported but
never selectable.  The frozen configuration `quartic_kde|linear_60|momentum`
was chosen among eight pre-declared candidates by a pre-declared selection
rule; the full exploration log is in `schrodinger_preregistration.md`.
Permanent solver tests (norm conservation, free-packet spreading vs the
analytic width, QHO coherent state and eigen-energies, split-step vs
Crank-Nicolson agreement, the Born-rule phase effect, and a causality guard)
are in `test_schrodinger.py` (9/9 passing).

```bash
price-prediction/.venv/bin/python price-prediction/test_schrodinger.py
price-prediction/.venv/bin/python price-prediction/train_schrodinger.py --stage explore
price-prediction/.venv/bin/python price-prediction/train_schrodinger.py --stage frozen
# one-shot holdout (Amendment A2: both arms fitted once on all dev rows, then scored on every holdout minute)
price-prediction/.venv/bin/python price-prediction/train_schrodinger.py --stage holdout \
  --db ../crypto-data/crypto_market.db --holdout-db ../crypto-data/holdout_klines_1m.db \
  --output price-prediction/results_schrodinger_holdout.json
```

**Verdict: null on dev and null on the holdout.**  Dev (hourly anchors,
walk-forward, 4999-replicate paired bootstrap): `Delta hit` 9/16 improving,
**0/16 Bonferroni survivors** (median +0.086 pp); `Delta IC` 11/16 improving,
**0/16 survivors** (median +0.00178); `Delta MSE` 8/16 improving with 1
survivor and 1 significantly worse.  Holdout (~18.5 days, every minute, both
arms fitted once on all dev rows): `Delta hit` 8/16 improving, **0/16
survivors** (median -0.025 pp); `Delta IC` 7/16 improving, **0/16 survivors**,
1/16 significantly worse (BTCUSDT 1m, p = 0.0004); `Delta MSE` 9/16 improving
but only **2/16 survivors** against the pre-registered >=9 requirement.  The
direct S0 readout carries a small signal whose **sign is not stable**: on dev
it is contrarian in 16/16 cells (hit rate 47.5-49.7%, IC negative in 16/16),
while on the holdout the h=1 cells flip to significantly *above* chance in 3/4
assets (BTCUSDT 51.909%, SOLUSDT 50.939%, BNBUSDT 51.040%) and only the h=15
cells stay contrarian.  The incremental value is ~0 because at h=1 the readout
is close to a monotone transform of the registered `ret_1m` (Spearman 0.965 on
dev; 0.331 at 5m, 0.170 at 15m, 0.046 at 60m) and the free-particle control
scores like the fitted potentials.  Numbers: `results_schrodinger_dev.json`,
`results_schrodinger_holdout.json`; protocol, exploration log, frozen spec,
and the pre-holdout (A2) and post-holdout reporting (A3) amendments:
`schrodinger_preregistration.md`.

## Edge lab (cost-aware exploration + holdout)

`edge_lab.py` is a separate cost-aware lab that pushed the mild leads from
`results_physics.json`.  It screened **27 configurations** (20 direction, 7
volatility) over the registered market/physics features, logistic/ridge/HGB
models, horizons 1-240 minutes, confidence filtering (`|p_up - 0.5| > theta`)
and taker/maker fee scenarios, on a dev-only split: EXPLORE = the first 70% of
dev minutes (`t_split = 2025-12-10 01:12 UTC`, with targets required to finish
inside EXPLORE), VALIDATE = the remainder after a 240-minute embargo (used only
to screen the shortlist and freeze three finalists).  The finalists were then
evaluated exactly once on the untouched 18.3-day holdout (26,584 minutes per
symbol; 26,343 evaluated minutes per symbol after the 240-minute embargo;
104,420-104,833 pooled direction rows per finalist), with a pre-registered
power rule that fewer than 30 holdout trades makes the net-return criterion
inconclusive.

```bash
python ../crypto-data/fetch_holdout.py   # refresh/extend the holdout DB (read-only w.r.t. crypto_market.db)
price-prediction/.venv/bin/python price-prediction/edge_lab.py --stage holdout \
  --holdout-db ../crypto-data/holdout_klines_1m.db
```

**Result: no edge claimed; the cost-aware criterion is inconclusive.**  On the
holdout the three finalists' full-coverage hit rates are 52.535% / 52.370% /
51.756% against persistence 48.08% and momentum 47.63% (paired 60-minute-block
bootstrap p at the 0.0004/0.0016 floor), but the gross edge at full coverage is
only a few tenths of a basis point per trade -- two orders of magnitude below
the 20 bp round-trip taker fee (10 bp/side) -- and every configuration is
net-negative at full coverage under every fee scenario.  The theta-filtered
trades that were the only route to a positive trade mean produced **0 / 3 / 1**
actual holdout trades against 2.6 / 8.6 / 5.1 expected from VALIDATE, so the
net-return criterion is underpowered for all three finalists and **no finalist
passes both pre-registered criteria**.  The full-coverage hit-rate advantage
persisting over 18.3 days is the one positive observation; it is not a tradable
claim, and the frozen test can be re-run unchanged on a longer holdout.
Details: `edge_lab_preregistration.md` (exploration log, frozen spec, holdout
table), `results_edge_lab_dev.json`, `results_edge_lab_holdout.json`.

## Mixed arm (crypto + GCP2 + Schrödinger)

`mix_lab.py` (protocol frozen in `mix_lab_preregistration.md` before any
result) asks whether GCP2 can be *coupled into* the literal Schrödinger wave
equation rather than added beside it as feature columns, and re-evaluates the
additive combination on identical rows.  It is a **null** on dev and on the
one-shot holdout.

- **Data.** Dev = `crypto_market.db` ∩ `gcp2_1m.db` over their common range
  (2024-03-01 00:00 -> 2026-09-02 23:59 UTC), hourly anchors.  Holdout = the
  untouched `holdout_klines_1m.db` bounded by the new `gcp2_holdout_1m.db`
  minute cache (2026-09-03 00:00 -> 2026-10-01 23:59 UTC, no gaps); the usable
  mixed window is 2026-09-14 10:09 -> 2026-10-02 00:00 UTC (~17.8 days,
  25,311 minutes per asset, ~421 60-minute blocks per cell).
- **Additive arms.** A0 `market_only`, A1 `+GCP`, A2 `+SCHR` (the frozen
  `quartic_kde|linear_60|momentum` columns), A3 `+GCP+SCHR`, all on the same
  complete-case rows; paired deltas A1-A0, A2-A0, A3-A0, A3-A2.
- **Coupled arms.** The registered causal `u = tanh(netvar_z_60m / 2)` drives
  exactly one element of the equation per pre-declared candidate: effective
  ħ = exp(±0.75 u), potential stiffness = exp(+u)·V, packet width =
  clip(σ·exp(±0.5 u), 0.35, 3), imaginary potential η = exp(+u) with
  W = (x/X_MAX)².  Each candidate is evaluated against **its own 30-day
  time-shifted GCP copy** (same marginal distribution, no alignment); the
  frozen candidate is `packet_widen`.  The absorbing candidate is deliberately
  non-unitary and its norm decay is verified against the analytic
  `exp(-2 η W τ)`.  A new coupled split-step propagator is tested to reduce
  bit-exactly to the registered solver at ħ = 1.
- **Dev results** (hourly anchors, 4999-replicate paired moving-block
  bootstrap, 3-anchor blocks): the GCP-driven couplings do not beat their own
  shifted controls — `packet_widen` gives Δhit 8/16 improving / 0 survivors
  (median -0.01 pp), ΔIC 12/16 / 0 (median +0.0014, smallest p = 0.0316),
  ΔMSE 9/16 / 0.  Additive arms: A2-A0 leans positive (Δhit 11/16, ΔIC 12/16,
  0 survivors), A1-A0 is flat-to-negative (6/16 both), and **A3-A2 shows GCP
  adds nothing on top of the Schrödinger columns** (Δhit 4/16, ΔIC 6/16).
  All 0/16 survivors at α = 0.003125.
- **One-shot holdout** (every holdout minute, `n_boot = 9999`, α_dir =
  0.05/160 = 3.125e-4, α_vol = 6.25e-4): **0/16 Bonferroni survivors for
  every metric in all five comparisons.**  Δhit / ΔIC improving cells: A1-A0
  2/16 and 2/16; A2-A0 8/16 and 6/16 (1 significantly worse IC: BTCUSDT 1m);
  A3-A0 5/16 and 2/16; A3-A2 5/16 and 3/16; `C_real-C_shift` 10/16 and 9/16
  (medians +0.06 pp and +0.00019, smallest p = 0.0420).  Volatility: 1 survivor
  each for A2-A0/A3-A0 (SOLUSDT 1m, below the ≥9 required).  The direct
  coupled Born-rule readout is contrarian on dev (16/16 cells, 46.9-48.6% hit)
  but at h = 1 flips to significantly *above* chance on the holdout (BTCUSDT
  51.867%, BNBUSDT 51.001%, SOLUSDT 50.940%) while h = 15/60 stays contrarian
  — the same sign instability the Schrödinger arm showed, reported
  descriptively only.
- **Deployment limitation.** The live `gcp2.net getcurrentnetvar` endpoint
  returns a different quantity (values of order 10-220) than the archive
  `netvar` series used here (order 1, standard deviation ~0.18).  The frozen
  coupling map and all features are defined on the archive quantity, so this
  arm is **not live-deployable** on that endpoint; it writes no live
  artifacts, and any future live use needs a separately pre-registered
  mapping between the two quantities.

```bash
price-prediction/.venv/bin/python price-prediction/test_mix_lab.py
price-prediction/.venv/bin/python price-prediction/mix_lab.py --stage explore   # exploration + selection + frozen dev tables
price-prediction/.venv/bin/python price-prediction/mix_lab.py --stage frozen    # frozen dev tables only
price-prediction/.venv/bin/python price-prediction/mix_lab.py --stage holdout \
  --db ../crypto-data/crypto_market.db --gcp ../crypto-data/gcp2_1m.db \
  --holdout-db ../crypto-data/holdout_klines_1m.db \
  --gcp-holdout-db ../crypto-data/gcp2_holdout_1m.db \
  --dev-results results_mix_lab_dev.json --output results_mix_lab_holdout.json --bootstrap 9999
```

## Seed lab (determinism, Gauss, folk TA, cycles, LPPLS)

`seed_lab.py` (protocol, formulas and decision rules frozen in
`seed_lab_preregistration.md` before any result) asks whether the market
behaves like a procedural generator with a fixed seed. A PRNG seed cannot be
recovered from prices because the generating process is aggregated order flow,
not a known generator. The testable version is that the 1m return path is
(close to) a low-dimensional deterministic system. If so, nearest-neighbour
forecasts on a delay embedding should have high skill that decays with
horizon and improves with library size, and the sign sequence should
compress.

- **Families** (all causal, same 1-minute shift as `features.py`): F1 simplex
  projection (E = 6) and S-map (θ = 2) forecasts plus in-library skill; F2
  Gaussian-process (RBF + white) drift and posterior std; F3 folk TA:
  Fibonacci retracement distance, Gann Square-of-9 phase, round-number
  distance, 369 digital roots, lunar phase, golden-ratio time since swing,
  Benford χ² of volumes; F4 FFT dominant-cycle extrapolation, Hurst (R/S),
  permutation entropy; F5 LPPLS grid fit (bubble confidence, fitted slope);
  KS = market + F1–F5 + frozen Schrödinger + physics (44 columns); CAL
  (lunar + `dr369_minute`) against a decorrelating time-shifted copy.
- **Determinism diagnostics (dev).** Simplex rho at h = 1 is −0.030…+0.029
  per asset and shows no growth from L = 250 to 8000. The S-map is best at
  θ = 0, a linear AR(6); only SOLUSDT reaches rho 0.20, and that is linear.
  lzma ratios are 0.984–1.001 against 200 shuffles. BDS rejects i.i.d., but
  vol-standardization roughly halves it, which matches volatility clustering
  rather than determinism. **None of the five pre-registered H_seed criteria
  hold.**
- **One-shot holdout** (never-evaluated window 2026-10-02 16:13 → 2026-10-10
  20:09 UTC, 11,757 minutes per cell, models fitted once on dev, n_boot =
  9999, α = 0.05/112): **0/16 survivors for Δhit and ΔIC in all seven
  comparisons**. Volatility: F2 16/16 improving with 4 survivors, F3 11/7,
  KS 11/2, all below the ≥9 surviving rule. Direct readouts (simplex, S-map,
  GP drift, FFT, LPPLS) are 47.98–53.61% with at most 1/16 significant: null.
  CAL−CAL_S meets the volatility rule (13/16 surviving), but every ΔMSE is
  8e-18…8e-16, a relative 4e-11…1e-9. The calendar terms are nearly
  constant over 8 days, so the paired difference never changes sign. Dev
  showed 1 improving and 1 worse cell. The note is recorded in the
  preregistration; this is not evidence for lunar or 369 effects.

```bash
price-prediction/.venv/bin/python price-prediction/test_seed_lab.py
price-prediction/.venv/bin/python price-prediction/seed_lab.py --stage explore   # diagnostics + dev tables (~1 h, 8 workers)
price-prediction/.venv/bin/python price-prediction/seed_lab.py --stage holdout \
  --holdout-db crypto-data/holdout_klines_1m.db                               # one-shot; refuses to re-run (~4.4 h)
```

## Data and alignment

- `features.py` verifies `PRAGMA table_info(klines_1m)` before reading. The verified columns are `symbol`, `open_time_s`, `open`, `high`, `low`, `close`, `volume`, `quote_volume`, and `trades`. The four assets each have about 1.32 million clean rows beginning at epoch `1709251200` (2024-03-01 00:00 UTC); the exact audit is in `results.json`.
- GCP minute data is the mean of 60 one-second netvar values. The verified cache has 1,273,377 rows from 2024-03-01 through 2026-09-02 23:59 UTC; `features.py` can reproduce it by streaming the raw zips when the cache is not supplied.
- A row's prediction timestamp is candle open `t`. Features are shifted one minute and use only candles with `open_time_s < t`; GCP minute `t-1` is the newest GCP observation used. Calendar features are deterministic and are shifted with the feature matrix (they use no market data).
- Direction target: `close[t+h-1] / close[t-1] - 1`; direction is its sign. Volatility target: `sqrt(sum(return[t:t+h-1]^2))`, where `return[t] = close[t]/close[t-1]-1`. Thus all target inputs are future relative to the feature row.
- The untouched out-of-sample holdout is `crypto-data/holdout_klines_1m.db` (same verified `klines_1m` schema; four assets; contiguous minutes from 2026-09-14 05:09 UTC, directly following the dev database's last candle at 05:08 UTC; 38,341 minutes per asset through 2026-10-10 20:09 UTC at the last refresh). `crypto-data/fetch_holdout.py` appends new closed candles to it (idempotent, read-only with respect to `crypto_market.db`; `--verify` runs integrity checks without HTTP). The holdout is used only as an evaluation source by the Schrödinger arm, edge lab, mixed arm and seed lab (`--stage holdout`), never for fitting. Minutes up to 2026-10-02 16:12 UTC were scored by the earlier arms; 2026-10-02 16:13 → 2026-10-10 20:09 UTC is the seed lab's primary window.

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
- `results_physics.json` -- the same machine-readable structure for the pre-registered `market_only` vs `with_physics` ablation, including the Bonferroni summary.
- `physics_preregistration.md` -- frozen formulas, constants, and decision rule for the physics analogue family.
- `schrodinger.py` -- literal 1-D time-dependent Schrödinger solver (split-step Fourier production solver + Crank-Nicolson reference) and the causal wave-packet feature builder.
- `train_schrodinger.py` -- Schrödinger arm evaluation: dev exploration grid, frozen dev run, and the one-shot holdout stage (Amendment A2 fixed-fit protocol).
- `test_schrodinger.py` -- permanent solver verification tests (norm conservation, free-packet spreading, QHO coherent state and eigen-energies, cross-solver agreement, phase effect, causality guard).
- `schrodinger_preregistration.md` -- frozen spec, exploration log, dev results, holdout plan and decision rule, and the A1/A2/A3 amendments.
- `results_schrodinger_dev.json` -- dev exploration log (9 configurations) plus the 4999-replicate frozen dev table, S0 chance tests, diagnostics, and controls.
- `results_schrodinger_holdout.json` -- the one-shot holdout results for the frozen Schrödinger configuration.
- `edge_lab.py` -- cost-aware exploration lab (27 configurations, confidence filtering, fee scenarios, frozen one-shot holdout stage).
- `edge_lab_preregistration.md` -- edge-lab protocol, exploration log, frozen finalists, and holdout results.
- `results_edge_lab_dev.json` / `results_edge_lab_holdout.json` -- edge-lab dev screening and one-shot holdout numbers.
- `mix_lab.py` -- mixed arm: additive market/GCP/Schrödinger arms, six GCP-driven wave-equation couplings each paired with a 30-day shifted-GCP control, and the one-shot holdout stage (`--holdout-db`/`--gcp-holdout-db`).
- `test_mix_lab.py` -- permanent tests for the coupled propagator (identity to the registered solver, norm conservation, analytic ħ scalings, absorbing-potential decay, coupling maps, shifted-control lag, causality).
- `mix_lab_preregistration.md` -- mixed-arm protocol, exploration log, frozen spec, dev results, holdout decision rule and one-shot holdout results, amendment log.
- `results_mix_lab_dev.json` / `results_mix_lab_holdout.json` -- mixed-arm dev exploration/frozen tables and the one-shot holdout numbers.
- `seed_lab.py` -- seed lab: determinism diagnostics (simplex/S-map, lzma, BDS) and the F1–F5/KS/CAL arms with dev and one-shot holdout stages.
- `test_seed_lab.py` -- permanent numeric tests (simplex skill on Lorenz/logistic map vs noise, GP trend recovery, FFT sinusoid continuation, LPPLS parameter recovery, lunar phase, BDS, chunked Schrödinger equivalence, causality).
- `seed_lab_preregistration.md` -- seed-lab frozen spec, H_seed signature, dev and holdout results, amendment log.
- `results_seed_lab_dev.json` / `results_seed_lab_holdout.json` -- seed-lab dev diagnostics/tables and the one-shot holdout numbers.
- `crypto-data/gcp2_holdout_1m.db` -- holdout GCP minute cache (2026-09-03 -> 2026-10-09 23:59 UTC, no gaps); `crypto-data/fetch_gcp_holdout.py` refreshes/extends it.
- `crypto-data/holdout_klines_1m.db` -- untouched holdout klines (Sept 2026 -> now); `crypto-data/fetch_holdout.py` refreshes/extends it.
- `model_*.joblib` -- final-history artifacts for each asset, horizon, task, and feature arm (now including `model_*_with_physics.joblib`).
- `research.md` -- time-boxed web research and source links.

## Live inference

`live.py` runs the persisted models on the newest **closed** Binance 1-minute
candles and predicts the next minute open:

```bash
price-prediction/.venv/bin/python price-prediction/live.py --once
price-prediction/.venv/bin/python price-prediction/live.py
```

Each cycle requests `limit=600` candles for BTCUSDT, ETHUSDT, SOLUSDT, and
BNBUSDT, filters by `close_time < now`, and keeps the inner intersection of
common minute opens (at least 61 contiguous minutes are required).  The
600-candle limit exists only so the physics arm's 479-candle OU warm-up is
available in one request; every market/GCP feature is trailing-only, so their
values do not depend on the limit.  The `--symbol` option limits emitted
records but still fetches all four assets because cross-asset dispersion is a
registered feature.  Feature construction is imported from `features.py`; no
live-only feature formulas are duplicated.

Whenever the `model_*_with_physics.joblib` artifacts are present and the
registered physics windows are warm, every horizon record also contains a
`with_physics` arm (direction score, probability, and volatility prediction)
next to `market_only`, plus a top-level `physics_available` flag.  The arm is
computed from the same closed candles, needs no external service, and is
omitted (with a log line) when its artifacts or warm-up history are missing;
it never blocks the market-only output.

The downstream live tools handle the extra arm: `live_eval.py` scores every arm
present in the log (`market_only`, `with_gcp`, `with_physics`) in
`scorecard[symbol][horizon].arms`, and `monitor.py` records `with_physics_n`
alongside `with_gcp_n`.

`significance.py` still tests **only the `market_only` arm in the automated
pipeline**: its default `--arm market_only` is what the orchestrator invokes
every cycle, and no scheduled run covers `with_physics` today. The same
pre-registered cell tests can be run manually for the physics arm with
`significance.py --arm with_physics [--windowed]`; records lacking the arm
yield empty cells and the default output schema is unchanged.

**Deployed artifacts:** the 2026-10-01 physics full run rewrote the 32
`model_*_market_only.joblib` files and added the 32
`model_*_with_physics.joblib` files (`with_gcp` artifacts and `results.json`
were not rewritten). The rewrite used the same source range and the same
market-feature code as the earlier run, and a from-scratch refit with the
current code reproduces the deployed market-only scalers and coefficients
exactly (checked on BTCUSDT 1m direction and 60m volatility), so live
`market_only` scores before and after that date come from the same fit in
substance; only the artifact files and their embedded manifest were
regenerated.

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
The 5,000-resample IC bootstrap ranks all resamples of a cell or window in one vectorized block: average-rank Pearson, the same definition as `scipy.stats.spearmanr`. On a frozen live snapshot it reproduced every field of the per-resample implementation to within 1e-12.
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
All three tools share `kline_cache.sqlite` (see *Live outcome evaluation*), so each cycle downloads only windows that matured since the previous cycle instead of every window ever logged.

## Live outcome evaluation

`live_eval.py` closes the loop from `live.py`'s append-only prediction log to
realized Binance outcomes.  It reads
`price-prediction/live_predictions.jsonl`, keeps the latest line for each
`(symbol, target_time_s, horizon)` key, and fetches the required 1-minute
klines from Binance's public REST API.  A target is scored only after every
required kline has `close_time < now`; predictions whose final candle is
still open are reported as pending and retained for a later continuous-loop
poll.  HTTP failures are retried once and then reported as skipped rather
than stopping the evaluator.  Candles are served first from the shared
persistent store `price-prediction/kline_cache.sqlite` (`--kline-cache PATH`;
WAL, `INSERT OR IGNORE`, holds only candles whose `close_time` was at least
10 s in the past when fetched, so values are immutable) and then from the
read-only `crypto-data/holdout_klines_1m.db` (`--kline-seed-db PATH`; same
Binance endpoint and `close` values, `close_time = open + 59,999 ms`); only the
still-missing open range of a window is requested from Binance.
`--no-kline-cache` restores the per-process in-memory cache; `--self-test`
never touches either file.

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
