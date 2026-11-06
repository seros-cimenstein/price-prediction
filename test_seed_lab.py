"""Permanent verification tests for the numerical machinery in ``seed_lab.py``.

Run with::

    price-prediction/.venv/bin/python price-prediction/test_seed_lab.py

(pytest is not installed in this environment; the file also works under pytest.)

1. Simplex projection has high skill on deterministic chaos (logistic map,
   Lorenz) and ~zero skill on i.i.d. noise; S-map beats the global linear
   map (theta = 0) on the logistic map.
2. The shared-distance ``f1_features`` equals the reference simplex / LOO /
   S-map functions exactly.
3. GP extrapolation recovers a linear trend; the precomputed linear filter
   equals the direct GP solve.
4. FFT extrapolation continues a pure (integer-bin) sinusoid exactly and an
   off-bin sinusoid approximately (frequency quantized to the nearest bin).
5. LPPLS recovers synthetic grid parameters and amplitudes; the analytic
   slope equals a finite difference of the true curve.
6. Lunar phase is ~full at the 2024-04-23 23:49 UTC full moon and ~new at the
   2024-04-08 18:21 UTC new moon (via the feature builder as well).
7. BDS is ~N(0,1) on i.i.d. data and huge on the logistic map; Hurst and
   permutation entropy behave as expected on noise / persistent / monotone
   inputs; Benford, digital root, round-number and Fibonacci helpers.
8. Causality guard: perturbing candles at/after an anchor leaves every seed
   feature at that anchor unchanged; off-grid anchors are rejected.
9. Splitting anchors into chunks (as the process pool does) reproduces the
   registered Schrodinger builder to floating-point rounding (< 1e-10).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from seed_lab import (  # noqa: E402
    FROZEN_SCHR,
    LPPLS_WINDOW,
    benford_chi2,
    bds_statistic,
    build_seed_features,
    digital_root,
    f1_features,
    fft_projection,
    folk_ta,
    gp_extrapolate,
    gp_filter,
    hurst_rs,
    lppls_derivative,
    lppls_features,
    lppls_fit,
    lppls_grid,
    lunar_phase_fraction,
    pearson,
    permutation_entropy,
    simplex_forecast,
    simplex_loo,
    smap_forecast,
)
from schrodinger import build_features_for_asset_multi  # noqa: E402


def _logistic(n: int, r: float = 3.9, x0: float = 0.3141) -> np.ndarray:
    x = np.empty(n)
    x[0] = x0
    for i in range(1, n):
        x[i] = r * x[i - 1] * (1.0 - x[i - 1])
    return x


def _lorenz_x(n: int, dt: float = 0.01, every: int = 5) -> np.ndarray:
    def f(s):
        x, y, z = s
        return np.array([10.0 * (y - x), x * (28.0 - z) - y, x * y - 8.0 / 3.0 * z])

    s = np.array([1.0, 1.0, 1.0])
    out = []
    for i in range((n + 200) * every):
        k1 = f(s)
        k2 = f(s + 0.5 * dt * k1)
        k3 = f(s + 0.5 * dt * k2)
        k4 = f(s + dt * k3)
        s = s + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        if i % every == 0:
            out.append(s[0])
    return np.asarray(out[200 : 200 + n])


def _std(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / x.std(ddof=1)


def test_simplex_skill_chaos_vs_noise() -> None:
    z_log = _std(_logistic(2000))
    rho_log = pearson(*simplex_loo(z_log, 2, 1, 500))
    z_lor = _std(_lorenz_x(3000))
    rho_lor = pearson(*simplex_loo(z_lor, 3, 1, 500))
    rng = np.random.default_rng(1)
    z_noise = rng.standard_normal(3000)
    rho_noise = pearson(*simplex_loo(z_noise, 6, 1, 1000))
    print(f"simplex LOO rho: logistic {rho_log:.4f}, Lorenz {rho_lor:.4f}, iid noise {rho_noise:+.4f}")
    assert rho_log > 0.95, rho_log
    assert rho_lor > 0.95, rho_lor
    assert abs(rho_noise) < 0.12, rho_noise
    # Registered F1 columns (E=6, L=2000) on the same kinds of input.
    f_log = f1_features(z_log)
    f_noise = f1_features(_std(z_noise[:2000]))
    print(f"F1 in-library rho (E=6): logistic h=1 {f_log[1][1]:.4f}, noise h=1 {f_noise[1][1]:+.4f}")
    assert f_log[1][1] > 0.9
    assert abs(f_noise[1][1]) < 0.25
    # Skill decays with horizon on chaos (finite Lyapunov horizon).
    assert f_log[1][1] > f_log[15][1]
    # Non-linear S-map beats the global linear AR on the logistic map.
    preds0, predsS, obs = [], [], []
    for end in range(1500, 1700):
        seg = z_log[: end + 1]
        preds0.append(smap_forecast(seg[-1000:], 2, 1, 0.0))
        predsS.append(smap_forecast(seg[-1000:], 2, 1, 4.0))
        obs.append(z_log[end + 1])
    r0, r4 = pearson(np.array(preds0), np.array(obs)), pearson(np.array(predsS), np.array(obs))
    print(f"S-map rho on logistic: theta=0 {r0:.4f}, theta=4 {r4:.4f}")
    assert r4 > r0 + 0.1


def test_f1_features_match_reference() -> None:
    rng = np.random.default_rng(7)
    z = _std(np.cumsum(rng.standard_normal(2000)) * 0.1 + rng.standard_normal(2000))
    fast = f1_features(z)
    for h in (1, 5, 15, 60):
        ref_simplex = simplex_forecast(z, 6, h) / np.sqrt(h)
        ref_smap = smap_forecast(z, 6, h, 2.0) / np.sqrt(h)
        ref_rho = pearson(*simplex_loo(z, 6, h, 250))
        assert np.isclose(fast[h][0], ref_simplex, rtol=0, atol=1e-12), (h, fast[h][0], ref_simplex)
        assert np.isclose(fast[h][2], ref_smap, rtol=1e-9, atol=1e-12), (h, fast[h][2], ref_smap)
        assert np.isclose(fast[h][1], ref_rho, rtol=0, atol=1e-12), (h, fast[h][1], ref_rho)
    print("f1_features == reference simplex/LOO/S-map for h in 1,5,15,60")


def test_gp_recovers_linear_trend() -> None:
    rng = np.random.default_rng(3)
    s = np.arange(-119, 1, dtype="float64")
    u = 2.0 + 0.5 * s + 0.01 * rng.standard_normal(len(s))
    for h in (1, 5, 15, 60):
        mean, sd = gp_extrapolate(u, s, float(h))
        assert abs(mean - (2.0 + 0.5 * h)) < 0.05, (h, mean)
        weights, sd_filter = gp_filter(h)
        assert np.isclose(weights @ u, mean, atol=1e-9)
        assert np.isclose(sd, sd_filter)
        assert sd > 1.0  # predictive sd includes the white term
    # Predictive sd grows with extrapolation distance.
    assert gp_filter(60)[1] > gp_filter(1)[1]
    print("GP: linear trend recovered at h=1..60; filter == direct solve; sd grows with h")


def test_fft_continues_sinusoid() -> None:
    k = np.arange(256 + 64)
    x = np.sin(2 * np.pi * 8 * k / 256 + 0.3) + 0.5 * np.cos(2 * np.pi * 20 * k / 256 - 1.1)
    proj = fft_projection(x[None, :256], 64)[0]
    err = np.max(np.abs(proj - x[256:]))
    print(f"FFT integer-bin sinusoid continuation max error {err:.2e}")
    assert err < 1e-9
    y = np.sin(2 * np.pi * k / 37.3 + 0.7)
    proj_y = fft_projection(y[None, :256], 15)[0]
    corr = pearson(proj_y, y[256 : 256 + 15])
    print(f"FFT off-bin sinusoid (period 37.3) 15-step correlation {corr:.4f}")
    # Peak picking quantizes the frequency to the nearest bin (6.86 -> 7), so the
    # off-bin continuation drifts in phase; it must still track the sinusoid.
    assert corr > 0.5


def _lppls_curve(tau, D, m, omega, A, B, C1, C2):
    x = (LPPLS_WINDOW - 1 + D) - tau
    f = x ** m
    return A + B * f + C1 * f * np.cos(omega * np.log(x)) + C2 * f * np.sin(omega * np.log(x))


def test_lppls_recovers_parameters() -> None:
    points, _d, _p = lppls_grid()
    tau = np.arange(LPPLS_WINDOW, dtype="float64")
    truth = dict(D=60, m=0.5, omega=9.0, A=0.4, B=-0.02, C1=0.003, C2=-0.002)
    rng = np.random.default_rng(11)
    y = _lppls_curve(tau, **truth) + 1e-6 * rng.standard_normal(len(tau))
    fit = lppls_fit(y[None, :])
    best = int(fit["best"][0])
    assert points[best] == (60, 0.5, 9.0), points[best]
    coefs = fit["coefs"][0, best]
    assert np.allclose(coefs, [truth["A"], truth["B"], truth["C1"], truth["C2"]], atol=1e-4), coefs
    # Analytic derivative vs a central finite difference of the true curve.
    tau_star = np.array([LPPLS_WINDOW - 1 + 15.0])
    analytic = lppls_derivative(np.array([60.0]), np.array([0.5]), np.array([9.0]),
                                np.array([[truth["A"], truth["B"], truth["C1"], truth["C2"]]]), tau_star)[0]
    eps = 1e-4
    numeric = (_lppls_curve(tau_star + eps, **truth) - _lppls_curve(tau_star - eps, **truth))[0] / (2 * eps)
    assert np.isclose(analytic, numeric, rtol=1e-6), (analytic, numeric)
    feats = lppls_features(y[None, :])
    assert int(feats["best_index"][0]) == best
    assert 0.0 <= feats["lppls_conf"][0] <= 1.0
    print(f"LPPLS: grid point {points[best]} and A,B,C1,C2 recovered; slope analytic {analytic:.6e} == numeric")


def test_lunar_phase() -> None:
    full = 1713916140  # 2024-04-23 23:49 UTC
    new = 1712600460  # 2024-04-08 18:21 UTC (total solar eclipse)
    f_full = float(lunar_phase_fraction(np.array([full]))[0])
    f_new = float(lunar_phase_fraction(np.array([new]))[0])
    print(f"lunar phase fraction: full moon {f_full:.4f}, new moon {f_new:.4f}")
    assert abs(f_full - 0.5) < 0.03
    assert min(f_new, 1.0 - f_new) < 0.03
    # Through the feature builder: the calendar columns use the open of candle t-1.
    times = np.arange(full - 2500 * 60, full + 120, 60, dtype="int64")
    close = np.full(len(times), 100.0)
    feats = build_seed_features(close, np.ones(len(times)), times, np.array([full + 60]))
    assert feats["lunar_cos"].iloc[0] < -0.98, feats["lunar_cos"].iloc[0]
    print(f"builder lunar_cos at full moon {feats['lunar_cos'].iloc[0]:.4f}")


def test_bds_hurst_entropy_and_helpers() -> None:
    rng = np.random.default_rng(5)
    iid = rng.standard_normal(1500)
    w_iid = bds_statistic(iid)
    w_log = bds_statistic(_std(_logistic(1500)))
    print(f"BDS m=2..5: iid {np.round(w_iid, 2)}, logistic {np.round(w_log, 1)}")
    assert np.all(np.abs(w_iid) < 3.5)
    assert np.all(w_log > 10.0)
    noise = rng.standard_normal((40, 512))
    h_noise = hurst_rs(noise)
    h_persist = hurst_rs(np.cumsum(noise, axis=1))
    print(f"Hurst R/S: iid mean {h_noise.mean():.3f}, cumulative-sum mean {h_persist.mean():.3f}")
    assert 0.45 < h_noise.mean() < 0.68
    assert h_persist.mean() > 0.9
    pe_noise = permutation_entropy(rng.standard_normal((20, 240)))
    pe_mono = permutation_entropy(np.arange(240.0)[None, :])
    assert pe_noise.min() > 0.95 and pe_mono[0] == 0.0
    benford_like = 10.0 ** rng.uniform(0, 5, size=(200, 60))
    flat = np.full((1, 60), 777.0)
    chi_b = benford_chi2(benford_like)
    chi_f = benford_chi2(flat)
    assert np.median(chi_b) < 15.5 and chi_f[0] > 100.0, (np.median(chi_b), chi_f)
    assert np.isnan(benford_chi2(np.zeros((1, 60)))[0])
    assert digital_root(np.array([0, 9, 10, 369, 6000000])).tolist() == [0, 9, 1, 9, 6]
    c = np.linspace(100.0, 200.0, 240)
    c[-2] = 200.0  # hi = 200, lo = 100
    c[-1] = 100.0 + 0.382 * 100.0  # p = 0.382 -> nearest level 0.382, distance 0
    out = folk_ta(c[None, :], np.ones((1, 60)))
    assert abs(out["fib_dist"][0]) < 1e-12, out["fib_dist"]
    flat_price = np.full((1, 240), 63421.0)
    flat_price[0, 0] = 63500.0
    out2 = folk_ta(flat_price, np.ones((1, 60)))
    assert np.isclose(out2["round_logdist"][0], np.log(63421.0 / 63000.0))
    print(f"Benford chi2 median (Benford data) {np.median(chi_b):.2f}, flat {chi_f[0]:.1f}; helpers ok")


def _synthetic_market(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    times = 1_700_000_000 - (1_700_000_000 % 60) + 60 * np.arange(n, dtype="int64")
    close = 30000.0 * np.exp(np.cumsum(0.001 * rng.standard_normal(n)))
    volume = rng.lognormal(3.0, 1.0, n)
    return times, close, volume


def test_causality_future_candles_do_not_leak() -> None:
    times, close, volume = _synthetic_market(4000)
    anchors = times[2600:2660:3]
    base = build_seed_features(close, volume, times, anchors)
    k = 9
    pos = int(np.searchsorted(times, anchors[k]))
    rng = np.random.default_rng(99)
    close2 = close.copy()
    volume2 = volume.copy()
    close2[pos:] *= np.exp(0.01 * rng.standard_normal(len(close) - pos))
    volume2[pos:] = rng.lognormal(5.0, 2.0, len(volume) - pos)
    perturbed = build_seed_features(close2, volume2, times, anchors)
    cols = [c for c in base.columns if c != "target_time_s"]
    a = base.iloc[: k + 1][cols].to_numpy(dtype="float64")
    b = perturbed.iloc[: k + 1][cols].to_numpy(dtype="float64")
    assert np.array_equal(np.isnan(a), np.isnan(b)) and np.allclose(a, b, rtol=0, atol=0, equal_nan=True)
    assert np.isfinite(a).all(), "features should be finite with 2600 candles of history"
    later = ~np.isclose(base.iloc[k + 1 :][cols].to_numpy(), perturbed.iloc[k + 1 :][cols].to_numpy(), equal_nan=True)
    assert later.any(), "perturbation should affect anchors after the perturbed candle"
    try:
        build_seed_features(close, volume, times, np.array([anchors[0] + 30]))
    except AssertionError:
        pass
    else:
        raise AssertionError("off-grid anchor was not rejected")
    print(f"causality: {len(cols)} columns x {k + 1} anchors unchanged by future candles; off-grid anchor rejected")


def test_chunked_schrodinger_equals_registered() -> None:
    times, close, _volume = _synthetic_market(1500, seed=4)
    series = pd.Series(close, index=pd.to_datetime(times, unit="s", utc=True))
    anchors = times[400:1500:7]
    for h in (1, 60):
        full, _a, _s = build_features_for_asset_multi(series, anchors, h, [FROZEN_SCHR])
        half = len(anchors) // 2
        p1, _a1, _s1 = build_features_for_asset_multi(series, anchors[:half], h, [FROZEN_SCHR])
        p2, _a2, _s2 = build_features_for_asset_multi(series, anchors[half:], h, [FROZEN_SCHR])
        joined = pd.concat([p1[FROZEN_SCHR.label], p2[FROZEN_SCHR.label]], ignore_index=True)
        diff = np.max(np.abs(full[FROZEN_SCHR.label].to_numpy() - joined.to_numpy()))
        # Batched FFT/BLAS kernels round differently with the batch size (~1e-13).
        assert diff < 1e-10, (h, diff)
    print("chunked Schrodinger features == single-call registered builder to <1e-10 (h=1, 60)")


def main() -> int:
    tests = [
        test_simplex_skill_chaos_vs_noise,
        test_f1_features_match_reference,
        test_gp_recovers_linear_trend,
        test_fft_continues_sinusoid,
        test_lppls_recovers_parameters,
        test_lunar_phase,
        test_bds_hurst_entropy_and_helpers,
        test_causality_future_candles_do_not_leak,
        test_chunked_schrodinger_equals_registered,
    ]
    failures = 0
    for test in tests:
        try:
            test()
        except AssertionError as exc:
            failures += 1
            print(f"FAILED {test.__name__}: {exc}")
            continue
        print(f"ok {test.__name__}")
    print(f"\n{len(tests) - failures}/{len(tests)} seed_lab tests passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
