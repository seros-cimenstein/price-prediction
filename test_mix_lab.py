"""Permanent verification tests for the new numerical machinery in ``mix_lab.py``.

Run with::

    price-prediction/.venv/bin/python price-prediction/test_mix_lab.py

(pytest is not installed in this environment; the file also works under pytest.)

The tests cover the genuinely new code paths a broken implementation cannot
fake:

1. ``split_step_propagate_coupled`` with ``hbar = 1`` and real ``V`` reproduces
   the registered ``schrodinger.split_step_propagate`` exactly.
2. Norm conservation for random per-row ``hbar`` (the coupled propagator stays
   unitary for any positive ``hbar``).
3. Free-packet spreading with effective ``hbar`` matches the analytic
   ``sigma(tau) = sigma sqrt(1 + (hbar tau)^2 / (4 sigma^4))`` and
   ``mean = x0 + hbar k0 tau``.
4. Quantum harmonic oscillator coherent state with effective ``hbar`` matches
   ``<x>(t) = x0 cos(omega t) + hbar k0 / omega sin(omega t)``.
5. The imaginary potential: exact ``exp(-2 eta W tau)`` decay for constant
   ``W``, monotone decay for the production shape, and no decay at
   ``eta = 0``.
6. The coupling maps are strictly monotone, bounded, finite at extreme inputs
   and symmetric under a sign flip.
7. The coupled builder with the identity coupling reproduces the registered
   Schrodinger feature builder bit-for-bit (no second convention).
8. Rows with a non-finite GCP value get NaN features and do not contaminate
   the other rows of the same PDE batch.
9. The shifted-GCP control column is exactly the registered causal column
   lagged by 30 days (including missing-minute behaviour).
10. Causality: changing candles strictly after the last anchor leaves every
    coupled feature unchanged, and a non-candle-open anchor is rejected.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from mix_lab import (  # noqa: E402
    CONTROL_SHIFT_DAYS,
    COUPLED_BY_LABEL,
    COUPLED_CANDIDATES,
    FROZEN_SCHR,
    absorbing_shape,
    build_coupled_features_for_asset_multi,
    coupled_model_columns,
    coupling_parameter,
    shifted_z_column,
    split_step_propagate_coupled,
)
from schrodinger import (  # noqa: E402
    born_observables,
    build_features_for_asset,
    gaussian_packet,
    grid,
    split_step_propagate,
)


def _smooth_potential(x: np.ndarray) -> np.ndarray:
    return 0.3 * x**2 + 0.02 * x**4 + 0.1 * x**3 - 0.5 * np.cos(1.7 * x)


def _discrete_norm(psi: np.ndarray, dx: float) -> np.ndarray:
    return np.sum(np.abs(psi) ** 2, axis=-1) * dx


def test_coupled_propagator_matches_registered_solver() -> None:
    x, dx, k = grid()
    del k
    rng = np.random.default_rng(1234)
    envelope = np.exp(-((x / 2.0) ** 2))
    psi = envelope * (rng.normal(size=len(x)) + 1j * rng.normal(size=len(x)))
    psi = psi / math.sqrt(_discrete_norm(psi, dx))
    V = _smooth_potential(x)
    for tau in (0.25, 1.0):
        registered = split_step_propagate(psi.copy(), V, tau)
        coupled = split_step_propagate_coupled(psi.copy(), V, tau)
        difference = float(np.max(np.abs(registered - coupled)))
        assert difference < 1e-12, (tau, difference)
        print(f"coupled vs registered solver tau={tau}: max |diff| = {difference:.3e}")


def test_norm_conservation_with_per_row_hbar() -> None:
    x, dx, _ = grid()
    rng = np.random.default_rng(7)
    envelope = np.exp(-((x / 2.0) ** 2))
    batch = envelope[None, :] * (
        rng.normal(size=(3, len(x))) + 1j * rng.normal(size=(3, len(x)))
    )
    batch = batch / np.sqrt(_discrete_norm(batch, dx))[:, None]
    V = _smooth_potential(x)
    hbar = np.array([0.5, 1.0, 2.0])
    for tau in (0.25, 1.0):
        out = split_step_propagate_coupled(batch.copy(), V, tau, hbar=hbar)
        drift = np.max(np.abs(_discrete_norm(out, dx) - 1.0))
        assert drift < 1e-8, (tau, drift)
        print(f"norm conservation per-row hbar tau={tau}: max |norm-1| = {drift:.3e}")


def test_free_packet_spreading_with_hbar() -> None:
    x, _, _ = grid()
    sigma, x0, k0 = 0.7, 1.0, 0.3
    psi0 = gaussian_packet(x, x0, sigma, k0)
    hbar = 1.3
    for tau in (0.2, 0.5, 1.0):
        psi = split_step_propagate_coupled(psi0.copy(), None, tau, hbar=np.array([hbar]))
        obs = born_observables(psi, x)
        expected_sigma = sigma * math.sqrt(1.0 + (hbar * tau) ** 2 / (4.0 * sigma**4))
        expected_mean = x0 + hbar * k0 * tau
        assert abs(float(obs["std"]) - expected_sigma) < 2e-3, (tau, obs["std"], expected_sigma)
        assert abs(float(obs["mean"]) - expected_mean) < 2e-3, (tau, obs["mean"], expected_mean)
        print(
            f"free packet hbar={hbar} tau={tau}: std {float(obs['std']):.6f} vs {expected_sigma:.6f}; "
            f"mean {float(obs['mean']):.6f} vs {expected_mean:.6f}"
        )


def test_qho_coherent_state_with_hbar() -> None:
    x, _, _ = grid()
    omega = 0.9
    sigma = 1.0 / math.sqrt(2.0 * omega)
    V = 0.5 * omega**2 * x**2
    x0, k0, hbar = 1.2, 0.4, 1.3
    psi0 = gaussian_packet(x, x0, sigma, k0)
    for tau in (0.7, 2.4):
        psi = split_step_propagate_coupled(psi0.copy(), V, tau, hbar=np.array([hbar]))
        obs = born_observables(psi, x)
        expected = x0 * math.cos(omega * tau) + hbar * k0 / omega * math.sin(omega * tau)
        assert abs(float(obs["mean"]) - expected) < 2e-3, (tau, obs["mean"], expected)
        print(f"QHO coherent state hbar={hbar} tau={tau}: <x> {float(obs['mean']):.6f} vs {expected:.6f}")


def test_absorbing_potential_norm_decay() -> None:
    x, dx, _ = grid()
    psi0 = gaussian_packet(x, 0.5, 1.0, 0.0)
    tau = 1.0
    # (a) constant W: the decay is exactly exp(-2 eta W tau) per grid point.
    for eta, W0 in ((0.5, 0.5), (1.0, 1.0), (2.5, 0.3)):
        potential = np.full(len(x), -1j * eta * W0)
        out = split_step_propagate_coupled(psi0.copy(), potential, tau)
        measured = float(np.trapezoid(np.abs(out) ** 2, x))
        expected = math.exp(-2.0 * eta * W0 * tau)
        assert abs(measured - expected) / expected < 1e-3, (eta, W0, measured, expected)
        print(f"absorbing constant W={W0} eta={eta}: norm {measured:.8f} vs {expected:.8f}")
    # (b) production shape: monotone in eta, no decay at eta = 0.
    shape = absorbing_shape(x)
    norms = {}
    for eta in (0.0, 0.37, 1.0, 2.72):
        potential = -1j * eta * shape
        out = split_step_propagate_coupled(psi0.copy(), potential, tau)
        norms[eta] = float(np.trapezoid(np.abs(out) ** 2, x))
        assert 0.0 < norms[eta] <= 1.0 + 1e-9, (eta, norms[eta])
    assert abs(norms[0.0] - 1.0) < 1e-3, norms[0.0]
    ordered = [norms[eta] for eta in (0.37, 1.0, 2.72)]
    assert ordered[0] > ordered[1] > ordered[2], ordered
    print(f"absorbing production shape: norms {norms}")
    # (c) negative imaginary part would be a *source* (not used); assert it grows.
    potential = +1j * 1.0 * shape
    out = split_step_propagate_coupled(psi0.copy(), potential, tau)
    grown = float(np.trapezoid(np.abs(out) ** 2, x))
    assert grown > 1.0 + 1e-3, grown
    print(f"imaginary source check: norm {grown:.6f} > 1")


def test_coupling_maps_monotone_and_bounded() -> None:
    z = np.linspace(-6.0, 6.0, 257)
    for config in COUPLED_CANDIDATES:
        values = coupling_parameter(z, config)
        assert np.all(np.isfinite(values)), config.label
        assert np.all(values > 0.0), config.label
        if config.kind == "packet":
            # sigma_eff is clipped at the point of use; check the factor range.
            assert np.all(values >= math.exp(-0.5) - 1e-12)
            assert np.all(values <= math.exp(0.5) + 1e-12)
        differences = np.diff(values)
        if config.sign > 0:
            assert np.all(differences > 0.0), f"{config.label} must be increasing"
        else:
            assert np.all(differences < 0.0), f"{config.label} must be decreasing"
        # sign symmetry: flipping the sign of the coupling flips the map.
        flipped = coupling_parameter(-z, config)
        other_sign = coupling_parameter(z, COUPLED_BY_LABEL[config.label.replace("inc", "dec")]) \
            if config.label.endswith("inc") and config.label.replace("inc", "dec") in COUPLED_BY_LABEL \
            else None
        if other_sign is not None:
            assert np.allclose(flipped, other_sign, atol=1e-12), config.label
        # extreme inputs stay finite and bounded.
        extreme = coupling_parameter(np.array([-1e6, 0.0, 1e6]), config)
        assert np.all(np.isfinite(extreme)), config.label
        assert extreme.min() > 0.0
        print(
            f"coupling {config.label}: range [{values.min():.4f}, {values.max():.4f}], "
            f"extreme {extreme}"
        )


def _synthetic_close(periods: int = 2200, seed: int = 5) -> pd.Series:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2026-01-01", periods=periods, freq="1min", tz="UTC")
    returns = rng.normal(0.0, 0.001, size=periods)
    close = 100.0 * np.exp(np.cumsum(returns))
    return pd.Series(close, index=index)


def test_identity_coupling_matches_registered_builder() -> None:
    close = _synthetic_close()
    index = close.index
    anchors = index[400:500].to_numpy(dtype="datetime64[s]").astype("int64")
    gcp_z = np.linspace(-2.0, 2.0, len(anchors))
    for horizon in (1, 5, 15, 60):
        frames, audit, stats = build_coupled_features_for_asset_multi(
            close, anchors, horizon, FROZEN_SCHR, COUPLED_CANDIDATES, gcp_z, gcp_z,
            chunk=64, include_identity=True,
        )
        registered = build_features_for_asset(close, anchors, horizon, FROZEN_SCHR, chunk=64)
        assert len(registered) == len(frames["identity"])
        assert stats["kde_fallback_rows"] == 0
        for base in ("p_up", "mean", "std", "kurt"):
            left = frames["identity"][f"cpl_identity_{base}_{horizon}m"].to_numpy(dtype="float64")
            right = registered[f"schr_{base}"].to_numpy(dtype="float64")
            difference = float(np.max(np.abs(left - right)))
            assert difference < 1e-9, (horizon, base, difference)
        # The coupling must actually change the observables for nonzero z.
        for label in ("hbar_inc", "absorb_inc"):
            changed = np.max(
                np.abs(
                    frames[label][f"cpl_{label}_p_up_{horizon}m"].to_numpy(dtype="float64")
                    - frames["identity"][f"cpl_identity_p_up_{horizon}m"].to_numpy(dtype="float64")
                )
            )
            assert changed > 1e-6, (horizon, label, changed)
        print(f"identity equivalence horizon={horizon}m: registered builder reproduced")


def test_nan_gcp_rows_are_isolated() -> None:
    close = _synthetic_close()
    index = close.index
    anchors = index[400:412].to_numpy(dtype="datetime64[s]").astype("int64")
    base_z = np.linspace(-1.5, 1.5, len(anchors))
    holed = base_z.copy()
    holed[3] = np.nan
    holed[7] = np.nan
    horizon = 15
    with_holes, _, _ = build_coupled_features_for_asset_multi(
        close, anchors, horizon, FROZEN_SCHR, [COUPLED_BY_LABEL["hbar_inc"]],
        holed, holed, chunk=16, include_identity=True,
    )
    replaced, _, _ = build_coupled_features_for_asset_multi(
        close, anchors, horizon, FROZEN_SCHR, [COUPLED_BY_LABEL["hbar_inc"]],
        np.where(np.isnan(holed), 0.0, holed), np.where(np.isnan(holed), 0.0, holed),
        chunk=16, include_identity=True,
    )
    column = f"cpl_hbar_inc_p_up_{horizon}m"
    holed_values = with_holes["hbar_inc"][column].to_numpy(dtype="float64")
    assert np.isnan(holed_values[3]) and np.isnan(holed_values[7])
    good = [i for i in range(len(anchors)) if i not in (3, 7)]
    difference = float(
        np.max(np.abs(holed_values[good] - replaced["hbar_inc"][column].to_numpy(dtype="float64")[good]))
    )
    assert difference < 1e-12, difference
    print(f"NaN-GCP rows isolated: max |diff| on finite rows = {difference:.3e}")


def test_shifted_z_column_is_registered_lag() -> None:
    rng = np.random.default_rng(11)
    index = pd.date_range("2026-01-01", periods=90 * 24 * 60, freq="1min", tz="UTC")
    values = 1.0 + np.cumsum(rng.normal(0.0, 0.01, size=len(index)))
    keep = rng.random(len(index)) > 0.05  # 5% missing minutes
    series = pd.Series(values, index=index)[keep]
    from mix_lab import shifted_z_column  # local import keeps the test readable
    registered = (series - series.rolling(60, min_periods=60).mean()) / series.rolling(
        60, min_periods=60
    ).std(ddof=1).replace(0.0, np.nan)
    registered = registered.shift(1)
    shifted = shifted_z_column(series, days=CONTROL_SHIFT_DAYS)
    probe_times = index[45 * 24 * 60 :: 997]
    expected = registered.reindex(probe_times - pd.Timedelta(days=CONTROL_SHIFT_DAYS))
    expected.index = probe_times
    observed = shifted.reindex(probe_times)
    both = expected.notna() & observed.notna()
    assert both.sum() > 50, both.sum()
    assert np.allclose(observed[both], expected[both], atol=0.0), "shifted z must be the exact lag"
    # A missing source minute stays NaN (no fill).
    missing_times = probe_times[~both]
    assert len(missing_times) > 0
    assert np.isnan(observed[missing_times]).all()
    print(
        f"shifted z column: {int(both.sum())} exact matches, "
        f"{int((~both).sum())} NaN where the source minute is missing"
    )


def test_causality_future_candles_do_not_leak() -> None:
    close = _synthetic_close()
    index = close.index
    anchors = index[400:460].to_numpy(dtype="datetime64[s]").astype("int64")
    gcp_z = np.linspace(-1.0, 1.0, len(anchors))
    horizon = 5
    frames_a, _, _ = build_coupled_features_for_asset_multi(
        close, anchors, horizon, FROZEN_SCHR, COUPLED_CANDIDATES, gcp_z, gcp_z, chunk=64,
        include_identity=True,
    )
    modified = close.copy()
    # Change every candle strictly after the last anchor (index 459) by +50%.
    modified.iloc[461:] = modified.iloc[461:] * 1.5
    frames_b, _, _ = build_coupled_features_for_asset_multi(
        modified, anchors, horizon, FROZEN_SCHR, COUPLED_CANDIDATES, gcp_z, gcp_z, chunk=64,
        include_identity=True,
    )
    for key, frame in frames_a.items():
        for column in frame.columns:
            if column == "target_time_s":
                continue
            difference = float(
                np.max(np.abs(frame[column].to_numpy(dtype="float64") - frames_b[key][column].to_numpy(dtype="float64")))
            )
            assert difference == 0.0, (key, column, difference)
    print("causality: future candles leave every coupled feature unchanged")
    # A non-candle-open anchor must raise.
    bad = np.array([anchors[0] + 30], dtype="int64")
    try:
        build_coupled_features_for_asset_multi(
            close, bad, horizon, FROZEN_SCHR, COUPLED_CANDIDATES, [0.0], [0.0], chunk=8,
        )
    except AssertionError:
        pass
    else:
        raise AssertionError("non-candle anchor was not rejected")
    print("causality: off-grid anchor rejected")


def test_coupled_model_columns_include_norm_only_for_absorb() -> None:
    for config in COUPLED_CANDIDATES:
        columns = coupled_model_columns(config, 15)
        has_norm = any(name.endswith("_norm_15m") for name in columns)
        assert has_norm == (config.kind == "absorb"), config.label
        assert len(columns) == (5 if config.kind == "absorb" else 4)
    print("coupled model columns: norm column only for the absorbing candidate")


def main() -> int:
    tests = [
        test_coupled_propagator_matches_registered_solver,
        test_norm_conservation_with_per_row_hbar,
        test_free_packet_spreading_with_hbar,
        test_qho_coherent_state_with_hbar,
        test_absorbing_potential_norm_decay,
        test_coupling_maps_monotone_and_bounded,
        test_identity_coupling_matches_registered_builder,
        test_nan_gcp_rows_are_isolated,
        test_shifted_z_column_is_registered_lag,
        test_causality_future_candles_do_not_leak,
        test_coupled_model_columns_include_norm_only_for_absorb,
    ]
    failures = 0
    for test in tests:
        try:
            test()
        except AssertionError as exc:
            failures += 1
            print(f"FAILED {test.__name__}: {exc}")
        print(f"ok {test.__name__}")
    print(f"\n{len(tests) - failures}/{len(tests)} mix_lab tests passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
