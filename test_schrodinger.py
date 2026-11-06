"""Permanent verification tests for the Schrodinger solver in ``schrodinger.py``.

Run with::

    price-prediction/.venv/bin/python price-prediction/test_schrodinger.py

(pytest is not installed in this environment; the file also works under pytest.)

The tests check real solver behaviour that a broken implementation cannot
fake:

1. Norm conservation ``| int |psi|^2 dx - 1 | < 1e-8`` for both solvers over a
   non-trivial potential.
2. Free Gaussian packet: width grows as the analytic
   ``sigma(tau) = sigma sqrt(1 + tau^2 / (4 sigma^4))`` and the centre moves as
   ``x0 + k0 tau`` (hbar = m = 1).
3. Harmonic oscillator coherent state: ``<x>(tau) = x0 cos(omega tau) +
   (k0/omega) sin(omega tau)``.
4. Quantum harmonic oscillator eigen-energies from the finite-difference
   Hamiltonian converge to ``(n + 1/2) hbar omega`` with the grid spacing.
5. The two independent solvers (split-step Fourier and Crank-Nicolson) agree
   on the same non-trivial potential.
6. The Born-rule phase effect is real: for a packet at ``x0 = 0`` with no
   potential, ``p_up(k0 > 0) > 0.5 > p_up(k0 < 0)``.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import eigh_tridiagonal

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from schrodinger import (  # noqa: E402
    born_observables,
    crank_nicolson_propagate,
    gaussian_packet,
    grid,
    split_step_propagate,
)


def _smooth_potential(x: np.ndarray) -> np.ndarray:
    """Deterministic, smooth, asymmetric potential for agreement tests."""
    return 0.3 * x**2 + 0.02 * x**4 + 0.1 * x**3 - 0.5 * np.cos(1.7 * x)


def _norm(psi: np.ndarray, x: np.ndarray) -> float:
    return float(np.trapezoid(np.abs(psi) ** 2, x))


def _discrete_norm(psi: np.ndarray, dx: float) -> float:
    """Grid L2 norm, the norm conserved exactly by both solvers."""
    return float(np.sum(np.abs(psi) ** 2) * dx)


def test_norm_conservation() -> None:
    x, dx, k = grid()
    del k
    rng = np.random.default_rng(1234)
    # Localized random state: smooth envelope times white noise.  A state that
    # does not vanish at the edges would legitimately lose its edge mass to the
    # Dirichlet boundaries of the Crank-Nicolson grid.
    envelope = np.exp(-((x / 2.0) ** 2))
    psi = envelope * (rng.normal(size=len(x)) + 1j * rng.normal(size=len(x)))
    psi = psi / math.sqrt(_discrete_norm(psi, dx))
    V = _smooth_potential(x)
    for name, solver in (("split_step", split_step_propagate), ("crank_nicolson", crank_nicolson_propagate)):
        for tau in (0.25, 1.0):
            out = solver(psi.copy(), V, tau)
            drift = abs(_discrete_norm(out, dx) - 1.0)
            assert drift < 1e-8, f"{name} tau={tau}: discrete norm drift {drift:g}"
            print(f"norm conservation {name} tau={tau}: |norm-1| = {drift:.3e}")
    # A localized packet must also conserve the trapezoidal (physical) norm,
    # which is the quantity the observables use.
    packet = gaussian_packet(x, 0.5, 0.8, 0.2)
    for name, solver in (("split_step", split_step_propagate), ("crank_nicolson", crank_nicolson_propagate)):
        out = solver(packet.copy(), V, 0.5)
        drift = abs(_norm(out, x) - 1.0)
        assert drift < 1e-8, f"{name} packet: trapezoidal norm drift {drift:g}"
        print(f"norm conservation {name} packet tau=0.5: |norm-1| = {drift:.3e}")


def test_free_packet_spreading() -> None:
    x, dx, _ = grid()
    sigma, x0, k0 = 0.7, 1.0, 0.3
    psi0 = gaussian_packet(x, x0, sigma, k0)
    for tau in (0.0, 0.5, 1.0):
        psi = split_step_propagate(psi0.copy(), None, tau) if tau > 0 else psi0
        obs = born_observables(psi, x)
        expected_sigma = sigma * math.sqrt(1.0 + tau**2 / (4.0 * sigma**4))
        expected_mean = x0 + k0 * tau
        assert abs(obs["std"] - expected_sigma) < 2e-3, (tau, obs["std"], expected_sigma)
        assert abs(obs["mean"] - expected_mean) < 2e-3, (tau, obs["mean"], expected_mean)
        print(
            f"free packet tau={tau}: std {obs['std']:.6f} vs {expected_sigma:.6f}; "
            f"mean {obs['mean']:.6f} vs {expected_mean:.6f}"
        )


def test_qho_coherent_state() -> None:
    x, dx, _ = grid()
    omega = 0.9
    sigma = 1.0 / math.sqrt(2.0 * omega)  # ground-state width (hbar = m = 1)
    V = 0.5 * omega**2 * x**2
    x0, k0 = 1.2, 0.4
    psi0 = gaussian_packet(x, x0, sigma, k0)
    for tau in (0.7, 2.4):
        psi = split_step_propagate(psi0.copy(), V, tau)
        obs = born_observables(psi, x)
        expected = x0 * math.cos(omega * tau) + (k0 / omega) * math.sin(omega * tau)
        assert abs(obs["mean"] - expected) < 2e-3, (tau, obs["mean"], expected)
        print(f"QHO coherent state tau={tau}: <x> {obs['mean']:.6f} vs {expected:.6f}")


def test_qho_eigenenergies() -> None:
    # Finite-difference Hamiltonian H = -1/2 d^2/dx^2 + omega^2 x^2 / 2 with
    # Dirichlet boundaries; eigenvalues converge to (n + 1/2) hbar omega with
    # O(dx^2) discretization error.
    omega = 1.0
    previous_error = None
    for n_grid, x_max, tolerance in ((513, 8.0, 5e-3), (2049, 8.0, 5e-4)):
        x = np.linspace(-x_max, x_max, n_grid)
        dx = x[1] - x[0]
        diagonal = 1.0 / dx**2 + 0.5 * omega**2 * x**2
        off = np.full(n_grid - 1, -0.5 / dx**2)
        # Dirichlet: drop the two boundary unknowns.
        w = eigh_tridiagonal(diagonal[1:-1], off[1:-1], select="i", select_range=(0, 5))[0]
        exact = np.array([(n + 0.5) * omega for n in range(len(w))])
        error = float(np.max(np.abs(w - exact)))
        print(f"QHO eigenenergies N={n_grid} (dx={dx:.5f}): max |E_n - (n+1/2)| = {error:.3e}")
        assert error < tolerance, (n_grid, w, exact)
        if previous_error is not None:
            assert error < previous_error / 2.0, "eigenvalue error must shrink with dx^2"
        previous_error = error


def test_solvers_agree() -> None:
    x, dx, _ = grid()
    # A smooth, grid-resolved state: the split-step kinetic operator is
    # spectrally exact, while Crank-Nicolson uses the second-order
    # finite-difference Laplacian, so agreement is expected only for states
    # resolved on the grid (as every packet used by the model is).
    psi = gaussian_packet(x, 0.5, 0.8, 0.2)
    V = _smooth_potential(x)
    tau = 0.6
    diff_floor = {}
    for n_steps in (40, 600):
        cn = crank_nicolson_propagate(psi.copy(), V, tau, n_steps=n_steps)
        ss = split_step_propagate(psi.copy(), V, tau)
        diff = float(np.sqrt(np.trapezoid(np.abs(ss - cn) ** 2, x)))
        diff_floor[n_steps] = diff
        print(f"split-step vs Crank-Nicolson L2 difference (n_steps={n_steps}): {diff:.3e}")
    # The residual difference is the Crank-Nicolson O(dx^2) spatial error; it
    # must shrink as the time step shrinks toward that floor.
    assert diff_floor[600] < diff_floor[40], "Crank-Nicolson must converge toward split-step with dt"
    assert diff_floor[600] < 2e-3, diff_floor
    ss = split_step_propagate(psi.copy(), V, tau)
    cn = crank_nicolson_propagate(psi.copy(), V, tau, n_steps=600)
    obs_ss = born_observables(ss, x)
    obs_cn = born_observables(cn, x)
    assert abs(obs_ss["p_up"] - obs_cn["p_up"]) < 1e-3
    assert abs(obs_ss["mean"] - obs_cn["mean"]) < 1e-3
    assert abs(obs_ss["std"] - obs_cn["std"]) < 1e-3
    print(
        f"  Born-rule agreement: p_up {obs_ss['p_up']:.6f} vs {obs_cn['p_up']:.6f}; "
        f"mean {obs_ss['mean']:.6f} vs {obs_cn['mean']:.6f}"
    )


def test_phase_effect_on_born_rule() -> None:
    # With no potential a phase gradient is a real momentum: it must move the
    # Born-rule mass across x = 0 in the direction of k0.
    x, _, _ = grid()
    for tau in (0.2, 0.6):
        up = born_observables(split_step_propagate(gaussian_packet(x, 0.0, 1.0, +0.8), None, tau), x)["p_up"]
        down = born_observables(split_step_propagate(gaussian_packet(x, 0.0, 1.0, -0.8), None, tau), x)["p_up"]
        assert up > 0.5, up
        assert down < 0.5, down
        assert abs((up - 0.5) + (down - 0.5)) < 1e-6, (up, down)  # antisymmetry
        print(f"phase effect tau={tau}: p_up(k0=+0.8)={up:.6f}, p_up(k0=-0.8)={down:.6f}")


def test_causality_guard_in_feature_builder() -> None:
    """The feature builder must refuse anchors that are not exact candle opens."""
    import pandas as pd

    from schrodinger import SchrodingerConfig, build_features_for_asset

    index = pd.date_range("2026-01-01", periods=500, freq="1min", tz="UTC")
    close = pd.Series(100.0 + np.arange(500) * 0.01, index=index)
    anchors = index[300:310].to_numpy(dtype="datetime64[s]").astype("int64")
    config = SchrodingerConfig()
    out = build_features_for_asset(close, anchors, 5, config, window=240, chunk=64)
    assert len(out) == 10
    log_close = np.log(close.to_numpy(dtype="float64"))
    trailing_5m = np.full_like(log_close, np.nan)
    trailing_5m[5:] = log_close[5:] - log_close[:-5]
    expected = (
        pd.Series(trailing_5m).rolling(240, min_periods=240).std(ddof=1).to_numpy()[299]
    )
    assert out["schr_trailing_vol"].notna().all()
    assert abs(out["schr_trailing_vol"].iloc[0] - expected) < 1e-12
    # The newest candle that may enter anchor index 300 is index 299.
    assert np.isfinite(out["schr_p_up"]).all()
    assert out["schr_p_up"].between(0.0, 1.0).all()
    # A non-candle-open anchor must raise.
    bad = np.array([anchors[0] + 30], dtype="int64")
    try:
        build_features_for_asset(close, bad, 5, config, window=240, chunk=16)
    except AssertionError:
        pass
    else:
        raise AssertionError("non-candle anchor was not rejected")
    print("causality guard: exact candle opens accepted, off-grid anchor rejected")


def test_quantum_potential_makes_kde_state_stationary() -> None:
    """V = (1/2) psi0''/psi0 must make psi0 = sqrt(density) an eigenstate.

    The inverse ("quantum") potential is built with the same second-difference
    stencil as the solver, so H psi0 = 0 exactly in the interior whenever the
    clip and the density floor do not bind; the test asserts the residual is at
    machine level there and that a propagation by tau = 1 leaves the Born
    density essentially unchanged (a stationary state only acquires a phase).
    """
    from schrodinger import kde_density, potential_quantum_kde

    x, dx, _ = grid()
    rng = np.random.default_rng(7)
    z = np.concatenate([rng.normal(-1.0, 0.7, 600), rng.normal(0.8, 0.5, 400)]).reshape(1, -1)
    density = kde_density(z, x)
    V = potential_quantum_kde(density, x)[0]
    psi0 = np.sqrt(density[0])
    d2 = np.empty_like(psi0)
    d2[1:-1] = (psi0[2:] - 2.0 * psi0[1:-1] + psi0[:-2]) / dx**2
    d2[0] = d2[1]
    d2[-1] = d2[-2]
    h_psi = -0.5 * d2 + V * psi0
    peak = density[0].max()
    bulk = density[0] > 1e-6 * peak
    bulk[:2] = False
    bulk[-2:] = False
    exact = bulk & (np.abs(V) < 60.0 - 1e-9)
    residual = float(np.max(np.abs(h_psi[exact])))
    scale = float(np.max(np.abs(V[exact] * psi0[exact])))
    print(
        f"quantum potential eigenstate residual {residual:.3e} (scale {scale:.3e}, "
        f"{int(exact.sum())}/{int(bulk.sum())} bulk bins unclipped, |V|max {np.max(np.abs(V[bulk])):.3f})"
    )
    assert int(exact.sum()) > 0.9 * int(bulk.sum()), "most of the bulk must be unclipped"
    assert residual < 1e-12 * scale, (residual, scale)
    # A stationary state only acquires a phase.  With the production
    # split-step solver the residual density change converges as dt -> 0 to the
    # O(dx^2) mismatch between the spectral kinetic operator and the
    # finite-difference stencil used to construct V (measured ~1e-3 L1 in the
    # core); it must shrink with dt and stay small in absolute terms.
    before = np.abs(psi0) ** 2
    changes = {}
    for dt_max in (0.02, 0.002):
        after = np.abs(split_step_propagate(psi0.astype(np.complex128), V, 1.0, dt_max=dt_max)) ** 2
        changes[dt_max] = float(np.sum(np.abs(after - before)[np.abs(x) < 3.0]) * dx)
    print(f"stationary density L1 change after tau=1 (|x|<3): dt=0.02 {changes[0.02]:.3e}, dt=0.002 {changes[0.002]:.3e}")
    assert changes[0.002] < changes[0.02], changes
    assert changes[0.002] < 2e-3, changes


def test_quartic_potential_matches_independent_fit() -> None:
    """Batched weighted quartic fit must match a per-row lstsq reference."""
    from schrodinger import kde_density, potential_quartic_kde

    x, dx, _ = grid()
    rng = np.random.default_rng(11)
    z = rng.normal(0.3, 1.0, size=(3, 240))
    z[1] = rng.standard_t(4, size=240)
    z[2] = np.concatenate([rng.normal(-2.0, 0.4, 120), rng.normal(1.5, 0.6, 120)])
    density = kde_density(z, x)
    V = potential_quartic_kde(density, x)
    mask = np.abs(x) <= 3.5
    xs = x[mask]
    A = np.stack([xs**2, xs**3, xs**4], axis=1)
    for row in range(len(z)):
        y = -np.log(np.maximum(density[row][mask], 1e-300))
        w = density[row][mask]
        reference = np.linalg.lstsq(A * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
        expected = reference[0] * x**2 + reference[1] * x**3 + reference[2] * x**4
        assert np.max(np.abs(np.clip(expected, -60.0, 60.0) - V[row])) < 1e-6
    print("quartic potential matches per-row lstsq reference")


def main() -> int:
    tests = [
        test_norm_conservation,
        test_free_packet_spreading,
        test_qho_coherent_state,
        test_qho_eigenenergies,
        test_solvers_agree,
        test_phase_effect_on_born_rule,
        test_quantum_potential_makes_kde_state_stationary,
        test_quartic_potential_matches_independent_fit,
        test_causality_guard_in_feature_builder,
    ]
    failures = 0
    for test in tests:
        try:
            test()
        except AssertionError as exc:
            failures += 1
            print(f"FAILED {test.__name__}: {exc}")
        print(f"ok {test.__name__}")
    print(f"\n{len(tests) - failures}/{len(tests)} solver tests passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
