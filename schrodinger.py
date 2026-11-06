"""Literal 1-D time-dependent Schrodinger equation solver and wave-packet model.

This module solves the dimensionless time-dependent Schrodinger equation

    i hbar dpsi/dt = [ -(hbar^2 / 2m) d^2/dx^2 + V(x) ] psi

on a finite grid for a complex wave function ``psi``.  It is used literally: the
prediction for horizon ``h`` at minute open ``t`` is read from the Born rule
``|psi(x, tau)|^2`` after evolving the packet by a dimensionless time ``tau``.

Units and conventions (fixed, never tuned on results)
-----------------------------------------------------
* ``hbar = m = 1``.  ``x`` is a *standardized* forward log-return coordinate:
  one unit of ``x`` is one trailing h-minute log-return standard deviation
  ``sigma_h(t)``, and ``x = 0`` is the current price.  ``x`` and ``tau`` are
  therefore dimensionless.
* Grid ``x in [-6, 6]`` with ``N_GRID = 193`` points (``dx = 0.0625``; ``x = 0``
  is exactly a grid point).  The box is large compared with every packet used
  (width <= 2, |k0| <= 4, tau <= 1).  Split-step Fourier propagation is
  periodic; wrap-around density is negligible (validated in the test file).
* Time step ``dt <= DT_MAX``, with ``n_steps = ceil(tau / DT_MAX)`` and uniform
  ``dt = tau / n_steps`` so the total evolution time is exact.

Solver
------
``split_step_propagate`` is the production solver: exact kinetic phase in
Fourier space and exact potential phase in real space, Strang-split
(``exp(-i V dt/2) exp(-i T dt) exp(-i V dt/2)`` per step), which is unitary for
any step size.  ``crank_nicolson_propagate`` is a second-order Crank-Nicolson
solver on the same grid (vectorized Thomas algorithm) used as an independent
reference in ``test_schrodinger.py``; the two agree for the test problems.

Model
-----
For horizon ``h`` at anchor ``t`` (candles with open time ``< t`` only):

* ``sigma_h(t)``: trailing standard deviation (ddof=1, window ``KDE_WINDOW``)
  of h-minute log returns ``L_s - L_{s-h}``, ending at candle ``t-60`` seconds.
* ``z`` samples: the same ``KDE_WINDOW`` trailing h-minute log returns divided
  by ``sigma_h(t)`` (exactly unit sample standard deviation).
* ``x0 = clip(z_lastest, +-X0_CLIP)``: current state displacement in sigma_h
  units; ``sigma_packet = clip(std_60 / sigma_h, .5, 2)``; ``k0`` is the
  pre-declared map of ``x0`` (phase-encoded drift/momentum).
* ``V(x)`` from trailing data (one of the pre-declared potentials):
  - ``quantum_kde``: ``psi0 = sqrt(p_hat)`` for the Gaussian KDE ``p_hat`` of
    the trailing ``z`` sample; ``V = E0 + (hbar^2/2m) psi0'' / psi0`` with the
    physically irrelevant constant ``E0 = 0`` (it only adds a global phase),
    computed with the same second-difference stencil as the solver and clipped
    to ``+-V_CLIP``.
  - ``quartic_kde``: quantum anharmonic oscillator
    ``V = a x^2 + c x^3 + b x^4``, weighted least squares fit of
    ``-log p_hat(x)`` on ``|x| <= QUARTIC_MASK`` with weights ``p_hat``.
  - ``free_particle``: ``V = 0`` (honesty control; no trailing data in V).
* Initial packet ``psi(x,0) = N exp(-(x-x0)^2/(4 sigma^2) + i k0 x)``,
  discretely normalized; evolved by ``tau = h / TAU_DENOMINATORS[tau_map]``.
* Observables of ``|psi(.,tau)|^2``: ``p_up = int_{x>0} |psi|^2 dx``,
  ``mean = <x>``, ``std = sqrt(<x^2>-<x>^2)``, ``kurt = excess kurtosis``.

Everything is computed from candles with ``open_time_s < t``; the causality
assertions at the bottom of :func:`build_features_for_asset` enforce it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Pre-declared constants (fixed before any result was computed).
# --------------------------------------------------------------------------
HBAR = 1.0
MASS = 1.0
X_MIN = -6.0
X_MAX = 6.0
N_GRID = 193
DT_MAX = 0.02
V_CLIP = 60.0

KDE_WINDOW = 240
# Silverman's rule evaluated at unit sample standard deviation: z samples are
# standardized within their own window, so the sample std is 1 by construction.
KDE_BANDWIDTH = 1.06 * KDE_WINDOW ** (-0.2)
# The inverse ("quantum") potential divides by psi0 = sqrt(density); the KDE
# histogram can be exactly zero in far-tail bins that no sample reaches within
# the Gaussian truncation radius.  The density is therefore floored at this
# fraction of its per-row peak before the division (only far tails where the
# packet density is ~0 are affected; documented in the pre-registration).
KDE_DENSITY_FLOOR_FRACTION = 1e-12
VOL_SHORT_WINDOW = 60
SIGMA_CLIP = (0.5, 2.0)
X0_CLIP = 4.0
QUARTIC_MASK = 3.5

POTENTIALS = ("quantum_kde", "quartic_kde", "free_particle")
TAU_DENOMINATORS = {"linear_60": 60.0, "slow_240": 240.0}
K0_MULTIPLIERS = {"zero": 0.0, "momentum": 1.0}

SCHR_BASE_FEATURES = ("schr_p_up", "schr_mean", "schr_std", "schr_kurt")


def schr_feature_columns(horizon: int) -> tuple[str, ...]:
    """Model columns for one horizon (per-horizon suffixed base features)."""
    return tuple(f"{name}_{horizon}m" for name in SCHR_BASE_FEATURES)


@dataclass(frozen=True)
class SchrodingerConfig:
    """One pre-declared (potential, tau map, k0 map) configuration."""

    potential: str = "quantum_kde"
    tau_map: str = "linear_60"
    k0_map: str = "momentum"

    def __post_init__(self) -> None:
        if self.potential not in POTENTIALS:
            raise ValueError(f"unknown potential: {self.potential!r}")
        if self.tau_map not in TAU_DENOMINATORS:
            raise ValueError(f"unknown tau map: {self.tau_map!r}")
        if self.k0_map not in K0_MULTIPLIERS:
            raise ValueError(f"unknown k0 map: {self.k0_map!r}")

    @property
    def label(self) -> str:
        return f"{self.potential}|{self.tau_map}|{self.k0_map}"

    def as_dict(self) -> dict:
        return {
            "potential": self.potential,
            "tau_map": self.tau_map,
            "k0_map": self.k0_map,
            "label": self.label,
        }


def tau_for(horizon: int, tau_map: str) -> float:
    """Dimensionless evolution time tau = h / denominator (h in minutes)."""
    return float(horizon) / float(TAU_DENOMINATORS[tau_map])


def k0_for(x0: np.ndarray, k0_map: str) -> np.ndarray:
    """Phase-encoded initial momentum k0 = multiplier * x0."""
    return float(K0_MULTIPLIERS[k0_map]) * np.asarray(x0, dtype="float64")


# --------------------------------------------------------------------------
# Solver
# --------------------------------------------------------------------------
@lru_cache(maxsize=4)
def grid(x_min: float = X_MIN, x_max: float = X_MAX, n: int = N_GRID):
    """Return ``(x, dx, k)`` for the uniform grid (``k`` = FFT frequencies)."""
    x = np.linspace(float(x_min), float(x_max), int(n))
    dx = float(x[1] - x[0])
    k = 2.0 * np.pi * np.fft.fftfreq(int(n), d=dx)
    return x, dx, k


def gaussian_packet(
    x: np.ndarray,
    x0: np.ndarray | float,
    sigma: np.ndarray | float,
    k0: np.ndarray | float,
) -> np.ndarray:
    """Complex Gaussian wave packet, discretely normalized on the grid.

    ``psi(x,0) = N exp(-(x-x0)^2/(4 sigma^2) + i k0 x)`` with ``N`` such that
    the trapezoidal integral of ``|psi|^2`` over the grid is 1.
    Supports broadcasting: scalar parameters give ``(N,)``, vectors ``(B,)``
    give ``(B, N)``.
    """
    x = np.asarray(x, dtype="float64")
    x0 = np.asarray(x0, dtype="float64")
    sigma = np.asarray(sigma, dtype="float64")
    k0 = np.asarray(k0, dtype="float64")
    if x0.ndim == 0:
        psi = np.exp(-((x - x0) ** 2) / (4.0 * sigma**2) + 1j * k0 * x)
        norm = math.sqrt(float(np.trapezoid(np.abs(psi) ** 2, x)))
        return psi / norm
    xb = x[None, :]
    psi = np.exp(
        -((xb - x0[:, None]) ** 2) / (4.0 * sigma[:, None] ** 2)
        + 1j * k0[:, None] * xb
    )
    norm = np.sqrt(np.trapezoid(np.abs(psi) ** 2, x, axis=1))
    return psi / norm[:, None]


def split_step_propagate(
    psi: np.ndarray,
    V: np.ndarray | None,
    tau: float,
    dt_max: float = DT_MAX,
) -> np.ndarray:
    """Strang-split Fourier propagation of ``psi`` by ``tau`` (hbar = m = 1).

    Kinetic part is exact in Fourier space (unitary), potential part is an
    exact phase multiplication in real space (unitary).  ``V`` may be ``(N,)``
    or ``(B, N)``; ``psi`` may be ``(N,)`` or ``(B, N)``.
    """
    x, dx, k = grid()
    tau = float(tau)
    if tau <= 0.0:
        raise ValueError("tau must be positive")
    n_steps = max(1, int(math.ceil(tau / float(dt_max) - 1e-12)))
    dt = tau / n_steps
    kinetic = np.exp(-0.5j * (k**2) * dt)  # -i * hbar*k^2/(2m) * dt
    potential = None if V is None else np.exp(-0.5j * np.asarray(V, dtype="float64") * dt)
    axis = 1 if psi.ndim == 2 else 0
    out = np.asarray(psi, dtype=np.complex128)
    for _ in range(n_steps):
        if potential is not None:
            out = out * potential
        out = np.fft.ifft(np.fft.fft(out, axis=axis) * kinetic, axis=axis)
        if potential is not None:
            out = out * potential
    return out


def crank_nicolson_propagate(
    psi: np.ndarray,
    V: np.ndarray | None,
    tau: float,
    n_steps: int = 100,
) -> np.ndarray:
    """Crank-Nicolson propagation with Dirichlet boundaries (reference solver).

    Solves ``(I + i dt/2 H) psi^{n+1} = (I - i dt/2 H) psi^n`` with the same
    second-order finite-difference Hamiltonian as :func:`split_step_propagate`
    and a vectorized Thomas algorithm.  ``psi`` may be ``(N,)`` or ``(B, N)``.
    """
    x, dx, k = grid()
    del k
    del x
    tau = float(tau)
    n_steps = int(n_steps)
    if tau <= 0.0 or n_steps <= 0:
        raise ValueError("tau and n_steps must be positive")
    dt = tau / n_steps
    psi = np.asarray(psi, dtype=np.complex128)
    batched = psi.ndim == 2
    if not batched:
        psi = psi[None, :]
    B, N = psi.shape
    if V is None:
        Vb = np.zeros((B, N), dtype="float64")
    else:
        Vb = np.asarray(V, dtype="float64")
        if Vb.ndim == 1:
            Vb = np.broadcast_to(Vb, (B, N))
    diag = 1.0 / dx**2 + Vb  # H diagonal before the i dt/2 factor
    off = -1.0 / (2.0 * dx**2)  # coefficient of psi_{j-1}, psi_{j+1} in H
    a_diag = 1.0 + 0.5j * dt * diag  # A diagonal
    a_off = 0.5j * dt * off  # A off-diagonal
    b_diag = 1.0 - 0.5j * dt * diag  # B diagonal
    b_off = -0.5j * dt * off  # B off-diagonal
    m = N - 2
    out = psi.copy()
    # Dirichlet boundaries are prescribed zero, so the initial state's edge
    # values are zeroed too; otherwise a nonzero edge value would leak a
    # boundary term into the first step and break exact unitarity.
    out[:, 0] = 0.0
    out[:, -1] = 0.0
    for _ in range(n_steps):
        interior = out[:, 1:-1]
        # rhs_k = b_diag_{k+1} psi_{k+1} + b_off (psi_k + psi_{k+2}) for the
        # interior unknowns k = 0..N-3; psi_0 and psi_{N-1} are the Dirichlet
        # zeros and appear as the edge entries of out[:, :-2] / out[:, 2:].
        rhs = b_diag[:, 1:-1] * interior + b_off * out[:, :-2] + b_off * out[:, 2:]
        c_prime = np.empty((B, m), dtype=np.complex128)
        d_prime = np.empty((B, m), dtype=np.complex128)
        denom = a_diag[:, 1:-1]
        c_prime[:, 0] = a_off / denom[:, 0]
        d_prime[:, 0] = rhs[:, 0] / denom[:, 0]
        for j in range(1, m):
            den = denom[:, j] - a_off * c_prime[:, j - 1]
            c_prime[:, j] = a_off / den
            d_prime[:, j] = (rhs[:, j] - a_off * d_prime[:, j - 1]) / den
        new = np.zeros((B, N), dtype=np.complex128)
        new[:, m] = d_prime[:, m - 1]
        for j in range(m - 2, -1, -1):
            new[:, j + 1] = d_prime[:, j] - c_prime[:, j] * new[:, j + 2]
        out = new
    return out if batched else out[0]


def born_observables(psi: np.ndarray, x: np.ndarray | None = None) -> dict:
    """Born-rule observables of ``|psi|^2`` on the grid.

    Returns arrays (or scalars for a single packet): ``p_up`` (integral over
    ``x > 0``), ``mean``, ``std``, ``kurt`` (excess kurtosis).  The density is
    renormalized by its trapezoidal integral first, so the numbers are
    insensitive to residual roundoff in the solver.
    """
    if x is None:
        x, _, _ = grid()
    psi = np.asarray(psi, dtype=np.complex128)
    single = psi.ndim == 1
    if single:
        psi = psi[None, :]
    dens = np.abs(psi) ** 2
    norm = np.trapezoid(dens, x, axis=1)
    dens = dens / norm[:, None]
    zero_index = int(np.argmin(np.abs(np.asarray(x, dtype="float64"))))
    p_up = np.trapezoid(dens[:, zero_index:], x[zero_index:], axis=1)
    mean = np.trapezoid(dens * x[None, :], x, axis=1)
    centred = x[None, :] - mean[:, None]
    var = np.trapezoid(dens * centred**2, x, axis=1)
    m4 = np.trapezoid(dens * centred**4, x, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        std = np.sqrt(var)
        kurt = m4 / var**2 - 3.0
    if single:
        return {
            "p_up": float(p_up[0]),
            "mean": float(mean[0]),
            "std": float(std[0]),
            "kurt": float(kurt[0]),
        }
    return {"p_up": p_up, "mean": mean, "std": std, "kurt": kurt}


# --------------------------------------------------------------------------
# Trailing-data potential
# --------------------------------------------------------------------------
def kde_density(
    z: np.ndarray,
    x: np.ndarray,
    bandwidth: float = KDE_BANDWIDTH,
    stats: dict | None = None,
) -> np.ndarray:
    """Gaussian KDE of each row of ``z`` (B, W) evaluated on grid ``x`` (N,).

    The KDE is evaluated on a padded grid (``5 * bandwidth`` on each side) so
    kernels of samples just outside the domain still contribute correctly, then
    truncated to ``x`` and renormalized so the trapezoidal integral over ``x``
    is exactly 1.  Samples further away than the padded domain are dropped
    (their kernel does not reach the domain).  No sample clipping is used.
    """
    from scipy.ndimage import gaussian_filter1d

    z = np.asarray(z, dtype="float64")
    x = np.asarray(x, dtype="float64")
    if z.ndim == 1:
        z = z[None, :]
    B, W = z.shape
    N = len(x)
    dx = float(x[1] - x[0])
    pad_bins = int(math.ceil(5.0 * bandwidth / dx))
    x_lo = float(x[0]) - pad_bins * dx
    nb = N + 2 * pad_bins
    edges = x_lo + (np.arange(nb + 1, dtype="float64") - 0.5) * dx
    idx = np.digitize(z, edges)  # 0..nb
    valid = (idx > 0) & (idx <= nb)
    rows = np.broadcast_to(np.arange(B)[:, None], idx.shape)[valid]
    counts = np.bincount(rows * nb + (idx[valid] - 1), minlength=B * nb).reshape(B, nb)
    smooth = gaussian_filter1d(
        counts.astype("float64"), sigma=bandwidth / dx, axis=1, mode="constant", truncate=4.0
    )
    density = smooth[:, pad_bins : pad_bins + N] / (float(W) * bandwidth * math.sqrt(2.0 * math.pi))
    mass = np.trapezoid(density, x, axis=1)
    bad = ~np.isfinite(mass) | (mass <= 0.0)
    if stats is not None:
        stats["kde_fallback_rows"] = int(stats.get("kde_fallback_rows", 0)) + int(np.sum(bad))
    if np.any(bad):
        # Degenerate row: no sample kernel reaches the domain.  Fall back to
        # the Gaussian with the sample's own mean and standard deviation
        # (documented degenerate-case fallback; it does not trigger anywhere in
        # the dev source range -- see the dev audit).
        mu = np.mean(z[bad], axis=1, keepdims=True)
        sd = np.std(z[bad], axis=1, ddof=1, keepdims=True)
        sd = np.where(np.isfinite(sd) & (sd > 0.0), sd, 1.0)
        mu = np.where(np.isfinite(mu), mu, 0.0)
        fallback = -((x[None, :] - mu) ** 2) / (2.0 * sd**2)
        fallback = fallback - fallback.max(axis=1, keepdims=True)
        fallback = np.exp(fallback)
        fallback /= np.trapezoid(fallback, x, axis=1)[:, None]
        density[bad] = fallback
        mass = np.where(bad, 1.0, mass)
    density /= np.where(mass > 0.0, mass, np.nan)[:, None]
    return density


def potential_quantum_kde(density: np.ndarray, x: np.ndarray) -> np.ndarray:
    """``V = E0 + (hbar^2/2m) psi0''/psi0`` with ``psi0 = sqrt(density)``, E0=0.

    Same second-difference stencil as the solvers; edge rows are copied from
    the adjacent interior row.  The density is floored at
    ``KDE_DENSITY_FLOOR_FRACTION`` of its per-row peak so the far tails (where
    the histogram+truncated Gaussian kernel can be exactly zero) produce a
    finite potential.  Clipped to ``+-V_CLIP``.
    """
    density = np.asarray(density, dtype="float64")
    x = np.asarray(x, dtype="float64")
    dx = float(x[1] - x[0])
    peak = np.max(density, axis=1, keepdims=True)
    density = np.maximum(density, peak * KDE_DENSITY_FLOOR_FRACTION)
    psi0 = np.sqrt(np.maximum(density, 0.0))
    d2 = np.empty_like(psi0)
    d2[:, 1:-1] = (psi0[:, 2:] - 2.0 * psi0[:, 1:-1] + psi0[:, :-2]) / dx**2
    d2[:, 0] = d2[:, 1]
    d2[:, -1] = d2[:, -2]
    with np.errstate(divide="ignore", invalid="ignore"):
        V = 0.5 * d2 / psi0
    return np.clip(V, -V_CLIP, V_CLIP)


def potential_quartic_kde(
    density: np.ndarray,
    x: np.ndarray,
    mask_halfwidth: float = QUARTIC_MASK,
) -> np.ndarray:
    """Weighted fit of ``-log density`` by ``a x^2 + c x^3 + b x^4``.

    Weights are the KDE density itself, so the fit follows the bulk.  The
    (physically irrelevant) additive constant is dropped.  Clipped to
    ``+-V_CLIP``.
    """
    density = np.asarray(density, dtype="float64")
    x = np.asarray(x, dtype="float64")
    y = -np.log(np.maximum(density, 1e-300))
    mask = np.abs(x) <= float(mask_halfwidth)
    xs = x[mask]
    A = np.stack([xs**2, xs**3, xs**4], axis=1)  # (M, 3)
    weights = density[:, mask]  # (B, M)
    AtWA = np.einsum("mi,bm,mj->bij", A, weights, A, optimize=True)
    AtWy = np.einsum("mi,bm,bm->bi", A, weights, y[:, mask], optimize=True)
    AtWA = AtWA + np.eye(3)[None, :, :] * 1e-10
    coef = np.linalg.solve(AtWA, AtWy[..., None])[..., 0]  # (B, 3)
    V = coef[:, 0:1] * x[None, :] ** 2 + coef[:, 1:2] * x[None, :] ** 3 + coef[:, 2:3] * x[None, :] ** 4
    return np.clip(V, -V_CLIP, V_CLIP)


def build_potential(
    density: np.ndarray,
    x: np.ndarray,
    potential: str,
) -> np.ndarray | None:
    """Dispatch to the pre-declared potential family (``None`` = free)."""
    if potential == "quantum_kde":
        return potential_quantum_kde(density, x)
    if potential == "quartic_kde":
        return potential_quartic_kde(density, x)
    if potential == "free_particle":
        return None
    raise ValueError(f"unknown potential: {potential!r}")


# --------------------------------------------------------------------------
# Causal feature construction
# --------------------------------------------------------------------------
def _rolling_std_of_trailing_returns(
    R: np.ndarray, window: int
) -> np.ndarray:
    """Rolling std (ddof=1) of ``R`` at index ``s`` over ``s-window+1..s``."""
    series = pd.Series(R, dtype="float64")
    return series.rolling(window, min_periods=window).std(ddof=1).to_numpy(dtype="float64")


def build_features_for_asset_multi(
    close: pd.Series,
    anchors_time_s: Sequence[int] | np.ndarray,
    horizon: int,
    configs: Sequence[SchrodingerConfig],
    window: int = KDE_WINDOW,
    chunk: int = 8192,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict]:
    """Schrodinger features for one asset/horizon, all configs in one pass.

    The causal inputs (sigma_h, trailing samples, KDE density, potentials) are
    computed once per anchor batch and shared by every config; only the packet
    propagation differs between configs.  Returns
    ``(features_by_config_label, audit_frame, stats)`` where each feature frame
    has ``target_time_s`` plus the four base features and ``schr_k0`` and the
    audit frame has ``schr_trailing_vol``, ``schr_x0``, ``schr_sigma``.

    Every input uses only candles with ``open_time_s < t``: the sample window
    ends at candle ``t - 60s`` and the assertions below fail loudly otherwise.
    """
    configs = list(configs)
    if not configs:
        raise ValueError("at least one config is required")
    close = pd.Series(close, dtype="float64").sort_index()
    times_s = close.index.to_numpy(dtype="datetime64[s]").astype("int64")
    L = np.log(close.to_numpy(dtype="float64"))
    n = len(L)
    h = int(horizon)
    anchors = np.asarray(anchors_time_s, dtype="int64")

    positions = np.searchsorted(times_s, anchors)
    if np.any(positions >= n) or not np.array_equal(times_s[positions], anchors):
        raise AssertionError("anchor times must be exact candle opens of this asset")
    last = positions - 1  # newest closed candle at anchor t is open t-60s
    # Anchors without a full history (window trailing h-returns ending at
    # t-60s) produce no row; the join in build_anchor_frame leaves them NaN and
    # every downstream complete-case filter drops them.
    usable = last >= h + window - 1
    anchors = anchors[usable]
    positions = positions[usable]
    last = last[usable]

    feature_columns = ["target_time_s", *SCHR_BASE_FEATURES, "schr_k0"]
    audit_columns = ["target_time_s", "schr_trailing_vol", "schr_x0", "schr_sigma"]
    if len(anchors) == 0:
        empty_features = {
            config.label: pd.DataFrame({column: np.empty(0, dtype="float64") for column in feature_columns})
            for config in configs
        }
        return empty_features, pd.DataFrame({column: np.empty(0, dtype="float64") for column in audit_columns}), {"rows": 0}

    R = np.full(n, np.nan, dtype="float64")
    R[h:] = L[h:] - L[:-h]  # trailing h-minute log return ending at each candle
    sigma_all = _rolling_std_of_trailing_returns(R, window)
    short_all = _rolling_std_of_trailing_returns(R, VOL_SHORT_WINDOW)

    sigma_h = sigma_all[last]
    short_h = short_all[last]

    # Sample matrix: the exact window used for sigma_h, all candles < t.
    offsets = np.arange(window, dtype=np.int64)
    sample_pos = last[:, None] - offsets[None, :]
    assert np.all(sample_pos < positions[:, None]), "sample uses candle at or after t"
    assert np.all(sample_pos >= 0), "sample before the start of the series"
    samples = R[sample_pos]  # (B, window)

    finite = (
        np.isfinite(sigma_h)
        & (sigma_h > 0.0)
        & np.isfinite(short_h)
        & np.all(np.isfinite(samples), axis=1)
    )
    anchors = anchors[finite]
    samples = samples[finite]
    sigma_h = sigma_h[finite]
    short_h = short_h[finite]
    if len(anchors) == 0:
        empty_features = {
            config.label: pd.DataFrame({column: np.empty(0, dtype="float64") for column in feature_columns})
            for config in configs
        }
        return empty_features, pd.DataFrame({column: np.empty(0, dtype="float64") for column in audit_columns}), {"rows": 0}

    z = samples / sigma_h[:, None]
    x0 = np.clip(z[:, 0], -X0_CLIP, X0_CLIP)
    sigma_packet = np.clip(short_h / sigma_h, SIGMA_CLIP[0], SIGMA_CLIP[1])

    x_grid, _, _ = grid()
    observable_keys = {"schr_p_up": "p_up", "schr_mean": "mean", "schr_std": "std", "schr_kurt": "kurt"}
    B = len(anchors)
    columns = {config.label: {name: np.empty(B, dtype="float64") for name in feature_columns[1:]} for config in configs}
    stats = {"rows": int(B), "kde_fallback_rows": 0}
    for start in range(0, B, int(chunk)):
        stop = min(B, start + int(chunk))
        z_chunk = z[start:stop]
        density = kde_density(z_chunk, x_grid, stats=stats)
        potentials: dict[str, np.ndarray | None] = {}
        for config in configs:
            if config.potential not in potentials:
                potentials[config.potential] = build_potential(density, x_grid, config.potential)
        packets: dict[str, np.ndarray] = {}
        for config in configs:
            if config.k0_map not in packets:
                packets[config.k0_map] = gaussian_packet(
                    x_grid, x0[start:stop], sigma_packet[start:stop], k0_for(x0[start:stop], config.k0_map)
                )
            psi_t = split_step_propagate(packets[config.k0_map], potentials[config.potential], tau_for(h, config.tau_map))
            obs = born_observables(psi_t, x_grid)
            for name in SCHR_BASE_FEATURES:
                columns[config.label][name][start:stop] = np.asarray(obs[observable_keys[name]], dtype="float64")
            columns[config.label]["schr_k0"][start:stop] = k0_for(x0[start:stop], config.k0_map)
    features_by_label: dict[str, pd.DataFrame] = {}
    for config in configs:
        data = {"target_time_s": anchors.astype("float64")}
        data.update(columns[config.label])
        features_by_label[config.label] = pd.DataFrame(data)
    audit = pd.DataFrame(
        {
            "target_time_s": anchors.astype("float64"),
            "schr_trailing_vol": sigma_h,
            "schr_x0": x0,
            "schr_sigma": sigma_packet,
        }
    )
    return features_by_label, audit, stats


def build_features_for_asset(
    close: pd.Series,
    anchors_time_s: Sequence[int] | np.ndarray,
    horizon: int,
    config: SchrodingerConfig,
    window: int = KDE_WINDOW,
    chunk: int = 8192,
) -> pd.DataFrame:
    """Convenience wrapper: one config, features plus audit columns merged."""
    features_by_label, audit, stats = build_features_for_asset_multi(
        close, anchors_time_s, horizon, [config], window=window, chunk=chunk
    )
    del stats
    out = features_by_label[config.label].merge(audit, on="target_time_s", how="left", validate="one_to_one")
    return out


def build_anchor_frame(
    db_path,
    anchor_minutes: int = 60,
    horizons: Sequence[int] = (1, 5, 15, 60),
    configs: Sequence[SchrodingerConfig] | SchrodingerConfig | None = None,
    start_time_s: int | None = None,
    end_time_s: int | None = None,
    chunk: int = 8192,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """Build evaluation frames from a database: causal features + Schr features.

    Loads the base market frames and the registered feature frame with
    :mod:`features`, then delegates to :func:`build_anchor_frame_from_market`.
    """
    import features as features_module

    frame = features_module.build_feature_frame(
        db_path, None, start_time_s=start_time_s, end_time_s=end_time_s, include_gcp=False
    )
    market = features_module._query_market(db_path, features_module.ASSETS, start_time_s, end_time_s)
    return build_anchor_frame_from_market(
        frame, market, anchor_minutes=anchor_minutes, horizons=horizons, configs=configs, chunk=chunk
    )


def build_anchor_frame_from_market(
    frame: pd.DataFrame,
    market: Mapping[str, pd.DataFrame],
    anchor_minutes: int = 60,
    horizons: Sequence[int] = (1, 5, 15, 60),
    configs: Sequence[SchrodingerConfig] | SchrodingerConfig | None = None,
    chunk: int = 8192,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """Build evaluation frames on supplied (possibly concatenated) market data.

    ``frame`` must be ``features._assemble_feature_frame(market)`` for the same
    ``market`` dict; the concatenated dev-tail + holdout case uses this entry
    point so trailing windows cross the source boundary while the loader,
    targets, and ``shift(1)`` stay the registered ones.  Anchor rows are those
    whose ``target_time_s`` is a multiple of ``anchor_minutes * 60``; features
    are joined on ``(symbol, target_time_s)``.  Returns
    ``(frame_by_config_label, audit)``.
    """
    import features as features_module

    if configs is None:
        raise ValueError("configs is required")
    if isinstance(configs, SchrodingerConfig):
        configs = [configs]
    configs = list(configs)
    horizons = tuple(int(h) for h in horizons)
    anchor_seconds = int(anchor_minutes) * 60
    anchors_mask = frame["target_time_s"].to_numpy(dtype="int64") % anchor_seconds == 0
    anchors = frame.loc[anchors_mask].copy()
    anchor_times = np.unique(anchors["target_time_s"].to_numpy(dtype="int64"))

    audit: dict = {
        "anchor_minutes": int(anchor_minutes),
        "anchor_convention": f"target_time_s % {anchor_seconds} == 0",
        "n_anchor_times": int(len(anchor_times)),
        "frame_min_time_s": int(frame["target_time_s"].min()),
        "frame_max_time_s": int(frame["target_time_s"].max()),
        "configs": [config.as_dict() for config in configs],
        "assets": {},
        "schrodinger_rows": {},
        "kde_fallback_rows": 0,
    }
    per_label_symbol: dict[str, list[pd.DataFrame]] = {config.label: [] for config in configs}
    for symbol in features_module.ASSETS:
        asset = anchors.loc[anchors["symbol"].eq(symbol)].copy()
        per_h: dict[str, dict[int, pd.DataFrame]] = {config.label: {} for config in configs}
        per_h_audit: dict[int, pd.DataFrame] = {}
        for h in horizons:
            features_by_label, h_audit, stats = build_features_for_asset_multi(
                market[symbol]["close"], asset["target_time_s"].to_numpy(), h, configs, chunk=chunk
            )
            audit["kde_fallback_rows"] += int(stats.get("kde_fallback_rows", 0))
            for config in configs:
                feats = features_by_label[config.label].rename(
                    columns={name: f"{name}_{h}m" for name in SCHR_BASE_FEATURES}
                )
                feats = feats[["target_time_s", *schr_feature_columns(h)]]
                per_h[config.label][h] = feats
                audit["schrodinger_rows"].setdefault(config.label, {}).setdefault(symbol, {})[f"{h}m"] = int(len(feats))
            rename_audit = {
                "schr_trailing_vol": f"schr_trailing_vol_{h}m",
                "schr_x0": f"schr_x0_{h}m",
                "schr_sigma": f"schr_sigma_{h}m",
            }
            h_audit = h_audit.rename(columns=rename_audit)
            audit.setdefault("anchor_rows", {}).setdefault(symbol, {})[f"{h}m"] = int(len(h_audit))
            per_h_audit[h] = h_audit
        for config in configs:
            merged = asset
            for h in horizons:
                merged = merged.merge(per_h[config.label][h], on="target_time_s", how="left", validate="one_to_one")
            for h in horizons:
                merged = merged.merge(per_h_audit[h], on="target_time_s", how="left", validate="one_to_one")
            per_label_symbol[config.label].append(merged)
    frames: dict[str, pd.DataFrame] = {}
    for config in configs:
        frames[config.label] = (
            pd.concat(per_label_symbol[config.label], ignore_index=True)
            .sort_values(["target_time_s", "symbol"], kind="mergesort")
            .reset_index(drop=True)
        )
    audit["rows"] = int(len(frames[configs[0].label]))
    return frames, audit
