"""Pre-registered "find the seed" arm (``seed_lab``).

The owner's premise ("the market is a procedural game with a fixed seed") is
false as stated: prices are aggregated orders, not PRNG output, and seed
recovery needs a known generator.  This driver tests the honest versions that
are frozen in ``seed_lab_preregistration.md``:

* **H_seed** - the 1-minute path is (close to) a low-dimensional deterministic
  system: Takens delay embedding + Sugihara-May simplex projection and S-map
  forecasts (family F1), plus descriptive determinism diagnostics (simplex
  skill vs E / horizon / library size, S-map theta curve, lzma compression of
  the return-sign sequence vs shuffles, BDS).
* **H_formula** - "esoteric" families carry out-of-sample skill on top of the
  registered market features: F2 Gaussian-process extrapolation, F3 folk TA
  (Fibonacci, Gann, round numbers, 369, lunar phase, golden-ratio time,
  Benford), F4 cycles/fractals (FFT extrapolation, Hurst R/S, permutation
  entropy), F5 Sornette LPPLS, and a kitchen sink with the frozen
  Schrodinger and physics columns.

Everything else is imported from the repository: the causal frame and targets
(``features.py``), folds/models/metrics/paired moving-block bootstrap and the
shuffled-target control (``backtest.py``), the Bonferroni majority summary
(``train._ablation_summary``), the complete-case policy, fit-once scoring and
one-sample chance tests (``train_schrodinger.py``), the frozen Schrodinger
builder (``schrodinger.py``) and the arm/delta helpers of ``mix_lab.py``.

Stages::

    .venv/bin/python seed_lab.py --stage explore   # determinism diagnostics + dev tables
    .venv/bin/python seed_lab.py --stage frozen    # dev tables only
    .venv/bin/python seed_lab.py --stage holdout \\
        --holdout-db ../crypto-data/holdout_klines_1m.db   # one-shot

Feature construction and cell evaluation run in a fork process pool (default
8 workers, one BLAS thread each).
"""
from __future__ import annotations

import os

# One BLAS thread per process: parallelism comes from the process pool.
for _thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_var] = "1"

import argparse  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import lzma  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from functools import lru_cache  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Mapping, Sequence  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from numpy.lib.stride_tricks import sliding_window_view  # noqa: E402
from scipy.spatial.distance import cdist  # noqa: E402
from scipy.linalg import solve_triangular  # noqa: E402
from scipy.stats import norm  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import features as features_module  # noqa: E402
from backtest import (  # noqa: E402
    HORIZONS,
    MARKET_FEATURES,
    direction_metrics,
    evaluate_predictions,
    shuffled_target_control,
)
from features import (  # noqa: E402
    ASSETS,
    PHYSICS_FEATURES,
    TARGET_COLUMNS,
    _assemble_feature_frame,
    assert_causal_features,
    market_db_audit,
)
from mix_lab import FROZEN_SCHR, _concat_market, _merge_write, _pair_deltas, _predict_arms  # noqa: E402
from schrodinger import SCHR_BASE_FEATURES, schr_feature_columns  # noqa: E402
from schrodinger import build_features_for_asset_multi as schr_build_features  # noqa: E402
from train import _ablation_summary, _jsonable  # noqa: E402
from train_schrodinger import block_len_for, complete_case_rows, fit_and_score, s0_chance_tests  # noqa: E402

DEFAULT_DB = HERE.parent / "crypto-data" / "crypto_market.db"
DEFAULT_OUTPUT = HERE / "results_seed_lab_dev.json"
DEFAULT_HOLDOUT_OUTPUT = HERE / "results_seed_lab_holdout.json"

# --------------------------------------------------------------------------
# Pre-registered constants (seed_lab_preregistration.md sections 4-8)
# --------------------------------------------------------------------------
ANCHOR_MINUTES_DEV = 60

F1_E = 6
F1_TAU = 1
F1_LIBRARY = 2000
F1_LOO_Q = 250
F1_SMAP_THETA = 2.0

GP_WINDOW = 120
GP_ELL = 30.0
GP_SF2 = 20.0
GP_SN2 = 1.0

FIB_WINDOW = 240
FIB_LEVELS = np.array([0.236, 0.382, 0.5, 0.618, 0.786])
SWING_WINDOW = 240
BENFORD_WINDOW = 60
BENFORD_MIN_COUNT = 30
SYNODIC_MONTH_DAYS = 29.530588853
NEW_MOON_REFERENCE_S = 947182440  # 2000-01-06 18:14 UTC
GOLDEN_RATIO = (1.0 + 5.0 ** 0.5) / 2.0

FFT_WINDOW = 256
FFT_HARMONICS = 3
HURST_WINDOW = 512
HURST_CHUNKS = (8, 16, 32, 64, 128, 256, 512)
PE_ORDER = 4
PE_WINDOW = 240

LPPLS_WINDOW = 360
LPPLS_TC_OFFSETS = (15, 30, 60, 120)
LPPLS_M = (0.3, 0.5, 0.7)
LPPLS_OMEGA = (6.0, 9.0, 12.0)
LPPLS_DAMPING_MIN = 0.8

CAL_SHIFT_MINUTES = 64462  # registered decorrelating shift (30 d + 14 d 18 h 22 min)
CAL_30D_MINUTES = 43200  # literal 30-day shift (descriptive only, degenerate)

# Newest closed candle index n must satisfy n >= MIN_HISTORY (F1 needs 2000 returns).
MIN_HISTORY = max(F1_LIBRARY, HURST_WINDOW, LPPLS_WINDOW, FFT_WINDOW, GP_WINDOW, FIB_WINDOW)

HOLDOUT_PRIMARY_START_S = 1790957580  # 2026-10-02 16:13 UTC (never evaluated)
HOLDOUT_END_S = 1791662940  # 2026-10-10 20:09 UTC
DEV_TAIL_DAYS = 3

N_CELLS = len(ASSETS) * len(HORIZONS)
REGISTERED_COMPARISONS: tuple[tuple[str, str], ...] = (
    ("F1", "M"),
    ("F2", "M"),
    ("F3", "M"),
    ("F4", "M"),
    ("F5", "M"),
    ("KS", "M"),
    ("CAL", "CAL_S"),
)
DESCRIPTIVE_COMPARISONS: tuple[tuple[str, str], ...] = (("CAL", "CAL_30"),)
ALL_COMPARISONS = REGISTERED_COMPARISONS + DESCRIPTIVE_COMPARISONS
ALPHA = 0.05 / float(N_CELLS * len(REGISTERED_COMPARISONS))  # 4.464e-4 per metric family
ALPHA_POOLED_DIRECTION = 0.05 / float(2 * N_CELLS * len(REGISTERED_COMPARISONS))  # 2.232e-4 (reported)
DIRECT_READOUTS = {
    "simplex": "seed_simplex_{h}m",
    "smap": "seed_smap_{h}m",
    "gp_drift": "gp_drift_{h}m",
    "fft": "fft_extrap_{h}m",
    "lppls_slope": "lppls_slope_{h}m",
}
ALPHA_DIRECT = 0.05 / float(N_CELLS * len(DIRECT_READOUTS))  # 6.25e-4

DIAG_ANCHOR_SECONDS = 43200
DIAG_MIN_HISTORY = 8000
DIAG_E = tuple(range(1, 11))
DIAG_H = (1, 2, 3, 5, 10, 15, 30, 60)
DIAG_L = (100, 250, 500, 1000, 2000, 4000, 8000)
DIAG_THETA = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
COMPRESSION_N = 262144
COMPRESSION_SHUFFLES = 200
COMPRESSION_SEED = 20261010
BDS_BLOCKS = 8
BDS_BLOCK_LEN = 4000
BDS_DISTANCE = 1.5
BDS_MAX_DIM = 5
BDS_VOL_WINDOW = 60

F3_COLUMNS: tuple[str, ...] = (
    "fib_dist",
    "gann_sin",
    "gann_cos",
    "round_logdist",
    "dr369_price",
    "dr369_minute",
    "lunar_sin",
    "lunar_cos",
    "phi_time_frac",
    "benford_chi2",
)
CAL_COLUMNS: tuple[str, ...] = ("lunar_sin", "lunar_cos", "dr369_minute")
CAL_SHIFT_COLUMNS = tuple(f"{name}_shift" for name in CAL_COLUMNS)
CAL_30D_COLUMNS = tuple(f"{name}_30d" for name in CAL_COLUMNS)
HORIZON_FREE_SEED_COLUMNS = (
    *F3_COLUMNS,
    "hurst_rs_512",
    "perm_entropy_4_240",
    "lppls_conf",
    *CAL_SHIFT_COLUMNS,
    *CAL_30D_COLUMNS,
)


def family_columns(family: str, horizon: int) -> tuple[str, ...]:
    h = int(horizon)
    if family == "F1":
        return (f"seed_simplex_{h}m", f"seed_simplex_rho_{h}m", f"seed_smap_{h}m")
    if family == "F2":
        return (f"gp_drift_{h}m", f"gp_std_{h}m")
    if family == "F3":
        return F3_COLUMNS
    if family == "F4":
        return (f"fft_extrap_{h}m", "hurst_rs_512", "perm_entropy_4_240")
    if family == "F5":
        return ("lppls_conf", f"lppls_slope_{h}m")
    raise ValueError(family)


def seed_columns_for_horizon(horizon: int) -> list[str]:
    columns: list[str] = []
    for family in ("F1", "F2", "F3", "F4", "F5"):
        for name in family_columns(family, horizon):
            if name not in columns:
                columns.append(name)
    return columns


def arms_for(horizon: int) -> dict[str, list[str]]:
    market = list(MARKET_FEATURES)
    arms = {"M": market}
    for family in ("F1", "F2", "F3", "F4", "F5"):
        arms[family] = market + list(family_columns(family, horizon))
    arms["KS"] = market + seed_columns_for_horizon(horizon) + list(schr_feature_columns(horizon)) + list(PHYSICS_FEATURES)
    arms["CAL"] = market + list(CAL_COLUMNS)
    arms["CAL_S"] = market + list(CAL_SHIFT_COLUMNS)
    arms["CAL_30"] = market + list(CAL_30D_COLUMNS)
    return arms


def union_columns(horizon: int) -> list[str]:
    seen: list[str] = []
    for columns in arms_for(horizon).values():
        for name in columns:
            if name not in seen:
                seen.append(name)
    return seen


# --------------------------------------------------------------------------
# F1: Takens embedding, simplex projection, S-map
# --------------------------------------------------------------------------
def embed(z: np.ndarray, E: int) -> np.ndarray:
    """Delay embedding rows ``x_i = (z_i, z_{i-1}, ..., z_{i-E+1})`` for i = E-1..L-1."""
    return sliding_window_view(np.asarray(z, dtype="float64"), int(E))[:, ::-1]


def cumulative_targets(z: np.ndarray, horizon: int) -> np.ndarray:
    """``T_h(i) = z_{i+1} + ... + z_{i+h}`` for i = 0..L-1-h."""
    z = np.asarray(z, dtype="float64")
    cs = np.concatenate([[0.0], np.cumsum(z)])
    i = np.arange(len(z) - int(horizon))
    return cs[i + int(horizon) + 1] - cs[i + 1]


def simplex_weights(d: np.ndarray) -> np.ndarray:
    """Sugihara-May weights ``exp(-d/d_min)``; indicator of ``d == 0`` if ``d_min == 0``."""
    d = np.asarray(d, dtype="float64")
    d1 = d.min(axis=-1, keepdims=True)
    safe = np.where(d1 > 0.0, d1, 1.0)
    return np.where(d1 > 0.0, np.exp(-d / safe), (d == 0.0).astype("float64"))


def _knn_forecast(D: np.ndarray, targets: np.ndarray, k: int) -> np.ndarray:
    idx = np.argpartition(D, k - 1, axis=1)[:, :k]
    d = np.take_along_axis(D, idx, axis=1)
    w = simplex_weights(d)
    return (w * targets[idx]).sum(axis=1) / w.sum(axis=1)


def _pairwise_dist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Exact Euclidean distances (direct differences, no |a|^2+|b|^2-2ab cancellation)."""
    return cdist(np.ascontiguousarray(A), np.ascontiguousarray(B), metric="euclidean")


def _smap(X: np.ndarray, y: np.ndarray, d: np.ndarray, query: np.ndarray, theta: float) -> float:
    dbar = float(d.mean())
    w = np.exp(-float(theta) * d / dbar) if dbar > 0.0 else np.ones_like(d)
    A = np.column_stack([np.ones(len(X)), X])
    sw = np.sqrt(w)
    beta = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)[0]
    return float(beta[0] + query @ beta[1:])


def _library(z: np.ndarray, E: int, horizon: int):
    L = len(z)
    emb = embed(z, E)
    T = cumulative_targets(z, horizon)
    n_lib = L - E - horizon + 1  # library i in [E-1, L-1-h]
    if n_lib < E + 1:
        raise ValueError("library too short")
    X = emb[:n_lib]
    y = T[E - 1 : E - 1 + n_lib]
    return emb, T, X, y, n_lib


def simplex_forecast(z: np.ndarray, E: int, horizon: int) -> float:
    """Simplex h-step forecast of T_h for the query embedding ending at z[-1]."""
    emb, _T, X, y, _n = _library(z, E, horizon)
    d = np.sqrt(((X - emb[-1]) ** 2).sum(axis=1))
    return float(_knn_forecast(d[None, :], y, E + 1)[0])


def smap_forecast(z: np.ndarray, E: int, horizon: int, theta: float) -> float:
    """S-map (locally weighted linear) h-step forecast for the query embedding."""
    emb, _T, X, y, _n = _library(z, E, horizon)
    d = np.sqrt(((X - emb[-1]) ** 2).sum(axis=1))
    return _smap(X, y, d, emb[-1], theta)


def simplex_loo(z: np.ndarray, E: int, horizon: int, Q: int, radius: int | None = None):
    """Leave-one-out simplex forecasts for the last Q library points (exclusion radius)."""
    z = np.asarray(z, dtype="float64")
    L = len(z)
    emb, T, X, y, n_lib = _library(z, E, horizon)
    radius = int(horizon) + int(E) - 1 if radius is None else int(radius)
    lib_i = np.arange(E - 1, E - 1 + n_lib)
    q_i = np.arange(L - horizon - Q, L - horizon)
    D = _pairwise_dist(emb[q_i - (E - 1)], X)
    D[np.abs(lib_i[None, :] - q_i[:, None]) <= radius] = np.inf
    return _knn_forecast(D, y, E + 1), T[q_i]


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    good = np.isfinite(a) & np.isfinite(b)
    a, b = a[good], b[good]
    if len(a) < 3 or a.std() == 0.0 or b.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def f1_features(z: np.ndarray, horizons: Sequence[int] = HORIZONS, E: int = F1_E, Q: int = F1_LOO_Q,
                theta: float = F1_SMAP_THETA) -> dict[int, tuple[float, float, float]]:
    """Registered F1 columns for one standardized library segment, all horizons.

    Returns ``{h: (simplex/sqrt(h), loo_rho, smap/sqrt(h))}``.  Identical to
    :func:`simplex_forecast`, :func:`simplex_loo` and :func:`smap_forecast`
    (asserted in ``test_seed_lab.py``) but shares one distance matrix.
    """
    z = np.asarray(z, dtype="float64")
    L = len(z)
    emb = embed(z, E)
    cs = np.concatenate([[0.0], np.cumsum(z)])
    hmax = int(max(horizons))
    q_lo = L - hmax - Q
    lib_all = emb[: L - E]  # i = E-1 .. L-2
    lib_i_all = np.arange(E - 1, L - 1)
    D_all = _pairwise_dist(emb[np.arange(q_lo, L - 1) - (E - 1)], lib_all)
    query = emb[-1]
    d_query = np.sqrt(((lib_all - query) ** 2).sum(axis=1))
    out: dict[int, tuple[float, float, float]] = {}
    for h in horizons:
        h = int(h)
        n_lib = L - E - h + 1
        i_lib = lib_i_all[:n_lib]
        y = cs[i_lib + h + 1] - cs[i_lib + 1]
        simplex = float(_knn_forecast(d_query[None, :n_lib], y, E + 1)[0])
        smap = _smap(lib_all[:n_lib], y, d_query[:n_lib], query, theta)
        q_i = np.arange(L - h - Q, L - h)
        D = D_all[q_i - q_lo, :n_lib]
        D[np.abs(i_lib[None, :] - q_i[:, None]) <= h + E - 1] = np.inf
        pred = _knn_forecast(D, y, E + 1)
        obs = cs[q_i + h + 1] - cs[q_i + 1]
        out[h] = (simplex / np.sqrt(h), pearson(pred, obs), smap / np.sqrt(h))
    return out


# --------------------------------------------------------------------------
# F2: Gaussian-process extrapolation (fixed hyperparameters)
# --------------------------------------------------------------------------
def gp_extrapolate(u: np.ndarray, s: np.ndarray, s_star: float, ell: float = GP_ELL, sf2: float = GP_SF2,
                   sn2: float = GP_SN2) -> tuple[np.ndarray, float]:
    """OLS-line mean + RBF/white residual GP; posterior mean and predictive sd at s_star.

    ``u`` may be ``(W,)`` or ``(W, B)``; the mean is linear in ``u``.
    """
    u = np.asarray(u, dtype="float64")
    s = np.asarray(s, dtype="float64")
    U = u[:, None] if u.ndim == 1 else u
    A = np.column_stack([np.ones(len(s)), s])
    coef = np.linalg.lstsq(A, U, rcond=None)[0]
    resid = U - A @ coef
    K = sf2 * np.exp(-((s[:, None] - s[None, :]) ** 2) / (2.0 * ell * ell)) + sn2 * np.eye(len(s))
    chol = np.linalg.cholesky(K)
    alpha = solve_triangular(chol.T, solve_triangular(chol, resid, lower=True), lower=False)
    k_star = sf2 * np.exp(-((s - s_star) ** 2) / (2.0 * ell * ell))
    mean = coef[0] + coef[1] * s_star + k_star @ alpha
    v = solve_triangular(chol, k_star, lower=True)
    sd = float(np.sqrt(max(sf2 - float(v @ v) + sn2, 0.0)))
    return (mean[0] if u.ndim == 1 else mean), sd


@lru_cache(maxsize=None)
def gp_filter(horizon: int) -> tuple[np.ndarray, float]:
    """The fixed linear filter ``mu(h) = w_h . u`` and the constant sd(h) (vol units)."""
    s = np.arange(-(GP_WINDOW - 1), 1, dtype="float64")
    weights, sd = gp_extrapolate(np.eye(GP_WINDOW), s, float(horizon))
    return np.asarray(weights, dtype="float64"), sd


# --------------------------------------------------------------------------
# F4: FFT extrapolation, Hurst R/S, permutation entropy
# --------------------------------------------------------------------------
def fft_projection(x: np.ndarray, n_ahead: int, n_harmonics: int = FFT_HARMONICS) -> np.ndarray:
    """Project the top Hann-windowed spectral peaks of each demeaned row ``n_ahead`` steps."""
    x = np.atleast_2d(np.asarray(x, dtype="float64"))
    B, N = x.shape
    xd = x - x.mean(axis=1, keepdims=True)
    window = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(N) / N)
    X = np.fft.rfft(xd * window[None, :], axis=1)
    A = np.abs(X)
    kmax = N // 2 - 1
    centre = A[:, 1 : kmax + 1]
    peak = (centre > A[:, 0:kmax]) & (centre >= A[:, 2 : kmax + 2])
    score = np.where(peak, centre, -np.inf)
    order = np.argsort(-score, axis=1, kind="stable")[:, :n_harmonics]
    valid = np.isfinite(np.take_along_axis(score, order, axis=1))
    bins = order + 1
    amp = 4.0 * np.take_along_axis(A, bins, axis=1) / N * valid
    phase = np.angle(np.take_along_axis(X, bins, axis=1))
    t = N + np.arange(int(n_ahead))
    arg = 2.0 * np.pi * bins[:, :, None] * t[None, None, :] / N + phase[:, :, None]
    return (amp[:, :, None] * np.cos(arg)).sum(axis=1)


def hurst_rs(x: np.ndarray, chunks: Sequence[int] = HURST_CHUNKS) -> np.ndarray:
    """Rescaled-range Hurst exponent per row (OLS slope of ln R/S on ln m)."""
    x = np.atleast_2d(np.asarray(x, dtype="float64"))
    B, N = x.shape
    log_rs = np.empty((B, len(chunks)), dtype="float64")
    with np.errstate(invalid="ignore", divide="ignore"):
        for j, m in enumerate(chunks):
            k = N // m
            X = x[:, : k * m].reshape(B, k, m)
            Y = np.cumsum(X - X.mean(axis=2, keepdims=True), axis=2)
            R = Y.max(axis=2) - Y.min(axis=2)
            S = X.std(axis=2)
            rs = np.where(S > 0.0, R / np.where(S > 0.0, S, 1.0), np.nan)
            count = np.sum(np.isfinite(rs), axis=1)
            mean_rs = np.where(count > 0, np.nansum(rs, axis=1) / np.maximum(count, 1), np.nan)
            log_rs[:, j] = np.log(mean_rs)
    lm = np.log(np.asarray(chunks, dtype="float64"))
    lmc = lm - lm.mean()
    slope = ((log_rs - log_rs.mean(axis=1, keepdims=True)) * lmc[None, :]).sum(axis=1) / float((lmc * lmc).sum())
    slope[~np.all(np.isfinite(log_rs), axis=1)] = np.nan
    return slope


def permutation_entropy(x: np.ndarray, order: int = PE_ORDER) -> np.ndarray:
    """Normalized Bandt-Pompe permutation entropy (delay 1, stable tie-break) per row."""
    x = np.atleast_2d(np.asarray(x, dtype="float64"))
    B = x.shape[0]
    patterns = sliding_window_view(x, order, axis=1)  # (B, P, order)
    perm = np.argsort(patterns, axis=2, kind="stable")
    codes = (perm * (order ** np.arange(order))[None, None, :]).sum(axis=2)
    n_codes = order ** order
    P = codes.shape[1]
    flat = (codes + n_codes * np.arange(B)[:, None]).ravel()
    counts = np.bincount(flat, minlength=B * n_codes).reshape(B, n_codes).astype("float64")
    p = counts / P
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.where(p > 0.0, p * np.log(p), 0.0).sum(axis=1)
    return ent / np.log(float(math.factorial(order)))


# --------------------------------------------------------------------------
# F5: LPPLS on a fixed grid
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def lppls_grid() -> tuple[tuple[tuple[int, float, float], ...], np.ndarray, np.ndarray]:
    """Grid points ``(D, m, omega)``, designs ``(G, W, 4)`` and pseudo-inverses ``(G, 4, W)``."""
    tau = np.arange(LPPLS_WINDOW, dtype="float64")
    points: list[tuple[int, float, float]] = []
    designs, pinvs = [], []
    for D in LPPLS_TC_OFFSETS:
        for m in LPPLS_M:
            for omega in LPPLS_OMEGA:
                x = (LPPLS_WINDOW - 1 + D) - tau
                f = x ** m
                theta = omega * np.log(x)
                X = np.column_stack([np.ones_like(f), f, f * np.cos(theta), f * np.sin(theta)])
                points.append((int(D), float(m), float(omega)))
                designs.append(X)
                pinvs.append(np.linalg.pinv(X))
    return tuple(points), np.stack(designs), np.stack(pinvs)


def lppls_fit(Y: np.ndarray) -> dict[str, np.ndarray]:
    """Linear OLS of ``A + B f + C1 g + C2 k`` at every grid point; ``Y`` is ``(B, 360)``."""
    Y = np.atleast_2d(np.asarray(Y, dtype="float64"))
    _points, designs, pinvs = lppls_grid()
    coefs = np.einsum("gkw,bw->bgk", pinvs, Y)
    fitted = np.einsum("gwk,bgk->bgw", designs, coefs)
    rss = ((Y[:, None, :] - fitted) ** 2).sum(axis=2)
    return {"coefs": coefs, "rss": rss, "best": np.argmin(rss, axis=1)}


def lppls_derivative(D: np.ndarray, m: np.ndarray, omega: np.ndarray, coefs: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """``dy/dtau`` of the LPPLS curve with tc = 359 + D (vectorized over rows)."""
    x = (LPPLS_WINDOW - 1 + D) - tau
    theta = omega * np.log(x)
    xm1 = x ** (m - 1.0)
    B, C1, C2 = coefs[:, 1], coefs[:, 2], coefs[:, 3]
    return (
        B * (-m * xm1)
        + C1 * xm1 * (-m * np.cos(theta) + omega * np.sin(theta))
        + C2 * xm1 * (-m * np.sin(theta) - omega * np.cos(theta))
    )


def lppls_features(Y: np.ndarray, horizons: Sequence[int] = HORIZONS) -> dict[str, np.ndarray]:
    """``lppls_conf`` and ``lppls_slope_{h}m`` for windows of log closes ``(B, 360)``."""
    Y = np.atleast_2d(np.asarray(Y, dtype="float64"))
    Y = Y - Y[:, -1:]
    points, _designs, _pinvs = lppls_grid()
    fit = lppls_fit(Y)
    coefs = fit["coefs"]
    m_all = np.array([p[1] for p in points])
    omega_all = np.array([p[2] for p in points])
    D_all = np.array([p[0] for p in points], dtype="float64")
    amp_c = np.sqrt(coefs[:, :, 2] ** 2 + coefs[:, :, 3] ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        damping = m_all[None, :] * np.abs(coefs[:, :, 1]) / (omega_all[None, :] * amp_c)
    damping = np.where(amp_c > 0.0, damping, np.inf)
    out = {"lppls_conf": np.mean((coefs[:, :, 1] < 0.0) & (damping >= LPPLS_DAMPING_MIN), axis=1)}
    best = fit["best"]
    rows = np.arange(len(Y))
    c_best = coefs[rows, best]
    D_b, m_b, w_b = D_all[best], m_all[best], omega_all[best]
    vol = np.std(np.diff(Y, axis=1), axis=1, ddof=1)
    vol = np.where(vol > 0.0, vol, np.nan)
    for h in horizons:
        tau_star = (LPPLS_WINDOW - 1) + np.minimum(float(h), D_b - 1.0)
        out[f"lppls_slope_{int(h)}m"] = lppls_derivative(D_b, m_b, w_b, c_best, tau_star) / vol
    out["best_index"] = best
    return out


# --------------------------------------------------------------------------
# F3: folk TA and calendar
# --------------------------------------------------------------------------
def digital_root(values: np.ndarray) -> np.ndarray:
    m = np.asarray(values, dtype="int64")
    return np.where(m == 0, 0, 1 + (m - 1) % 9)


def lunar_phase_fraction(epoch_s: np.ndarray) -> np.ndarray:
    """Mean synodic phase in [0, 1): 0 = new moon, 0.5 = full moon."""
    t = np.asarray(epoch_s, dtype="float64")
    return np.mod((t - NEW_MOON_REFERENCE_S) / (SYNODIC_MONTH_DAYS * 86400.0), 1.0)


def calendar_columns(epoch_s: np.ndarray, suffix: str = "") -> dict[str, np.ndarray]:
    """Lunar sin/cos and the 369 minute-of-day indicator at the given instants."""
    t = np.asarray(epoch_s, dtype="int64")
    f = lunar_phase_fraction(t)
    minute_of_day = (t // 60) % 1440
    return {
        f"lunar_sin{suffix}": np.sin(2.0 * np.pi * f),
        f"lunar_cos{suffix}": np.cos(2.0 * np.pi * f),
        f"dr369_minute{suffix}": np.isin(digital_root(minute_of_day), (3, 6, 9)).astype("float64"),
    }


def folk_ta(close_win: np.ndarray, volume_win: np.ndarray) -> dict[str, np.ndarray]:
    """Price/volume folk-TA columns. ``close_win`` (B, 240) ends at c_n; ``volume_win`` (B, 60)."""
    c = np.atleast_2d(np.asarray(close_win, dtype="float64"))
    v = np.atleast_2d(np.asarray(volume_win, dtype="float64"))
    cn = c[:, -1]
    hi = c.max(axis=1)
    lo = c.min(axis=1)
    rng = hi - lo
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(rng > 0.0, (cn - lo) / np.where(rng > 0.0, rng, 1.0), np.nan)
    nearest = FIB_LEVELS[np.argmin(np.abs(p[:, None] - FIB_LEVELS[None, :]), axis=1)]
    out = {"fib_dist": p - nearest}
    phase = 2.0 * np.pi * (np.mod(np.sqrt(cn), 2.0) / 2.0)
    out["gann_sin"] = np.sin(phase)
    out["gann_cos"] = np.cos(phase)
    step = 10.0 ** (np.floor(np.log10(cn)) - 1.0)
    out["round_logdist"] = np.log(cn / (step * np.round(cn / step)))
    cents = np.round(cn * 100.0).astype("int64")
    out["dr369_price"] = np.isin(digital_root(cents), (3, 6, 9)).astype("float64")
    age_max = np.argmax(c[:, ::-1], axis=1)
    age_min = np.argmin(c[:, ::-1], axis=1)
    age = np.minimum(age_max, age_min).astype("float64")
    out["phi_time_frac"] = np.mod(np.log(age + 1.0) / np.log(GOLDEN_RATIO), 1.0)
    out["benford_chi2"] = benford_chi2(v)
    return out


def benford_chi2(volumes: np.ndarray) -> np.ndarray:
    v = np.atleast_2d(np.asarray(volumes, dtype="float64"))
    positive = v > 0.0
    safe = np.where(positive, v, 1.0)
    lead = np.floor(safe / 10.0 ** np.floor(np.log10(safe))).astype("int64")
    lead = np.clip(lead, 1, 9)
    digits = np.arange(1, 10)
    counts = ((lead[:, :, None] == digits[None, None, :]) & positive[:, :, None]).sum(axis=1).astype("float64")
    total = positive.sum(axis=1).astype("float64")
    expected = total[:, None] * np.log10(1.0 + 1.0 / digits)[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        chi2 = ((counts - expected) ** 2 / expected).sum(axis=1)
    return np.where(total >= BENFORD_MIN_COUNT, chi2, np.nan)


# --------------------------------------------------------------------------
# Per-asset seed feature builder (causal)
# --------------------------------------------------------------------------
def build_seed_features(
    close: np.ndarray,
    volume: np.ndarray,
    times_s: np.ndarray,
    anchor_times: np.ndarray,
    horizons: Sequence[int] = HORIZONS,
) -> pd.DataFrame:
    """Every seed-lab column for one asset at the given anchors (candle opens ``t``).

    Inputs are sliced to candles strictly before the latest anchor, and every
    window ends at ``n = pos(t) - 1``, so a feature at ``t`` only reads candles
    with ``open_time_s < t``.  Calendar columns are evaluated at ``t - 60 s``.
    """
    times_s = np.asarray(times_s, dtype="int64")
    anchors = np.asarray(anchor_times, dtype="int64")
    positions = np.searchsorted(times_s, anchors)
    if np.any(positions >= len(times_s)) or not np.array_equal(times_s[positions], anchors):
        raise AssertionError("anchor times must be exact candle opens of this asset")
    cut = int(positions.max())
    close = np.asarray(close, dtype="float64")[:cut]
    volume = np.asarray(volume, dtype="float64")[:cut]
    n = positions - 1
    B = len(anchors)
    horizons = tuple(int(h) for h in horizons)
    names = list(HORIZON_FREE_SEED_COLUMNS)
    for h in horizons:
        names += [f"seed_simplex_{h}m", f"seed_simplex_rho_{h}m", f"seed_smap_{h}m", f"gp_drift_{h}m",
                  f"gp_std_{h}m", f"fft_extrap_{h}m", f"lppls_slope_{h}m"]
    cols = {name: np.full(B, np.nan, dtype="float64") for name in names}

    # Calendar / astronomical columns (no market data), at the open of candle n.
    stamp = anchors - 60
    cols.update(calendar_columns(stamp))
    cols.update(calendar_columns(stamp - CAL_SHIFT_MINUTES * 60, suffix="_shift"))
    cols.update(calendar_columns(stamp - CAL_30D_MINUTES * 60, suffix="_30d"))

    ok = n >= MIN_HISTORY
    sel = np.flatnonzero(ok)
    if len(sel):
        nn = n[sel]
        logc = np.log(close)
        r = np.empty_like(logc)
        r[0] = np.nan
        r[1:] = np.diff(logc)

        # F1
        for j, end in zip(sel, nn):
            seg = r[end - F1_LIBRARY + 1 : end + 1]
            sd = seg.std(ddof=1)
            if not np.isfinite(sd) or sd <= 0.0:
                continue
            values = f1_features((seg - seg.mean()) / sd, horizons)
            for h in horizons:
                simplex, rho, smap = values[h]
                cols[f"seed_simplex_{h}m"][j] = simplex
                cols[f"seed_simplex_rho_{h}m"][j] = rho
                cols[f"seed_smap_{h}m"][j] = smap

        # F2
        Y = sliding_window_view(logc, GP_WINDOW)[nn - GP_WINDOW + 1]
        sigma1 = np.std(np.diff(Y, axis=1), axis=1, ddof=1)
        sigma1 = np.where(sigma1 > 0.0, sigma1, np.nan)
        U = (Y - Y[:, -1:]) / sigma1[:, None]
        for h in horizons:
            weights, sd_h = gp_filter(h)
            cols[f"gp_drift_{h}m"][sel] = (U @ weights) / np.sqrt(h)
            cols[f"gp_std_{h}m"][sel] = sd_h * sigma1

        # F3 (price/volume part)
        cwin = sliding_window_view(close, FIB_WINDOW)[nn - FIB_WINDOW + 1]
        vwin = sliding_window_view(volume, BENFORD_WINDOW)[nn - BENFORD_WINDOW + 1]
        for name, values in folk_ta(cwin, vwin).items():
            cols[name][sel] = values

        # F4
        rwin = sliding_window_view(r, FFT_WINDOW)[nn - FFT_WINDOW + 1]
        proj = fft_projection(rwin, max(horizons))
        rsd = np.std(rwin, axis=1, ddof=1)
        rsd = np.where(rsd > 0.0, rsd, np.nan)
        for h in horizons:
            cols[f"fft_extrap_{h}m"][sel] = proj[:, :h].sum(axis=1) / (rsd * np.sqrt(h))
        cols["hurst_rs_512"][sel] = hurst_rs(sliding_window_view(r, HURST_WINDOW)[nn - HURST_WINDOW + 1])
        cols["perm_entropy_4_240"][sel] = permutation_entropy(
            sliding_window_view(close, PE_WINDOW)[nn - PE_WINDOW + 1]
        )

        # F5
        lp = lppls_features(sliding_window_view(logc, LPPLS_WINDOW)[nn - LPPLS_WINDOW + 1], horizons)
        cols["lppls_conf"][sel] = lp["lppls_conf"]
        for h in horizons:
            cols[f"lppls_slope_{h}m"][sel] = lp[f"lppls_slope_{h}m"]

    out = pd.DataFrame({"target_time_s": anchors})
    for name in names:
        out[name] = cols[name]
    return out


# --------------------------------------------------------------------------
# Process pool plumbing (fork; workers read module-level arrays)
# --------------------------------------------------------------------------
_ARRAYS: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}


def _set_arrays(key: str, market: Mapping[str, pd.DataFrame]) -> None:
    _ARRAYS[key] = {}
    for symbol in ASSETS:
        frame = market[symbol]
        times = frame.index.to_numpy(dtype="datetime64[s]").astype("int64")
        if not np.all(np.diff(times) == 60):
            raise AssertionError(f"{symbol}: candles are not contiguous ({key})")
        _ARRAYS[key][symbol] = (
            frame["close"].to_numpy(dtype="float64"),
            frame["volume"].to_numpy(dtype="float64"),
            times,
        )


def _pool(workers: int):
    return mp.get_context("fork").Pool(processes=int(workers), maxtasksperchild=None)


def _dispatch(task: tuple) -> tuple:
    kind = task[0]
    if kind == "seed":
        _, key, asset, anchors = task
        close, volume, times = _ARRAYS[key][asset]
        return ("seed", asset, None, build_seed_features(close, volume, times, anchors))
    if kind == "schr":
        _, key, asset, horizon, anchors = task
        close, _volume, times = _ARRAYS[key][asset]
        series = pd.Series(close, index=pd.to_datetime(times, unit="s", utc=True))
        feats, _audit, _stats = schr_build_features(series, anchors, horizon, [FROZEN_SCHR])
        frame = feats[FROZEN_SCHR.label][["target_time_s", *SCHR_BASE_FEATURES]].rename(
            columns={name: f"{name}_{horizon}m" for name in SCHR_BASE_FEATURES}
        )
        frame["target_time_s"] = frame["target_time_s"].to_numpy(dtype="float64").astype("int64")
        return ("schr", asset, horizon, frame)
    if kind == "diag":
        _, asset, ends = task
        return ("diag", asset, None, _diag_anchor_block(asset, ends))
    if kind == "compress":
        _, asset, asset_index = task
        return ("compress", asset, None, compression_test(asset, asset_index))
    if kind == "bds":
        _, asset = task
        return ("bds", asset, None, bds_blocks(asset))
    if kind == "dev_cell":
        return ("dev_cell", task[1], task[2], _dev_cell(*task[1:]))
    if kind == "holdout_cell":
        return ("holdout_cell", task[1], task[2], _holdout_cell(*task[1:]))
    raise ValueError(kind)


def add_seed_columns(frame: pd.DataFrame, key: str, pool, chunk: int) -> pd.DataFrame:
    """Merge seed-lab and frozen Schrodinger columns onto an anchor frame."""
    tasks: list[tuple] = []
    for asset in ASSETS:
        anchors = np.unique(frame.loc[frame["symbol"].eq(asset), "target_time_s"].to_numpy(dtype="int64"))
        for start in range(0, len(anchors), int(chunk)):
            part = anchors[start : start + int(chunk)]
            tasks.append(("seed", key, asset, part))
            for h in HORIZONS:
                tasks.append(("schr", key, asset, int(h), part))
    seed_parts: dict[str, list[pd.DataFrame]] = {asset: [] for asset in ASSETS}
    schr_parts: dict[tuple[str, int], list[pd.DataFrame]] = {}
    for kind, asset, horizon, part in pool.imap_unordered(_dispatch, tasks):
        if kind == "seed":
            seed_parts[asset].append(part)
        else:
            schr_parts.setdefault((asset, horizon), []).append(part)
    out_parts = []
    for asset in ASSETS:
        asset_frame = frame.loc[frame["symbol"].eq(asset)].copy()
        seed = pd.concat(seed_parts[asset], ignore_index=True)
        asset_frame = asset_frame.merge(seed, on="target_time_s", how="left", validate="one_to_one")
        for h in HORIZONS:
            schr = pd.concat(schr_parts[(asset, int(h))], ignore_index=True)
            asset_frame = asset_frame.merge(schr, on="target_time_s", how="left", validate="one_to_one")
        out_parts.append(asset_frame)
    out = pd.concat(out_parts, ignore_index=True)
    return out.sort_values(["target_time_s", "symbol"], kind="mergesort").reset_index(drop=True)


KEEP_BASE_COLUMNS = (
    "time",
    "symbol",
    "target_time_s",
    "feature_source_max_time_s",
    "ewma_vol_60m",
    *MARKET_FEATURES,
    *PHYSICS_FEATURES,
    *TARGET_COLUMNS,
)


def _epoch(frame: pd.DataFrame) -> np.ndarray:
    return features_module._epoch_seconds(frame.index)


def anchor_base_frame(market: Mapping[str, pd.DataFrame], anchor_minutes: int,
                      min_time_s: int | None = None, max_time_s: int | None = None) -> pd.DataFrame:
    base = _assemble_feature_frame(market)
    times = base["target_time_s"].to_numpy(dtype="int64")
    mask = times % (int(anchor_minutes) * 60) == 0
    if min_time_s is not None:
        mask &= times >= int(min_time_s)
    if max_time_s is not None:
        mask &= times <= int(max_time_s)
    frame = base.loc[mask, list(KEEP_BASE_COLUMNS)].copy().reset_index(drop=True)
    del base
    gc.collect()
    return frame


# --------------------------------------------------------------------------
# Determinism diagnostics (dev, descriptive)
# --------------------------------------------------------------------------
def _returns(close: np.ndarray) -> np.ndarray:
    logc = np.log(np.asarray(close, dtype="float64"))
    r = np.empty_like(logc)
    r[0] = np.nan
    r[1:] = np.diff(logc)
    return r


def _diag_anchor_block(asset: str, ends: np.ndarray) -> dict[str, np.ndarray]:
    close, _volume, _times = _ARRAYS["dev"][asset]
    r = _returns(close)
    B = len(ends)
    out = {
        "pred_E": np.full((B, len(DIAG_E)), np.nan),
        "pred_h": np.full((B, len(DIAG_H)), np.nan),
        "obs_h": np.full((B, len(DIAG_H)), np.nan),
        "pred_L": np.full((B, len(DIAG_L)), np.nan),
        "obs_L": np.full((B, len(DIAG_L)), np.nan),
        "pred_theta": np.full((B, len(DIAG_THETA)), np.nan),
    }
    for b, n in enumerate(ends):
        for li, L in enumerate(DIAG_L):
            seg = r[n - L + 1 : n + 1]
            mean, sd = seg.mean(), seg.std(ddof=1)
            if not np.isfinite(sd) or sd <= 0.0:
                continue
            z = (seg - mean) / sd
            fut = (r[n + 1 : n + 1 + max(DIAG_H)] - mean) / sd
            out["pred_L"][b, li] = simplex_forecast(z, F1_E, 1)
            out["obs_L"][b, li] = fut[0]
            if L == F1_LIBRARY:
                for ei, E in enumerate(DIAG_E):
                    out["pred_E"][b, ei] = simplex_forecast(z, E, 1)
                for hi, h in enumerate(DIAG_H):
                    out["pred_h"][b, hi] = simplex_forecast(z, F1_E, h)
                    out["obs_h"][b, hi] = fut[:h].sum()
                for ti, theta in enumerate(DIAG_THETA):
                    out["pred_theta"][b, ti] = smap_forecast(z, F1_E, 1, theta)
    return out


def _rho_entry(pred: np.ndarray, obs: np.ndarray) -> dict:
    good = np.isfinite(pred) & np.isfinite(obs)
    n = int(good.sum())
    rho = pearson(pred[good], obs[good])
    if n > 3 and np.isfinite(rho):
        zf = np.arctanh(np.clip(rho, -0.999999, 0.999999))
        half = 1.959963984540054 / np.sqrt(n - 3)
        ci = [float(np.tanh(zf - half)), float(np.tanh(zf + half))]
    else:
        ci = [float("nan"), float("nan")]
    return {"rho": rho, "n": n, "ci95": ci}


def compression_test(asset: str, asset_index: int) -> dict:
    close, _volume, _times = _ARRAYS["dev"][asset]
    r = _returns(close)[-COMPRESSION_N:]
    signs = (np.sign(r) + 1).astype("uint8")
    real = len(lzma.compress(signs.tobytes()))
    rng = np.random.default_rng(COMPRESSION_SEED + int(asset_index))
    shuffled = np.array([len(lzma.compress(rng.permutation(signs).tobytes())) for _ in range(COMPRESSION_SHUFFLES)])
    return {
        "n_returns": int(len(signs)),
        "fraction_zero_returns": float(np.mean(signs == 1)),
        "fraction_up": float(np.mean(signs == 2)),
        "compressed_bytes_real": int(real),
        "compressed_bytes_shuffled_mean": float(shuffled.mean()),
        "compressed_bytes_shuffled_min": int(shuffled.min()),
        "compressed_bytes_shuffled_max": int(shuffled.max()),
        "ratio_real_over_shuffled_mean": float(real / shuffled.mean()),
        "p_one_sided": float((1 + np.sum(shuffled <= real)) / (COMPRESSION_SHUFFLES + 1)),
        "n_shuffles": COMPRESSION_SHUFFLES,
        "seed": COMPRESSION_SEED + int(asset_index),
    }


def _upper_mean(indicator: np.ndarray) -> float:
    n = indicator.shape[0]
    total = float(indicator.sum()) - float(np.trace(indicator))
    return total / 2.0 / (n * (n - 1) / 2.0)


def bds_statistic(x: np.ndarray, max_dim: int = BDS_MAX_DIM, distance: float = BDS_DISTANCE) -> np.ndarray:
    """BDS statistics for m = 2..max_dim (statsmodels ``bds`` formula, eps = distance * sd)."""
    x = np.asarray(x, dtype="float64")
    nobs_full = len(x)
    eps = distance * x.std(ddof=1)
    ind = np.abs(x[:, None] - x[None, :]) < eps
    # correlation sums C_m on the progressively trimmed joint indicators
    corrsums = np.zeros(max_dim)
    joint = ind
    corrsums[0] = _upper_mean(joint)
    for m in range(1, max_dim):
        joint = joint[1:, 1:] & joint[:-1, :-1]
        corrsums[m] = _upper_mean(joint)
    n = nobs_full
    c1 = corrsums[0]
    row = ind.sum(axis=1).astype("float64")
    k = ((row ** 2).sum() - 3.0 * float(ind.sum()) + 2.0 * n) / (n * (n - 1.0) * (n - 2.0))
    stats = np.zeros(max_dim - 1)
    for m in range(2, max_dim + 1):
        tmp = sum((k ** (m - j)) * (c1 ** (2 * j)) for j in range(1, m))
        var = 4.0 * (k ** m + 2.0 * tmp + (m - 1) ** 2 * c1 ** (2 * m) - m ** 2 * k * c1 ** (2 * m - 2))
        trim = m - 1
        c1_trim = _upper_mean(ind[trim:, trim:])
        effect = corrsums[m - 1] - c1_trim ** m
        stats[m - 2] = np.sqrt(nobs_full - trim) * effect / np.sqrt(var)
    return stats


def bds_blocks(asset: str) -> dict:
    close, _volume, _times = _ARRAYS["dev"][asset]
    r = _returns(close)
    starts = np.linspace(BDS_VOL_WINDOW + 1, len(r) - BDS_BLOCK_LEN - 1, BDS_BLOCKS).astype("int64")
    raw, volstd = [], []
    for s in starts:
        block = r[s : s + BDS_BLOCK_LEN]
        raw.append(bds_statistic((block - block.mean()) / block.std(ddof=1)))
        trailing = sliding_window_view(r[s - BDS_VOL_WINDOW : s + BDS_BLOCK_LEN - 1], BDS_VOL_WINDOW).std(axis=1, ddof=1)
        volstd.append(bds_statistic(block / trailing))
    raw = np.array(raw)
    volstd = np.array(volstd)
    dims = list(range(2, BDS_MAX_DIM + 1))
    return {
        "block_starts_index": starts.tolist(),
        "dims": dims,
        "standardized": {
            "stats": raw.tolist(),
            "median": np.median(raw, axis=0).tolist(),
            "n_blocks_abs_gt_1_96": np.sum(np.abs(raw) > 1.96, axis=0).tolist(),
        },
        "vol_standardized": {
            "stats": volstd.tolist(),
            "median": np.median(volstd, axis=0).tolist(),
            "n_blocks_abs_gt_1_96": np.sum(np.abs(volstd) > 1.96, axis=0).tolist(),
        },
        "two_sided_p_of_median": {
            "standardized": (2.0 * norm.sf(np.abs(np.median(raw, axis=0)))).tolist(),
            "vol_standardized": (2.0 * norm.sf(np.abs(np.median(volstd, axis=0)))).tolist(),
        },
    }


def run_diagnostics(pool, dev_frame: pd.DataFrame) -> dict:
    started = time.time()
    tasks: list[tuple] = []
    for asset in ASSETS:
        _close, _volume, times = _ARRAYS["dev"][asset]
        positions = np.flatnonzero(times % DIAG_ANCHOR_SECONDS == 0)
        ends = positions - 1
        ends = ends[(ends >= DIAG_MIN_HISTORY) & (ends + max(DIAG_H) <= len(times) - 1)]
        for part in np.array_split(ends, 8):
            tasks.append(("diag", asset, part))
    for index, asset in enumerate(ASSETS):
        tasks.append(("compress", asset, index))
        tasks.append(("bds", asset))
    blocks: dict[str, list[dict]] = {asset: [] for asset in ASSETS}
    compression: dict[str, dict] = {}
    bds: dict[str, dict] = {}
    for kind, asset, _h, payload in pool.imap_unordered(_dispatch, tasks):
        if kind == "diag":
            blocks[asset].append(payload)
        elif kind == "compress":
            compression[asset] = payload
        else:
            bds[asset] = payload
    merged: dict[str, dict[str, np.ndarray]] = {}
    for asset in ASSETS:
        merged[asset] = {key: np.concatenate([b[key] for b in blocks[asset]]) for key in blocks[asset][0]}
    pooled = {key: np.concatenate([merged[a][key] for a in ASSETS]) for key in merged[ASSETS[0]]}
    groups = {**merged, "pooled": pooled}

    def panel(pred_key: str, obs_key: str | None, values: Sequence, obs_index_fixed: int | None = None) -> dict:
        result: dict[str, Any] = {"values": list(values), "by_asset": {}}
        for name, data in groups.items():
            entries = []
            for i, _v in enumerate(values):
                if obs_key is None:
                    obs = data["obs_h"][:, 0]
                elif obs_index_fixed is not None:
                    obs = data[obs_key][:, obs_index_fixed]
                else:
                    obs = data[obs_key][:, i]
                entries.append(_rho_entry(data[pred_key][:, i], obs))
            result["by_asset"][name] = entries
        return result

    diagnostics = {
        "anchors": f"target_time_s % {DIAG_ANCHOR_SECONDS} == 0, newest candle index >= {DIAG_MIN_HISTORY}",
        "n_anchors": {asset: int(len(merged[asset]["obs_h"])) for asset in ASSETS},
        "simplex_rho_vs_E": panel("pred_E", None, DIAG_E),
        "simplex_rho_vs_horizon": panel("pred_h", "obs_h", DIAG_H),
        "simplex_rho_vs_library": panel("pred_L", "obs_L", DIAG_L),
        "smap_rho_vs_theta": panel("pred_theta", None, DIAG_THETA),
        "compression": compression,
        "bds": bds,
    }
    loo = {}
    for asset in ASSETS:
        rows = dev_frame.loc[dev_frame["symbol"].eq(asset)]
        loo[asset] = {f"{h}m": float(np.nanmean(rows[f"seed_simplex_rho_{h}m"].to_numpy(dtype="float64"))) for h in HORIZONS}
    diagnostics["mean_in_library_loo_rho_hourly"] = loo
    diagnostics["h_seed_signature"] = h_seed_signature(diagnostics)
    diagnostics["runtime_seconds"] = time.time() - started
    return diagnostics


def h_seed_signature(diag: Mapping[str, Any]) -> dict:
    """Apply the pre-registered dev signature (preregistration section 7)."""
    e_idx = list(DIAG_E).index(F1_E)
    rho_e6 = {a: diag["simplex_rho_vs_E"]["by_asset"][a][e_idx]["rho"] for a in ASSETS}
    h_rho = diag["simplex_rho_vs_horizon"]["by_asset"]
    decays = {a: bool(h_rho[a][0]["rho"] > h_rho[a][-1]["rho"]) for a in ASSETS}
    l_rho = diag["simplex_rho_vs_library"]["by_asset"]
    i250, i8000 = list(DIAG_L).index(250), list(DIAG_L).index(8000)
    l_gain = {a: float(l_rho[a][i8000]["rho"] - l_rho[a][i250]["rho"]) for a in ASSETS}
    t_rho = diag["smap_rho_vs_theta"]["by_asset"]
    theta_gain = {a: float(max(e["rho"] for e in t_rho[a][1:]) - t_rho[a][0]["rho"]) for a in ASSETS}
    ratios = {a: diag["compression"][a]["ratio_real_over_shuffled_mean"] for a in ASSETS}
    criteria = {
        "rho_h1_E6_ge_0.10_in_3of4": int(sum(v >= 0.10 for v in rho_e6.values())) >= 3,
        "rho_decays_with_h_in_3of4": int(sum(decays.values())) >= 3,
        "library_gain_8000_vs_250_gt_0.02_in_3of4": int(sum(v > 0.02 for v in l_gain.values())) >= 3,
        "smap_best_theta_gain_gt_0.02_in_3of4": int(sum(v > 0.02 for v in theta_gain.values())) >= 3,
        "lzma_ratio_lt_0.90_in_3of4": int(sum(v < 0.90 for v in ratios.values())) >= 3,
    }
    return {
        "rho_h1_E6": rho_e6,
        "rho_decays_h1_vs_h60": decays,
        "library_gain_8000_minus_250": l_gain,
        "smap_best_theta_minus_theta0": theta_gain,
        "lzma_ratio": ratios,
        "criteria": criteria,
        "supported_on_dev": bool(all(criteria.values())),
        "note": "3-of-4-asset reading of the section-7 signature; every component is also reported per asset",
    }


# --------------------------------------------------------------------------
# Cell evaluation
# --------------------------------------------------------------------------
def direct_readouts(valid: pd.DataFrame, horizon: int, n_boot: int, block_len: int) -> dict:
    """No-fit sign readouts vs chance, reusing ``train_schrodinger.s0_chance_tests``.

    The readout score is presented as ``0.5 + score`` under the column name that
    helper reads; this preserves the sign and the ranks.
    """
    target = f"forward_return_{horizon}m"
    y = valid[target].to_numpy(dtype="float64")
    out: dict[str, Any] = {}
    for name, template in DIRECT_READOUTS.items():
        score = valid[template.format(h=horizon)].to_numpy(dtype="float64")
        frame = pd.DataFrame({target: y, f"schr_p_up_{horizon}m": 0.5 + score})
        out[name] = {
            "metrics": direction_metrics(y, score),
            "chance": s0_chance_tests(frame, horizon, n_boot, block_len),
        }
    return out


def _dev_cell(asset: str, horizon: int, task: str, asset_frame: pd.DataFrame, n_boot: int,
              n_folds: int, initial_train_fraction: float, embargo: int | None) -> dict:
    target_col = f"forward_return_{horizon}m" if task == "direction" else f"forward_vol_{horizon}m"
    block_len = block_len_for(ANCHOR_MINUTES_DEV)
    arms = arms_for(horizon)
    valid = complete_case_rows(asset_frame, union_columns(horizon), target_col, task)
    predictions = _predict_arms(valid, arms, target_col, horizon, task, n_folds, initial_train_fraction, embargo)
    y = predictions["M"][target_col].to_numpy(dtype="float64")
    for name in arms:
        assert np.array_equal(
            predictions[name]["target_time_s"].to_numpy(dtype="int64"),
            predictions["M"]["target_time_s"].to_numpy(dtype="int64"),
        ), "arms are not on identical rows"
    cell: dict[str, Any] = {
        "n_complete_case": int(len(valid)),
        "n_oos": int(len(y)),
        "arms": {name: {"evaluation": evaluate_predictions(pred, horizon, task)} for name, pred in predictions.items()},
        "deltas": {
            f"{a}-{b}": _pair_deltas(y, predictions[a]["score"], predictions[b]["score"], task, horizon, n_boot, block_len)
            for a, b in ALL_COMPARISONS
        },
    }
    if task == "direction":
        cell["direct"] = direct_readouts(valid, horizon, n_boot, block_len)
        if asset == "BTCUSDT":
            cell["shuffled_target_control_KS"] = shuffled_target_control(
                valid, arms["KS"], horizon, kind="direction", n_folds=3, seed=20260914 + horizon * 100
            )
    return cell


def _holdout_cell(asset: str, horizon: int, task: str, window: str, train_frame: pd.DataFrame,
                  eval_frame: pd.DataFrame, n_boot: int, holdout_start_s: int) -> dict:
    target_col = f"forward_return_{horizon}m" if task == "direction" else f"forward_vol_{horizon}m"
    block_len = 60
    arms = arms_for(horizon)
    common = union_columns(horizon)
    train_valid = complete_case_rows(train_frame, common, target_col, task)
    train_valid = train_valid.loc[
        train_valid["target_time_s"].to_numpy(dtype="int64") + horizon * 60 <= holdout_start_s
    ].reset_index(drop=True)
    assert int(train_valid["target_time_s"].max()) + horizon * 60 <= holdout_start_s
    eval_valid = complete_case_rows(eval_frame, common, target_col, task)
    scored = {name: fit_and_score(train_valid, eval_valid, cols, target_col, task)[2] for name, cols in arms.items()}
    y = eval_valid[target_col].to_numpy(dtype="float64")
    cell: dict[str, Any] = {
        "window": window,
        "n_train": int(len(train_valid)),
        "n_eval": int(len(eval_valid)),
        "eval_min_time_s": int(eval_valid["target_time_s"].min()),
        "eval_max_time_s": int(eval_valid["target_time_s"].max()),
        "arms": {
            name: {"evaluation": evaluate_predictions(
                eval_valid.assign(score=score, fold=1, prediction=np.sign(score)), horizon, task
            )}
            for name, score in scored.items()
        },
        "deltas": {
            f"{a}-{b}": _pair_deltas(y, scored[a], scored[b], task, horizon, n_boot, block_len)
            for a, b in ALL_COMPARISONS
        },
    }
    if task == "direction":
        cell["direct"] = direct_readouts(eval_valid, horizon, n_boot, block_len)
    return cell


# --------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------
def comparison_summaries(cells: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> dict:
    """``train._ablation_summary`` per comparison at the registered per-metric alpha."""
    out: dict[str, Any] = {}
    for a, b in ALL_COMPARISONS:
        name = f"{a}-{b}"
        results = {
            "ablation": {"candidate_arm": a, "reference_arm": b, "delta_key": "delta", "bonferroni_alpha": ALPHA},
            "assets": {
                asset: {
                    f"{h}m": {
                        task: {"delta": cells[asset][f"{h}m"][task]["deltas"][name]}
                        for task in ("direction", "volatility")
                    }
                    for h in HORIZONS
                }
                for asset in ASSETS
            },
        }
        summary = _ablation_summary(results)
        summary["registered"] = (a, b) in REGISTERED_COMPARISONS
        for metric in ("hit_rate", "ic_spearman"):
            stats = summary["metrics"][metric]
            stats["n_survivors_pooled_direction_alpha"] = int(
                sum(1 for c in stats["cells"] if c["improves"] and c["p_value"] < ALPHA_POOLED_DIRECTION)
            )
        summary["adds_direction_skill"] = summary.pop("physics_adds_direction_skill")
        summary["adds_volatility_skill"] = summary.pop("physics_adds_volatility_skill")
        out[name] = summary
    return out


def direct_summaries(cells: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> dict:
    out: dict[str, Any] = {}
    for name in DIRECT_READOUTS:
        rows = []
        for asset in ASSETS:
            for h in HORIZONS:
                d = cells[asset][f"{h}m"]["direction"]["direct"][name]
                rows.append(
                    {
                        "cell": f"{asset}_{h}m",
                        "hit_rate": d["metrics"]["hit_rate"],
                        "hit_minus_half": d["chance"]["hit_rate_vs_chance"]["delta"],
                        "hit_p": d["chance"]["hit_rate_vs_chance"]["p_value"],
                        "ic": d["chance"]["ic_vs_zero"]["delta"],
                        "ic_p": d["chance"]["ic_vs_zero"]["p_value"],
                        "n": d["chance"]["hit_rate_vs_chance"]["n"],
                    }
                )
        above_hit = sum(1 for r in rows if r["hit_minus_half"] > 0 and r["hit_p"] < ALPHA_DIRECT)
        below_hit = sum(1 for r in rows if r["hit_minus_half"] < 0 and r["hit_p"] < ALPHA_DIRECT)
        above_ic = sum(1 for r in rows if r["ic"] > 0 and r["ic_p"] < ALPHA_DIRECT)
        below_ic = sum(1 for r in rows if r["ic"] < 0 and r["ic_p"] < ALPHA_DIRECT)
        if above_hit >= 9 or above_ic >= 9:
            verdict = "skill (>=9/16 significantly above chance)"
        elif below_hit >= 9 or below_ic >= 9:
            verdict = "contrarian (>=9/16 significantly below chance; not skill, sign never flipped)"
        else:
            verdict = "null"
        out[name] = {
            "alpha": ALPHA_DIRECT,
            "n_hit_above_half": int(sum(1 for r in rows if r["hit_minus_half"] > 0)),
            "n_hit_sig_above": int(above_hit),
            "n_hit_sig_below": int(below_hit),
            "n_ic_positive": int(sum(1 for r in rows if r["ic"] > 0)),
            "n_ic_sig_positive": int(above_ic),
            "n_ic_sig_negative": int(below_ic),
            "hit_rate_range": [float(min(r["hit_rate"] for r in rows)), float(max(r["hit_rate"] for r in rows))],
            "median_hit_rate": float(np.median([r["hit_rate"] for r in rows])),
            "median_ic": float(np.median([r["ic"] for r in rows])),
            "verdict": verdict,
            "cells": rows,
        }
    return out


def run_dev_tables(pool, frame: pd.DataFrame, args: argparse.Namespace, n_boot: int) -> dict:
    started = time.time()
    asset_frames = {asset: frame.loc[frame["symbol"].eq(asset)].copy() for asset in ASSETS}
    tasks = []
    for task in ("direction", "volatility"):
        for asset in ASSETS:
            for h in HORIZONS:
                tasks.append(("dev_cell", asset, int(h), task, asset_frames[asset], n_boot,
                              int(args.n_folds), float(args.initial_train_fraction), args.embargo))
    cells: dict[str, dict[str, dict[str, Any]]] = {asset: {f"{h}m": {} for h in HORIZONS} for asset in ASSETS}
    results = pool.map(_dispatch, tasks, chunksize=1)
    for task_tuple, (_kind, asset, horizon, cell) in zip(tasks, results):
        cells[asset][f"{horizon}m"][task_tuple[3]] = cell
        print(f"  dev cell {asset} {horizon}m {task_tuple[3]} n_oos={cell['n_oos']}", flush=True)
    shuffled = {
        f"{h}m": cells["BTCUSDT"][f"{h}m"]["direction"]["shuffled_target_control_KS"]["metrics"] for h in HORIZONS
    }
    return {
        "n_boot": int(n_boot),
        "block_len_anchors": block_len_for(ANCHOR_MINUTES_DEV),
        "cells": cells,
        "comparisons": comparison_summaries(cells),
        "direct": direct_summaries(cells),
        "shuffled_target_control_KS_BTCUSDT": shuffled,
        "runtime_seconds": time.time() - started,
    }


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------
def protocol_block(n_boot: int, stage: str) -> dict:
    return {
        "preregistration": "seed_lab_preregistration.md",
        "families": {fam: list(family_columns(fam, 1)) for fam in ("F1", "F2", "F3", "F4", "F5")},
        "families_note": "columns shown for h=1; per-horizon columns carry the _{h}m suffix",
        "arms_h1": arms_for(1),
        "registered_comparisons": [f"{a}-{b}" for a, b in REGISTERED_COMPARISONS],
        "descriptive_comparisons": [f"{a}-{b}" for a, b in DESCRIPTIVE_COMPARISONS],
        "constants": {
            "F1": {"E": F1_E, "tau": F1_TAU, "library": F1_LIBRARY, "loo_Q": F1_LOO_Q, "exclusion_radius": "h+E-1",
                   "neighbours": F1_E + 1, "smap_theta": F1_SMAP_THETA},
            "F2": {"window": GP_WINDOW, "ell_minutes": GP_ELL, "sf2_vol_units": GP_SF2, "sn2_vol_units": GP_SN2},
            "F3": {"fib_window": FIB_WINDOW, "fib_levels": FIB_LEVELS.tolist(), "swing_window": SWING_WINDOW,
                   "benford_window": BENFORD_WINDOW, "synodic_month_days": SYNODIC_MONTH_DAYS,
                   "new_moon_reference_s": NEW_MOON_REFERENCE_S},
            "F4": {"fft_window": FFT_WINDOW, "harmonics": FFT_HARMONICS, "hurst_window": HURST_WINDOW,
                   "hurst_chunks": list(HURST_CHUNKS), "pe_order": PE_ORDER, "pe_window": PE_WINDOW},
            "F5": {"window": LPPLS_WINDOW, "tc_offsets": list(LPPLS_TC_OFFSETS), "m": list(LPPLS_M),
                   "omega": list(LPPLS_OMEGA), "damping_min": LPPLS_DAMPING_MIN},
            "calendar_shift_minutes": CAL_SHIFT_MINUTES,
            "calendar_30d_minutes": CAL_30D_MINUTES,
            "schrodinger_config": FROZEN_SCHR.as_dict(),
        },
        "cv": "expanding walk-forward, n_folds=3, initial_train_fraction=0.50, embargo=horizon (dev)",
        "models": "per-fold StandardScaler + L2 logistic C=1 (direction) / ridge alpha=1 (volatility)",
        "bootstrap": {
            "module": "backtest.paired_bootstrap_delta",
            "n_boot": int(n_boot),
            "p_value_floor": 2.0 / (int(n_boot) + 1.0),
            "seeds": "20260914+h (hit), 20260915+h (IC), 20260916+h (MSE)",
        },
        "bonferroni": {
            "per_metric_alpha": ALPHA,
            "pooled_direction_alpha_reported": ALPHA_POOLED_DIRECTION,
            "direct_readout_alpha": ALPHA_DIRECT,
        },
        "decision_rule": ">=9/16 improving AND >=9/16 improving with p < alpha (train._ablation_summary)",
        "stage": stage,
    }


def load_dev(args: argparse.Namespace, pool_factory) -> tuple[pd.DataFrame, dict, Any]:
    started = time.time()
    audit = market_db_audit(args.db)
    market = features_module._query_market(args.db, ASSETS, args.start_time_s, args.end_time_s)
    if args.start_time_s is not None or args.end_time_s is not None:
        lo = args.start_time_s if args.start_time_s is not None else -np.inf
        hi = args.end_time_s if args.end_time_s is not None else np.inf
        market = {s: f.loc[(_epoch(f) >= lo) & (_epoch(f) <= hi)] for s, f in market.items()}
    frame = anchor_base_frame(market, ANCHOR_MINUTES_DEV)
    _set_arrays("dev", market)
    del market
    gc.collect()
    print(f"dev anchor frame: {len(frame):,} rows ({time.time() - started:.1f}s)", flush=True)
    pool = pool_factory()
    t0 = time.time()
    frame = add_seed_columns(frame, "dev", pool, args.chunk)
    feature_seconds = time.time() - t0
    causality = assert_causal_features(frame, sample_rows=256, require_physics=True)
    info = {
        "db": str(Path(args.db).resolve()),
        "source_audit": audit,
        "anchor_convention": f"target_time_s % {ANCHOR_MINUTES_DEV * 60} == 0",
        "frame_rows": int(len(frame)),
        "rows_per_asset": {a: int(frame["symbol"].eq(a).sum()) for a in ASSETS},
        "complete_seed_rows_per_asset": {
            a: int(frame.loc[frame["symbol"].eq(a), union_columns(60)].notna().all(axis=1).sum()) for a in ASSETS
        },
        "causality": causality,
        "feature_build_seconds": feature_seconds,
        "load_seconds": time.time() - started,
    }
    print(f"seed features built in {feature_seconds:.1f}s", flush=True)
    return frame, info, pool


def run_explore(args: argparse.Namespace) -> dict:
    started = time.time()
    frame, info, pool = load_dev(args, lambda: _pool(args.workers))
    try:
        print("determinism diagnostics ...", flush=True)
        diagnostics = run_diagnostics(pool, frame)
        print(f"diagnostics done in {diagnostics['runtime_seconds']:.1f}s", flush=True)
        dev = run_dev_tables(pool, frame, args, int(args.bootstrap))
    finally:
        pool.close()
        pool.join()
    results = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "explore",
        "data": info,
        "protocol": protocol_block(int(args.bootstrap), "explore"),
        "diagnostics": diagnostics,
        "dev": dev,
        "runtime_seconds": time.time() - started,
    }
    _merge_write(Path(args.output), results)
    print_report(results)
    print(f"Wrote {Path(args.output).resolve()} ({results['runtime_seconds']:.1f}s)")
    return results


def run_frozen(args: argparse.Namespace) -> dict:
    started = time.time()
    frame, info, pool = load_dev(args, lambda: _pool(args.workers))
    try:
        dev = run_dev_tables(pool, frame, args, int(args.bootstrap))
    finally:
        pool.close()
        pool.join()
    results = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "frozen",
        "data": info,
        "protocol": protocol_block(int(args.bootstrap), "frozen"),
        "dev": dev,
        "runtime_seconds": time.time() - started,
    }
    _merge_write(Path(args.output), results)
    print_report(results)
    print(f"Wrote {Path(args.output).resolve()} ({results['runtime_seconds']:.1f}s)")
    return results


def run_holdout(args: argparse.Namespace) -> dict:
    started = time.time()
    if args.holdout_db is None:
        raise SystemExit("--holdout-db is required for --stage holdout")
    output_path = Path(args.output)
    if output_path.exists():
        try:
            existing = json.loads(output_path.read_text())
        except json.JSONDecodeError:
            existing = {}
        if existing.get("stage") == "holdout":
            raise SystemExit(f"{output_path} already holds a holdout run; the holdout is one-shot and is not re-run")
    dev_results_path = Path(args.dev_results)
    if not dev_results_path.exists():
        raise SystemExit(f"{dev_results_path} not found: run --stage explore first")
    dev_payload = json.loads(dev_results_path.read_text())
    dev_signature = dev_payload.get("diagnostics", {}).get("h_seed_signature")

    audit_dev = market_db_audit(args.db)
    audit_holdout = market_db_audit(args.holdout_db)
    dev_end_s = max(int(row["max_time_s"]) for row in audit_dev["ranges"].values())
    holdout_start_s = min(int(row["min_time_s"]) for row in audit_holdout["ranges"].values())
    holdout_db_end_s = min(int(row["max_time_s"]) for row in audit_holdout["ranges"].values())
    if holdout_start_s != dev_end_s + 60:
        raise SystemExit(f"holdout must directly follow the dev source (gap {holdout_start_s - dev_end_s - 60}s)")
    end_s = int(args.holdout_end_s)
    primary_start_s = int(args.primary_start_s)
    if holdout_db_end_s < end_s:
        raise SystemExit(f"holdout DB ends at {holdout_db_end_s} < registered end {end_s}")
    if not holdout_start_s < primary_start_s <= end_s:
        raise SystemExit("primary window outside the holdout range")
    print(f"dev end {dev_end_s}; holdout {holdout_start_s}..{end_s}; primary from {primary_start_s}", flush=True)

    # Training frame: dev hourly anchors (the frozen dev cadence).
    dev_market = features_module._query_market(args.db, ASSETS)
    dev_frame = anchor_base_frame(dev_market, ANCHOR_MINUTES_DEV)
    _set_arrays("dev", dev_market)
    del dev_market
    gc.collect()
    # Evaluation frame: dev tail + holdout (capped at the registered end), every minute.
    tail_seconds = int(args.dev_tail_days) * 86400
    dev_tail = features_module._query_market(args.db, ASSETS, start_time_s=dev_end_s - tail_seconds + 60, end_time_s=dev_end_s)
    dev_tail = {s: f.loc[_epoch(f) >= dev_end_s - tail_seconds + 60] for s, f in dev_tail.items()}
    holdout_market = features_module._query_market(args.holdout_db, ASSETS)
    holdout_market = {s: f.loc[_epoch(f) <= end_s] for s, f in holdout_market.items()}
    concatenated = _concat_market(dev_tail, holdout_market)
    eval_frame = anchor_base_frame(concatenated, 1, min_time_s=holdout_start_s, max_time_s=end_s)
    _set_arrays("eval", concatenated)
    pool = _pool(args.workers)
    try:
        t0 = time.time()
        dev_frame = add_seed_columns(dev_frame, "dev", pool, args.chunk)
        eval_frame = add_seed_columns(eval_frame, "eval", pool, args.chunk)
        feature_seconds = time.time() - t0
        print(f"features built in {feature_seconds:.1f}s (dev {len(dev_frame):,}, eval {len(eval_frame):,})", flush=True)
        causality = {
            "dev": assert_causal_features(dev_frame, sample_rows=256, require_physics=True),
            "eval": assert_causal_features(eval_frame, sample_rows=256, require_physics=True),
        }
        required = sorted({c for h in HORIZONS for c in union_columns(h)} | {"ewma_vol_60m"})
        for label, t_first in (("holdout_first_row", holdout_start_s), ("primary_first_row", primary_start_s)):
            first = eval_frame.loc[eval_frame["target_time_s"].eq(t_first)]
            if len(first) != len(ASSETS):
                raise AssertionError(f"{label} missing")
            bad = [c for c in required if not np.all(np.isfinite(first[c].to_numpy(dtype="float64")))]
            if bad:
                raise AssertionError(f"features not finite at {label}: {bad}")
        times = eval_frame["target_time_s"].to_numpy(dtype="int64")
        windows = {
            "primary": eval_frame.loc[(times >= primary_start_s) & (times <= end_s)].copy(),
            "secondary": eval_frame.loc[(times >= holdout_start_s) & (times <= end_s)].copy(),
        }
        boots = {"primary": int(args.bootstrap), "secondary": int(args.bootstrap_secondary)}
        dev_assets = {a: dev_frame.loc[dev_frame["symbol"].eq(a)].copy() for a in ASSETS}
        tasks = []
        for window, wframe in windows.items():
            w_assets = {a: wframe.loc[wframe["symbol"].eq(a)].copy() for a in ASSETS}
            for task in ("direction", "volatility"):
                for asset in ASSETS:
                    for h in HORIZONS:
                        tasks.append(("holdout_cell", asset, int(h), task, window, dev_assets[asset], w_assets[asset],
                                      boots[window], holdout_start_s))
        # Heaviest (secondary direction) first for load balance.
        order = sorted(range(len(tasks)), key=lambda i: (tasks[i][4] != "secondary", tasks[i][3] != "direction"))
        ordered = [tasks[i] for i in order]
        t1 = time.time()
        outputs = pool.map(_dispatch, ordered, chunksize=1)
        eval_seconds = time.time() - t1
    finally:
        pool.close()
        pool.join()
    blocks: dict[str, dict[str, dict[str, dict[str, Any]]]] = {
        w: {a: {f"{h}m": {} for h in HORIZONS} for a in ASSETS} for w in windows
    }
    for task_tuple, (_kind, asset, horizon, cell) in zip(ordered, outputs):
        blocks[task_tuple[4]][asset][f"{horizon}m"][task_tuple[3]] = cell
    results: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "holdout",
        "data": {
            "dev_db": str(Path(args.db).resolve()),
            "holdout_db": str(Path(args.holdout_db).resolve()),
            "source_audit_dev": audit_dev,
            "source_audit_holdout": audit_holdout,
            "bounds": {
                "dev_end_time_s": dev_end_s,
                "holdout_start_time_s": holdout_start_s,
                "holdout_db_end_time_s": holdout_db_end_s,
                "registered_end_time_s": end_s,
                "primary_start_time_s": primary_start_s,
                "dev_tail_days": int(args.dev_tail_days),
            },
            "rows": {w: {a: int(f["symbol"].eq(a).sum()) for a in ASSETS} for w, f in windows.items()},
            "causality": causality,
            "feature_build_seconds": feature_seconds,
            "evaluation_seconds": eval_seconds,
        },
        "protocol": {
            **protocol_block(int(args.bootstrap), "holdout"),
            "design": (
                "one-shot frozen holdout; every arm fitted once on all dev hourly complete-case rows whose targets "
                "finish before the holdout start, then scored on every minute of the primary window "
                "(decisive) and of the full holdout window (secondary, descriptive); direct readouts fit nothing"
            ),
            "block_len_rows": 60,
            "n_boot_primary": int(args.bootstrap),
            "n_boot_secondary": int(args.bootstrap_secondary),
        },
    }
    for window in windows:
        results[window] = {
            "n_boot": boots[window],
            "cells": blocks[window],
            "comparisons": comparison_summaries(blocks[window]),
            "direct": direct_summaries(blocks[window]),
        }
    results["verdicts"] = holdout_verdicts(results["primary"], dev_signature)
    results["runtime_seconds"] = time.time() - started
    _merge_write(output_path, results)
    print_report(results)
    print(f"Wrote {output_path.resolve()} ({results['runtime_seconds']:.1f}s)")
    return results


def holdout_verdicts(primary: Mapping[str, Any], dev_signature: Mapping[str, Any] | None) -> dict:
    comps = primary["comparisons"]
    direct = primary["direct"]
    families = {}
    for a, b in REGISTERED_COMPARISONS:
        s = comps[f"{a}-{b}"]
        families[f"{a}-{b}"] = {
            "adds_direction_skill": s["adds_direction_skill"],
            "adds_volatility_skill": s["adds_volatility_skill"],
            "verdict": s["verdict"],
        }
    direct_v = {name: direct[name]["verdict"] for name in DIRECT_READOUTS}
    dev_ok = bool(dev_signature and dev_signature.get("supported_on_dev"))
    holdout_ok = bool(
        comps["F1-M"]["adds_direction_skill"]
        or direct["simplex"]["verdict"].startswith("skill")
        or direct["smap"]["verdict"].startswith("skill")
    )
    if dev_ok and holdout_ok:
        h_seed = "supported"
    elif not dev_ok and not holdout_ok:
        h_seed = "not supported (refuted at the tested scale)"
    else:
        h_seed = f"mixed (dev signature {'met' if dev_ok else 'not met'}, holdout F1/direct {'met' if holdout_ok else 'not met'})"
    return {"families": families, "direct": direct_v, "h_seed": h_seed,
            "dev_signature_met": dev_ok, "holdout_f1_or_direct_met": holdout_ok}


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "NA" if not np.isfinite(v) else f"{v:.{digits}f}"


def _print_block(title: str, block: Mapping[str, Any]) -> None:
    print(f"\n== {title} (n_boot={block['n_boot']}, alpha={ALPHA:.3e}) ==")
    for name, summary in block["comparisons"].items():
        tag = "" if summary["registered"] else " [descriptive]"
        parts = []
        for metric in ("hit_rate", "ic_spearman", "mse"):
            m = summary["metrics"][metric]
            parts.append(f"{metric}: {m['n_improving']}/16 impr, {m['n_survivors']} surv, "
                         f"{m['n_significantly_worse']} worse, med {_fmt(m['median_delta'], 6)}")
        print(f"  {name}{tag}: " + " | ".join(parts))
    print(f"  direct readouts (alpha={ALPHA_DIRECT:.3e}):")
    for name, d in block["direct"].items():
        print(f"    {name}: hit {d['hit_rate_range'][0]*100:.2f}-{d['hit_rate_range'][1]*100:.2f}% "
              f"(median {d['median_hit_rate']*100:.2f}%), sig above {d['n_hit_sig_above']}, below {d['n_hit_sig_below']}; "
              f"IC>0 {d['n_ic_positive']}/16, sig+ {d['n_ic_sig_positive']}, sig- {d['n_ic_sig_negative']} -> {d['verdict']}")


def print_report(results: Mapping[str, Any]) -> None:
    diag = results.get("diagnostics")
    if diag:
        print("\n== Determinism diagnostics (dev, descriptive) ==")
        for key, label in (("simplex_rho_vs_E", "E"), ("simplex_rho_vs_horizon", "h"),
                           ("simplex_rho_vs_library", "L"), ("smap_rho_vs_theta", "theta")):
            panel = diag[key]
            print(f"  {key}: values {panel['values']}")
            for name, entries in panel["by_asset"].items():
                print(f"    {name:8s} " + " ".join(f"{e['rho']:+.4f}" for e in entries))
        for asset, c in diag["compression"].items():
            print(f"  lzma {asset}: ratio {c['ratio_real_over_shuffled_mean']:.5f} p={c['p_one_sided']:.4f}")
        for asset, b in diag["bds"].items():
            print(f"  BDS {asset}: std median {['%.1f' % v for v in b['standardized']['median']]}, "
                  f"vol-std median {['%.1f' % v for v in b['vol_standardized']['median']]}")
        print(f"  H_seed dev signature: {diag['h_seed_signature']['criteria']} -> "
              f"supported_on_dev={diag['h_seed_signature']['supported_on_dev']}")
    if results.get("dev"):
        _print_block("dev hourly walk-forward", results["dev"])
        print("  shuffled KS control BTCUSDT: " + ", ".join(
            f"{h}: hit {v['hit_rate']*100:.2f}% IC {v['ic_spearman']:+.5f}"
            for h, v in results["dev"]["shuffled_target_control_KS_BTCUSDT"].items()))
    for window in ("primary", "secondary"):
        if results.get(window):
            _print_block(f"holdout {window}", results[window])
    if results.get("verdicts"):
        print(f"\nverdicts: {json.dumps(results['verdicts'], indent=1)}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("explore", "frozen", "holdout"), default="explore")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--holdout-db", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dev-results", type=Path, default=DEFAULT_OUTPUT,
                        help="dev results file required before the holdout stage")
    parser.add_argument("--start-time-s", type=int, default=None, help="smoke runs only")
    parser.add_argument("--end-time-s", type=int, default=None, help="smoke runs only")
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--initial-train-fraction", type=float, default=0.50)
    parser.add_argument("--embargo", type=int, default=None)
    parser.add_argument("--bootstrap", type=int, default=None, help="dev 4999 / holdout primary 9999")
    parser.add_argument("--bootstrap-secondary", type=int, default=4999)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk", type=int, default=1500, help="anchors per feature task")
    parser.add_argument("--dev-tail-days", type=int, default=DEV_TAIL_DAYS)
    parser.add_argument("--primary-start-s", type=int, default=HOLDOUT_PRIMARY_START_S)
    parser.add_argument("--holdout-end-s", type=int, default=HOLDOUT_END_S)
    args = parser.parse_args(argv)
    if args.bootstrap is None:
        args.bootstrap = 9999 if args.stage == "holdout" else 4999
    if args.stage == "holdout" and args.output == DEFAULT_OUTPUT:
        args.output = DEFAULT_HOLDOUT_OUTPUT
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.stage == "explore":
        run_explore(args)
    elif args.stage == "frozen":
        run_frozen(args)
    else:
        run_holdout(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
