"""Mixed Schrodinger + GCP2 + market evaluation arm (``mix_lab``).

This driver pre-registers and evaluates three ways of combining the existing
market features, the registered GCP2 columns, and the literal Schrodinger
wave-packet arm (``schrodinger.py``):

* additive arms on identical rows: A0 ``market_only``, A1 ``+GCP``,
  A2 ``+SCHR`` and A3 ``+GCP+SCHR`` (deltas A1-A0, A2-A0, A3-A0, A3-A2);
* a genuinely coupled arm where the registered causal GCP ``netvar_z_60m``
  column drives the wave equation itself (effective ``hbar``, potential
  stiffness, initial packet width, or an imaginary potential), each candidate
  paired against its own 30-day time-shifted GCP control;
* the direct (S0-style) Born-rule readout of the coupled arm.

Everything else is imported from the repository: the frame/feature assembly
(``features.py``), the solver, KDE, potentials and registered feature builder
(``schrodinger.py``), and the folds, models, metrics, paired moving-block
bootstrap, shuffled-target control and S0 chance tests
(``backtest.py`` / ``train_schrodinger.py``).  The only new numerical
machinery is the coupled split-step propagator and the coupling maps, both
covered by ``test_mix_lab.py``.

Stages::

    .venv/bin/python mix_lab.py --stage explore    # exploration + selection + frozen dev tables
    .venv/bin/python mix_lab.py --stage frozen     # frozen dev tables (re-runs the selected label)
    .venv/bin/python mix_lab.py --stage holdout \
        --db ../crypto-data/crypto_market.db --gcp ../crypto-data/gcp2_1m.db \
        --holdout-db ../crypto-data/holdout_klines_1m.db \
        --gcp-holdout-db ../crypto-data/<gcp-holdout-cache>.db \
        --output results_mix_lab_holdout.json     # one-shot, only after freezing

The protocol, constants, selection rule and holdout decision rule are frozen
in ``mix_lab_preregistration.md``.  CPU use is capped to half the cores.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

# CPU cap to at most half the cores, applied before numpy is imported.
MAX_THREADS = max(1, (os.cpu_count() or 2) // 2)
for _thread_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    _current = os.environ.get(_thread_var)
    if _current is None:
        os.environ[_thread_var] = str(MAX_THREADS)
    else:
        try:
            if int(_current) > MAX_THREADS:
                os.environ[_thread_var] = str(MAX_THREADS)
        except ValueError:
            os.environ[_thread_var] = str(MAX_THREADS)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import features as features_module  # noqa: E402
from backtest import (  # noqa: E402
    HORIZONS,
    MARKET_FEATURES,
    _valid_frame,
    direction_metrics,
    evaluate_predictions,
    paired_bootstrap_delta,
    shuffled_target_control,
    volatility_metrics,
    walk_forward_predict,
)
from features import (  # noqa: E402
    ASSETS,
    GCP_FEATURES,
    _assemble_feature_frame,
    _finite_zscore,
    assert_causal_features,
    load_netvar_minute,
    market_db_audit,
)
from schrodinger import (  # noqa: E402
    DT_MAX,
    KDE_WINDOW,
    SIGMA_CLIP,
    VOL_SHORT_WINDOW,
    X0_CLIP,
    X_MAX,
    SchrodingerConfig,
    _rolling_std_of_trailing_returns,
    born_observables,
    build_anchor_frame_from_market,
    build_potential,
    gaussian_packet,
    grid,
    k0_for,
    kde_density,
    schr_feature_columns,
    tau_for,
)
from train import _ablation_summary, _jsonable, _same_oos_rows  # noqa: E402
from train_schrodinger import (  # noqa: E402
    block_len_for,
    complete_case_rows,
    fit_and_score,
    s0_chance_tests,
    warmup_minutes_exact,
)

DEFAULT_DB = HERE.parent / "crypto-data" / "crypto_market.db"
DEFAULT_GCP = HERE.parent / "crypto-data" / "gcp2_1m.db"
DEFAULT_OUTPUT = HERE / "results_mix_lab_dev.json"
DEFAULT_HOLDOUT_OUTPUT = HERE / "results_mix_lab_holdout.json"

# --------------------------------------------------------------------------
# Pre-declared constants (mix_lab_preregistration.md sections 3-8).
# --------------------------------------------------------------------------
FROZEN_SCHR = SchrodingerConfig(potential="quartic_kde", tau_map="linear_60", k0_map="momentum")
CONTROL_SHIFT_DAYS = 30
CONTROL_SHIFT_MINUTES = CONTROL_SHIFT_DAYS * 24 * 60  # 43200
BONFERRONI_DEV = 0.05 / float(len(ASSETS) * len(HORIZONS))  # 0.003125
ANCHOR_MINUTES_DEV = 60
COUPLED_NORM_COLUMN = "cpl_norm"

# V_eff = V - i eta W(x); the fixed shape is (x / X_MAX)^2 (0 at centre, 1 at edge).
def absorbing_shape(x: np.ndarray) -> np.ndarray:
    """Pre-declared imaginary-potential shape ``(x / X_MAX)^2``."""
    return (np.asarray(x, dtype="float64") / float(X_MAX)) ** 2


@dataclass(frozen=True)
class CouplingConfig:
    """One pre-declared GCP-driven modification of the wave equation."""

    label: str
    kind: str  # "hbar" | "stiffness" | "packet" | "absorb"
    sign: int
    strength: float

    def __post_init__(self) -> None:
        if self.kind not in ("hbar", "stiffness", "packet", "absorb"):
            raise ValueError(f"unknown coupling kind: {self.kind!r}")
        if self.sign not in (-1, 1):
            raise ValueError("coupling sign must be +1 or -1")
        if not self.strength > 0.0:
            raise ValueError("coupling strength must be positive")

    @property
    def beta(self) -> float:
        return float(self.sign) * float(self.strength)

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "kind": self.kind,
            "sign": int(self.sign),
            "strength": float(self.strength),
            "map": "u = tanh(netvar_z_60m / 2)",
        }


# The full pre-declared grid (six candidates, nothing else is run).
COUPLED_CANDIDATES: tuple[CouplingConfig, ...] = (
    CouplingConfig(label="hbar_inc", kind="hbar", sign=+1, strength=0.75),
    CouplingConfig(label="hbar_dec", kind="hbar", sign=-1, strength=0.75),
    CouplingConfig(label="stiff_inc", kind="stiffness", sign=+1, strength=1.0),
    CouplingConfig(label="packet_widen", kind="packet", sign=+1, strength=0.5),
    CouplingConfig(label="packet_narrow", kind="packet", sign=-1, strength=0.5),
    CouplingConfig(label="absorb_inc", kind="absorb", sign=+1, strength=1.0),
)
COUPLED_BY_LABEL = {config.label: config for config in COUPLED_CANDIDATES}

PACKET_SIGMA_CLIP = (0.35, 3.0)
ABSORB_ETA0 = 1.0


def coupling_u(gcp_z: np.ndarray) -> np.ndarray:
    """Bounded strictly monotone map of the registered causal GCP z-score."""
    return np.tanh(np.asarray(gcp_z, dtype="float64") / 2.0)


def coupling_parameter(gcp_z: np.ndarray, config: CouplingConfig) -> np.ndarray:
    """The single scalar the candidate feeds into the equation.

    ``hbar`` -> effective hbar; ``stiffness`` -> multiplicative V scale;
    ``packet`` -> multiplicative packet-width factor; ``absorb`` -> eta.
    """
    u = coupling_u(gcp_z)
    if config.kind == "packet":
        return np.exp(config.beta * u)
    if config.kind == "absorb":
        return float(ABSORB_ETA0) * np.exp(config.beta * u)
    return np.exp(config.beta * u)


def split_step_propagate_coupled(
    psi: np.ndarray,
    V: np.ndarray | None,
    tau: float,
    hbar: np.ndarray | float | None = None,
    dt_max: float = DT_MAX,
) -> np.ndarray:
    """Strang-split Fourier propagation with per-row effective hbar.

    ``psi`` may be ``(N,)`` or ``(B, N)``; ``V`` may be ``None``, ``(N,)`` or
    ``(B, N)`` and may be complex (an imaginary potential is a legitimate
    input, in which case the propagation is deliberately non-unitary);
    ``hbar`` may be ``None`` (registered ``hbar = 1``), a scalar, or ``(B,)``.
    With ``hbar = 1`` and real ``V`` this reproduces
    :func:`schrodinger.split_step_propagate` to floating-point noise.
    """
    x, dx, k = grid()
    del x, dx
    tau = float(tau)
    if tau <= 0.0:
        raise ValueError("tau must be positive")
    n_steps = max(1, int(math.ceil(tau / float(dt_max) - 1e-12)))
    dt = tau / n_steps
    out = np.asarray(psi, dtype=np.complex128)
    batched = out.ndim == 2
    if not batched:
        out = out[None, :]
    batch, n_grid = out.shape
    if hbar is None:
        hbar_rows = np.ones(batch, dtype="float64")
    else:
        hbar_rows = np.asarray(hbar, dtype="float64").reshape(-1)
        if hbar_rows.size == 1:
            hbar_rows = np.full(batch, float(hbar_rows[0]), dtype="float64")
        if hbar_rows.shape != (batch,):
            raise ValueError("hbar must be scalar or have one entry per row")
        if not np.all(np.isfinite(hbar_rows)) or np.any(hbar_rows <= 0.0):
            raise ValueError("hbar rows must be finite and positive")
    kinetic = np.exp(-0.5j * hbar_rows[:, None] * (k[None, :] ** 2) * dt)
    if V is None:
        potential = None
    else:
        potential_values = np.asarray(V, dtype=np.complex128)
        if potential_values.ndim == 1:
            potential_values = np.broadcast_to(potential_values, (batch, n_grid))
        potential = np.exp(-0.5j * potential_values * dt / hbar_rows[:, None])
    for _ in range(n_steps):
        if potential is not None:
            out = out * potential
        out = np.fft.ifft(np.fft.fft(out, axis=1) * kinetic, axis=1)
        if potential is not None:
            out = out * potential
    return out if batched else out[0]


# --------------------------------------------------------------------------
# Coupled feature construction
# --------------------------------------------------------------------------
def coupled_columns(
    label: str,
    horizon: int,
    shifted: bool = False,
    include_norm: bool = False,
) -> list[str]:
    """Feature column names for one candidate/horizon/variant."""
    prefix = "cpls" if shifted else "cpl"
    columns = [
        f"{prefix}_{label}_p_up_{horizon}m",
        f"{prefix}_{label}_mean_{horizon}m",
        f"{prefix}_{label}_std_{horizon}m",
        f"{prefix}_{label}_kurt_{horizon}m",
    ]
    if include_norm:
        columns.append(f"{prefix}_{label}_norm_{horizon}m")
    return columns


def coupled_model_columns(config: CouplingConfig, horizon: int, shifted: bool = False) -> list[str]:
    """Model columns of the coupled arm for one candidate/horizon/variant."""
    include_norm = config.kind == "absorb"
    return coupled_columns(config.label, horizon, shifted=shifted, include_norm=include_norm)


def build_coupled_features_for_asset_multi(
    close: pd.Series,
    anchors_time_s: Sequence[int] | np.ndarray,
    horizon: int,
    base_config: SchrodingerConfig,
    couplings: Sequence[CouplingConfig],
    gcp_z_real: Sequence[float] | np.ndarray,
    gcp_z_shift: Sequence[float] | np.ndarray,
    window: int = KDE_WINDOW,
    chunk: int = 8192,
    include_identity: bool = True,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict]:
    """Coupled GCP-driven Schrodinger features for one asset/horizon.

    Mirrors :func:`schrodinger.build_features_for_asset_multi` exactly for the
    causal inputs (same warm-up rule, same rolling std, same KDE, same
    potential, same packet construction); only the propagation is modified by
    the per-anchor coupling scalar ``u = tanh(gcp_z/2)``.  Returns
    ``(frames_by_variant, audit_frame, stats)`` where the variant keys are the
    candidate labels, ``{label}_shift30d`` for the shifted-GCP copies, and
    ``identity`` for the uncoupled consistency check.

    Anchors whose GCP z value is not finite produce NaN features (never a
    fabricated value); the GCP value must be the registered causal column, so
    no additional shift is applied here.
    """
    base_config = base_config
    couplings = list(couplings)
    gcp_real = np.asarray(gcp_z_real, dtype="float64").reshape(-1)
    gcp_shift = np.asarray(gcp_z_shift, dtype="float64").reshape(-1)
    close = pd.Series(close, dtype="float64").sort_index()
    times_s = close.index.to_numpy(dtype="datetime64[s]").astype("int64")
    L = np.log(close.to_numpy(dtype="float64"))
    n = len(L)
    h = int(horizon)
    anchors = np.asarray(anchors_time_s, dtype="int64")
    if not (len(anchors) == len(gcp_real) == len(gcp_shift)):
        raise ValueError("anchors and GCP arrays must have the same length")

    positions = np.searchsorted(times_s, anchors)
    if np.any(positions >= n) or not np.array_equal(times_s[positions], anchors):
        raise AssertionError("anchor times must be exact candle opens of this asset")
    last = positions - 1
    usable = last >= h + window - 1
    anchors, positions, last = anchors[usable], positions[usable], last[usable]
    gcp_real, gcp_shift = gcp_real[usable], gcp_shift[usable]

    if len(anchors) == 0:
        frames = {
            key: pd.DataFrame({"target_time_s": np.empty(0, dtype="float64")})
            for key in _variant_keys(couplings, include_identity)
        }
        return frames, pd.DataFrame({"target_time_s": np.empty(0, dtype="float64")}), {"rows": 0}

    R = np.full(n, np.nan, dtype="float64")
    R[h:] = L[h:] - L[:-h]
    sigma_all = _rolling_std_of_trailing_returns(R, window)
    short_all = _rolling_std_of_trailing_returns(R, VOL_SHORT_WINDOW)

    sigma_h = sigma_all[last]
    short_h = short_all[last]
    offsets = np.arange(window, dtype=np.int64)
    sample_pos = last[:, None] - offsets[None, :]
    assert np.all(sample_pos < positions[:, None]), "sample uses candle at or after t"
    assert np.all(sample_pos >= 0), "sample before the start of the series"
    samples = R[sample_pos]

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
    gcp_real = gcp_real[finite]
    gcp_shift = gcp_shift[finite]
    if len(anchors) == 0:
        frames = {
            key: pd.DataFrame({"target_time_s": np.empty(0, dtype="float64")})
            for key in _variant_keys(couplings, include_identity)
        }
        return frames, pd.DataFrame({"target_time_s": np.empty(0, dtype="float64")}), {"rows": 0}

    z = samples / sigma_h[:, None]
    x0 = np.clip(z[:, 0], -X0_CLIP, X0_CLIP)
    sigma_packet = np.clip(short_h / sigma_h, SIGMA_CLIP[0], SIGMA_CLIP[1])

    x_grid, _, _ = grid()
    tau = tau_for(h, base_config.tau_map)
    k0 = k0_for(x0, base_config.k0_map)
    shape = absorbing_shape(x_grid)
    observable_keys = {"p_up": "p_up", "mean": "mean", "std": "std", "kurt": "kurt"}

    variant_keys = _variant_keys(couplings, include_identity)
    B = len(anchors)
    data: dict[str, dict[str, np.ndarray]] = {}
    for key in variant_keys:
        data[key] = {
            name: np.empty(B, dtype="float64") for name in ("p_up", "mean", "std", "kurt", "norm")
        }
    stats: dict[str, Any] = {"rows": int(B), "kde_fallback_rows": 0}
    for start in range(0, B, int(chunk)):
        stop = min(B, start + int(chunk))
        z_chunk = z[start:stop]
        density = kde_density(z_chunk, x_grid, stats=stats)
        V = build_potential(density, x_grid, base_config.potential)
        if V is not None and V.ndim == 1:
            V = np.broadcast_to(V, (stop - start, V.shape[0]))
        for key in variant_keys:
            if key == "identity":
                psi0 = gaussian_packet(x_grid, x0[start:stop], sigma_packet[start:stop], k0[start:stop])
                psi_t = split_step_propagate_coupled(psi0, V, tau)
            else:
                shifted = key.endswith("_shift30d")
                label = key[: -len("_shift30d")] if shifted else key
                config = COUPLED_BY_LABEL[label]
                gcp_values = gcp_shift[start:stop] if shifted else gcp_real[start:stop]
                good = np.isfinite(gcp_values)
                parameter = coupling_parameter(np.where(good, gcp_values, 0.0), config)
                if config.kind == "hbar":
                    psi0 = gaussian_packet(x_grid, x0[start:stop], sigma_packet[start:stop], k0[start:stop])
                    psi_t = split_step_propagate_coupled(psi0, V, tau, hbar=parameter)
                elif config.kind == "stiffness":
                    if V is None:
                        raise ValueError("stiffness coupling requires a non-free potential")
                    psi0 = gaussian_packet(x_grid, x0[start:stop], sigma_packet[start:stop], k0[start:stop])
                    psi_t = split_step_propagate_coupled(psi0, parameter[:, None] * V, tau)
                elif config.kind == "packet":
                    sigma_eff = np.clip(
                        sigma_packet[start:stop] * parameter,
                        PACKET_SIGMA_CLIP[0],
                        PACKET_SIGMA_CLIP[1],
                    )
                    psi0 = gaussian_packet(x_grid, x0[start:stop], sigma_eff, k0[start:stop])
                    psi_t = split_step_propagate_coupled(psi0, V, tau)
                elif config.kind == "absorb":
                    potential = V - 1j * parameter[:, None] * shape[None, :]
                    psi0 = gaussian_packet(x_grid, x0[start:stop], sigma_packet[start:stop], k0[start:stop])
                    psi_t = split_step_propagate_coupled(psi0, potential, tau)
                else:  # pragma: no cover - CouplingConfig validates the kind
                    raise ValueError(f"unknown coupling kind: {config.kind!r}")
                # Rows without a finite GCP value get NaN features, never the identity.
                mask = ~good
                if np.any(mask):
                    psi_t = np.array(psi_t, dtype="complex128", copy=True)
                    psi_t[mask] = np.nan
            obs = born_observables(psi_t, x_grid)
            norm = np.trapezoid(np.abs(psi_t) ** 2, x_grid, axis=1)
            for name, key_name in observable_keys.items():
                data[key][name][start:stop] = np.asarray(obs[key_name], dtype="float64")
            data[key]["norm"][start:stop] = np.asarray(norm, dtype="float64")

    frames: dict[str, pd.DataFrame] = {}
    for key in variant_keys:
        shifted = key.endswith("_shift30d")
        label = key[: -len("_shift30d")] if shifted else key
        include_norm = label == "identity" or COUPLED_BY_LABEL[label].kind == "absorb"
        payload: dict[str, np.ndarray] = {"target_time_s": anchors.astype("float64")}
        for name in ("p_up", "mean", "std", "kurt"):
            payload[f"{'cpls' if shifted else 'cpl'}_{label}_{name}_{h}m"] = data[key][name]
        if include_norm:
            payload[f"{'cpls' if shifted else 'cpl'}_{label}_norm_{h}m"] = data[key]["norm"]
        frames[key] = pd.DataFrame(payload)
    audit = pd.DataFrame(
        {
            "target_time_s": anchors.astype("float64"),
            "cpl_trailing_vol": sigma_h,
            "cpl_x0": x0,
            "cpl_sigma": sigma_packet,
        }
    )
    return frames, audit, stats


def _variant_keys(couplings: Sequence[CouplingConfig], include_identity: bool) -> list[str]:
    keys: list[str] = []
    if include_identity:
        keys.append("identity")
    for config in couplings:
        keys.append(config.label)
        keys.append(f"{config.label}_shift30d")
    return keys


def shifted_z_column(gcp: pd.Series, days: int = CONTROL_SHIFT_DAYS) -> pd.Series:
    """The registered causal ``netvar_z_60m`` column lagged by exactly ``days``.

    ``features._assemble_feature_frame`` joins the GCP frame (indexed by the
    present minute labels, shifted one row) onto the market rows.  Here the
    same registered column is computed and its index moved forward ``days``,
    so a lookup on a market row ``t`` returns the registered column of row
    ``t - days`` (NaN when that minute label is missing).  Values are never
    filled; this is the time-shifted control, not a second feature formula.
    """
    registered = _finite_zscore(gcp).shift(1)
    shifted = registered.copy()
    shifted.index = shifted.index + pd.Timedelta(days=int(days))
    return shifted.sort_index()


def assert_shift_alignment(frame: pd.DataFrame, gcp: pd.Series, sample_rows: int = 256) -> dict:
    """Check the merged shifted column equals the registered column 30 days earlier."""
    registered = _finite_zscore(gcp).shift(1)
    left = frame.loc[
        frame["netvar_z_60m_shift30d"].notna(), ["symbol", "time", "netvar_z_60m_shift30d"]
    ]
    if left.empty:
        span = pd.DatetimeIndex(frame["time"]).max() - pd.DatetimeIndex(frame["time"]).min()
        if span <= pd.Timedelta(days=CONTROL_SHIFT_DAYS + 1):
            # A frame shorter than the shift cannot carry the control column;
            # only acceptable for tiny smoke ranges, never for the dev range.
            return {"rows_checked": 0, "insufficient_span": True, "shift_days": int(CONTROL_SHIFT_DAYS)}
        raise AssertionError("shifted GCP column is entirely NaN")
    take = np.linspace(0, len(left) - 1, min(int(sample_rows), len(left)), dtype=int)
    sample = left.iloc[np.unique(take)]
    earlier = pd.DatetimeIndex(sample["time"]) - pd.Timedelta(days=CONTROL_SHIFT_DAYS)
    expected = registered.reindex(earlier)
    observed = sample["netvar_z_60m_shift30d"].to_numpy(dtype="float64")
    expected_values = expected.to_numpy(dtype="float64")
    both_nan = np.isnan(observed) & np.isnan(expected_values)
    difference = np.where(
        both_nan, 0.0, np.abs(np.where(both_nan, 0.0, observed - expected_values))
    )
    if not np.all((difference == 0.0) | (np.isnan(difference) & np.isnan(observed))):
        raise AssertionError("shifted GCP column is not the registered column lagged 30 days")
    # Full-series check: the shifted series is the registered series moved 30 days.
    moved = registered.copy()
    moved.index = moved.index + pd.Timedelta(days=CONTROL_SHIFT_DAYS)
    overlap = registered.index.intersection(moved.index - pd.Timedelta(days=CONTROL_SHIFT_DAYS))
    del overlap
    return {
        "rows_checked": int(len(sample)),
        "max_abs_difference": float(np.nanmax(difference)),
        "shift_days": int(CONTROL_SHIFT_DAYS),
    }


@dataclass
class DevBundle:
    """Everything the dev stages need, built once per process."""

    frame: pd.DataFrame
    market: dict[str, pd.DataFrame]
    market_audit: dict
    gcp_audit: dict
    schr_audit: dict
    coupled_audit: dict
    causality: dict
    shift_check: dict
    runtime_seconds: float
    couplings: tuple[CouplingConfig, ...]


def load_dev_frames(args: argparse.Namespace) -> DevBundle:
    started = time.time()
    market_audit = market_db_audit(args.db)
    market = features_module._query_market(args.db, ASSETS, args.start_time_s, args.end_time_s)
    gcp = load_netvar_minute(args.gcp, args.start_time_s, args.end_time_s)
    print(f"GCP minutes loaded: {len(gcp):,} {gcp.index.min()} .. {gcp.index.max()}", flush=True)
    frame = _assemble_feature_frame(market, gcp=gcp)
    print(f"Base frame assembled: {len(frame):,} rows", flush=True)
    z_shift = shifted_z_column(gcp)
    lookup = z_shift.rename("netvar_z_60m_shift30d").rename_axis("time").reset_index()
    frame = frame.merge(lookup, on="time", how="left", validate="many_to_one")
    frame = frame.sort_values(["target_time_s", "symbol"], kind="mergesort").reset_index(drop=True)
    shift_check = assert_shift_alignment(frame, gcp)

    couplings = tuple(COUPLED_BY_LABEL[label] for label in args.coupling_labels) if args.coupling_labels else COUPLED_CANDIDATES
    print(f"Building registered Schrodinger features for {FROZEN_SCHR.label} ...", flush=True)
    schr_frames, schr_audit = build_anchor_frame_from_market(
        frame,
        market,
        anchor_minutes=ANCHOR_MINUTES_DEV,
        horizons=HORIZONS,
        configs=[FROZEN_SCHR],
        chunk=args.chunk,
    )
    frame = schr_frames[FROZEN_SCHR.label]
    print(f"Schrodinger rows: {schr_audit['rows']:,}", flush=True)

    coupled_audit: dict[str, Any] = {"candidates": [config.as_dict() for config in couplings], "assets": {}}
    coupled_columns_all: list[str] = []
    identity_check: dict[str, Any] = {"max_abs_difference": 0.0, "cells": 0}
    asset_frames: dict[str, pd.DataFrame] = {}
    for symbol in ASSETS:
        asset = frame.loc[frame["symbol"].eq(symbol)].copy().reset_index(drop=True)
        anchor_times = asset["target_time_s"].to_numpy(dtype="int64")
        gcp_real = asset["netvar_z_60m"].to_numpy(dtype="float64")
        gcp_shift = asset["netvar_z_60m_shift30d"].to_numpy(dtype="float64")
        for horizon in HORIZONS:
            print(f"  coupled features {symbol} {horizon}m ...", flush=True)
            frames, audit, _stats = build_coupled_features_for_asset_multi(
                market[symbol]["close"],
                anchor_times,
                horizon,
                FROZEN_SCHR,
                couplings,
                gcp_real,
                gcp_shift,
                chunk=args.chunk,
                include_identity=True,
            )
            # Consistency check: identity coupling == registered Schrodinger builder.
            identity = frames["identity"].set_index("target_time_s")
            registered_cols = schr_feature_columns(horizon)
            registered = asset.set_index("target_time_s")[list(registered_cols)]
            joined = identity.join(registered, how="inner")
            for base, column in zip(("p_up", "mean", "std", "kurt"), registered_cols):
                left = f"cpl_identity_{base}_{horizon}m"
                difference = np.max(np.abs(joined[left].to_numpy(dtype="float64") - joined[column].to_numpy(dtype="float64")))
                identity_check["max_abs_difference"] = max(identity_check["max_abs_difference"], float(difference))
                assert difference < 1e-9, (
                    f"identity coupling differs from the registered builder: {symbol} {horizon}m {left} {difference}"
                )
            identity_check["cells"] += 1
            merged = pd.DataFrame({"target_time_s": asset["target_time_s"].to_numpy(dtype="float64")})
            for key in _variant_keys(couplings, include_identity=True):
                merged = merged.merge(frames[key], on="target_time_s", how="left", validate="one_to_one")
            audit_renamed = audit.rename(
                columns={
                    "cpl_trailing_vol": f"cpl_trailing_vol_{horizon}m",
                    "cpl_x0": f"cpl_x0_{horizon}m",
                    "cpl_sigma": f"cpl_sigma_{horizon}m",
                }
            )
            merged = merged.merge(audit_renamed, on="target_time_s", how="left", validate="one_to_one")
            asset = asset.merge(merged, on="target_time_s", how="left", validate="one_to_one")
            coupled_columns_all = [name for name in merged.columns if name != "target_time_s"]
            coupled_audit["assets"].setdefault(symbol, {})[f"{horizon}m"] = int(_stats["rows"])
        asset_frames[symbol] = asset
    frame = pd.concat([agent for agent in asset_frames.values()], ignore_index=True)
    frame = frame.sort_values(["target_time_s", "symbol"], kind="mergesort").reset_index(drop=True)
    coupled_audit["identity_check"] = identity_check
    coupled_audit["n_coupled_columns"] = int(len(coupled_columns_all))

    leakage = assert_causal_features(frame, require_gcp=True, sample_rows=256)
    return DevBundle(
        frame=frame,
        market=market,
        market_audit=market_audit,
        gcp_audit={
            "rows": int(len(gcp)),
            "min_time_s": int(gcp.index[0].timestamp()),
            "max_time_s": int(gcp.index[-1].timestamp()),
        },
        schr_audit=schr_audit,
        coupled_audit=coupled_audit,
        causality=leakage,
        shift_check=shift_check,
        runtime_seconds=time.time() - started,
        couplings=couplings,
    )


# --------------------------------------------------------------------------
# Cell evaluation helpers
# --------------------------------------------------------------------------
def _pair_deltas(
    y: np.ndarray,
    candidate: pd.Series,
    reference: pd.Series,
    task: str,
    horizon: int,
    n_boot: int,
    block_len: int,
) -> dict:
    if task == "direction":
        return {
            "hit_rate": paired_bootstrap_delta(
                y, candidate, reference, metric="hit_rate", n_boot=n_boot, block_len=block_len,
                seed=20260914 + horizon,
            ),
            "ic_spearman": paired_bootstrap_delta(
                y, candidate, reference, metric="ic_spearman", n_boot=n_boot, block_len=block_len,
                seed=20260915 + horizon,
            ),
        }
    return {
        "mse": paired_bootstrap_delta(
            y, candidate, reference, metric="mse", n_boot=n_boot, block_len=block_len,
            seed=20260916 + horizon,
        )
    }


def _predict_arms(
    valid: pd.DataFrame,
    arms: Mapping[str, Sequence[str]],
    target_col: str,
    horizon: int,
    task: str,
    n_folds: int,
    initial_train_fraction: float,
    embargo: int | None,
) -> dict[str, pd.DataFrame]:
    predictions: dict[str, pd.DataFrame] = {}
    for name, columns in arms.items():
        pred, _folds = walk_forward_predict(
            valid,
            list(columns),
            target_col,
            horizon,
            task,
            n_folds=n_folds,
            initial_train_fraction=initial_train_fraction,
            embargo=embargo,
        )
        predictions[name] = pred
    names = list(predictions)
    for name in names[1:]:
        predictions[names[0]], predictions[name] = _same_oos_rows(predictions[names[0]], predictions[name])
    return predictions


def evaluate_cell_comparison(
    asset_frame: pd.DataFrame,
    horizon: int,
    task: str,
    common_columns: Sequence[str],
    arms: Mapping[str, Sequence[str]],
    comparisons: Sequence[tuple[str, str]],
    args: argparse.Namespace,
    n_boot: int,
) -> dict:
    """Walk-forward evaluation of several arms on one identical-row set."""
    target_col = f"forward_return_{horizon}m" if task == "direction" else f"forward_vol_{horizon}m"
    block_len = block_len_for(ANCHOR_MINUTES_DEV)
    valid = complete_case_rows(asset_frame, common_columns, target_col, task)
    if valid.empty:
        raise RuntimeError(f"empty row set for {horizon}m {task}")
    predictions = _predict_arms(
        valid,
        arms,
        target_col,
        horizon,
        task,
        n_folds=args.n_folds,
        initial_train_fraction=args.initial_train_fraction,
        embargo=args.embargo,
    )
    y = predictions[next(iter(arms))][target_col].to_numpy(dtype="float64")
    cell: dict[str, Any] = {
        "n_rows": int(len(y)),
        "arms": {
            name: {"evaluation": evaluate_predictions(pred, horizon, task)}
            for name, pred in predictions.items()
        },
        "deltas": {},
    }
    for candidate, reference in comparisons:
        cell["deltas"][f"{candidate}-{reference}"] = _pair_deltas(
            y,
            predictions[candidate]["score"],
            predictions[reference]["score"],
            task,
            horizon,
            n_boot,
            block_len,
        )
    return cell


def additive_arms(horizon: int) -> dict[str, list[str]]:
    schr = list(schr_feature_columns(horizon))
    return {
        "A0": list(MARKET_FEATURES),
        "A1": list(MARKET_FEATURES) + list(GCP_FEATURES),
        "A2": list(MARKET_FEATURES) + schr,
        "A3": list(MARKET_FEATURES) + list(GCP_FEATURES) + schr,
    }


def additive_common_columns(horizon: int) -> list[str]:
    """Complete-case column set shared by A0..A3 for one horizon."""
    return list(MARKET_FEATURES) + list(GCP_FEATURES) + list(schr_feature_columns(horizon))


ADDITIVE_COMPARISONS: tuple[tuple[str, str], ...] = (
    ("A1", "A0"),
    ("A2", "A0"),
    ("A3", "A0"),
    ("A3", "A2"),
)


def coupled_arms(config: CouplingConfig, horizon: int) -> dict[str, list[str]]:
    return {
        "C0": list(MARKET_FEATURES),
        "C_real": list(MARKET_FEATURES) + coupled_model_columns(config, horizon, shifted=False),
        "C_shift": list(MARKET_FEATURES) + coupled_model_columns(config, horizon, shifted=True),
    }


def coupled_common_columns(config: CouplingConfig, horizon: int) -> list[str]:
    return list(MARKET_FEATURES) + coupled_model_columns(
        config, horizon, shifted=False
    ) + coupled_model_columns(config, horizon, shifted=True)


COUPLED_COMPARISONS: tuple[tuple[str, str], ...] = (("C_real", "C_shift"), ("C_real", "C0"))


def direct_readout(
    valid: pd.DataFrame,
    config: CouplingConfig,
    horizon: int,
    n_boot: int,
) -> dict:
    """S0-style direct Born-rule readout of the coupled arm (no fitted model)."""
    block_len = block_len_for(ANCHOR_MINUTES_DEV)
    target_dir = f"forward_return_{horizon}m"
    target_vol = f"forward_vol_{horizon}m"
    direction_rows = complete_case_rows(valid, coupled_common_columns(config, horizon), target_dir, "direction")
    p_up = direction_rows[f"cpl_{config.label}_p_up_{horizon}m"].to_numpy(dtype="float64")
    out: dict[str, Any] = {
        "direction": direction_metrics(
            direction_rows[target_dir].to_numpy(dtype="float64"), p_up - 0.5
        )
    }
    # Reuse the registered one-sample chance tests by presenting the coupled
    # probability under the column name that helper reads.
    renamed = direction_rows.assign(**{f"schr_p_up_{horizon}m": p_up})
    out["direction_chance"] = s0_chance_tests(renamed, horizon, n_boot, block_len)
    vol_rows = complete_case_rows(valid, coupled_common_columns(config, horizon), target_vol, "volatility")
    extra = coupled_model_columns(config, horizon, shifted=False)
    vol_rows = vol_rows.dropna(subset=[target_vol, *extra, "ewma_vol_60m"])
    y_vol = vol_rows[target_vol].to_numpy(dtype="float64")
    pred_vol = (
        vol_rows[f"cpl_{config.label}_std_{horizon}m"].to_numpy(dtype="float64")
        * vol_rows[f"cpl_trailing_vol_{horizon}m"].to_numpy(dtype="float64")
    )
    ewma_vol = (vol_rows["ewma_vol_60m"] * np.sqrt(horizon)).to_numpy(dtype="float64")
    metrics = volatility_metrics(y_vol, pred_vol, ewma=ewma_vol)
    metrics["vs_ewma_paired"] = paired_bootstrap_delta(
        y_vol, pred_vol, ewma_vol, metric="mse", n_boot=n_boot, block_len=block_len,
        seed=20260916 + horizon,
    )
    out["volatility"] = metrics
    out["n_direction"] = int(len(direction_rows))
    out["n_volatility"] = int(len(y_vol))
    return out


def evaluate_coupled_cell(
    asset_frame: pd.DataFrame,
    config: CouplingConfig,
    horizon: int,
    args: argparse.Namespace,
    n_boot: int,
) -> dict:
    """C_real vs C_shift vs C0 plus the direct readout for one cell."""
    cell: dict[str, Any] = {}
    common = coupled_common_columns(config, horizon)
    arms = coupled_arms(config, horizon)
    for task in ("direction", "volatility"):
        cell[task] = evaluate_cell_comparison(
            asset_frame, horizon, task, common, arms, COUPLED_COMPARISONS, args, n_boot
        )
    cell["direct"] = direct_readout(asset_frame, config, horizon, n_boot)
    return cell


def exploration_entry(cell: dict, horizon: int) -> dict:
    direction = cell["direction"]
    volatility = cell["volatility"]
    vs_shift = direction["deltas"]["C_real-C_shift"]
    vs_market = direction["deltas"]["C_real-C0"]
    return {
        "delta_hit_pp": float(vs_shift["hit_rate"]["delta"]) * 100.0,
        "delta_hit_p": float(vs_shift["hit_rate"]["p_value"]),
        "delta_ic": float(vs_shift["ic_spearman"]["delta"]),
        "delta_ic_p": float(vs_shift["ic_spearman"]["p_value"]),
        "delta_mse": float(volatility["deltas"]["C_real-C_shift"]["mse"]["delta"]),
        "delta_mse_p": float(volatility["deltas"]["C_real-C_shift"]["mse"]["p_value"]),
        "candidate_hit_rate": float(direction["arms"]["C_real"]["evaluation"]["model"]["hit_rate"]),
        "control_hit_rate": float(direction["arms"]["C_shift"]["evaluation"]["model"]["hit_rate"]),
        "candidate_ic": float(direction["arms"]["C_real"]["evaluation"]["model"]["ic_spearman"]),
        "control_ic": float(direction["arms"]["C_shift"]["evaluation"]["model"]["ic_spearman"]),
        "delta_hit_vs_market_pp": float(vs_market["hit_rate"]["delta"]) * 100.0,
        "delta_ic_vs_market": float(vs_market["ic_spearman"]["delta"]),
        "s0_hit_rate": float(cell["direct"]["direction"]["hit_rate"]),
        "s0_ic": float(cell["direct"]["direction"]["ic_spearman"]),
        "s0_mse": float(cell["direct"]["volatility"]["mse"]),
        "s0_ewma_mse": float(cell["direct"]["volatility"].get("ewma_mse", float("nan"))),
        "n_direction": int(direction["n_rows"]),
    }


def summarise_coupled(summaries: Mapping[str, dict]) -> dict:
    """Selection statistics (same aggregation as the registered Schr rule)."""
    out: dict[str, dict] = {}
    for label, cells in summaries.items():
        deltas_ic, deltas_hit, deltas_mse, weights = [], [], [], []
        for asset in ASSETS:
            for horizon in HORIZONS:
                entry = cells[f"{asset}_{horizon}m"]
                deltas_ic.append(entry["delta_ic"])
                deltas_hit.append(entry["delta_hit_pp"])
                deltas_mse.append(entry["delta_mse"])
                weights.append(entry["n_direction"])
        weights = np.asarray(weights, dtype="float64")
        deltas_ic = np.asarray(deltas_ic, dtype="float64")
        deltas_hit = np.asarray(deltas_hit, dtype="float64")
        deltas_mse = np.asarray(deltas_mse, dtype="float64")
        out[label] = {
            "n_positive_delta_ic": int(np.sum(deltas_ic > 0.0)),
            "n_positive_delta_hit": int(np.sum(deltas_hit > 0.0)),
            "n_negative_delta_mse": int(np.sum(deltas_mse < 0.0)),
            "pooled_delta_ic": float(np.sum(weights * deltas_ic) / np.sum(weights)),
            "pooled_delta_hit_pp": float(np.sum(weights * deltas_hit) / np.sum(weights)),
            "median_delta_ic": float(np.median(deltas_ic)),
            "median_delta_hit_pp": float(np.median(deltas_hit)),
            "median_delta_mse": float(np.median(deltas_mse)),
            "n_cells": int(len(deltas_ic)),
        }
    return out


def select_coupled(summaries: Mapping[str, dict]) -> str:
    """Pre-declared ranking rule (mix_lab_preregistration.md section 7)."""
    labels = sorted(summaries)
    labels.sort(
        key=lambda label: (
            summaries[label]["n_positive_delta_ic"],
            summaries[label]["pooled_delta_ic"],
            summaries[label]["median_delta_ic"],
            summaries[label]["n_positive_delta_hit"],
            summaries[label]["pooled_delta_hit_pp"],
        ),
        reverse=True,
    )
    return labels[0]


# --------------------------------------------------------------------------
# Dev stages
# --------------------------------------------------------------------------
def _comparison_summary(
    asset_blocks: Mapping[str, Mapping[str, Mapping[str, Any]]],
    candidate_arm: str,
    delta_key: str,
    comparisons: Mapping[str, str],
) -> dict:
    """Bonferroni/majority summary for one comparison via train._ablation_summary."""
    results = {
        "ablation": {
            "candidate_arm": candidate_arm,
            "reference_arm": delta_key,
            "delta_key": "delta",
            "bonferroni_alpha": BONFERRONI_DEV,
        },
        "assets": {},
    }
    for asset in ASSETS:
        results["assets"][asset] = {}
        for horizon in HORIZONS:
            cell = asset_blocks[asset][f"{horizon}m"]
            entry: dict[str, Any] = {}
            for task in ("direction", "volatility"):
                if task in comparisons:
                    source = cell[task]["deltas"][comparisons[task]]
                entry[task] = {"delta": {name: source[name] for name in source}}
            results["assets"][asset][f"{horizon}m"] = entry
    summary = _ablation_summary(results)
    return summary


def run_dev_evaluation(
    bundle: DevBundle,
    args: argparse.Namespace,
    selected_label: str | None,
    n_boot: int,
    stage_name: str,
) -> dict:
    """Frozen dev tables: additive A0-A3, frozen coupled C_real/C_shift/C0, controls."""
    started = time.time()
    frame = bundle.frame
    config = COUPLED_BY_LABEL[selected_label] if selected_label else None
    output: dict[str, Any] = {
        "n_boot": int(n_boot),
        "block_len_anchors": block_len_for(ANCHOR_MINUTES_DEV),
        "stage": stage_name,
        "additive": {"deltas": {}},
        "coupled": {"deltas": {}},
        "direct": {},
        "leakage_checks": {},
        "runtime_seconds": None,
    }
    asset_frames = {asset: frame.loc[frame["symbol"].eq(asset)].copy() for asset in ASSETS}
    additive_blocks: dict[str, dict[str, Any]] = {asset: {} for asset in ASSETS}
    for asset in ASSETS:
        for horizon in HORIZONS:
            print(f"[{stage_name}] additive {asset} {horizon}m ...", flush=True)
            additive_blocks[asset][f"{horizon}m"] = {
                "direction": evaluate_cell_comparison(
                    asset_frames[asset],
                    horizon,
                    "direction",
                    additive_common_columns(horizon),
                    {name: additive_arms(horizon)[name] for name in ("A0", "A1", "A2", "A3")},
                    ADDITIVE_COMPARISONS,
                    args,
                    n_boot,
                ),
                "volatility": evaluate_cell_comparison(
                    asset_frames[asset],
                    horizon,
                    "volatility",
                    additive_common_columns(horizon),
                    {name: additive_arms(horizon)[name] for name in ("A0", "A1", "A2", "A3")},
                    ADDITIVE_COMPARISONS,
                    args,
                    n_boot,
                ),
            }
    output["additive"]["cells"] = additive_blocks
    for name, (candidate, reference) in zip(
        ("A1-A0", "A2-A0", "A3-A0", "A3-A2"), ADDITIVE_COMPARISONS
    ):
        output["additive"]["deltas"][name] = _comparison_summary(
            additive_blocks,
            candidate_arm=candidate,
            delta_key=reference,
            comparisons={
                "direction": f"{candidate}-{reference}",
                "volatility": f"{candidate}-{reference}",
            },
        )
    if config is not None:
        coupled_blocks: dict[str, dict[str, Any]] = {asset: {} for asset in ASSETS}
        for asset in ASSETS:
            for horizon in HORIZONS:
                print(f"[{stage_name}] coupled {config.label} {asset} {horizon}m ...", flush=True)
                coupled_blocks[asset][f"{horizon}m"] = evaluate_coupled_cell(
                    asset_frames[asset], config, horizon, args, n_boot
                )
        output["coupled"]["config"] = config.as_dict()
        output["coupled"]["cells"] = coupled_blocks
        for name, (candidate, reference) in zip(("C_real-C_shift", "C_real-C0"), COUPLED_COMPARISONS):
            output["coupled"]["deltas"][name] = _comparison_summary(
                coupled_blocks,
                candidate_arm=candidate,
                delta_key=reference,
                comparisons={
                    "direction": f"{candidate}-{reference}",
                    "volatility": f"{candidate}-{reference}",
                },
            )
        output["direct"] = {
            asset: {f"{h}m": coupled_blocks[asset][f"{h}m"]["direct"] for h in HORIZONS}
            for asset in ASSETS
        }
        # Shuffled-target controls: coupled features and the A3 features, BTCUSDT.
        controls: dict[str, Any] = {}
        for horizon in HORIZONS:
            target_col = f"forward_return_{horizon}m"
            coupled_common = coupled_common_columns(config, horizon)
            coupled_valid = complete_case_rows(
                asset_frames["BTCUSDT"], coupled_common, target_col, "direction"
            )
            controls[f"coupled_{horizon}m"] = shuffled_target_control(
                coupled_valid,
                list(MARKET_FEATURES) + coupled_model_columns(config, horizon, shifted=False),
                horizon,
                kind="direction",
                n_folds=3,
                seed=20260914 + horizon * 100,
            )
            a3_valid = complete_case_rows(
                asset_frames["BTCUSDT"], additive_common_columns(horizon), target_col, "direction"
            )
            controls[f"A3_{horizon}m"] = shuffled_target_control(
                a3_valid,
                list(MARKET_FEATURES) + list(GCP_FEATURES) + list(schr_feature_columns(horizon)),
                horizon,
                kind="direction",
                n_folds=3,
                seed=20260914 + horizon * 100,
            )
        output["leakage_checks"]["shuffled_target_controls"] = controls
    output["runtime_seconds"] = time.time() - started
    return output


def run_explore(args: argparse.Namespace) -> dict:
    started = time.time()
    bundle = load_dev_frames(args)
    frame = bundle.frame
    candidates = bundle.couplings
    ranking_boot = int(args.bootstrap_explore)
    exploration: dict[str, Any] = {
        "selection_rule": (
            "rank candidates by paired dev C_real - C_shift: (1) cells with positive delta IC, "
            "(2) pooled delta IC, (3) median delta IC, (4) cells with positive delta hit, (5) pooled delta hit"
        ),
        "ranking_bootstrap_n": ranking_boot,
        "anchor_convention": f"target_time_s % {ANCHOR_MINUTES_DEV * 60} == 0",
        "candidates": {},
    }
    asset_frames = {asset: frame.loc[frame["symbol"].eq(asset)].copy() for asset in ASSETS}
    summaries: dict[str, dict[str, Any]] = {}
    for config in candidates:
        print(f"exploring {config.label} ...", flush=True)
        t0 = time.time()
        cells: dict[str, Any] = {}
        for asset in ASSETS:
            for horizon in HORIZONS:
                cell = evaluate_coupled_cell(asset_frames[asset], config, horizon, args, ranking_boot)
                cells[f"{asset}_{horizon}m"] = exploration_entry(cell, horizon)
        summaries[config.label] = summarise_coupled({config.label: cells})[config.label]
        exploration["candidates"][config.label] = {
            "config": config.as_dict(),
            "cells": cells,
            "summary": summaries[config.label],
            "runtime_seconds": time.time() - t0,
        }
        summary = summaries[config.label]
        print(
            f"  {config.label}: positive delta IC {summary['n_positive_delta_ic']}/16, "
            f"pooled delta IC {summary['pooled_delta_ic']:+.5f}, "
            f"positive delta hit {summary['n_positive_delta_hit']}/16 ({time.time() - t0:.1f}s)",
            flush=True,
        )
    selected = select_coupled(summaries)
    exploration["selected_label"] = selected
    exploration["ranking"] = [
        {"label": label, **summaries[label]}
        for label in sorted(summaries, key=lambda label: -summaries[label]["n_positive_delta_ic"])
    ]
    exploration["runtime_seconds"] = time.time() - started
    print(f"Selected coupled config by pre-declared rule: {selected}", flush=True)

    frozen = run_dev_evaluation(bundle, args, selected, int(args.bootstrap), "explore+frozen")
    results: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "explore+frozen",
        "data": data_block(bundle, args),
        "protocol": protocol_block(args, int(args.bootstrap), ranking_boot),
        "exploration": exploration,
        "frozen": frozen,
        "runtime_seconds": time.time() - started,
    }
    _merge_write(Path(args.output), results)
    _print_tables(results)
    print(f"Wrote {Path(args.output).resolve()}")
    return results


def run_frozen(args: argparse.Namespace) -> dict:
    started = time.time()
    output_path = Path(args.output)
    dev_results_path = Path(args.dev_results)
    if not dev_results_path.exists():
        raise SystemExit("run --stage explore first; the selected coupled label is read from the dev results file")
    payload = json.loads(dev_results_path.read_text())
    selected = payload.get("exploration", {}).get("selected_label") or payload.get("selected_label")
    if not selected:
        raise SystemExit(f"{dev_results_path} has no exploration.selected_label; run --stage explore first")
    if selected not in COUPLED_BY_LABEL:
        raise SystemExit(f"unknown selected label {selected!r}")
    bundle = load_dev_frames(args)
    frozen = run_dev_evaluation(bundle, args, selected, int(args.bootstrap), "frozen")
    results: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "frozen",
        "selected_label": selected,
        "data": data_block(bundle, args),
        "protocol": protocol_block(args, int(args.bootstrap), None),
        "frozen": frozen,
        "runtime_seconds": time.time() - started,
    }
    _merge_write(output_path, results)
    _print_frozen_tables(results)
    print(f"Wrote {output_path.resolve()}")
    return results


# --------------------------------------------------------------------------
# Holdout stage (one-shot; requires --holdout-db and --gcp-holdout-db)
# --------------------------------------------------------------------------
def _concat_market(dev_market: Mapping[str, pd.DataFrame], holdout_market: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    concatenated: dict[str, pd.DataFrame] = {}
    for symbol in ASSETS:
        joint = pd.concat([dev_market[symbol], holdout_market[symbol]], axis=0)
        joint = joint[~joint.index.duplicated(keep="first")].sort_index()
        step = np.diff(joint.index.to_numpy(dtype="datetime64[s]").astype("int64"))
        if not np.all(step == 60):
            raise AssertionError(f"{symbol}: concatenated candles are not contiguous")
        concatenated[symbol] = joint
    return concatenated


def run_holdout(args: argparse.Namespace) -> dict:
    started = time.time()
    if args.holdout_db is None:
        raise SystemExit("--holdout-db is required for --stage holdout")
    if args.gcp_holdout_db is None:
        raise SystemExit("--gcp-holdout-db is required for --stage holdout")
    output_path = Path(args.output)
    dev_results_path = Path(args.dev_results)
    if output_path.exists():
        try:
            existing = json.loads(output_path.read_text())
        except json.JSONDecodeError:
            existing = {}
        if existing.get("stage") == "holdout":
            raise SystemExit(
                f"{output_path} already holds a holdout run; the frozen holdout is one-shot and is not re-run"
            )
    if not dev_results_path.exists():
        raise SystemExit(
            f"{dev_results_path} not found: run --stage explore (or --stage frozen) first; "
            "the frozen label is read from there"
        )
    payload = json.loads(dev_results_path.read_text())
    selected = payload.get("selected_label") or payload.get("exploration", {}).get("selected_label")
    if not selected:
        raise SystemExit(f"{dev_results_path} has no frozen label; run --stage explore first")
    if selected not in COUPLED_BY_LABEL:
        raise SystemExit(f"unknown selected label {selected!r}")
    config = COUPLED_BY_LABEL[selected]

    audit_dev = market_db_audit(args.db)
    audit_holdout = market_db_audit(args.holdout_db)
    dev_end_s = max(int(row["max_time_s"]) for row in audit_dev["ranges"].values())
    holdout_start_s = min(int(row["min_time_s"]) for row in audit_holdout["ranges"].values())
    holdout_end_s = max(int(row["max_time_s"]) for row in audit_holdout["ranges"].values())
    if holdout_start_s != dev_end_s + 60:
        raise SystemExit(
            f"holdout must directly follow the dev source (gap {holdout_start_s - dev_end_s - 60}s)"
        )
    warmup = warmup_minutes_exact(HORIZONS)
    eval_start_s = max(holdout_start_s, dev_end_s) + warmup * 60
    print(f"Frozen coupled config: {config.label}; train dev hourly rows, score holdout minutes from {eval_start_s}")

    # Dev training frame at the frozen dev cadence (hourly anchors).
    dev_market = features_module._query_market(args.db, ASSETS)
    gcp_dev = load_netvar_minute(args.gcp)
    dev_frame = _assemble_feature_frame(dev_market, gcp=gcp_dev)
    dev_z_shift = shifted_z_column(gcp_dev)
    dev_frame = dev_frame.merge(
        dev_z_shift.rename("netvar_z_60m_shift30d").rename_axis("time").reset_index(),
        on="time", how="left", validate="many_to_one",
    )
    dev_schr_frames, _dev_schr_audit = build_anchor_frame_from_market(
        dev_frame, dev_market, anchor_minutes=ANCHOR_MINUTES_DEV, horizons=HORIZONS,
        configs=[FROZEN_SCHR], chunk=args.chunk,
    )
    dev_frame = dev_schr_frames[FROZEN_SCHR.label]
    dev_frame = _add_coupled_columns(dev_frame, dev_market, [config], args)

    # Evaluation frame: dev tail + holdout market concatenated.
    tail_seconds = int(args.dev_tail_days) * 86400
    dev_tail = features_module._query_market(
        args.db, ASSETS, start_time_s=dev_end_s - tail_seconds + 60, end_time_s=dev_end_s
    )
    holdout_market = features_module._query_market(args.holdout_db, ASSETS)
    concatenated = _concat_market(dev_tail, holdout_market)
    gcp_tail = load_netvar_minute(
        args.gcp, start_time_s=dev_end_s - (CONTROL_SHIFT_DAYS + 45) * 86400, end_time_s=dev_end_s
    )
    gcp_holdout = load_netvar_minute(args.gcp_holdout_db)
    gcp_holdout_start_s = int(gcp_holdout.index[0].timestamp())
    gcp_holdout_end_s = int(gcp_holdout.index[-1].timestamp())
    if gcp_holdout_start_s > holdout_start_s:
        raise SystemExit(
            "GCP holdout series starts after the market holdout; refusing to substitute dev data "
            f"({gcp_holdout_start_s} > {holdout_start_s})"
        )
    # Evaluation must stop where the GCP series ends: the GCP column at row t
    # is the minute labelled t - 60s, so the last scorable row is
    # last_gcp_minute + 60s (and never beyond the market holdout).
    eval_end_s = min(holdout_end_s, gcp_holdout_end_s + 60)
    print(
        f"GCP holdout {gcp_holdout_start_s}..{gcp_holdout_end_s}; "
        f"evaluation window {eval_start_s}..{eval_end_s}"
    )
    gcp_combined = pd.concat([gcp_tail, gcp_holdout])
    gcp_combined = gcp_combined[~gcp_combined.index.duplicated(keep="first")].sort_index()
    concat_frame = _assemble_feature_frame(concatenated, gcp=gcp_combined)
    z_shift_eval = shifted_z_column(gcp_combined)
    concat_frame = concat_frame.merge(
        z_shift_eval.rename("netvar_z_60m_shift30d").rename_axis("time").reset_index(),
        on="time", how="left", validate="many_to_one",
    )
    eval_schr_frames, eval_audit = build_anchor_frame_from_market(
        concat_frame, concatenated, anchor_minutes=int(args.holdout_anchor_minutes),
        horizons=HORIZONS, configs=[FROZEN_SCHR], chunk=args.chunk,
    )
    eval_frame = eval_schr_frames[FROZEN_SCHR.label]
    eval_frame = _add_coupled_columns(eval_frame, concatenated, [config], args)
    eval_frame = eval_frame.loc[eval_frame["target_time_s"].to_numpy(dtype="int64") >= holdout_start_s].copy()
    warm_row = eval_frame.loc[eval_frame["target_time_s"].to_numpy(dtype="int64") == eval_start_s]
    if len(warm_row) != len(ASSETS):
        raise AssertionError("eval_start row missing from the holdout frame")
    required = [*MARKET_FEATURES, *GCP_FEATURES, "ewma_vol_60m"]
    for h in HORIZONS:
        required.extend(schr_feature_columns(h))
        required.extend(coupled_common_columns(config, h))
    missing = [name for name in required if not np.isfinite(float(warm_row.iloc[0][name]))]
    if missing:
        raise AssertionError(f"features not finite at the first evaluated row: {missing}")
    eval_times = eval_frame["target_time_s"].to_numpy(dtype="int64")
    eval_frame = eval_frame.loc[(eval_times >= eval_start_s) & (eval_times <= eval_end_s)].copy()

    n_boot = int(args.bootstrap)
    block_len = 60  # minute anchors: the repository convention
    results: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "holdout",
        "selected_label": selected,
        "data": {
            "dev_db": str(Path(args.db).resolve()),
            "holdout_db": str(Path(args.holdout_db).resolve()),
            "gcp_dev": str(Path(args.gcp).resolve()),
            "gcp_holdout_db": str(Path(args.gcp_holdout_db).resolve()),
            "source_audit_dev": audit_dev,
            "source_audit_holdout": audit_holdout,
            "bounds": {
                "dev_end_time_s": dev_end_s,
                "holdout_start_time_s": holdout_start_s,
                "holdout_end_time_s": holdout_end_s,
                "gcp_holdout_start_time_s": gcp_holdout_start_s,
                "gcp_holdout_end_time_s": gcp_holdout_end_s,
                "warmup_minutes": warmup,
                "eval_start_time_s": eval_start_s,
                "eval_end_time_s": eval_end_s,
                "eval_end_rule": "min(market holdout end, last GCP holdout minute + 60s)",
            },
            "eval_frame_audit": eval_audit,
        },
        "protocol": {
            "design": (
                "one-shot frozen holdout; every arm fitted once on all dev hourly rows whose targets "
                "finish before the holdout start, then scored on every holdout minute from eval_start; "
                "the direct Born-rule readout fits nothing"
            ),
            "anchors": f"holdout target_time_s % {int(args.holdout_anchor_minutes) * 60} == 0",
            "models": "per-arm StandardScaler + L2 logistic C=1 (direction) / ridge alpha=1 (volatility), fitted once",
            "bootstrap": {
                "module": "backtest.paired_bootstrap_delta",
                "n_boot": n_boot,
                "block_len_rows": block_len,
                "p_value_floor": 2.0 / (n_boot + 1.0),
            },
            "bonferroni": {"direction_alpha": 0.05 / 160.0, "volatility_alpha": 0.05 / 80.0},
            "preregistration": "mix_lab_preregistration.md",
        },
        "cells": {},
    }
    dev_assets = {asset: dev_frame.loc[dev_frame["symbol"].eq(asset)].copy() for asset in ASSETS}
    eval_assets = {asset: eval_frame.loc[eval_frame["symbol"].eq(asset)].copy() for asset in ASSETS}
    for asset in ASSETS:
        results["cells"][asset] = {}
        for horizon in HORIZONS:
            print(f"holdout {config.label} {asset} {horizon}m ...", flush=True)
            cell: dict[str, Any] = {}
            for task in ("direction", "volatility"):
                target_col = f"forward_return_{horizon}m" if task == "direction" else f"forward_vol_{horizon}m"
                arms = {**additive_arms(horizon), **coupled_arms(config, horizon)}
                common = sorted(
                    set(additive_common_columns(horizon))
                    | set(coupled_common_columns(config, horizon))
                )
                train_valid = complete_case_rows(dev_assets[asset], common, target_col, task)
                train_valid = train_valid.loc[
                    train_valid["target_time_s"].to_numpy(dtype="int64") + horizon * 60 <= holdout_start_s
                ]
                if len(train_valid):
                    max_target_end = int(train_valid["target_time_s"].max()) + horizon * 60
                    assert max_target_end <= holdout_start_s, "a training target ends at/after the holdout start"
                eval_valid = complete_case_rows(eval_assets[asset], common, target_col, task)
                scored = {
                    name: fit_and_score(train_valid, eval_valid, cols, target_col, task)[2]
                    for name, cols in arms.items()
                }
                y = eval_valid[target_col].to_numpy(dtype="float64")
                comparisons = {**{f"{a}-{b}": (a, b) for a, b in ADDITIVE_COMPARISONS},
                               "C_real-C_shift": ("C_real", "C_shift")}
                cell[task] = {
                    "n_train": int(len(train_valid)),
                    "n_eval": int(len(eval_valid)),
                    "train_max_target_time_s": int(train_valid["target_time_s"].max()) if len(train_valid) else None,
                    "arms": {
                        name: {"evaluation": evaluate_predictions(
                            eval_valid.assign(score=score, fold=1, prediction=np.sign(score)), horizon, task
                        )} for name, score in scored.items()
                    },
                    "deltas": {
                        name: _pair_deltas(y, scored[a], scored[b], task, horizon, n_boot, block_len)
                        for name, (a, b) in comparisons.items()
                    },
                }
            cell["direct"] = direct_readout(eval_assets[asset], config, horizon, n_boot)
            results["cells"][asset][f"{horizon}m"] = cell
    results["runtime_seconds"] = time.time() - started
    _merge_write(output_path, results)
    print(f"Wrote {output_path.resolve()}")
    return results


def _add_coupled_columns(
    anchor_frame: pd.DataFrame,
    market: Mapping[str, pd.DataFrame],
    couplings: Sequence[CouplingConfig],
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Merge coupled feature columns onto an anchor frame, per asset/horizon."""
    asset_frames: dict[str, pd.DataFrame] = {}
    for symbol in ASSETS:
        asset = anchor_frame.loc[anchor_frame["symbol"].eq(symbol)].copy().reset_index(drop=True)
        if asset.empty:
            continue
        anchor_times = asset["target_time_s"].to_numpy(dtype="int64")
        gcp_real = asset["netvar_z_60m"].to_numpy(dtype="float64")
        gcp_shift_col = asset.get("netvar_z_60m_shift30d")
        gcp_shift = (
            gcp_shift_col.to_numpy(dtype="float64")
            if gcp_shift_col is not None
            else np.full(len(asset), np.nan)
        )
        for horizon in HORIZONS:
            frames, audit, _stats = build_coupled_features_for_asset_multi(
                market[symbol]["close"], anchor_times, horizon, FROZEN_SCHR, couplings,
                gcp_real, gcp_shift, chunk=args.chunk, include_identity=False,
            )
            merged = pd.DataFrame({"target_time_s": asset["target_time_s"].to_numpy(dtype="float64")})
            for key in _variant_keys(couplings, False):
                merged = merged.merge(frames[key], on="target_time_s", how="left", validate="one_to_one")
            audit_renamed = audit.rename(
                columns={
                    "cpl_trailing_vol": f"cpl_trailing_vol_{horizon}m",
                    "cpl_x0": f"cpl_x0_{horizon}m",
                    "cpl_sigma": f"cpl_sigma_{horizon}m",
                }
            )
            merged = merged.merge(audit_renamed, on="target_time_s", how="left", validate="one_to_one")
            asset = asset.merge(merged, on="target_time_s", how="left", validate="one_to_one")
        asset_frames[symbol] = asset
    out = pd.concat(asset_frames.values(), ignore_index=True)
    return out.sort_values(["target_time_s", "symbol"], kind="mergesort").reset_index(drop=True)


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def data_block(bundle: DevBundle, args: argparse.Namespace) -> dict:
    ranges = bundle.market_audit["ranges"]
    common_start = max(int(row["min_time_s"]) for row in ranges.values())
    common_end = bundle.gcp_audit["max_time_s"]
    return {
        "db": str(Path(args.db).resolve()),
        "gcp": str(Path(args.gcp).resolve()),
        "source_audit": bundle.market_audit,
        "gcp_audit": bundle.gcp_audit,
        "gcp_audit_min_time_s": bundle.gcp_audit["min_time_s"],
        "common_range": {"start_time_s": int(common_start), "end_time_s": int(common_end)},
        "anchor_convention": f"target_time_s % {ANCHOR_MINUTES_DEV * 60} == 0",
        "schrodinger_audit": bundle.schr_audit,
        "coupled_audit": bundle.coupled_audit,
        "causality": bundle.causality,
        "shifted_gcp_check": bundle.shift_check,
        "frame_rows": int(len(bundle.frame)),
    }


def protocol_block(args: argparse.Namespace, n_boot: int, ranking_boot: int | None) -> dict:
    return {
        "preregistration": "mix_lab_preregistration.md",
        "schrodinger_config": FROZEN_SCHR.as_dict(),
        "coupled_candidates": [config.as_dict() for config in COUPLED_CANDIDATES],
        "control_shift_days": CONTROL_SHIFT_DAYS,
        "cv": "expanding walk-forward, n_folds=3, initial_train_fraction=0.50, embargo=horizon",
        "models": "per-fold StandardScaler + L2 logistic C=1 (direction) / ridge alpha=1 (volatility)",
        "bootstrap": {
            "module": "backtest.paired_bootstrap_delta",
            "n_boot_frozen": int(n_boot),
            "n_boot_ranking": None if ranking_boot is None else int(ranking_boot),
            "block_len_anchors": block_len_for(ANCHOR_MINUTES_DEV),
            "seeds": "20260914+h (hit), 20260915+h (IC), 20260916+h (MSE)",
            "p_value_floor_frozen": 2.0 / (int(n_boot) + 1.0),
        },
        "bonferroni_dev": BONFERRONI_DEV,
        "cpu_thread_cap": MAX_THREADS,
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def _print_tables(results: dict) -> None:
    _print_additive(results)
    if results.get("frozen", {}).get("coupled"):
        _print_coupled(results)


def _print_frozen_tables(results: dict) -> None:
    _print_additive(results)
    if results.get("frozen", {}).get("coupled"):
        _print_coupled(results)


def _print_additive(results: dict) -> None:
    frozen = results.get("frozen")
    if not frozen or not frozen.get("additive", {}).get("cells"):
        return
    cells = frozen["additive"]["cells"]
    print(f"\nAdditive arms ({results.get('stage', 'dev')}; deltas on identical rows; n_boot={frozen['n_boot']}):")
    header = "asset  h | A0 hit A1 hit A2 hit A3 hit | dA1 dA2 dA3 dA3-dA2 (pp) | dIC A1 A2 A3 A3-A2"
    print(header)
    print("-" * len(header))
    for asset in ASSETS:
        for horizon in HORIZONS:
            direction = cells[asset][f"{horizon}m"]["direction"]
            hits = [
                direction["arms"][arm]["evaluation"]["model"]["hit_rate"] for arm in ("A0", "A1", "A2", "A3")
            ]
            deltas = direction["deltas"]
            print(
                f"{asset:7s} {horizon:>2d} | " + " ".join(f"{100 * h:.2f}" for h in hits)
                + " | " + " ".join(
                    f"{100 * deltas[key]['hit_rate']['delta']:+.4f}" for key in ("A1-A0", "A2-A0", "A3-A0", "A3-A2")
                )
                + " | " + " ".join(
                    f"{deltas[key]['ic_spearman']['delta']:+.5f}" for key in ("A1-A0", "A2-A0", "A3-A0", "A3-A2")
                )
            )
    for name, summary in frozen["additive"]["deltas"].items():
        print(f"  {name}: {summary['verdict']}")
        for metric, stats in summary["metrics"].items():
            print(
                f"    {metric}: improving {stats['n_improving']}/{stats['n_cells']}, "
                f"survivors {stats['n_survivors']}, worse {stats['n_significantly_worse']}, "
                f"median {_fmt(stats['median_delta'], 8)}"
            )


def _print_coupled(results: dict) -> None:
    frozen = results["frozen"]
    cells = frozen["coupled"]["cells"]
    config = frozen["coupled"]["config"]
    print(f"\nCoupled arm {config['label']} vs shifted-GCP control (n_boot={frozen['n_boot']}):")
    header = "asset  h | real hit shift hit | dHit pp (p) | dIC (p) | dMSE (p) | S0 hit% S0 IC"
    print(header)
    print("-" * len(header))
    for asset in ASSETS:
        for horizon in HORIZONS:
            cell = cells[asset][f"{horizon}m"]
            direction = cell["direction"]
            d = direction["deltas"]["C_real-C_shift"]
            v = cell["volatility"]["deltas"]["C_real-C_shift"]
            direct = cell["direct"]
            print(
                f"{asset:7s} {horizon:>2d} | "
                f"{100 * direction['arms']['C_real']['evaluation']['model']['hit_rate']:.2f} "
                f"{100 * direction['arms']['C_shift']['evaluation']['model']['hit_rate']:.2f} | "
                f"{100 * d['hit_rate']['delta']:+.4f} ({_fmt(d['hit_rate']['p_value'])}) | "
                f"{d['ic_spearman']['delta']:+.5f} ({_fmt(d['ic_spearman']['p_value'])}) | "
                f"{v['mse']['delta']:+.3e} ({_fmt(v['mse']['p_value'])}) | "
                f"{100 * direct['direction']['hit_rate']:.3f} {_fmt(direct['direction']['ic_spearman'], 5)}"
            )
    print(f"  C_real-C_shift: {frozen['coupled']['deltas']['C_real-C_shift']['verdict']}")
    for metric, stats in frozen["coupled"]["deltas"]["C_real-C_shift"]["metrics"].items():
        print(
            f"    {metric}: improving {stats['n_improving']}/{stats['n_cells']}, "
            f"survivors {stats['n_survivors']}, worse {stats['n_significantly_worse']}, "
            f"median {_fmt(stats['median_delta'], 8)}"
        )


def _merge_write(path: Path, block: dict) -> None:
    payload: dict = {}
    if path.exists():
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            payload = {}
    payload.update(block)
    payload["updated_utc"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(_jsonable(payload), indent=2, allow_nan=False) + "\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("explore", "frozen", "holdout"), default="explore")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--gcp", type=Path, default=DEFAULT_GCP, help="dev GCP minute cache (netvar_1m)")
    parser.add_argument("--holdout-db", type=Path, default=None)
    parser.add_argument("--gcp-holdout-db", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--dev-results",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="dev results file the frozen label is read from (holdout stage)",
    )
    parser.add_argument("--start-time-s", type=int, default=None)
    parser.add_argument("--end-time-s", type=int, default=None)
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--initial-train-fraction", type=float, default=0.50)
    parser.add_argument("--embargo", type=int, default=None)
    parser.add_argument("--bootstrap", type=int, default=4999, help="frozen dev / holdout replicates")
    parser.add_argument("--bootstrap-explore", type=int, default=300, help="ranking-only replicates")
    parser.add_argument("--chunk", type=int, default=8192)
    parser.add_argument("--dev-tail-days", type=int, default=2)
    parser.add_argument("--holdout-anchor-minutes", type=int, default=1)
    parser.add_argument(
        "--coupling-labels",
        type=str,
        default=None,
        help="diagnostic subset of coupled labels (default: the full pre-declared grid)",
    )
    args = parser.parse_args(argv)
    if args.coupling_labels:
        args.coupling_labels = [part for part in args.coupling_labels.split(",") if part]
        unknown = [part for part in args.coupling_labels if part not in COUPLED_BY_LABEL]
        if unknown:
            parser.error(f"unknown coupling labels: {unknown}")
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
