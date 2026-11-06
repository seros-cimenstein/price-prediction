"""Evaluation driver for the literal Schrodinger wave-packet arm.

Usage from the repository root::

    price-prediction/.venv/bin/python price-prediction/train_schrodinger.py --stage explore
    price-prediction/.venv/bin/python price-prediction/train_schrodinger.py --stage frozen
    price-prediction/.venv/bin/python price-prediction/train_schrodinger.py --stage holdout \
        --db ../crypto-data/crypto_market.db --holdout-db ../crypto-data/holdout_klines_1m.db \
        --output price-prediction/results_schrodinger_holdout.json

``--stage explore`` runs the pre-registered grid of (potential, tau map, k0 map)
configurations on the dev database, logs every configuration's dev result, and
selects the frozen configuration by the pre-declared rule in
``schrodinger_preregistration.md``.  ``--stage frozen`` runs the frozen
configuration with confirmatory bootstrap resolution and the shuffled-target
control on dev.

``--stage holdout`` implements Amendment A2: both S1 arms are fitted **once**
on all dev rows at the frozen dev anchor cadence whose horizon targets finish
before the holdout starts, then frozen and scored on every holdout minute; the
S0 direct Born-rule readout is evaluated on the same holdout rows.  Features
are built on the last ``--dev-tail-days`` of dev candles concatenated with the
holdout so trailing windows cross the source boundary, and evaluation starts
at ``max(first holdout minute, dev end) + warm-up`` where the warm-up is the
exact number of trailing candles every horizon needs (300 minutes for h=60).
The holdout database is opened read-only.

The frame, folds, models, metrics, paired bootstrap, and shuffled control are
the repository's own (``features.py``, ``backtest.py``); this driver adds only
the Schrodinger feature columns and the pre-registered tables.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from backtest import (  # noqa: E402
    HORIZONS,
    MARKET_FEATURES,
    _block_mean_bootstrap,
    _block_sample_indices,
    _spearman,
    _valid_frame,
    direction_metrics,
    evaluate_predictions,
    make_walk_forward_splits,
    paired_bootstrap_delta,
    shuffled_target_control,
    volatility_metrics,
    walk_forward_predict,
)
from features import ASSETS, assert_causal_features, market_db_audit  # noqa: E402
from schrodinger import (  # noqa: E402
    SchrodingerConfig,
    build_anchor_frame,
    schr_feature_columns,
)
from train import _ablation_summary, _jsonable, _same_oos_rows  # noqa: E402

DEFAULT_DB = HERE.parent / "crypto-data" / "crypto_market.db"
DEFAULT_OUTPUT = HERE / "results_schrodinger_dev.json"

EXPLORATION_CONFIGS: tuple[SchrodingerConfig, ...] = tuple(
    SchrodingerConfig(potential=potential, tau_map=tau_map, k0_map=k0_map)
    for potential in ("quantum_kde", "quartic_kde")
    for tau_map in ("linear_60", "slow_240")
    for k0_map in ("zero", "momentum")
)
CONTROL_CONFIG = SchrodingerConfig(potential="free_particle", tau_map="linear_60", k0_map="momentum")

# Frozen after the exploration run by the pre-declared selection rule in
# ``schrodinger_preregistration.md`` (rank by number of cells with positive
# S1 Delta IC, then pooled Delta IC, then median Delta IC).  Selected from the
# 8 candidates: quartic_kde|linear_60|momentum (11/16 positive Delta IC,
# pooled +0.00087).  The frozen stage refuses to run while it is None.
FROZEN_CONFIG: SchrodingerConfig | None = SchrodingerConfig(
    potential="quartic_kde", tau_map="linear_60", k0_map="momentum"
)

CANDIDATE_ARM = "with_schrodinger"
REFERENCE_ARM = "market_only"
DELTA_KEY = "schrodinger_delta"
BONFERRONI_ALPHA = 0.05 / float(len(ASSETS) * len(HORIZONS))


def block_len_for(anchor_minutes: int) -> int:
    """Moving-block length in anchor rows (repository convention at 1-minute)."""
    return 60 if int(anchor_minutes) <= 1 else 3


def complete_case_rows(
    frame: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    kind: str,
) -> pd.DataFrame:
    """Same complete-case selection as ``backtest._valid_frame``, all columns.

    ``_valid_frame`` projects onto the model columns, but the direct (S0) arm
    also needs the audit columns (``schr_trailing_vol_{h}m``).  The selection is
    identical and is asserted against ``_valid_frame``.
    """
    valid = (
        frame.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=[*feature_cols, target_col])
        .copy()
    )
    if kind == "direction":
        valid = valid.loc[valid[target_col].ne(0.0)]
    valid = valid.reset_index(drop=True)
    reference = _valid_frame(frame, feature_cols, target_col, kind)
    if len(valid) != len(reference) or not np.array_equal(
        valid["target_time_s"].to_numpy(dtype="int64"), reference["target_time_s"].to_numpy(dtype="int64")
    ):
        raise AssertionError("complete_case_rows selection differs from backtest._valid_frame")
    return valid


def oos_rows(
    frame: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    horizon: int,
    kind: str,
    n_folds: int,
    initial_train_fraction: float,
    embargo: int | None,
) -> pd.DataFrame:
    """The exact OOS rows of ``walk_forward_predict``, with all frame columns."""
    valid = complete_case_rows(frame, feature_cols, target_col, kind)
    folds = make_walk_forward_splits(
        valid["target_time_s"].to_numpy(),
        horizon=horizon,
        n_folds=n_folds,
        initial_train_fraction=initial_train_fraction,
        embargo=embargo,
    )
    index = np.concatenate([fold.test_idx for fold in folds])
    return valid.iloc[index].reset_index(drop=True)


def _two_sided_bootstrap_p(boot: np.ndarray) -> float:
    finite = np.asarray(boot, dtype="float64")
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return float("nan")
    return float(
        min(
            1.0,
            2.0
            * min(
                (1.0 + np.sum(finite <= 0.0)) / (len(finite) + 1.0),
                (1.0 + np.sum(finite >= 0.0)) / (len(finite) + 1.0),
            ),
        )
    )


def s0_chance_tests(
    oos: pd.DataFrame,
    horizon: int,
    n_boot: int,
    block_len: int,
    seed_offset: int = 0,
) -> dict:
    """One-sample block-bootstrap tests for the direct (S0) direction arm.

    Uses the same resampling blocks as ``backtest.paired_bootstrap_delta``:
    ``_block_mean_bootstrap`` on the per-row hit indicator minus 0.5 for the hit
    rate, and ``_block_sample_indices`` + ``_spearman`` for the IC (the paired
    IC branch without the reference arm).  The two-sided p-value tests the
    direct Born-rule readout against chance (0.5 / 0.0) on exactly the OOS rows.
    """
    y = oos[f"forward_return_{horizon}m"].to_numpy(dtype="float64")
    p_up = oos[f"schr_p_up_{horizon}m"].to_numpy(dtype="float64")
    good = np.isfinite(y) & np.isfinite(p_up) & (y != 0.0)
    y = y[good]
    p_up = p_up[good]
    hits = (np.sign(p_up - 0.5) == np.sign(y)).astype("float64")
    rng = np.random.default_rng(seed_offset + 20260914 + horizon)
    boot_hit = _block_mean_bootstrap(hits - 0.5, n_boot, block_len, rng)
    rng_ic = np.random.default_rng(seed_offset + 20260915 + horizon)
    n = len(y)
    boot_ic = np.empty(int(n_boot), dtype="float64")
    for i in range(int(n_boot)):
        idx = _block_sample_indices(n, rng_ic, block_len)
        boot_ic[i] = _spearman(y[idx], p_up[idx])
    return {
        "hit_rate_vs_chance": {
            "delta": float(hits.mean() - 0.5),
            "p_value": _two_sided_bootstrap_p(boot_hit),
            "n": int(n),
            "n_boot": int(n_boot),
            "block_len": int(block_len),
        },
        "ic_vs_zero": {
            "delta": _spearman(y, p_up),
            "p_value": _two_sided_bootstrap_p(boot_ic),
            "n": int(n),
            "n_boot": int(n_boot),
            "block_len": int(block_len),
        },
    }


def evaluate_cell(
    asset_frame: pd.DataFrame,
    horizon: int,
    args: argparse.Namespace,
    n_boot: int,
    candidate_cols: Sequence[str],
    folds_n: int = 3,
    robustness_block_len: int | None = None,
) -> dict:
    """One (asset, horizon) cell: S1 paired arms, deltas, and the S0 direct arm."""
    market_cols = list(MARKET_FEATURES)
    candidate_cols = list(candidate_cols)
    cell: dict[str, Any] = {}
    pair_valid_by_task: dict[str, pd.DataFrame] = {}
    pred_by_task: dict[str, pd.DataFrame] = {}
    block_len = block_len_for(args.anchor_minutes)
    for task in ("direction", "volatility"):
        target_col = f"forward_return_{horizon}m" if task == "direction" else f"forward_vol_{horizon}m"
        pair_valid = (
            asset_frame.replace([np.inf, -np.inf], np.nan)
            .dropna(subset=[*candidate_cols, target_col])
            .copy()
        )
        if task == "direction":
            pair_valid = pair_valid.loc[pair_valid[target_col].ne(0.0)]
        pair_valid_by_task[task] = pair_valid
        candidate_pred, candidate_folds = walk_forward_predict(
            pair_valid,
            candidate_cols,
            target_col,
            horizon,
            task,
            n_folds=folds_n,
            initial_train_fraction=args.initial_train_fraction,
            embargo=args.embargo,
        )
        market_pred, market_folds = walk_forward_predict(
            pair_valid,
            market_cols,
            target_col,
            horizon,
            task,
            n_folds=folds_n,
            initial_train_fraction=args.initial_train_fraction,
            embargo=args.embargo,
        )
        candidate_pred, market_pred = _same_oos_rows(candidate_pred, market_pred)
        pred_by_task[task] = candidate_pred
        y = candidate_pred[target_col].to_numpy(dtype="float64")
        if task == "direction":
            deltas = {
                "hit_rate": paired_bootstrap_delta(
                    y, candidate_pred["score"], market_pred["score"],
                    metric="hit_rate", n_boot=n_boot, block_len=block_len, seed=20260914 + horizon,
                ),
                "ic_spearman": paired_bootstrap_delta(
                    y, candidate_pred["score"], market_pred["score"],
                    metric="ic_spearman", n_boot=n_boot, block_len=block_len, seed=20260915 + horizon,
                ),
            }
        else:
            deltas = {
                "mse": paired_bootstrap_delta(
                    y, candidate_pred["score"], market_pred["score"],
                    metric="mse", n_boot=n_boot, block_len=block_len, seed=20260916 + horizon,
                )
            }
        cell[task] = {
            CANDIDATE_ARM: {"evaluation": evaluate_predictions(candidate_pred, horizon, task), "folds": candidate_folds},
            REFERENCE_ARM: {"evaluation": evaluate_predictions(market_pred, horizon, task), "folds": market_folds},
            DELTA_KEY: deltas,
        }
        if robustness_block_len is not None:
            seed_offset = 1_000_000
            if task == "direction":
                cell[task]["robustness"] = {
                    "block_len_rows": int(robustness_block_len),
                    "hit_rate": paired_bootstrap_delta(
                        y, candidate_pred["score"], market_pred["score"], metric="hit_rate",
                        n_boot=n_boot, block_len=int(robustness_block_len), seed=seed_offset + 20260914 + horizon,
                    ),
                    "ic_spearman": paired_bootstrap_delta(
                        y, candidate_pred["score"], market_pred["score"], metric="ic_spearman",
                        n_boot=n_boot, block_len=int(robustness_block_len), seed=seed_offset + 20260915 + horizon,
                    ),
                }
            else:
                cell[task]["robustness"] = {
                    "block_len_rows": int(robustness_block_len),
                    "mse": paired_bootstrap_delta(
                        y, candidate_pred["score"], market_pred["score"], metric="mse",
                        n_boot=n_boot, block_len=int(robustness_block_len), seed=seed_offset + 20260916 + horizon,
                    ),
                }
    # S0: the Born-rule readouts used directly, on the identical OOS rows of the
    # matching S1 task (direction rows for direction metrics, volatility rows
    # for volatility metrics), asserted against the S1 row keys.
    s0: dict[str, Any] = {}
    oos_by_task: dict[str, pd.DataFrame] = {}
    for task in ("direction", "volatility"):
        target_col = f"forward_return_{horizon}m" if task == "direction" else f"forward_vol_{horizon}m"
        oos = oos_rows(
            pair_valid_by_task[task], candidate_cols, target_col, horizon, task,
            folds_n, args.initial_train_fraction, args.embargo,
        )
        oos_by_task[task] = oos
        s1_keys = set(zip(pred_by_task[task]["target_time_s"].astype("int64"), pred_by_task[task]["symbol"]))
        oos_keys = set(zip(oos["target_time_s"].astype("int64"), oos["symbol"]))
        if s1_keys != oos_keys:
            raise AssertionError("S0 rows differ from S1 OOS rows")
        if task == "direction":
            # backtest.direction_metrics treats the score as a *signed* score
            # (sign(score) vs sign(return)); p_up is a probability, so it must
            # be centred first or sign(p_up) would be +1 for every row.
            s0["direction"] = direction_metrics(
                oos[target_col].to_numpy(dtype="float64"),
                oos[f"schr_p_up_{horizon}m"].to_numpy(dtype="float64") - 0.5,
            )
            s0["direction_chance"] = s0_chance_tests(oos, horizon, n_boot, block_len)
        else:
            valid_vol = oos.dropna(
                subset=[target_col, f"schr_std_{horizon}m", f"schr_trailing_vol_{horizon}m", "ewma_vol_60m"]
            )
            y_vol = valid_vol[target_col].to_numpy(dtype="float64")
            pred_vol = (
                valid_vol[f"schr_std_{horizon}m"] * valid_vol[f"schr_trailing_vol_{horizon}m"]
            ).to_numpy(dtype="float64")
            ewma_vol = (valid_vol["ewma_vol_60m"] * np.sqrt(horizon)).to_numpy(dtype="float64")
            trailing_vol = valid_vol[f"schr_trailing_vol_{horizon}m"].to_numpy(dtype="float64")
            metrics = volatility_metrics(y_vol, pred_vol, ewma=ewma_vol)
            trailing = volatility_metrics(y_vol, trailing_vol)
            metrics["trailing_mse"] = trailing["mse"]
            metrics["trailing_mae"] = trailing["mae"]
            metrics["mse_delta_vs_trailing"] = metrics["mse"] - trailing["mse"]
            metrics["vs_ewma_paired"] = paired_bootstrap_delta(
                y_vol, pred_vol, ewma_vol, metric="mse", n_boot=n_boot, block_len=block_len, seed=20260916 + horizon
            )
            metrics["vs_trailing_paired"] = paired_bootstrap_delta(
                y_vol, pred_vol, trailing_vol, metric="mse", n_boot=n_boot, block_len=block_len, seed=20260917 + horizon
            )
            s0["volatility"] = metrics
    if robustness_block_len is not None:
        robust_block = int(robustness_block_len)
        oos_dir = oos_by_task["direction"]
        oos_vol = oos_by_task["volatility"].dropna(
            subset=[
                f"forward_vol_{horizon}m",
                f"schr_std_{horizon}m",
                f"schr_trailing_vol_{horizon}m",
                "ewma_vol_60m",
            ]
        )
        y_vol = oos_vol[f"forward_vol_{horizon}m"].to_numpy(dtype="float64")
        pred_vol = (
            oos_vol[f"schr_std_{horizon}m"] * oos_vol[f"schr_trailing_vol_{horizon}m"]
        ).to_numpy(dtype="float64")
        ewma_vol = (oos_vol["ewma_vol_60m"] * np.sqrt(horizon)).to_numpy(dtype="float64")
        s0["robustness"] = {
            "block_len_rows": robust_block,
            "direction_chance": s0_chance_tests(oos_dir, horizon, n_boot, robust_block, seed_offset=1_000_000),
            "volatility_vs_ewma": paired_bootstrap_delta(
                y_vol, pred_vol, ewma_vol, metric="mse", n_boot=n_boot, block_len=robust_block,
                seed=1_000_000 + 20260916 + horizon,
            ),
        }
    cell["s0"] = s0
    return cell


def exploration_entry(cell: dict, horizon: int) -> dict:
    """Compact per-cell row for the exploration log."""
    direction = cell["direction"]
    volatility = cell["volatility"]
    return {
        "delta_hit_pp": float(direction[DELTA_KEY]["hit_rate"]["delta"]) * 100.0,
        "delta_hit_p": float(direction[DELTA_KEY]["hit_rate"]["p_value"]),
        "delta_ic": float(direction[DELTA_KEY]["ic_spearman"]["delta"]),
        "delta_ic_p": float(direction[DELTA_KEY]["ic_spearman"]["p_value"]),
        "delta_mse": float(volatility[DELTA_KEY]["mse"]["delta"]),
        "delta_mse_p": float(volatility[DELTA_KEY]["mse"]["p_value"]),
        "candidate_hit_rate": float(direction[CANDIDATE_ARM]["evaluation"]["model"]["hit_rate"]),
        "market_hit_rate": float(direction[REFERENCE_ARM]["evaluation"]["model"]["hit_rate"]),
        "candidate_ic": float(direction[CANDIDATE_ARM]["evaluation"]["model"]["ic_spearman"]),
        "market_ic": float(direction[REFERENCE_ARM]["evaluation"]["model"]["ic_spearman"]),
        "s0_hit_rate": float(cell["s0"]["direction"]["hit_rate"]),
        "s0_ic": float(cell["s0"]["direction"]["ic_spearman"]),
        "s0_mse": float(cell["s0"]["volatility"]["mse"]),
        "s0_ewma_mse": float(cell["s0"]["volatility"]["ewma_mse"]),
        "n_direction": int(direction[CANDIDATE_ARM]["evaluation"]["model"]["n"]),
    }


def summarise_config(cells: dict) -> dict:
    """Selection statistics for one config over its 16 cells."""
    deltas_ic = []
    deltas_hit = []
    deltas_mse = []
    weights = []
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
    return {
        "n_positive_delta_ic": int(np.sum(deltas_ic > 0.0)),
        "n_positive_delta_hit": int(np.sum(deltas_hit > 0.0)),
        "n_negative_delta_mse": int(np.sum(deltas_mse < 0.0)),
        "pooled_delta_ic": float(np.sum(weights * deltas_ic) / np.sum(weights)),
        "median_delta_ic": float(np.median(deltas_ic)),
        "median_delta_hit_pp": float(np.median(deltas_hit)),
        "median_delta_mse": float(np.median(deltas_mse)),
        "n_cells": int(len(deltas_ic)),
    }


def select_config(summaries: dict[str, dict]) -> str:
    """Apply the pre-declared selection rule to candidate configs."""
    labels = [label for label in summaries if label != CONTROL_CONFIG.label]
    labels.sort(
        key=lambda label: (
            summaries[label]["n_positive_delta_ic"],
            summaries[label]["pooled_delta_ic"],
            summaries[label]["median_delta_ic"],
        ),
        reverse=True,
    )
    return labels[0]


def run_explore(args: argparse.Namespace) -> dict:
    started = time.time()
    output_path = Path(args.output).resolve()
    audit = market_db_audit(args.db)
    print("Market schema/ranges verified:")
    for symbol, row in audit["ranges"].items():
        print(f"  {symbol}: {row['rows']:,} rows {row['min_time_s']}..{row['max_time_s']}")
    configs = list(EXPLORATION_CONFIGS) + [CONTROL_CONFIG]
    print(f"Building Schrodinger features for {len(configs)} configs on dev anchors ...", flush=True)
    t_features = time.time()
    frames, frame_audit = build_anchor_frame(
        args.db,
        anchor_minutes=args.anchor_minutes,
        horizons=HORIZONS,
        configs=configs,
        start_time_s=args.start_time_s,
        end_time_s=args.end_time_s,
        chunk=args.chunk,
    )
    base_frame = frames[configs[0].label]
    leakage = assert_causal_features(base_frame, require_gcp=False, require_physics=False)
    print(f"Leakage self-check passed: {leakage}")
    print(f"Schrodinger rows per asset/horizon: {frame_audit['schrodinger_rows'][configs[0].label]}")
    print(f"KDE degenerate-fallback rows: {frame_audit['kde_fallback_rows']}")

    exploration: dict[str, Any] = {
        "selection_rule": (
            "rank candidates (control excluded) by (1) number of the 16 cells with positive delta IC, "
            "(2) pooled row-weighted delta IC, (3) median delta IC"
        ),
        "anchor_convention": frame_audit["anchor_convention"],
        "configs": {},
        "runtime_seconds_features": time.time() - t_features,
    }
    summaries: dict[str, dict] = {}
    for config in configs:
        label = config.label
        frame = frames[label]
        print(f"Evaluating config {label} ...", flush=True)
        t0 = time.time()
        cells: dict[str, Any] = {}
        for asset in ASSETS:
            asset_frame = frame.loc[frame["symbol"].eq(asset)].copy()
            for horizon in HORIZONS:
                candidate_cols = MARKET_FEATURES + schr_feature_columns(horizon)
                cell = evaluate_cell(asset_frame, horizon, args, n_boot=args.bootstrap_explore, candidate_cols=candidate_cols)
                cells[f"{asset}_{horizon}m"] = exploration_entry(cell, horizon)
        summary = summarise_config(cells)
        summaries[label] = summary
        exploration["configs"][label] = {
            "config": config.as_dict(),
            "cells": cells,
            "summary": summary,
            "runtime_seconds": time.time() - t0,
        }
        print(
            f"  {label}: positive delta IC {summary['n_positive_delta_ic']}/16, "
            f"pooled delta IC {summary['pooled_delta_ic']:+.5f}, "
            f"positive delta hit {summary['n_positive_delta_hit']}/16, "
            f"negative delta MSE {summary['n_negative_delta_mse']}/16 ({time.time() - t0:.1f}s)"
        )
    selected = select_config(summaries)
    exploration["selected_label"] = selected
    exploration["control_label"] = CONTROL_CONFIG.label
    exploration["selection_ranking"] = [
        {"label": label, **summaries[label]}
        for label in sorted(
            (label for label in summaries if label != CONTROL_CONFIG.label),
            key=lambda label: (
                summaries[label]["n_positive_delta_ic"],
                summaries[label]["pooled_delta_ic"],
                summaries[label]["median_delta_ic"],
            ),
            reverse=True,
        )
    ]
    exploration["runtime_seconds"] = time.time() - started
    results: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "explore",
        "data": {
            "db": str(Path(args.db).resolve()),
            "source_audit": audit,
            "frame_audit": frame_audit,
            "causality": leakage,
        },
        "protocol": protocol_block(args, n_boot=args.bootstrap_explore),
        "exploration": exploration,
    }
    _merge_write(output_path, results)
    print(f"Selected config by pre-declared rule: {selected}")
    print(f"Wrote {output_path}")
    return results


def protocol_block(args: argparse.Namespace, n_boot: int) -> dict:
    return {
        "anchors": f"target_time_s % {int(args.anchor_minutes) * 60} == 0",
        "cv": "expanding walk-forward, n_folds=3, initial_train_fraction=0.50, embargo=horizon",
        "models": "per-fold StandardScaler + L2 logistic C=1 (direction) / ridge alpha=1 (volatility)",
        "arms": {
            "s0": "Born-rule observables used directly (p_up for direction, std*sigma_h for volatility)",
            "s1": f"{REFERENCE_ARM} vs {REFERENCE_ARM}+SCHR_h (four columns)",
            "s2": f"free-particle control config {CONTROL_CONFIG.label} (diagnostic only)",
        },
        "bootstrap": {
            "module": "backtest.paired_bootstrap_delta",
            "n_boot": int(n_boot),
            "block_len_rows": block_len_for(args.anchor_minutes),
            "seeds": "20260914+h (hit), 20260915+h (IC), 20260916+h (MSE)",
            "p_value_floor": 2.0 / (int(n_boot) + 1.0),
        },
        "bonferroni_alpha": BONFERRONI_ALPHA,
        "robustness_block_len_rows": None if getattr(args, "robustness_block_len", None) is None else int(args.robustness_block_len),
        "preregistration": "schrodinger_preregistration.md",
    }


def warmup_minutes_exact(horizons: Sequence[int] = HORIZONS) -> int:
    """Exact trailing candles needed before an anchor for every horizon.

    ``schrodinger.build_features_for_asset_multi`` keeps an anchor only when
    ``last >= h + KDE_WINDOW - 1`` where ``last`` is the index of candle
    ``t - 60s``; the anchor candle therefore needs ``h + KDE_WINDOW`` candles
    before it.  For h = 60 and the 240-candle KDE window this is 300 minutes.
    """
    from schrodinger import KDE_WINDOW

    return int(max(int(h) + int(KDE_WINDOW) for h in horizons))


def fit_and_score(
    train_valid: pd.DataFrame,
    eval_valid: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    task: str,
):
    """One frozen fit (house model/scaler policy) scored on the eval rows."""
    from backtest import _fit_score

    x_train = train_valid.loc[:, feature_cols].to_numpy(dtype="float64")
    y_train = train_valid[target_col].to_numpy(dtype="float64")
    x_test = eval_valid.loc[:, feature_cols].to_numpy(dtype="float64")
    scaler, model, score = _fit_score(x_train, y_train, x_test, task)
    return scaler, model, score


def run_holdout(args: argparse.Namespace) -> dict:
    """Amendment A2: fit once on all dev rows, score every holdout minute."""
    import features as features_module
    from schrodinger import build_anchor_frame_from_market

    config = FROZEN_CONFIG if args.config is None else _parse_config(args.config)
    assert config is not None
    started = time.time()
    output_path = Path(args.output).resolve()
    audit_dev = market_db_audit(args.db)
    audit_holdout = market_db_audit(args.holdout_db)
    dev_end_s = max(int(row["max_time_s"]) for row in audit_dev["ranges"].values())
    holdout_start_s = min(int(row["min_time_s"]) for row in audit_holdout["ranges"].values())
    holdout_end_s = max(int(row["max_time_s"]) for row in audit_holdout["ranges"].values())
    assert holdout_start_s == dev_end_s + 60, (
        "holdout must directly follow the dev source (gap "
        f"{holdout_start_s - dev_end_s - 60} seconds)"
    )
    warmup_minutes = warmup_minutes_exact(HORIZONS)
    eval_start_s = max(holdout_start_s, dev_end_s) + warmup_minutes * 60
    print(f"Frozen config: {config.label}")
    print(
        f"dev end {dev_end_s}, holdout {holdout_start_s}..{holdout_end_s}, "
        f"exact warm-up {warmup_minutes} minutes, eval from {eval_start_s}"
    )

    # Training frame: the registered dev anchors at the frozen dev cadence.
    train_frames, train_audit = build_anchor_frame(
        args.db,
        anchor_minutes=args.train_anchor_minutes,
        horizons=HORIZONS,
        configs=[config],
        chunk=args.chunk,
    )
    train_frame = train_frames[config.label]
    print(f"Training frame rows: {len(train_frame)} ({train_audit['anchor_convention']})")

    # Evaluation frame: dev tail + holdout concatenated, so trailing windows
    # cross the source boundary; only holdout rows at/after eval_start are scored.
    tail_seconds = int(args.dev_tail_days) * 86400
    dev_tail = features_module._query_market(
        args.db, features_module.ASSETS, start_time_s=dev_end_s - tail_seconds + 60, end_time_s=dev_end_s
    )
    holdout_market = features_module._query_market(args.holdout_db, features_module.ASSETS)
    concatenated: dict[str, pd.DataFrame] = {}
    for symbol in features_module.ASSETS:
        joint = pd.concat([dev_tail[symbol], holdout_market[symbol]], axis=0)
        joint = joint[~joint.index.duplicated(keep="first")].sort_index()
        step = np.diff(joint.index.to_numpy(dtype="datetime64[s]").astype("int64"))
        assert np.all(step == 60), f"{symbol}: concatenated candles are not contiguous"
        concatenated[symbol] = joint
    concat_frame = features_module._assemble_feature_frame(concatenated)
    eval_frames, eval_audit = build_anchor_frame_from_market(
        concat_frame,
        concatenated,
        anchor_minutes=args.anchor_minutes,
        horizons=HORIZONS,
        configs=[config],
        chunk=args.chunk,
    )
    eval_frame = eval_frames[config.label]
    eval_frame = eval_frame.loc[eval_frame["target_time_s"].to_numpy(dtype="int64") >= holdout_start_s].copy()
    print(f"Holdout rows before warm-up filter: {len(eval_frame)}")

    # Assertions required by Amendment A2.
    warm_row = eval_frame.loc[eval_frame["target_time_s"].to_numpy(dtype="int64") == eval_start_s]
    assert len(warm_row) == len(features_module.ASSETS), "eval_start row missing from the holdout frame"
    first_eval = warm_row.iloc[0]
    required_at_first_row = [*MARKET_FEATURES, "ewma_vol_60m"]
    for h in HORIZONS:
        required_at_first_row.extend(schr_feature_columns(h))
    for h in HORIZONS:
        required_at_first_row.append(f"schr_trailing_vol_{h}m")
    missing_finite = [name for name in required_at_first_row if not np.isfinite(first_eval[name])]
    assert not missing_finite, f"features not finite at the first evaluated row: {missing_finite}"
    print(f"First evaluated row {eval_start_s} has all {len(required_at_first_row)} features finite")
    eval_frame = eval_frame.loc[eval_frame["target_time_s"].to_numpy(dtype="int64") >= eval_start_s].copy()

    n_boot = int(args.bootstrap)
    block_len = block_len_for(args.anchor_minutes)
    results: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "holdout",
        "amendment": "A2",
        "data": {
            "dev_db": str(Path(args.db).resolve()),
            "holdout_db": str(Path(args.holdout_db).resolve()),
            "source_audit_dev": audit_dev,
            "source_audit_holdout": audit_holdout,
            "bounds": {
                "dev_end_time_s": dev_end_s,
                "holdout_start_time_s": holdout_start_s,
                "holdout_end_time_s": holdout_end_s,
                "warmup_minutes": warmup_minutes,
                "eval_start_time_s": eval_start_s,
                "dev_tail_days": int(args.dev_tail_days),
            },
            "train_frame_audit": train_audit,
            "eval_frame_audit": eval_audit,
            "causality": {
                "holdout_directly_follows_dev": True,
                "training_targets_end_before_holdout": None,  # filled per cell below
                "first_eval_row_features_finite": True,
                "schrodinger_assertions": "sample candles strictly before each anchor (schrodinger.py)",
            },
        },
        "protocol": {
            "design": (
                "Amendment A2: both S1 arms are fitted once on ALL dev rows at the frozen dev anchor "
                "cadence (hourly) whose horizon targets finish before the holdout starts, then frozen and "
                "scored on every holdout minute from eval_start; S0 is the direct Born-rule readout on the "
                "same holdout rows (nothing to fit)."
            ),
            "anchors": f"holdout target_time_s % {int(args.anchor_minutes) * 60} == 0",
            "models": "per-arm StandardScaler + L2 logistic C=1 (direction) / ridge alpha=1 (volatility), fitted once",
            "bootstrap": {
                "module": "backtest.paired_bootstrap_delta",
                "n_boot": n_boot,
                "block_len_rows": block_len,
                "seeds": "20260914+h (hit), 20260915+h (IC), 20260916+h (MSE)",
                "p_value_floor": 2.0 / (n_boot + 1.0),
            },
            "bonferroni_alpha": BONFERRONI_ALPHA,
            "preregistration": "schrodinger_preregistration.md",
        },
        "ablation": {
            "reference_arm": REFERENCE_ARM,
            "candidate_arm": CANDIDATE_ARM,
            "delta_key": DELTA_KEY,
            "candidate_config": config.as_dict(),
            "candidate_feature_columns": list(schr_feature_columns(1)),
            "bonferroni_alpha": BONFERRONI_ALPHA,
            "bootstrap": {
                "hit_rate": n_boot,
                "mse": n_boot,
                "ic_spearman": n_boot,
                "block_minutes": block_len,
                "p_value_floor": {
                    "hit_rate": 2.0 / (n_boot + 1.0),
                    "mse": 2.0 / (n_boot + 1.0),
                    "ic_spearman": 2.0 / (n_boot + 1.0),
                },
            },
            "preregistration": "schrodinger_preregistration.md",
        },
        "assets": {},
        "leakage_checks": {"shuffled_target_controls": {}, "design": "training-target permutation, held-out scoring"},
    }
    max_target_end = 0
    for asset in ASSETS:
        asset_frame = eval_frame.loc[eval_frame["symbol"].eq(asset)].copy()
        asset_train = train_frame.loc[train_frame["symbol"].eq(asset)].copy()
        results["assets"][asset] = {}
        for horizon in HORIZONS:
            print(f"Holdout {config.label} {asset} {horizon}m ...", flush=True)
            candidate_cols = MARKET_FEATURES + schr_feature_columns(horizon)
            cell: dict[str, Any] = {"training": {}, "s0": {}}
            for task in ("direction", "volatility"):
                target_col = f"forward_return_{horizon}m" if task == "direction" else f"forward_vol_{horizon}m"
                train_rows = asset_train.loc[
                    asset_train["target_time_s"].to_numpy(dtype="int64") + horizon * 60 <= holdout_start_s
                ]
                # Amendment A2 assertion: every fitted target finishes before
                # the holdout source starts (checked on the unfiltered rows).
                if len(train_rows):
                    rows_max_end = int(train_rows["target_time_s"].max()) + horizon * 60
                    assert rows_max_end <= holdout_start_s, "a training target ends at/after the holdout start"
                    max_target_end = max(max_target_end, rows_max_end)
                train_valid = _valid_frame(train_rows, candidate_cols, target_col, task)
                eval_valid = complete_case_rows(asset_frame, candidate_cols, target_col, task)
                if len(train_valid) == 0 or len(eval_valid) == 0:
                    raise RuntimeError(f"empty train/eval rows for {asset} {horizon}m {task}")
                if task == "direction":
                    cell["training"]["direction"] = {
                        "n_rows": int(len(train_valid)),
                        "max_target_time_s": int(train_valid["target_time_s"].max()),
                        "max_target_end_time_s": int(train_valid["target_time_s"].max()) + horizon * 60,
                        "n_eval_rows": int(len(eval_valid)),
                        "eval_min_target_time_s": int(eval_valid["target_time_s"].min()),
                        "eval_max_target_time_s": int(eval_valid["target_time_s"].max()),
                    }
                _, _, cand_score = fit_and_score(train_valid, eval_valid, candidate_cols, target_col, task)
                _, _, mkt_score = fit_and_score(train_valid, eval_valid, MARKET_FEATURES, target_col, task)
                y = eval_valid[target_col].to_numpy(dtype="float64")
                if task == "direction":
                    candidate_eval = {"model": direction_metrics(y, cand_score), "folds": {"frozen_fit": int(len(eval_valid))}}
                    market_eval = {"model": direction_metrics(y, mkt_score), "folds": {"frozen_fit": int(len(eval_valid))}}
                    deltas = {
                        "hit_rate": paired_bootstrap_delta(
                            y, cand_score, mkt_score, metric="hit_rate", n_boot=n_boot, block_len=block_len,
                            seed=20260914 + horizon,
                        ),
                        "ic_spearman": paired_bootstrap_delta(
                            y, cand_score, mkt_score, metric="ic_spearman", n_boot=n_boot, block_len=block_len,
                            seed=20260915 + horizon,
                        ),
                    }
                else:
                    ewma = eval_valid["ewma_vol_60m"].to_numpy(dtype="float64") * np.sqrt(horizon)
                    candidate_eval = {"model": volatility_metrics(y, cand_score, ewma=ewma), "folds": {"frozen_fit": int(len(eval_valid))}}
                    market_eval = {"model": volatility_metrics(y, mkt_score, ewma=ewma), "folds": {"frozen_fit": int(len(eval_valid))}}
                    deltas = {
                        "mse": paired_bootstrap_delta(
                            y, cand_score, mkt_score, metric="mse", n_boot=n_boot, block_len=block_len,
                            seed=20260916 + horizon,
                        )
                    }
                cell[task] = {
                    CANDIDATE_ARM: {"evaluation": candidate_eval},
                    REFERENCE_ARM: {"evaluation": market_eval},
                    DELTA_KEY: deltas,
                }
            # S0: direct Born-rule readout on exactly the same holdout rows.
            target_dir = f"forward_return_{horizon}m"
            eval_s0 = complete_case_rows(asset_frame, candidate_cols, target_dir, "direction")
            s0 = {
                "direction": direction_metrics(
                    eval_s0[target_dir].to_numpy(dtype="float64"),
                    eval_s0[f"schr_p_up_{horizon}m"].to_numpy(dtype="float64") - 0.5,
                ),
                "direction_chance": s0_chance_tests(eval_s0, horizon, n_boot, block_len),
            }
            target_vol = f"forward_vol_{horizon}m"
            eval_vol = complete_case_rows(asset_frame, candidate_cols, target_vol, "volatility")
            valid_vol = eval_vol.dropna(
                subset=[target_vol, f"schr_std_{horizon}m", f"schr_trailing_vol_{horizon}m", "ewma_vol_60m"]
            )
            y_vol = valid_vol[target_vol].to_numpy(dtype="float64")
            pred_vol = (valid_vol[f"schr_std_{horizon}m"] * valid_vol[f"schr_trailing_vol_{horizon}m"]).to_numpy(dtype="float64")
            ewma_vol = (valid_vol["ewma_vol_60m"] * np.sqrt(horizon)).to_numpy(dtype="float64")
            trailing_vol = valid_vol[f"schr_trailing_vol_{horizon}m"].to_numpy(dtype="float64")
            metrics = volatility_metrics(y_vol, pred_vol, ewma=ewma_vol)
            trailing = volatility_metrics(y_vol, trailing_vol)
            metrics["trailing_mse"] = trailing["mse"]
            metrics["trailing_mae"] = trailing["mae"]
            metrics["mse_delta_vs_trailing"] = metrics["mse"] - trailing["mse"]
            metrics["vs_ewma_paired"] = paired_bootstrap_delta(
                y_vol, pred_vol, ewma_vol, metric="mse", n_boot=n_boot, block_len=block_len, seed=20260916 + horizon
            )
            metrics["vs_trailing_paired"] = paired_bootstrap_delta(
                y_vol, pred_vol, trailing_vol, metric="mse", n_boot=n_boot, block_len=block_len, seed=20260917 + horizon
            )
            s0["volatility"] = metrics
            cell["s0"] = s0
            # Shuffled training-target control (BTCUSDT only): fit the candidate
            # arm on permuted training targets, score the same holdout rows.
            if asset == "BTCUSDT":
                control = shuffled_training_control(
                    asset_train, asset_frame, horizon, candidate_cols, n_boot, block_len, holdout_start_s
                )
                results["leakage_checks"]["shuffled_target_controls"][f"{asset}_{horizon}m"] = control
            results["assets"][asset][f"{horizon}m"] = cell
    assert max_target_end <= holdout_start_s, "a training target ends at/after the holdout start"
    results["data"]["causality"]["training_targets_end_before_holdout"] = True
    results["data"]["causality"]["max_training_target_end_time_s"] = int(max_target_end)
    results["diagnostics"] = feature_diagnostics(eval_frame)
    results["ablation_summary"] = _ablation_summary(results)
    results["runtime_seconds"] = time.time() - started
    _print_table(results)
    _print_summary(results["ablation_summary"])
    _merge_write(output_path, results)
    print(f"Wrote {output_path}")
    return results


def shuffled_training_control(
    train_frame: pd.DataFrame,
    eval_frame: pd.DataFrame,
    horizon: int,
    candidate_cols: Sequence[str],
    n_boot: int,
    block_len: int,
    holdout_start_s: int,
) -> dict:
    """Permute the training targets, refit, score the holdout rows (control)."""
    target_col = f"forward_return_{horizon}m"
    train_rows = train_frame.loc[
        train_frame["target_time_s"].to_numpy(dtype="int64") + horizon * 60 <= holdout_start_s
    ]
    train_valid = _valid_frame(train_rows, candidate_cols, target_col, "direction")
    eval_valid = _valid_frame(eval_frame, candidate_cols, target_col, "direction")
    rng = np.random.default_rng(20260914 + horizon * 100)
    shuffled = train_valid.copy()
    shuffled[target_col] = rng.permutation(shuffled[target_col].to_numpy())
    _, _, score = fit_and_score(shuffled, eval_valid, candidate_cols, target_col, "direction")
    metrics = direction_metrics(eval_valid[target_col].to_numpy(dtype="float64"), score)
    return {
        "metrics": metrics,
        "n_train": int(len(train_valid)),
        "n_eval": int(len(eval_valid)),
        "n_boot": int(n_boot),
        "block_len_rows": int(block_len),
    }


def run_frozen(args: argparse.Namespace) -> dict:
    if FROZEN_CONFIG is None and args.config is None:
        raise SystemExit(
            "FROZEN_CONFIG is not set in schrodinger.py yet; run --stage explore first "
            "or pass --config explicitly (not allowed for the pre-registered holdout run)"
        )
    config = FROZEN_CONFIG if args.config is None else _parse_config(args.config)
    assert config is not None
    started = time.time()
    output_path = Path(args.output).resolve()
    audit = market_db_audit(args.db)
    print(f"Frozen config: {config.label}")
    frames, frame_audit = build_anchor_frame(
        args.db,
        anchor_minutes=args.anchor_minutes,
        horizons=HORIZONS,
        configs=[config],
        start_time_s=args.start_time_s,
        end_time_s=args.end_time_s,
        chunk=args.chunk,
    )
    frame = frames[config.label]
    leakage = assert_causal_features(frame, require_gcp=False, require_physics=False)
    print(f"Leakage self-check passed: {leakage}")

    n_boot = args.bootstrap
    results: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "frozen",
        "data": {
            "db": str(Path(args.db).resolve()),
            "source_audit": audit,
            "frame_audit": frame_audit,
            "causality": leakage,
        },
        "protocol": protocol_block(args, n_boot=n_boot),
        "ablation": {
            "reference_arm": REFERENCE_ARM,
            "candidate_arm": CANDIDATE_ARM,
            "delta_key": DELTA_KEY,
            "candidate_config": config.as_dict(),
            "candidate_feature_columns": list(schr_feature_columns(1)),
            "bonferroni_alpha": BONFERRONI_ALPHA,
            "bootstrap": {
                "hit_rate": n_boot,
                "mse": n_boot,
                "ic_spearman": n_boot,
                "block_minutes": block_len_for(args.anchor_minutes),
                "p_value_floor": {
                    "hit_rate": 2.0 / (n_boot + 1.0),
                    "mse": 2.0 / (n_boot + 1.0),
                    "ic_spearman": 2.0 / (n_boot + 1.0),
                },
            },
            "preregistration": "schrodinger_preregistration.md",
        },
        "assets": {},
        "leakage_checks": dict(leakage),
    }
    results["leakage_checks"]["shuffled_target_controls"] = {}
    for asset in ASSETS:
        asset_frame = frame.loc[frame["symbol"].eq(asset)].copy()
        results["assets"][asset] = {}
        for horizon in HORIZONS:
            print(f"Evaluating frozen {config.label} {asset} {horizon}m ...", flush=True)
            candidate_cols = MARKET_FEATURES + schr_feature_columns(horizon)
            cell = evaluate_cell(
                asset_frame,
                horizon,
                args,
                n_boot=n_boot,
                candidate_cols=candidate_cols,
                robustness_block_len=args.robustness_block_len,
            )
            results["assets"][asset][f"{horizon}m"] = cell
            if asset == "BTCUSDT":
                target_col = f"forward_return_{horizon}m"
                pair_valid = (
                    asset_frame.replace([np.inf, -np.inf], np.nan)
                    .dropna(subset=[*candidate_cols, target_col])
                    .copy()
                )
                pair_valid = pair_valid.loc[pair_valid[target_col].ne(0.0)]
                control = shuffled_target_control(
                    pair_valid, candidate_cols, horizon, kind="direction", n_folds=3, seed=20260914 + horizon * 100
                )
                results["leakage_checks"]["shuffled_target_controls"][f"{asset}_{horizon}m"] = control
    results["diagnostics"] = feature_diagnostics(frame)
    results["ablation_summary"] = _ablation_summary(results)
    results["runtime_seconds"] = time.time() - started
    _print_table(results)
    _print_summary(results["ablation_summary"])
    _merge_write(output_path, results)
    print(f"Wrote {output_path}")
    return results


def _print_summary(summary: dict) -> None:
    print(
        f"\nAblation summary: candidate={summary['candidate_arm']} reference={summary['reference_arm']} "
        f"Bonferroni alpha={summary['bonferroni_alpha']:.6f} over {summary['n_cells_expected']} cells"
    )
    for name, metrics in summary["metrics"].items():
        print(
            f"  {name}: improving {metrics['n_improving']}/{metrics['n_cells']}, "
            f"Bonferroni survivors {metrics['n_survivors']}, "
            f"significantly worse {metrics['n_significantly_worse']}, "
            f"median delta {_fmt(metrics['median_delta'], 6)}"
        )
    print(f"  verdict: {summary['verdict']}")


def feature_diagnostics(frame: pd.DataFrame) -> dict:
    """Descriptive Spearman correlations between Schr features and market ones."""
    out: dict[str, Any] = {}
    cols = ["schr_p_up", "schr_std", "schr_mean", "schr_kurt"]
    market = ["ret_1m", "realized_vol_60m"]
    for horizon in HORIZONS:
        block: dict[str, Any] = {}
        for base in cols:
            series = frame[f"{base}_{horizon}m"]
            for other in market:
                pair = frame[[f"{base}_{horizon}m", other]].dropna()
                if len(pair) < 100:
                    block[f"rho_{base}_{other}"] = None
                    continue
                rho = pair[f"{base}_{horizon}m"].rank().corr(pair[other].rank())
                block[f"rho_{base}_{other}"] = float(rho)
        out[f"{horizon}m"] = block
    return out


def _parse_config(label: str) -> SchrodingerConfig:
    parts = label.split("|")
    if len(parts) != 3:
        raise SystemExit(f"config label must be potential|tau_map|k0_map, got {label!r}")
    return SchrodingerConfig(potential=parts[0], tau_map=parts[1], k0_map=parts[2])


def _merge_write(path: Path, block: dict) -> None:
    """Merge a stage block into an existing results JSON (never erase logs)."""
    payload = {}
    if path.exists():
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            payload = {}
    payload.update(block)
    payload["updated_utc"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(_jsonable(payload), indent=2, allow_nan=False) + "\n")


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def _print_table(results: dict) -> None:
    print(f"\nPer-asset x horizon OOS table ({CANDIDATE_ARM} minus {REFERENCE_ARM}):")
    header = "asset h cand_hit [95% CI] cand_IC dHit_pp (p) dIC (p) | vol_MSE EWMA_MSE dMSE (p) | S0_hit S0_IC S0_MSE"
    print(header)
    print("-" * len(header))
    for asset in ASSETS:
        for horizon in HORIZONS:
            cell = results["assets"][asset][f"{horizon}m"]
            d = cell["direction"]
            v = cell["volatility"]
            dm = d[CANDIDATE_ARM]["evaluation"]["model"]
            dd = d[DELTA_KEY]["hit_rate"]
            di = d[DELTA_KEY]["ic_spearman"]
            vm = v[CANDIDATE_ARM]["evaluation"]["model"]
            vd = v[DELTA_KEY]["mse"]
            s0 = cell["s0"]
            ci = dm.get("hit_ci95", [None, None])
            print(
                f"{asset:7s} {horizon:>3d}m {100 * dm['hit_rate']:.3f} "
                f"[{100 * ci[0]:.3f},{100 * ci[1]:.3f}] {_fmt(dm.get('ic_spearman'), 5)} "
                f"{100 * dd['delta']:+.4f} ({_fmt(dd['p_value'])}) "
                f"{_fmt(di['delta'], 5)} ({_fmt(di['p_value'])}) | "
                f"{_fmt(vm.get('mse'), 8)} {_fmt(vm.get('ewma_mse'), 8)} "
                f"{vd['delta']:+.3e} ({_fmt(vd['p_value'])}) | "
                f"{100 * s0['direction']['hit_rate']:.3f} {_fmt(s0['direction']['ic_spearman'], 5)} {_fmt(s0['volatility']['mse'], 8)}"
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("explore", "frozen", "holdout"), default="explore")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--holdout-db", type=Path, default=None, help="holdout source (required for --stage holdout)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--anchor-minutes",
        type=int,
        default=None,
        help="anchor cadence in minutes (default: 60 for explore/frozen, 1 for holdout)",
    )
    parser.add_argument(
        "--train-anchor-minutes",
        type=int,
        default=60,
        help="dev anchor cadence used to fit the frozen models in --stage holdout (the frozen dev cadence)",
    )
    parser.add_argument(
        "--dev-tail-days",
        type=int,
        default=2,
        help="days of dev candles concatenated before the holdout so trailing windows cross the boundary",
    )
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--initial-train-fraction", type=float, default=0.50)
    parser.add_argument("--embargo", type=int, default=None, help="minimum embargo minutes; horizon always enforced")
    parser.add_argument("--bootstrap", type=int, default=4999, help="frozen/holdout bootstrap replicates (floor 2/(n+1))")
    parser.add_argument("--bootstrap-explore", type=int, default=300, help="exploration-stage bootstrap replicates (ranking only)")
    parser.add_argument("--chunk", type=int, default=8192, help="anchor batch size for the PDE batches")
    parser.add_argument(
        "--robustness-block-len",
        type=int,
        default=None,
        help="optional alternative block length (anchor rows) for a descriptive robustness view (holdout: 1440)",
    )
    parser.add_argument("--config", type=str, default=None, help="explicit config label (diagnostics only)")
    parser.add_argument("--start-time-s", type=int, default=None)
    parser.add_argument("--end-time-s", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.stage in ("explore", "frozen"):
        args.anchor_minutes = int(args.anchor_minutes or 60)
    else:
        args.anchor_minutes = int(args.anchor_minutes or 1)
        if args.holdout_db is None:
            raise SystemExit("--holdout-db is required for --stage holdout")
    if args.stage == "explore":
        run_explore(args)
    elif args.stage == "frozen":
        run_frozen(args)
    else:
        run_holdout(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
