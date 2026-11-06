"""Run the pre-registered crypto direction/volatility evaluation.

Usage from the repository root::

    price-prediction/.venv/bin/python price-prediction/train.py
    price-prediction/.venv/bin/python price-prediction/train.py --ablation physics

The default run reads the raw SQLite candles and one-second GCP zip archives,
builds causal minute features, performs expanding walk-forward CV with an
embargo, runs a market-only vs with-GCP ablation on the same rows, writes
results.json, and stores final full-history joblib artifacts for live
inference.  ``--ablation physics`` instead evaluates the pre-registered
physics/geometry analogue family (``physics_preregistration.md``) as
market_only vs with_physics, writes results_physics.json, and stores the
with_physics artifacts; GCP data is not used by that arm.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from backtest import (  # noqa: E402
    direction_metrics,
    evaluate_predictions,
    paired_bootstrap_delta,
    shuffled_target_control,
    volatility_metrics,
    walk_forward_predict,
)
from features import (  # noqa: E402
    ASSETS,
    FEATURES,
    GCP_FEATURES,
    HORIZONS,
    MARKET_FEATURES,
    TARGET_COLUMNS,
    assert_causal_features,
    build_feature_frame,
    feature_columns_for_arm,
    feature_manifest,
    market_db_audit,
)

DEFAULT_DB = HERE.parent / "crypto-data" / "crypto_market.db"
DEFAULT_GCP_ZIPS = HERE.parent / "gcp2-data" / "network_coherence" / "global_network"
DEFAULT_GCP_CACHE = HERE.parent / "crypto-data" / "gcp2_1m.db"
DEFAULT_GCP = DEFAULT_GCP_CACHE if DEFAULT_GCP_CACHE.exists() else DEFAULT_GCP_ZIPS
DEFAULT_EVENTS = HERE.parent / "gcp2-data" / "event_hypothesis_archives" / "events.json"


def _jsonable(value: Any):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def _fmt_num(value: Any, digits: int = 4) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{float(value):.{digits}f}"
def _fmt_pct(value: Any, digits: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{100.0 * float(value):.{digits}f}%"


def _model_for(task: str):
    if task == "direction":
        return LogisticRegression(C=1.0, solver="lbfgs", max_iter=300, random_state=0)
    return Ridge(alpha=1.0)


def _fit_final(frame: pd.DataFrame, feature_cols: Sequence[str], target_col: str, task: str):
    valid = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=[*feature_cols, target_col]).copy()
    if task == "direction":
        valid = valid.loc[valid[target_col].ne(0.0)]
    if valid.empty:
        raise ValueError(f"no valid rows for final {task} fit")
    x = valid.loc[:, feature_cols].to_numpy(dtype="float64")
    y = valid[target_col].to_numpy(dtype="float64")
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    model = _model_for(task)
    if task == "direction":
        model.fit(x_scaled, (y > 0).astype("int8"))
    else:
        model.fit(x_scaled, y)
    return scaler, model, int(valid["target_time_s"].max()), int(len(valid))


def _save_artifact(
    output_dir: Path,
    asset: str,
    horizon: int,
    task: str,
    arm: str,
    frame: pd.DataFrame,
    feature_cols: Sequence[str],
):
    target_col = f"forward_return_{horizon}m" if task == "direction" else f"forward_vol_{horizon}m"
    scaler, model, fit_end, n_rows = _fit_final(frame, feature_cols, target_col, task)
    artifact = {
        "artifact_version": 1,
        "asset": asset,
        "horizon_minutes": horizon,
        "task": task,
        "feature_arm": arm,
        "feature_names": list(feature_cols),
        "scaler": scaler,
        "model": model,
        "fit_rows": n_rows,
        "fit_end_time_s": fit_end,
        "feature_manifest": feature_manifest(),
        "target_column": target_col,
        "target_definition": (
            "sign(close[t+h-1]/close[t-1]-1), with prediction timestamp t at candle open"
            if task == "direction"
            else "sqrt(sum(return[t:t+h-1]^2)), where return[t]=close[t]/close[t-1]-1"
        ),
        "causality": "all registered features use market/GCP observations strictly before target_time_s",
    }
    path = output_dir / f"model_{asset}_{horizon}m_{task}_{arm}.joblib"
    joblib.dump(artifact, path, compress=3)
    return path.name, n_rows, fit_end


def _same_oos_rows(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["target_time_s", "symbol", "fold"]
    aa = a.sort_values(keys).reset_index(drop=True)
    bb = b.sort_values(keys).reset_index(drop=True)
    if len(aa) != len(bb) or not aa[keys].equals(bb[keys]):
        merged = aa.merge(bb[keys], on=keys, how="inner")
        aa = aa.merge(merged[keys], on=keys, how="inner").sort_values(keys).reset_index(drop=True)
        bb = bb.merge(merged[keys], on=keys, how="inner").sort_values(keys).reset_index(drop=True)
    if not aa[keys].equals(bb[keys]):
        raise RuntimeError("ablation arms do not share OOS rows")
    return aa, bb


def _event_study(frame: pd.DataFrame, events_path: Path) -> list[dict]:
    """Descriptive 24-hour BTC volatility around registered event dates."""
    if not events_path.exists():
        return []
    payload = json.loads(events_path.read_text())
    events = payload.get("data", payload if isinstance(payload, list) else [])
    btc = frame.loc[frame["symbol"].eq("BTCUSDT")].copy()
    if btc.empty:
        return []
    times = btc["target_time_s"].to_numpy(dtype="int64")
    vol = btc["forward_vol_60m"].to_numpy(dtype="float64")
    output: list[dict] = []
    for event in events:
        raw_date = event.get("event_date")
        if not raw_date:
            continue
        try:
            dt = pd.Timestamp(raw_date).tz_convert("UTC") if pd.Timestamp(raw_date).tzinfo else pd.Timestamp(raw_date, tz="UTC")
        except Exception:
            continue
        t = int(dt.timestamp())
        pre = vol[(times >= t - 24 * 3600) & (times < t)]
        post = vol[(times >= t) & (times < t + 24 * 3600)]
        pre = pre[np.isfinite(pre)]
        post = post[np.isfinite(post)]
        if len(pre) == 0 or len(post) == 0:
            continue
        pre_median = float(np.median(pre))
        post_median = float(np.median(post))
        output.append(
            {
                "id": event.get("id"),
                "title": event.get("title"),
                "event_date": dt.isoformat(),
                "pre_24h_median_forward_60m_vol": pre_median,
                "post_24h_median_forward_60m_vol": post_median,
                "post_minus_pre": post_median - pre_median,
                "post_pre_ratio": post_median / pre_median if pre_median else None,
                "n_pre": int(len(pre)),
                "n_post": int(len(post)),
            }
        )
    return output


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    physics_ablation = args.ablation == "physics"
    candidate_arm = "with_physics" if physics_ablation else "with_gcp"
    delta_key = "physics_delta" if physics_ablation else "gcp_delta"
    candidate_features = feature_columns_for_arm(candidate_arm)
    market_features = feature_columns_for_arm("market_only")
    results_name = "results_physics.json" if physics_ablation else "results.json"
    bootstrap_hit_mse = args.bootstrap if args.bootstrap_hit_mse is None else int(args.bootstrap_hit_mse)
    bootstrap_ic = args.bootstrap if args.bootstrap_ic is None else int(args.bootstrap_ic)
    audit = market_db_audit(args.db)
    print("Market schema/ranges verified:")
    for symbol, row in audit["ranges"].items():
        print(f"  {symbol}: {row['rows']:,} rows {row['min_time_s']}..{row['max_time_s']}")
    if physics_ablation:
        print("Building causal features from raw market files (physics ablation; GCP unused) ...", flush=True)
    else:
        print("Building causal features from raw market and GCP files ...", flush=True)
    frame = build_feature_frame(
        args.db,
        None if physics_ablation else args.gcp,
        start_time_s=args.start_time_s,
        end_time_s=args.end_time_s,
        include_gcp=not physics_ablation,
    )
    leakage = assert_causal_features(
        frame,
        require_gcp=not physics_ablation,
        require_physics=physics_ablation,
    )
    print(f"Leakage self-check passed: {leakage}")
    results: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_audit": audit,
        "feature_manifest": feature_manifest(),
        "ablation": {
            "reference_arm": "market_only",
            "candidate_arm": candidate_arm,
            "delta_key": delta_key,
            "candidate_feature_columns": list(candidate_features),
            "bonferroni_alpha": 0.05 / float(len(ASSETS) * len(HORIZONS)),
            "bootstrap": {
                "hit_rate": bootstrap_hit_mse,
                "mse": bootstrap_hit_mse,
                "ic_spearman": bootstrap_ic,
                "block_minutes": max(1, args.block_minutes),
                "p_value_floor": {
                    "hit_rate": 2.0 / (bootstrap_hit_mse + 1.0),
                    "mse": 2.0 / (bootstrap_hit_mse + 1.0),
                    "ic_spearman": 2.0 / (bootstrap_ic + 1.0),
                },
            },
            "preregistration": "physics_preregistration.md" if physics_ablation else None,
        },
        "cv": {
            "method": "expanding walk-forward",
            "n_folds": args.n_folds,
            "initial_train_fraction": args.initial_train_fraction,
            "embargo_minutes": "max(horizon, --embargo)",
        },
        "leakage_checks": leakage,
        "assets": {},
        "event_study": [],
    }
    # Controls are intentionally pre-registered as a sanity check, not used to
    # choose features.  One asset at every registered horizon is sufficient to
    # catch accidental target access while keeping the full run manageable.
    results["leakage_checks"]["shuffled_target_controls"] = {}
    for asset in ASSETS:
        asset_frame = frame.loc[frame["symbol"].eq(asset)].copy()
        results["assets"][asset] = {}
        for horizon in HORIZONS:
            print(f"Evaluating {asset} {horizon}m ...", flush=True)
            result_h: dict[str, Any] = {}
            for task in ("direction", "volatility"):
                arm_results: dict[str, Any] = {}
                target_col = f"forward_return_{horizon}m" if task == "direction" else f"forward_vol_{horizon}m"
                # Candidate and reference arms must use the identical
                # complete-case rows.  Otherwise differing missing values
                # would alter fold boundaries and make a paired delta invalid.
                pair_valid = (
                    asset_frame.replace([np.inf, -np.inf], np.nan)
                    .dropna(subset=[*candidate_features, target_col])
                    .copy()
                )
                if task == "direction":
                    pair_valid = pair_valid.loc[pair_valid[target_col].ne(0.0)]
                candidate_pred, candidate_folds = walk_forward_predict(
                    pair_valid,
                    candidate_features,
                    target_col,
                    horizon,
                    task,
                    n_folds=args.n_folds,
                    initial_train_fraction=args.initial_train_fraction,
                    embargo=args.embargo,
                )
                market_pred, market_folds = walk_forward_predict(
                    pair_valid,
                    market_features,
                    target_col,
                    horizon,
                    task,
                    n_folds=args.n_folds,
                    initial_train_fraction=args.initial_train_fraction,
                    embargo=args.embargo,
                )
                candidate_pred, market_pred = _same_oos_rows(candidate_pred, market_pred)
                candidate_eval = evaluate_predictions(candidate_pred, horizon, task)
                market_eval = evaluate_predictions(market_pred, horizon, task)
                arm_results[candidate_arm] = {"evaluation": candidate_eval, "folds": candidate_folds}
                arm_results["market_only"] = {"evaluation": market_eval, "folds": market_folds}
                y = candidate_pred[target_col].to_numpy(dtype="float64")
                if task == "direction":
                    arm_results[delta_key] = {
                        "hit_rate": paired_bootstrap_delta(
                            y,
                            candidate_pred["score"],
                            market_pred["score"],
                            metric="hit_rate",
                            n_boot=bootstrap_hit_mse,
                            block_len=max(1, args.block_minutes),
                            seed=20260914 + horizon,
                        ),
                        "ic_spearman": paired_bootstrap_delta(
                            y,
                            candidate_pred["score"],
                            market_pred["score"],
                            metric="ic_spearman",
                            n_boot=bootstrap_ic,
                            block_len=max(1, args.block_minutes),
                            seed=20260915 + horizon,
                        ),
                    }
                else:
                    arm_results[delta_key] = {
                        "mse": paired_bootstrap_delta(
                            y,
                            candidate_pred["score"],
                            market_pred["score"],
                            metric="mse",
                            n_boot=bootstrap_hit_mse,
                            block_len=max(1, args.block_minutes),
                            seed=20260916 + horizon,
                        )
                    }
                # Final full-history artifacts: both arms are retained so a
                # live consumer can inspect whether the candidate feature
                # family was helpful.
                artifact_names = {}
                for arm, cols in ((candidate_arm, candidate_features), ("market_only", market_features)):
                    name, n_rows, fit_end = _save_artifact(
                        output_dir, asset, horizon, task, arm, asset_frame, cols
                    )
                    artifact_names[arm] = {"path": name, "fit_rows": n_rows, "fit_end_time_s": fit_end}
                arm_results["artifacts"] = artifact_names
                result_h[task] = arm_results
                if task == "direction" and asset == "BTCUSDT":
                    control = shuffled_target_control(
                        pair_valid,
                        candidate_features,
                        horizon,
                        kind="direction",
                        n_folds=max(2, min(args.n_folds, 3)),
                        seed=20260914 + horizon * 100,
                    )
                    results["leakage_checks"]["shuffled_target_controls"][f"{asset}_{horizon}m"] = control
            results["assets"][asset][f"{horizon}m"] = result_h
    results["ablation_summary"] = _ablation_summary(results)
    results["event_study"] = _event_study(frame, Path(args.events))
    results["runtime_seconds"] = time.time() - started
    path = output_dir / results_name
    path.write_text(json.dumps(_jsonable(results), indent=2, allow_nan=False) + "\n")
    _print_table(results)
    _print_ablation_summary(results)
    print(f"Wrote {path}")
    print(f"Saved {len(list(output_dir.glob('model_*.joblib')))} model artifacts")
    return results


def _ablation_summary(results: dict) -> dict:
    """Pre-registered Bonferroni summary for the registered ablation arms.

    The rule is frozen in ``physics_preregistration.md`` (for the physics arm)
    and mirrors the per-cell paired deltas already stored in the results: a
    metric supports a claim only when at least 9 of the 16 cells improve and at
    least 9 improve with a bootstrap p-value below ``0.05/16``.
    """
    ablation = results["ablation"]
    candidate = ablation["candidate_arm"]
    delta_key = ablation["delta_key"]
    alpha = float(ablation["bonferroni_alpha"])
    metrics: dict[str, dict[str, str]] = {
        "hit_rate": {"task": "direction", "better": "higher"},
        "ic_spearman": {"task": "direction", "better": "higher"},
        "mse": {"task": "volatility", "better": "lower"},
    }
    cells: dict[str, list[dict[str, Any]]] = {name: [] for name in metrics}
    for asset in ASSETS:
        for horizon in HORIZONS:
            cell = results["assets"][asset][f"{horizon}m"]
            for name, spec in metrics.items():
                stat = cell[spec["task"]][delta_key].get(name)
                if not isinstance(stat, dict):
                    continue
                delta = stat.get("delta")
                p_value = stat.get("p_value")
                if delta is None or p_value is None or not np.isfinite(delta) or not np.isfinite(p_value):
                    continue
                improves = bool(delta > 0.0) if spec["better"] == "higher" else bool(delta < 0.0)
                worsens = bool(delta < 0.0) if spec["better"] == "higher" else bool(delta > 0.0)
                cells[name].append(
                    {
                        "asset": asset,
                        "horizon_minutes": int(horizon),
                        "delta": float(delta),
                        "p_value": float(p_value),
                        "improves": improves,
                        "worsens": worsens,
                        "survives_bonferroni": bool(improves and p_value < alpha),
                        "significantly_worse": bool(worsens and p_value < alpha),
                    }
                )
    summary: dict[str, Any] = {
        "candidate_arm": candidate,
        "reference_arm": ablation["reference_arm"],
        "bonferroni_alpha": alpha,
        "n_cells_expected": len(ASSETS) * len(HORIZONS),
        "rule": (
            "claim only if a metric has >=9/16 improving cells and >=9/16 improving cells "
            "with paired bootstrap p-value below the Bonferroni threshold"
        ),
        "metrics": {},
    }
    verdicts: dict[str, bool] = {}
    for name, spec in metrics.items():
        items = cells[name]
        improving = [item for item in items if item["improves"]]
        survivors = [item for item in items if item["survives_bonferroni"]]
        worse = [item for item in items if item["significantly_worse"]]
        deltas = [item["delta"] for item in items]
        summary["metrics"][name] = {
            "task": spec["task"],
            "better": spec["better"],
            "n_cells": len(items),
            "n_improving": len(improving),
            "n_survivors": len(survivors),
            "n_significantly_worse": len(worse),
            "median_delta": float(np.median(deltas)) if deltas else None,
            "survivor_cells": [f"{item['asset']}_{item['horizon_minutes']}m" for item in survivors],
            "significantly_worse_cells": [f"{item['asset']}_{item['horizon_minutes']}m" for item in worse],
            "cells": items,
        }
        verdicts[name] = bool(
            len(items) == len(ASSETS) * len(HORIZONS)
            and len(improving) >= 9
            and len(survivors) >= 9
        )
    summary["verdicts"] = verdicts
    summary["physics_adds_direction_skill"] = bool(verdicts["hit_rate"] or verdicts["ic_spearman"])
    summary["physics_adds_volatility_skill"] = bool(verdicts["mse"])
    summary["verdict"] = (
        f"{candidate} adds skill (pre-registered majority + Bonferroni rule met)"
        if (summary["physics_adds_direction_skill"] or summary["physics_adds_volatility_skill"])
        else f"null: no evidence the {candidate} arm adds skill (pre-registered majority + Bonferroni rule not met)"
    )
    return summary


def _print_table(results: dict) -> None:
    ablation = results.get("ablation", {})
    candidate = ablation.get("candidate_arm", "with_gcp")
    delta_key = ablation.get("delta_key", "gcp_delta")
    print(f"\nPer-asset x horizon OOS table ({candidate}; deltas are {candidate} minus market_only):")
    header = (
        "asset horizon cand_hit [95% CI] cand_IC dHit_pp (p) dIC (p) | "
        "vol_MSE EWMA_MSE dMSE (p)"
    )
    print(header)
    print("-" * len(header))
    for asset in ASSETS:
        for horizon in HORIZONS:
            h = results["assets"][asset][f"{horizon}m"]
            d = h["direction"]
            v = h["volatility"]
            dm = d[candidate]["evaluation"]["model"]
            dd = d[delta_key]["hit_rate"]
            di = d[delta_key]["ic_spearman"]
            vm = v[candidate]["evaluation"]["model"]
            vd = v[delta_key]["mse"]
            ci = dm.get("hit_ci95", [None, None])
            print(
                f"{asset:7s} {horizon:>3d}m {_fmt_pct(dm.get('hit_rate'))} "
                f"[{_fmt_pct(ci[0])},{_fmt_pct(ci[1])}] "
                f"{_fmt_num(dm.get('ic_spearman'))} "
                f"{_fmt_pct(dd.get('delta'))} ({_fmt_num(dd.get('p_value'))}) "
                f"{_fmt_num(di.get('delta'), 5)} ({_fmt_num(di.get('p_value'))}) | "
                f"{_fmt_num(vm.get('mse'), 6)} {_fmt_num(vm.get('ewma_mse'), 6)} "
                f"{_fmt_num(vd.get('delta'), 6)} ({_fmt_num(vd.get('p_value'))})"
            )


def _print_ablation_summary(results: dict) -> None:
    summary = results.get("ablation_summary")
    if not summary:
        return
    print(
        f"\nAblation summary: candidate={summary['candidate_arm']} "
        f"reference={summary['reference_arm']} "
        f"Bonferroni alpha={summary['bonferroni_alpha']:.6f} over "
        f"{summary['n_cells_expected']} cells"
    )
    for name, metrics in summary["metrics"].items():
        print(
            f"  {name}: improving {metrics['n_improving']}/{metrics['n_cells']}, "
            f"Bonferroni survivors {metrics['n_survivors']}, "
            f"significantly worse {metrics['n_significantly_worse']}, "
            f"median delta {_fmt_num(metrics['median_delta'], 6)}"
        )
    print(f"  verdict: {summary['verdict']}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--gcp", type=Path, default=DEFAULT_GCP)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument(
        "--ablation",
        choices=("gcp", "physics"),
        default="gcp",
        help=(
            "gcp (default): market_only vs market+GCP, writes results.json; "
            "physics: market_only vs market+PHYSICS_FEATURES, writes results_physics.json"
        ),
    )
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--initial-train-fraction", type=float, default=0.50)
    parser.add_argument("--embargo", type=int, default=None, help="minimum embargo minutes; horizon is always enforced")
    parser.add_argument("--bootstrap", type=int, default=300, help="paired moving-block bootstrap replicates")
    parser.add_argument(
        "--bootstrap-hit-mse",
        type=int,
        default=None,
        help=(
            "replicates for the hit-rate and MSE deltas (default: --bootstrap). "
            "The two-sided p-value floor is 2/(n_boot+1), so thresholds below "
            "2/(n_boot+1) need a larger value (e.g. 4999 for a 0.0004 floor)"
        ),
    )
    parser.add_argument(
        "--bootstrap-ic",
        type=int,
        default=None,
        help="replicates for the Spearman IC delta (default: --bootstrap)",
    )
    parser.add_argument("--block-minutes", type=int, default=60)
    parser.add_argument("--start-time-s", type=int, default=None)
    parser.add_argument("--end-time-s", type=int, default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
