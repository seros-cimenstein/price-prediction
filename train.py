"""Run the pre-registered crypto direction/volatility evaluation.

Usage from the repository root::

    price-prediction/.venv/bin/python price-prediction/train.py

The default run reads the raw SQLite candles and one-second GCP zip archives,
builds causal minute features, performs expanding walk-forward CV with an
embargo, runs a market-only ablation on the same rows, writes results.json,
and stores final full-history joblib artifacts for live inference.
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
    registered_feature_columns,
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
        raise RuntimeError("GCP ablation arms do not share OOS rows")
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
    audit = market_db_audit(args.db)
    print("Market schema/ranges verified:")
    for symbol, row in audit["ranges"].items():
        print(f"  {symbol}: {row['rows']:,} rows {row['min_time_s']}..{row['max_time_s']}")
    print("Building causal features from raw market and GCP files ...", flush=True)
    frame = build_feature_frame(
        args.db,
        args.gcp,
        start_time_s=args.start_time_s,
        end_time_s=args.end_time_s,
        include_gcp=True,
    )
    leakage = assert_causal_features(frame, require_gcp=True)
    print(f"Leakage self-check passed: {leakage}")
    results: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_audit": audit,
        "feature_manifest": feature_manifest(),
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
                full_features = registered_feature_columns(True)
                market_features = registered_feature_columns(False)
                # Full and ablation arms must use the identical complete-case
                # rows.  Otherwise differing missing-GCP months would alter
                # fold boundaries and make a paired delta invalid.
                pair_valid = (
                    asset_frame.replace([np.inf, -np.inf], np.nan)
                    .dropna(subset=[*full_features, target_col])
                    .copy()
                )
                if task == "direction":
                    pair_valid = pair_valid.loc[pair_valid[target_col].ne(0.0)]
                full_pred, full_folds = walk_forward_predict(
                    pair_valid,
                    full_features,
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
                full_pred, market_pred = _same_oos_rows(full_pred, market_pred)
                full_eval = evaluate_predictions(full_pred, horizon, task)
                market_eval = evaluate_predictions(market_pred, horizon, task)
                arm_results["with_gcp"] = {"evaluation": full_eval, "folds": full_folds}
                arm_results["market_only"] = {"evaluation": market_eval, "folds": market_folds}
                y = full_pred[target_col].to_numpy(dtype="float64")
                if task == "direction":
                    arm_results["gcp_delta"] = {
                        "hit_rate": paired_bootstrap_delta(
                            y,
                            full_pred["score"],
                            market_pred["score"],
                            metric="hit_rate",
                            n_boot=args.bootstrap,
                            block_len=max(1, args.block_minutes),
                            seed=20260914 + horizon,
                        ),
                        "ic_spearman": paired_bootstrap_delta(
                            y,
                            full_pred["score"],
                            market_pred["score"],
                            metric="ic_spearman",
                            n_boot=args.bootstrap,
                            block_len=max(1, args.block_minutes),
                            seed=20260915 + horizon,
                        ),
                    }
                else:
                    arm_results["gcp_delta"] = {
                        "mse": paired_bootstrap_delta(
                            y,
                            full_pred["score"],
                            market_pred["score"],
                            metric="mse",
                            n_boot=args.bootstrap,
                            block_len=max(1, args.block_minutes),
                            seed=20260916 + horizon,
                        )
                    }
                # Final full-history artifacts: both arms are retained so a
                # live consumer can inspect whether GCP was helpful.
                artifact_names = {}
                for arm, cols in (("with_gcp", full_features), ("market_only", market_features)):
                    name, n_rows, fit_end = _save_artifact(
                        output_dir, asset, horizon, task, arm, asset_frame, cols
                    )
                    artifact_names[arm] = {"path": name, "fit_rows": n_rows, "fit_end_time_s": fit_end}
                arm_results["artifacts"] = artifact_names
                result_h[task] = arm_results
                if task == "direction" and asset == "BTCUSDT":
                    control = shuffled_target_control(
                        pair_valid,
                        full_features,
                        horizon,
                        kind="direction",
                        n_folds=max(2, min(args.n_folds, 3)),
                        seed=20260914 + horizon * 100,
                    )
                    results["leakage_checks"]["shuffled_target_controls"][f"{asset}_{horizon}m"] = control
            results["assets"][asset][f"{horizon}m"] = result_h
    results["event_study"] = _event_study(frame, Path(args.events))
    results["runtime_seconds"] = time.time() - started
    path = output_dir / "results.json"
    path.write_text(json.dumps(_jsonable(results), indent=2, allow_nan=False) + "\n")
    _print_table(results)
    print(f"Wrote {path}")
    print(f"Saved {len(list(output_dir.glob('model_*.joblib')))} model artifacts")
    return results


def _print_table(results: dict) -> None:
    print("\nPer-asset x horizon OOS table (with-GCP model; deltas are with-GCP minus market-only):")
    header = "asset horizon direction_hit [95% CI] direction_IC GCP_hit_delta (p) | vol_MSE EWMA_MSE GCP_MSE_delta (p)"
    print(header)
    print("-" * len(header))
    for asset in ASSETS:
        for horizon in HORIZONS:
            h = results["assets"][asset][f"{horizon}m"]
            d = h["direction"]
            v = h["volatility"]
            dm = d["with_gcp"]["evaluation"]["model"]
            dd = d["gcp_delta"]["hit_rate"]
            vm = v["with_gcp"]["evaluation"]["model"]
            vd = v["gcp_delta"]["mse"]
            ci = dm.get("hit_ci95", [None, None])
            print(
                f"{asset:7s} {horizon:>3d}m {_fmt_pct(dm.get('hit_rate'))} "
                f"[{_fmt_pct(ci[0])},{_fmt_pct(ci[1])}] "
                f"{_fmt_num(dm.get('ic_spearman'))} "
                f"{_fmt_pct(dd.get('delta'))} ({_fmt_num(dd.get('p_value'))}) | "
                f"{_fmt_num(vm.get('mse'), 6)} {_fmt_num(vm.get('ewma_mse'), 6)} "
                f"{_fmt_num(vd.get('delta'), 6)} ({_fmt_num(vd.get('p_value'))})"
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--gcp", type=Path, default=DEFAULT_GCP)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--initial-train-fraction", type=float, default=0.50)
    parser.add_argument("--embargo", type=int, default=None, help="minimum embargo minutes; horizon is always enforced")
    parser.add_argument("--bootstrap", type=int, default=300, help="paired moving-block bootstrap replicates")
    parser.add_argument("--block-minutes", type=int, default=60)
    parser.add_argument("--start-time-s", type=int, default=None)
    parser.add_argument("--end-time-s", type=int, default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
