#!/usr/bin/env python3
"""Append cumulative live-evaluation metrics to a JSONL track record."""
from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

# Keep this wrapper from creating bytecode alongside the existing source files.
sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import live_eval  # noqa: E402

LOGGER = logging.getLogger("price_prediction.monitor")
DEFAULT_INTERVAL = 300.0
DEFAULT_TIMEOUT = 10.0
TRACK_RECORD_PATH = HERE / "track_record.jsonl"
LOCK_PATH = HERE / ".track_record.lock"


def _mapping_value(mapping: Any, key: str) -> Any:
    return mapping.get(key) if isinstance(mapping, Mapping) else None


def snapshot_from_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Project one evaluator payload into the append-only monitor schema."""
    prediction_rows = _mapping_value(payload, "prediction_rows") or {}
    scorecard = _mapping_value(payload, "scorecard") or {}
    per_symbol_horizon: dict[str, dict[str, Any]] = {}

    for symbol, horizons in scorecard.items():
        if not isinstance(horizons, Mapping):
            continue
        for horizon, entry in horizons.items():
            if not isinstance(entry, Mapping):
                continue
            arms = _mapping_value(entry, "arms") or {}
            market = _mapping_value(arms, "market_only")
            if not isinstance(market, Mapping):
                continue
            direction = _mapping_value(market, "direction") or {}
            volatility = _mapping_value(market, "volatility") or {}
            baselines = _mapping_value(entry, "baselines") or {}
            persistence = _mapping_value(baselines, "persistence_sign") or {}
            momentum = _mapping_value(baselines, "momentum_sign") or {}
            summary: dict[str, Any] = {
                "n": _mapping_value(market, "n") or 0,
                "hit_rate": _mapping_value(direction, "hit_rate"),
                "persist_hit_rate": _mapping_value(persistence, "hit_rate"),
                "momentum_hit_rate": _mapping_value(momentum, "hit_rate"),
                "pred_mae": _mapping_value(volatility, "mae"),
                "ewma_mae": _mapping_value(volatility, "ewma_mae"),
            }
            with_gcp = _mapping_value(arms, "with_gcp")
            if isinstance(with_gcp, Mapping):
                summary["with_gcp_n"] = _mapping_value(with_gcp, "n") or 0
            per_symbol_horizon[f"{symbol}_{horizon}"] = summary

    snapshot = {
        "generated_utc": _mapping_value(payload, "generated_utc"),
        "records_read": _mapping_value(payload, "records_read") or 0,
        "dedup_rows": _mapping_value(prediction_rows, "deduplicated") or 0,
        "evaluated_rows": _mapping_value(prediction_rows, "evaluated") or 0,
        "pending_rows": _mapping_value(prediction_rows, "pending") or 0,
        "per_symbol_horizon": per_symbol_horizon,
    }
    return live_eval._json_safe(snapshot)


def append_snapshot(
    snapshot: Mapping[str, Any],
    *,
    path: str | Path = TRACK_RECORD_PATH,
    lock_path: str | Path = LOCK_PATH,
) -> None:
    """Append one complete JSON object while holding the shared file lock."""
    output = Path(path).expanduser()
    lock_file = Path(lock_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(snapshot, sort_keys=True, allow_nan=False, separators=(",", ":"))
    with lock_file.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            with output.open("a", encoding="utf-8") as stream:
                stream.write(encoded)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def collect_snapshot(
    predictions_path: str | Path,
    *,
    cache: Any,
    existing: dict[Any, Any],
    fingerprints: dict[Any, str],
) -> dict[str, Any]:
    """Run the existing scorer and return its cumulative summary."""
    payload, _, _ = live_eval.evaluate_predictions(
        predictions_path,
        out_path=None,
        cache=cache,
        existing=existing,
        fingerprints=fingerprints,
    )
    return snapshot_from_payload(payload)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=HERE / "live_predictions.jsonl",
        help="JSONL prediction log",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_INTERVAL,
        help="seconds between continuous polls (default: 300)",
    )
    parser.add_argument("--once", action="store_true", help="append one snapshot and exit")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="Binance request timeout in seconds")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.interval <= 0.0:
        print("--interval must be positive", file=sys.stderr)
        return 2
    if args.timeout <= 0.0:
        print("--timeout must be positive", file=sys.stderr)
        return 2

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
        stream=sys.stderr,
    )
    cache = live_eval.KlineCache(timeout=args.timeout)
    existing: dict[Any, Any] = {}
    fingerprints: dict[Any, str] = {}

    while True:
        try:
            snapshot = collect_snapshot(
                args.predictions,
                cache=cache,
                existing=existing,
                fingerprints=fingerprints,
            )
            append_snapshot(snapshot)
            print(json.dumps(snapshot, sort_keys=True, allow_nan=False, separators=(",", ":")), flush=True)
        except Exception:
            LOGGER.exception("monitor evaluation cycle failed")
            if args.once:
                return 1
        if args.once:
            return 0
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("Stopped.", file=sys.stderr, flush=True)
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
