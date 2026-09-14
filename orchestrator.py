#!/usr/bin/env python3
"""Periodically refresh and verify the live prediction artifacts.

The orchestrator deliberately composes the existing monitor, evaluator, and
significance commands.  It does not reimplement any scoring or statistical
logic; it only parses their JSON artifacts into an append-only history.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
VENV_PYTHON = HERE / ".venv" / "bin" / "python"

MONITOR = HERE / "monitor.py"
LIVE_EVAL = HERE / "live_eval.py"
SIGNIFICANCE = HERE / "significance.py"
TRACK_RECORD = HERE / "track_record.jsonl"
LIVE_EVAL_OUTPUT = HERE / "live_eval.json"
SIGNIFICANCE_OUTPUT = HERE / "live_edge_significance.json"
VERDICT_HISTORY = HERE / "verdict_history.jsonl"
ORCHESTRATOR_LOG = HERE / "orchestrator.log"
ORCHESTRATOR_LOCK = HERE / ".orchestrator.lock"

DEFAULT_INTERVAL = 1800.0
DEFAULT_MAX_CYCLES = 0

MONITOR_KEYS = ("records_read", "evaluated_rows", "pending_rows")
LIVE_EVAL_KEYS = ("records_read", "evaluated", "pending")
SIGNIFICANCE_KEYS = (
    "evaluated_rows",
    "surviving_both",
    "surviving_vs_chance_count",
    "surviving_vs_persistence_count",
    "windowed_consistent",
    "windowed_marginal",
    "windowed_noise",
    "overall_verdict",
)
WINDOWED_VERDICTS = frozenset(("consistent", "marginal", "noise"))


@dataclass(frozen=True)
class ToolResult:
    """Exit information for one composed tool invocation."""

    tool: str
    exit_code: int
    duration_s: float
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _error_text(exc: BaseException) -> str:
    detail = str(exc).replace("\n", " ").strip()
    return f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__


def _append_json_line(path: str | Path, payload: Mapping[str, Any]) -> None:
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, sort_keys=True, allow_nan=False, separators=(",", ":"))
    with output.open("a", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _log_event(
    tool: str,
    exit_code: int | None,
    duration_s: float,
    *,
    log_path: str | Path = ORCHESTRATOR_LOG,
    error: str | None = None,
) -> None:
    """Append one machine-readable event line to the orchestrator log."""
    event: dict[str, Any] = {
        "timestamp": _utc_now(),
        "tool": tool,
        "exit_code": exit_code,
        "duration_s": round(max(0.0, float(duration_s)), 3),
    }
    if error:
        event["error"] = error
    _append_json_line(log_path, event)


def run_tool(
    tool: str,
    command: Sequence[str | Path],
    *,
    log_path: str | Path = ORCHESTRATOR_LOG,
    cwd: str | Path = PROJECT_ROOT,
) -> ToolResult:
    """Run one existing tool, log its exit event, and never raise on failure."""
    started = time.monotonic()
    exit_code: int
    error: str | None = None
    try:
        completed = subprocess.run(
            [str(argument) for argument in command],
            cwd=str(cwd),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        exit_code = int(completed.returncode)
    except OSError as exc:
        # 127 is the conventional command-not-found status and keeps the
        # failure visible in the same schema as a child process exit.
        exit_code = 127
        error = _error_text(exc)
    duration_s = time.monotonic() - started
    _log_event(tool, exit_code, duration_s, log_path=log_path, error=error)
    return ToolResult(tool, exit_code, duration_s, error)


def _read_last_jsonl(path: str | Path) -> Any:
    """Decode only the final nonblank JSONL line without reading the whole file."""
    source = Path(path).expanduser()
    try:
        with source.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            position = stream.tell()
            if position == 0:
                raise ValueError(f"{source} is empty")
            data = b""
            chunk_size = 64 * 1024
            while position > 0:
                size = min(chunk_size, position)
                position -= size
                stream.seek(position)
                data = stream.read(size) + data

                candidate_data = data.rstrip(b"\r\n")
                while candidate_data:
                    separator = candidate_data.rfind(b"\n")
                    candidate = candidate_data[separator + 1 :].strip()
                    if candidate:
                        try:
                            return json.loads(candidate.decode("utf-8"))
                        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                            raise ValueError(f"malformed JSON in {source}") from exc
                    if separator < 0:
                        break
                    candidate_data = candidate_data[:separator]

                # No complete nonblank line was found yet.  Prepend another
                # chunk; this also handles a single JSON line larger than the
                # normal tail chunk size.
            raise ValueError(f"{source} has no nonblank JSONL line")
    except FileNotFoundError as exc:
        raise ValueError(f"missing artifact: {source}") from exc


def _read_json_object(path: str | Path) -> Mapping[str, Any]:
    source = Path(path).expanduser()
    try:
        with source.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except FileNotFoundError as exc:
        raise ValueError(f"missing artifact: {source}") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"malformed JSON in {source}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON artifact is not an object: {source}")
    return payload


def _required_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} is not an object")
    return value


def _required_int(mapping: Mapping[str, Any], key: str, name: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name}.{key} is not an integer")
    return int(value)


def _required_string(mapping: Mapping[str, Any], key: str, name: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{name}.{key} is not a string")
    return value


def _required_list(mapping: Mapping[str, Any], key: str, name: str) -> list[Any]:
    value = mapping.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{name}.{key} is not an array")
    return value


def parse_monitor_snapshot(path: str | Path = TRACK_RECORD) -> dict[str, int]:
    """Project the newest monitor JSONL line into the history schema."""
    payload = _read_last_jsonl(path)
    mapping = _required_mapping(payload, str(path))
    return {key: _required_int(mapping, key, "monitor_snapshot") for key in MONITOR_KEYS}


def parse_live_eval_summary(path: str | Path = LIVE_EVAL_OUTPUT) -> dict[str, int]:
    """Project the live evaluator's top-level run counts."""
    payload = _read_json_object(path)
    return {key: _required_int(payload, key, "live_eval_summary") for key in LIVE_EVAL_KEYS}


def parse_significance(path: str | Path = SIGNIFICANCE_OUTPUT) -> dict[str, Any]:
    """Project significance counts directly from its cumulative/windowed JSON."""
    payload = _read_json_object(path)
    input_payload = _required_mapping(payload.get("input"), "significance.input")
    surviving = _required_mapping(payload.get("surviving_cells"), "significance.surviving_cells")
    windowed = _required_mapping(payload.get("windowed"), "significance.windowed")
    cells = _required_mapping(windowed.get("cells"), "significance.windowed.cells")

    counts = {"consistent": 0, "marginal": 0, "noise": 0}
    for cell_name, cell_value in cells.items():
        cell = _required_mapping(cell_value, f"significance.windowed.cells[{cell_name!r}]")
        verdict = _required_string(cell, "stability_verdict", f"significance.windowed.cells[{cell_name!r}]")
        if verdict not in WINDOWED_VERDICTS:
            raise ValueError(f"unknown windowed stability verdict: {verdict!r}")
        counts[verdict] += 1

    return {
        "evaluated_rows": _required_int(input_payload, "evaluated_rows", "significance.input"),
        "surviving_both": len(_required_list(surviving, "both", "significance.surviving_cells")),
        "surviving_vs_chance_count": len(
            _required_list(surviving, "vs_chance", "significance.surviving_cells")
        ),
        "surviving_vs_persistence_count": len(
            _required_list(surviving, "vs_persistence", "significance.surviving_cells")
        ),
        "windowed_consistent": counts["consistent"],
        "windowed_marginal": counts["marginal"],
        "windowed_noise": counts["noise"],
        "overall_verdict": _required_string(payload, "overall_verdict", "significance"),
    }


def build_verdict(
    cycle_utc: str,
    cycle_duration_s: float,
    monitor_snapshot: Mapping[str, Any] | None,
    live_eval_summary: Mapping[str, Any] | None,
    significance: Mapping[str, Any] | None,
    *,
    error: str | None = None,
) -> dict[str, Any]:
    """Construct one history record from parsed artifact projections."""
    record: dict[str, Any] = {
        "cycle_utc": cycle_utc,
        "cycle_duration_s": round(max(0.0, float(cycle_duration_s)), 3),
        "monitor_snapshot": dict(monitor_snapshot) if monitor_snapshot is not None else None,
        "live_eval_summary": dict(live_eval_summary) if live_eval_summary is not None else None,
        "significance": dict(significance) if significance is not None else None,
    }
    if error:
        record["error"] = error
    return record


def build_verdict_from_artifacts(
    *,
    cycle_utc: str,
    cycle_duration_s: float,
    track_record_path: str | Path = TRACK_RECORD,
    live_eval_path: str | Path = LIVE_EVAL_OUTPUT,
    significance_path: str | Path = SIGNIFICANCE_OUTPUT,
) -> dict[str, Any]:
    """Parse all three fresh artifacts and construct a successful history line."""
    return build_verdict(
        cycle_utc,
        cycle_duration_s,
        parse_monitor_snapshot(track_record_path),
        parse_live_eval_summary(live_eval_path),
        parse_significance(significance_path),
    )


def append_verdict(record: Mapping[str, Any], path: str | Path = VERDICT_HISTORY) -> None:
    """Append one verdict line; the process lock is held by the caller."""
    _append_json_line(path, record)


def _tool_commands() -> tuple[tuple[str, list[str | Path]], ...]:
    return (
        ("monitor", [VENV_PYTHON, MONITOR, "--once"]),
        ("live_eval", [VENV_PYTHON, LIVE_EVAL, "--once"]),
        ("significance", [VENV_PYTHON, SIGNIFICANCE, "--windowed"]),
    )


def run_cycle(
    *,
    track_record_path: str | Path = TRACK_RECORD,
    live_eval_path: str | Path = LIVE_EVAL_OUTPUT,
    significance_path: str | Path = SIGNIFICANCE_OUTPUT,
    verdict_history_path: str | Path = VERDICT_HISTORY,
    log_path: str | Path = ORCHESTRATOR_LOG,
) -> dict[str, Any]:
    """Run monitor, evaluator, and significance in order and record the result."""
    cycle_started = time.monotonic()
    cycle_utc = _utc_now()
    results: list[ToolResult] = []
    for tool, command in _tool_commands():
        results.append(run_tool(tool, command, log_path=log_path))

    cycle_duration_s = time.monotonic() - cycle_started
    failure_messages = [
        f"{result.tool} exited with code {result.exit_code}"
        + (f" ({result.error})" if result.error else "")
        for result in results
        if not result.ok
    ]
    if failure_messages:
        record = build_verdict(
            cycle_utc,
            cycle_duration_s,
            None,
            None,
            None,
            error="; ".join(failure_messages),
        )
    else:
        try:
            record = build_verdict_from_artifacts(
                cycle_utc=cycle_utc,
                cycle_duration_s=cycle_duration_s,
                track_record_path=track_record_path,
                live_eval_path=live_eval_path,
                significance_path=significance_path,
            )
        except Exception as exc:
            parse_error = _error_text(exc)
            # Parsing is an observable event too; record it without allowing a
            # malformed artifact to terminate the long-lived scheduler.
            _log_event(
                "parse_artifacts",
                1,
                time.monotonic() - cycle_started,
                log_path=log_path,
                error=parse_error,
            )
            record = build_verdict(
                cycle_utc,
                cycle_duration_s,
                None,
                None,
                None,
                error=parse_error,
            )

    append_verdict(record, verdict_history_path)
    return record


@contextmanager
def orchestrator_lock(path: str | Path = ORCHESTRATOR_LOCK) -> Iterator[None]:
    """Hold an exclusive process lock for the entire scheduling loop."""
    lock_path = Path(path).expanduser()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def self_test() -> int:
    """Run network-free parser and subprocess failure-path assertions."""
    failures: list[str] = []

    def check(label: str, condition: bool, detail: Any = None) -> None:
        if condition:
            print(f"PASS {label}")
        else:
            failures.append(label)
            suffix = f": {detail}" if detail is not None else ""
            print(f"FAIL {label}{suffix}")

    with tempfile.TemporaryDirectory(prefix="price-prediction-orchestrator-") as directory:
        root = Path(directory)
        track_path = root / "track_record.jsonl"
        live_path = root / "live_eval.json"
        significance_path = root / "live_edge_significance.json"
        log_path = root / "orchestrator.log"

        track_path.write_text(
            "{\"records_read\": 1, \"evaluated_rows\": 0, \"pending_rows\": 1}\n"
            "{\"records_read\": 17, \"evaluated_rows\": 11, \"pending_rows\": 6}\n",
            encoding="utf-8",
        )
        live_path.write_text(
            json.dumps({"records_read": 23, "evaluated": 14, "pending": 9}) + "\n",
            encoding="utf-8",
        )
        significance_path.write_text(
            json.dumps(
                {
                    "input": {"evaluated_rows": 31},
                    "surviving_cells": {
                        "both": ["BTCUSDT_60m", "SOLUSDT_60m"],
                        "vs_chance": ["BTCUSDT_60m", "SOLUSDT_15m", "SOLUSDT_60m", "ETHUSDT_15m"],
                        "vs_persistence": ["BTCUSDT_60m", "SOLUSDT_60m", "ETHUSDT_60m"],
                    },
                    "windowed": {
                        "cells": {
                            "a": {"stability_verdict": "consistent"},
                            "b": {"stability_verdict": "consistent"},
                            "c": {"stability_verdict": "marginal"},
                            "d": {"stability_verdict": "noise"},
                            "e": {"stability_verdict": "noise"},
                        }
                    },
                    "overall_verdict": "canned verdict",
                }
            )
            + "\n",
            encoding="utf-8",
        )

        try:
            monitor = parse_monitor_snapshot(track_path)
            live_eval = parse_live_eval_summary(live_path)
            significance = parse_significance(significance_path)
            record = build_verdict(
                "2026-01-01T00:00:00+00:00",
                1.25,
                monitor,
                live_eval,
                significance,
            )
            check(
                "canned monitor counts",
                monitor == {"records_read": 17, "evaluated_rows": 11, "pending_rows": 6},
                monitor,
            )
            check(
                "canned live-eval counts",
                live_eval == {"records_read": 23, "evaluated": 14, "pending": 9},
                live_eval,
            )
            check(
                "canned significance counts",
                significance
                == {
                    "evaluated_rows": 31,
                    "surviving_both": 2,
                    "surviving_vs_chance_count": 4,
                    "surviving_vs_persistence_count": 3,
                    "windowed_consistent": 2,
                    "windowed_marginal": 1,
                    "windowed_noise": 2,
                    "overall_verdict": "canned verdict",
                },
                significance,
            )
            check(
                "canned verdict line construction",
                record["monitor_snapshot"] == monitor
                and record["live_eval_summary"] == live_eval
                and record["significance"] == significance
                and record["cycle_duration_s"] == 1.25,
                record,
            )
        except Exception as exc:
            failures.append("canned parser snippets")
            print(f"FAIL canned parser snippets: {_error_text(exc)}")

        try:
            result = run_tool(
                "failure-test",
                [sys.executable, "-c", "import sys; sys.exit(7)"],
                log_path=log_path,
                cwd=PROJECT_ROOT,
            )
            events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            check(
                "nonzero tool failure is contained and logged",
                result.exit_code == 7
                and len(events) == 1
                and events[0]["tool"] == "failure-test"
                and events[0]["exit_code"] == 7
                and "duration_s" in events[0],
                events,
            )
        except Exception as exc:
            failures.append("nonzero tool failure is contained and logged")
            print(f"FAIL nonzero tool failure is contained and logged: {_error_text(exc)}")

    if failures:
        print(f"SELF-TEST FAIL ({len(failures)} assertion(s))")
        return 1
    print("SELF-TEST PASS")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_INTERVAL,
        help="seconds between cycle starts (default: 1800)",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=DEFAULT_MAX_CYCLES,
        help="stop after this many cycles; 0 means unlimited (default: 0)",
    )
    parser.add_argument("--once", action="store_true", help="run one cycle and exit")
    parser.add_argument("--self-test", action="store_true", help="run network-free assertions and exit")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return self_test()
    if args.interval <= 0.0:
        print("--interval must be positive", file=sys.stderr)
        return 2
    if args.max_cycles < 0:
        print("--max-cycles must be nonnegative", file=sys.stderr)
        return 2

    cycle_limit: int | None
    if args.once:
        cycle_limit = 1
    elif args.max_cycles:
        cycle_limit = args.max_cycles
    else:
        cycle_limit = None

    with orchestrator_lock():
        completed = 0
        while cycle_limit is None or completed < cycle_limit:
            started = time.monotonic()
            record = run_cycle()
            completed += 1
            print(json.dumps(record, sort_keys=True, allow_nan=False, separators=(",", ":")), flush=True)
            if cycle_limit is not None and completed >= cycle_limit:
                break
            remaining = args.interval - (time.monotonic() - started)
            if remaining > 0.0:
                try:
                    time.sleep(remaining)
                except KeyboardInterrupt:
                    print("Stopped.", file=sys.stderr, flush=True)
                    break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
