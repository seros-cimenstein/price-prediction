#!/usr/bin/env python3
"""Score logged live predictions against the realized Binance outcomes.

Semantics mapping (kept next to the evaluator so the live-data loop cannot
silently drift from the registered contract): ``README.md`` lines 130-143
and 155-160 say that ``live.py`` uses closed one-minute candles and predicts
the next minute open; ``live.py`` lines 643-646 set ``target_time_s`` to the
newest closed open plus 60 seconds, while lines 509-517 record the newest
open ``t`` uses ``current_close = close[t-1]``.  The exact target formulas
are in ``features.py`` lines 250-264: for horizon ``h``, realized direction
is ``close[t+h-1] / close[t-1] - 1`` and realized volatility is the square
root of the sum of squared one-minute returns for opens ``t`` through
``t+h-1``.  This evaluator fetches the anchor candle at ``t-1`` and those
``h`` future candles, and never substitutes a different close-to-close
window.  Directional hit rates exclude exact-zero realized returns just as
``backtest.py`` does; IC is reported only with at least ten valid rows.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sqlite3
import sys
import tempfile
import time
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from features import ASSETS, HORIZONS  # noqa: E402

LOGGER = logging.getLogger("price_prediction.live_eval")
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
RETRY_ATTEMPTS = 2
RETRY_BACKOFF_SECONDS = 0.5
CANDLE_MS = 60_000
CANDLE_CLOSE_OFFSET_MS = 59_999
HORIZON_KEYS = {f"{h}m": h for h in HORIZONS}
DEFAULT_KLINE_CACHE = HERE / "kline_cache.sqlite"
DEFAULT_KLINE_SEED_DB = PROJECT_ROOT / "crypto-data" / "holdout_klines_1m.db"
# Persist a candle only once its close time is this far behind the local
# clock, so ordinary clock skew against Binance cannot freeze an open candle.
PERSIST_CLOSE_MARGIN_MS = 10_000
SQLITE_BUSY_TIMEOUT_S = 30.0


class EvaluationError(RuntimeError):
    """Raised when a Binance response cannot be used for one target window."""


@dataclass(frozen=True)
class Candle:
    """The fields needed from one Binance kline."""

    close: float
    close_time_ms: int


@dataclass(frozen=True)
class PredictionRow:
    """One deduplicated ``(symbol, target, horizon)`` log entry."""

    symbol: str
    target_time_s: int
    horizon_key: str
    horizon: int
    values: Mapping[str, Any]
    record: Mapping[str, Any]
    line_number: int

    @property
    def key(self) -> tuple[str, int, str]:
        return (self.symbol, self.target_time_s, self.horizon_key)

    @property
    def record_key(self) -> tuple[str, int]:
        return (self.symbol, self.target_time_s)

    @property
    def fingerprint(self) -> str:
        return json.dumps(self.values, sort_keys=True, separators=(",", ":"), default=str)


@dataclass(frozen=True)
class Outcome:
    """Realized target values using the formulas registered in features.py."""

    forward_return: float
    realized_vol: float


@dataclass(frozen=True)
class EvaluatedRow:
    """Prediction row paired with its realized outcome."""

    prediction: PredictionRow
    outcome: Outcome


@dataclass
class LoadedPredictions:
    rows: list[PredictionRow]
    records_read: int
    invalid_lines: int
    invalid_horizon_rows: int


def _finite(value: Any) -> float | None:
    """Return a finite float, or ``None`` for absent/non-numeric values."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _integer(value: Any) -> int | None:
    """Parse an integral JSON number without accepting fractional targets."""
    number = _finite(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def _safe_error(exc: BaseException) -> str:
    """Keep errors useful without including a request URL or response body."""
    if isinstance(exc, HTTPError):
        return f"HTTP {exc.code}"
    if isinstance(exc, URLError):
        return "network error"
    return type(exc).__name__


def _http_json(url: str, *, timeout: float = 10.0) -> Any:
    """Fetch one JSON response with stdlib networking and no body logging."""
    request = Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "price-prediction-live-eval/1"},
        method="GET",
    )
    with urlopen(request, timeout=timeout) as response:
        body = response.read()
    try:
        return json.loads(body)
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationError("Binance response was not valid JSON") from exc


def _parse_kline_payload(payload: Any, symbol: str) -> dict[int, Candle]:
    """Parse Binance's kline array, retaining valid rows keyed by open time."""
    if not isinstance(payload, list):
        raise EvaluationError(f"Binance {symbol} returned an unexpected payload")
    candles: dict[int, Candle] = {}
    for raw in payload:
        if not isinstance(raw, (list, tuple)) or len(raw) < 7:
            continue
        try:
            open_time_ms = int(raw[0])
            close_time_ms = int(raw[6])
        except (TypeError, ValueError, OverflowError):
            continue
        close = _finite(raw[4])
        if close is None or close_time_ms < open_time_ms:
            continue
        candles[open_time_ms // 1000] = Candle(close=close, close_time_ms=close_time_ms)
    if not candles:
        raise EvaluationError(f"Binance {symbol} returned no valid candles")
    return candles


def fetch_binance_klines(
    symbol: str,
    first_open_s: int,
    last_open_s: int,
    *,
    timeout: float = 10.0,
    fetcher: Callable[..., Any] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[int, Candle]:
    """Fetch the candles opening ``first_open_s..last_open_s``, retrying once.

    For a whole target window the range is the anchor candle at ``t-1``
    through the ``h`` future candles.  ``endTime`` is the open after the last
    required candle and ``limit`` allows that one extra candle, so Binance's
    inclusive ``endTime`` behavior cannot truncate the range.
    """
    if symbol not in ASSETS:
        raise EvaluationError(f"unregistered symbol {symbol!r}")
    first_open_s = int(first_open_s)
    last_open_s = int(last_open_s)
    if last_open_s < first_open_s:
        raise EvaluationError(f"invalid kline range {first_open_s}..{last_open_s}")
    candle_count = (last_open_s - first_open_s) // 60 + 1
    query = urlencode(
        {
            "symbol": symbol,
            "interval": "1m",
            "startTime": first_open_s * 1000,
            "endTime": (last_open_s + 60) * 1000,
            "limit": candle_count + 1,
        }
    )
    url = f"{BINANCE_KLINES_URL}?{query}"
    request_fetcher = _http_json if fetcher is None else fetcher
    last_error: BaseException | None = None
    for attempt in range(RETRY_ATTEMPTS):
        try:
            payload = request_fetcher(url, timeout=timeout)
            return _parse_kline_payload(payload, symbol)
        except Exception as exc:  # network, HTTP, malformed payload, or parse failure
            last_error = exc
            if attempt + 1 < RETRY_ATTEMPTS:
                sleep_fn(RETRY_BACKOFF_SECONDS * (attempt + 1))
    assert last_error is not None
    raise EvaluationError(f"Binance {symbol} request failed after {RETRY_ATTEMPTS} attempts: {_safe_error(last_error)}") from last_error


def _wall_ms() -> int:
    return int(time.time() * 1000)


class KlineStore:
    """Persistent SQLite store of closed Binance candles keyed by ``(symbol, open_time_s)``.

    A closed kline never changes, so a stored row is served instead of HTTP
    forever.  Writes use WAL plus ``INSERT OR IGNORE`` so concurrently running
    evaluators can share one file.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(self.path), timeout=SQLITE_BUSY_TIMEOUT_S)
        self._connection.execute("PRAGMA journal_mode=WAL")
        with self._connection:
            self._connection.execute(
                "CREATE TABLE IF NOT EXISTS klines_1m ("
                "symbol TEXT NOT NULL, open_time_s INTEGER NOT NULL, close REAL NOT NULL, "
                "close_time_ms INTEGER NOT NULL, PRIMARY KEY (symbol, open_time_s)) WITHOUT ROWID"
            )

    def load(self, symbol: str, first_open_s: int, last_open_s: int) -> dict[int, Candle]:
        rows = self._connection.execute(
            "SELECT open_time_s, close, close_time_ms FROM klines_1m "
            "WHERE symbol = ? AND open_time_s BETWEEN ? AND ?",
            (symbol, int(first_open_s), int(last_open_s)),
        )
        return {int(open_time_s): Candle(float(close), int(close_time_ms)) for open_time_s, close, close_time_ms in rows}

    def save(self, symbol: str, candles: Mapping[int, Candle], *, closed_before_ms: int) -> None:
        """Persist only candles whose close time is before ``closed_before_ms``."""
        rows = [
            (symbol, int(open_time_s), candle.close, candle.close_time_ms)
            for open_time_s, candle in candles.items()
            if candle.close_time_ms < closed_before_ms
        ]
        if rows:
            with self._connection:
                self._connection.executemany("INSERT OR IGNORE INTO klines_1m VALUES (?, ?, ?, ?)", rows)

    def close(self) -> None:
        self._connection.close()


class KlineSeed:
    """Read-only ``crypto-data`` klines_1m DB filled from Binance ``/api/v3/klines``.

    ``fetch_holdout.py`` stores ``float(kline[4])`` for closed candles only,
    which is exactly what ``_parse_kline_payload`` keeps; Binance one-minute
    close times are always ``open + 59_999`` ms.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self._connection = sqlite3.connect(f"{self.path.as_uri()}?mode=ro", uri=True, timeout=SQLITE_BUSY_TIMEOUT_S)

    def load(self, symbol: str, first_open_s: int, last_open_s: int) -> dict[int, Candle]:
        rows = self._connection.execute(
            "SELECT open_time_s, close FROM klines_1m WHERE symbol = ? AND open_time_s BETWEEN ? AND ?",
            (symbol, int(first_open_s), int(last_open_s)),
        )
        candles: dict[int, Candle] = {}
        for open_time_s, close in rows:
            value = _finite(close)
            if value is not None:
                candles[int(open_time_s)] = Candle(value, int(open_time_s) * 1000 + CANDLE_CLOSE_OFFSET_MS)
        return candles

    def close(self) -> None:
        self._connection.close()


class KlineCache:
    """Candle cache keyed by ``(symbol, open_time_s)``.

    Lookups go memory, then the optional persistent ``KlineStore``, then the
    optional read-only ``KlineSeed``; only the still-missing open range is
    requested from Binance, and only closed candles are persisted.
    """

    def __init__(
        self,
        *,
        timeout: float = 10.0,
        fetcher: Callable[..., Any] | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        store_path: str | Path | None = None,
        seed_path: str | Path | None = None,
        clock_ms: Callable[[], int] = _wall_ms,
    ) -> None:
        self.timeout = float(timeout)
        self.fetcher = fetcher
        self.sleep_fn = sleep_fn
        self.clock_ms = clock_ms
        self.store = None if store_path is None else KlineStore(store_path)
        self.seed = None if seed_path is None else KlineSeed(seed_path)
        self.candles: dict[tuple[str, int], Candle] = {}
        self.http_fetches = 0
        self._attempted_windows: set[tuple[str, int, int]] = set()
        self._window_errors: dict[tuple[str, int, int], EvaluationError] = {}

    def begin_cycle(self) -> None:
        """Allow a later continuous-loop cycle to retry failed windows."""
        self._attempted_windows.clear()
        self._window_errors.clear()

    def close(self) -> None:
        for source in (self.store, self.seed):
            if source is not None:
                source.close()

    def _missing(self, symbol: str, opens: Iterable[int]) -> list[int]:
        return [open_time_s for open_time_s in opens if (symbol, open_time_s) not in self.candles]

    def ensure_window(self, symbol: str, target_time_s: int, horizon: int) -> None:
        """Load a target window once in this evaluation cycle, fetching only missing candles."""
        window_key = (symbol, int(target_time_s), int(horizon))
        required_opens = _required_open_times(int(target_time_s), int(horizon))
        missing = self._missing(symbol, required_opens)
        if not missing:
            return
        if window_key in self._attempted_windows:
            error = self._window_errors.get(window_key)
            if error is not None:
                raise error
            return
        self._attempted_windows.add(window_key)
        closed_before_ms = self.clock_ms() - PERSIST_CLOSE_MARGIN_MS
        for source in (self.store, self.seed):
            if source is None or not missing:
                continue
            for open_time_s, candle in source.load(symbol, missing[0], missing[-1]).items():
                if candle.close_time_ms < closed_before_ms:
                    self.candles.setdefault((symbol, open_time_s), candle)
            missing = self._missing(symbol, missing)
        if not missing:
            return
        fetch_started_ms = self.clock_ms()
        self.http_fetches += 1
        try:
            fetched = fetch_binance_klines(
                symbol,
                missing[0],
                missing[-1],
                timeout=self.timeout,
                fetcher=self.fetcher,
                sleep_fn=self.sleep_fn,
            )
        except EvaluationError as exc:
            self._window_errors[window_key] = exc
            raise
        for open_time_s, candle in fetched.items():
            self.candles[(symbol, int(open_time_s))] = candle
        if self.store is not None:
            self.store.save(symbol, fetched, closed_before_ms=fetch_started_ms - PERSIST_CLOSE_MARGIN_MS)

    def get(self, symbol: str, open_time_s: int) -> Candle | None:
        return self.candles.get((symbol, int(open_time_s)))


def add_kline_cache_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the shared candle-cache options used by every live CLI."""
    parser.add_argument(
        "--kline-cache",
        type=Path,
        default=DEFAULT_KLINE_CACHE,
        help="persistent SQLite store of closed Binance candles (default: %(default)s)",
    )
    parser.add_argument(
        "--kline-seed-db",
        type=Path,
        default=DEFAULT_KLINE_SEED_DB,
        help="read-only Binance klines_1m DB consulted before HTTP; skipped if absent (default: %(default)s)",
    )
    parser.add_argument(
        "--no-kline-cache",
        action="store_true",
        help="disable the persistent store and seed DB (per-process in-memory cache only)",
    )


def kline_cache_from_args(args: argparse.Namespace, *, timeout: float) -> KlineCache:
    if args.no_kline_cache:
        return KlineCache(timeout=timeout)
    seed_path = args.kline_seed_db if args.kline_seed_db.is_file() else None
    return KlineCache(timeout=timeout, store_path=args.kline_cache, seed_path=seed_path)


def _required_open_times(target_time_s: int, horizon: int) -> range:
    return range(target_time_s - 60, target_time_s + horizon * 60, 60)


def _outcome_for_row(
    prediction: PredictionRow,
    *,
    now_ms: int,
    cache: KlineCache,
) -> tuple[str, Outcome | None, str | None]:
    """Return ``evaluated``, ``pending``, or ``skipped`` for one prediction."""
    target = prediction.target_time_s
    horizon = prediction.horizon
    final_open_s = target + (horizon - 1) * 60
    final_close_ms = final_open_s * 1000 + CANDLE_CLOSE_OFFSET_MS
    # This deterministic gate avoids requesting windows whose final candle is
    # necessarily still open.  We still inspect each Binance close_time below.
    if final_close_ms >= now_ms:
        return "pending", None, "required future candle is not fully closed"
    try:
        cache.ensure_window(prediction.symbol, target, horizon)
    except EvaluationError as exc:
        return "skipped", None, str(exc)

    opens = list(_required_open_times(target, horizon))
    candles = [cache.get(prediction.symbol, open_time_s) for open_time_s in opens]
    if any(candle is None for candle in candles):
        return "skipped", None, "required candle missing from Binance response"
    closed = [candle for candle in candles if candle is not None]
    if any(candle.close_time_ms >= now_ms for candle in closed):
        return "pending", None, "Binance close_time is not earlier than now"
    closes = [candle.close for candle in closed]
    if any(not math.isfinite(close) or close <= 0.0 for close in closes):
        return "skipped", None, "required candle close is not positive and finite"
    anchor = closes[0]
    future_closes = closes[1:]
    one_minute_returns: list[float] = []
    previous = anchor
    for close in future_closes:
        one_minute_returns.append(close / previous - 1.0)
        previous = close
    forward_return = future_closes[-1] / anchor - 1.0
    realized_vol = math.sqrt(sum(value * value for value in one_minute_returns))
    if not math.isfinite(forward_return) or not math.isfinite(realized_vol):
        return "skipped", None, "realized target is not finite"
    return "evaluated", Outcome(forward_return, realized_vol), None


def _read_predictions(path: str | Path) -> LoadedPredictions:
    """Read JSONL and retain the last line for each deduplication key."""
    source = Path(path).expanduser()
    latest: dict[tuple[str, int, str], PredictionRow] = {}
    records_read = 0
    invalid_lines = 0
    invalid_horizon_rows = 0
    with source.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            if not raw_line.strip():
                continue
            records_read += 1
            try:
                record = json.loads(raw_line)
            except (json.JSONDecodeError, TypeError):
                invalid_lines += 1
                continue
            if not isinstance(record, Mapping):
                invalid_lines += 1
                continue
            symbol = record.get("symbol")
            target_time_s = _integer(record.get("target_time_s"))
            horizons = record.get("horizons")
            if not isinstance(symbol, str) or symbol not in ASSETS or target_time_s is None or not isinstance(horizons, Mapping):
                invalid_lines += 1
                continue
            for horizon_key, horizon in HORIZON_KEYS.items():
                values = horizons.get(horizon_key)
                if not isinstance(values, Mapping):
                    invalid_horizon_rows += 1
                    continue
                latest[(symbol, target_time_s, horizon_key)] = PredictionRow(
                    symbol=symbol,
                    target_time_s=target_time_s,
                    horizon_key=horizon_key,
                    horizon=horizon,
                    values=values,
                    record=record,
                    line_number=line_number,
                )
    rows = sorted(latest.values(), key=lambda row: (row.target_time_s, row.symbol, row.horizon))
    return LoadedPredictions(rows, records_read, invalid_lines, invalid_horizon_rows)


def _direction_metrics(rows: Sequence[EvaluatedRow], arm_name: str) -> dict[str, Any]:
    """Score p_up classification and direction_score rank correlation."""
    returns: list[float] = []
    probabilities: list[float] = []
    ic_returns: list[float] = []
    ic_scores: list[float] = []
    for row in rows:
        arm = row.prediction.values.get(arm_name)
        if not isinstance(arm, Mapping):
            continue
        realized = row.outcome.forward_return
        if not math.isfinite(realized) or realized == 0.0:
            continue
        probability = _finite(arm.get("p_up"))
        if probability is not None:
            returns.append(realized)
            probabilities.append(probability)
        score = _finite(arm.get("direction_score"))
        if score is not None:
            ic_returns.append(realized)
            ic_scores.append(score)
    hits = sum((probability > 0.5) == (realized > 0.0) for probability, realized in zip(probabilities, returns))
    n = len(returns)
    ic: float | None = None
    if len(ic_scores) >= 10:
        # scipy returns NaN for constant inputs; that is an undefined IC, not
        # a failed evaluation.  Keep scipy.stats.spearmanr as the registered
        # implementation rather than approximating ranks locally.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            correlation_result = spearmanr(ic_scores, ic_returns)
        correlation = (
            correlation_result.statistic
            if hasattr(correlation_result, "statistic")
            else correlation_result[0]
        )
        if math.isfinite(float(correlation)):
            ic = float(correlation)
    return {
        "n": n,
        "hits": int(hits),
        "hit_rate": float(hits / n) if n else None,
        "ic_spearman": ic,
    }


def _baseline_value(values: Mapping[str, Any], name: str) -> int | None:
    """Read explicit sign fields, with raw-score fallback for old logs."""
    explicit = values.get(name + "_sign")
    if explicit is not None:
        number = _finite(explicit)
        if number is not None:
            return int(np.sign(number))
    raw = _finite(values.get(name))
    return None if raw is None else int(np.sign(raw))


def _baseline_direction_metrics(rows: Sequence[EvaluatedRow], name: str) -> dict[str, Any]:
    actual: list[float] = []
    predicted: list[int] = []
    for row in rows:
        realized = row.outcome.forward_return
        if not math.isfinite(realized) or realized == 0.0:
            continue
        baselines = row.prediction.values.get("baselines")
        if not isinstance(baselines, Mapping):
            continue
        value = _baseline_value(baselines, name)
        if value is None:
            continue
        actual.append(realized)
        predicted.append(value)
    hits = sum((prediction > 0) == (realized > 0.0) for prediction, realized in zip(predicted, actual))
    n = len(actual)
    return {
        "n": n,
        "hits": int(hits),
        "hit_rate": float(hits / n) if n else None,
    }


def _volatility_metrics(rows: Sequence[EvaluatedRow], arm_name: str) -> dict[str, Any]:
    prediction_pairs: list[tuple[float, float]] = []
    ewma_pairs: list[tuple[float, float]] = []
    for row in rows:
        realized = row.outcome.realized_vol
        if not math.isfinite(realized):
            continue
        arm = row.prediction.values.get(arm_name)
        if isinstance(arm, Mapping):
            prediction = _finite(arm.get("predicted_vol"))
            if prediction is not None:
                prediction_pairs.append((prediction, realized))
        baseline = row.prediction.values.get("baselines")
        ewma = _finite(baseline.get("ewma_vol")) if isinstance(baseline, Mapping) else None
        if ewma is not None:
            ewma_pairs.append((ewma, realized))
    mae = (
        float(np.mean([abs(prediction - realized) for prediction, realized in prediction_pairs]))
        if prediction_pairs
        else None
    )
    ewma_mae = (
        float(np.mean([abs(ewma - realized) for ewma, realized in ewma_pairs]))
        if ewma_pairs
        else None
    )
    return {
        "n": len(prediction_pairs),
        "mae": mae,
        "mean_abs_error": mae,
        "ewma_n": len(ewma_pairs),
        "ewma_mae": ewma_mae,
        "ewma_mean_abs_error": ewma_mae,
        "mae_delta_vs_ewma": mae - ewma_mae if mae is not None and ewma_mae is not None else None,
    }


def build_scorecard(rows: Sequence[EvaluatedRow]) -> dict[str, dict[str, dict[str, Any]]]:
    """Aggregate evaluated rows by symbol and horizon, including every arm."""
    grouped: dict[tuple[str, int], list[EvaluatedRow]] = {}
    for row in rows:
        grouped.setdefault((row.prediction.symbol, row.prediction.horizon), []).append(row)
    scorecard: dict[str, dict[str, dict[str, Any]]] = {}
    for symbol in ASSETS:
        symbol_rows = {
            horizon: grouped[(symbol, horizon)]
            for horizon in HORIZONS
            if (symbol, horizon) in grouped
        }
        if not symbol_rows:
            continue
        scorecard[symbol] = {}
        for horizon in HORIZONS:
            grouped_rows = symbol_rows.get(horizon)
            if not grouped_rows:
                continue
            arm_names = ["market_only"]
            if any(isinstance(row.prediction.values.get("with_gcp"), Mapping) for row in grouped_rows):
                arm_names.append("with_gcp")
            if any(isinstance(row.prediction.values.get("with_physics"), Mapping) for row in grouped_rows):
                arm_names.append("with_physics")
            arms: dict[str, Any] = {}
            for arm_name in arm_names:
                available_rows = [
                    row for row in grouped_rows if isinstance(row.prediction.values.get(arm_name), Mapping)
                ]
                arms[arm_name] = {
                    "n": len(available_rows),
                    "direction": _direction_metrics(grouped_rows, arm_name),
                    "volatility": _volatility_metrics(grouped_rows, arm_name),
                }
            persistence = _baseline_direction_metrics(grouped_rows, "persistence")
            momentum = _baseline_direction_metrics(grouped_rows, "momentum")
            market = arms["market_only"]
            scorecard[symbol][f"{horizon}m"] = {
                "n": len(grouped_rows),
                "arms": arms,
                "baselines": {
                    "persistence_sign": persistence,
                    "momentum_sign": momentum,
                },
                # These summaries make the requested per-(symbol,horizon)
                # values easy to consume while ``arms`` retains the complete
                # market-only/with-GCP breakdown.
                "direction": {
                    "model_call": market["direction"],
                    "persistence_sign": persistence,
                    "momentum_sign": momentum,
                },
                "volatility": {"model_call": market["volatility"]},
            }
    return scorecard


def _record_counts(
    rows: Sequence[PredictionRow],
    states: Mapping[tuple[str, int, str], str],
    *,
    invalid_lines: int,
    invalid_horizon_rows: int,
) -> dict[str, int]:
    by_record: dict[tuple[str, int], set[str]] = {}
    for row in rows:
        by_record.setdefault(row.record_key, set()).add(states.get(row.key, "skipped"))
    evaluated_records = sum("evaluated" in values for values in by_record.values())
    pending_records = sum("pending" in values for values in by_record.values())
    skipped_records = sum(
        "evaluated" not in values and "pending" not in values for values in by_record.values()
    )
    evaluated_rows = sum(state == "evaluated" for state in states.values())
    pending_rows = sum(state == "pending" for state in states.values())
    skipped_rows = sum(state == "skipped" for state in states.values())
    return {
        "evaluated": evaluated_records,
        "pending": pending_records,
        "skipped": skipped_records + invalid_lines,
        "evaluated_rows": evaluated_rows,
        "pending_rows": pending_rows,
        "skipped_rows": skipped_rows,
        "invalid_lines": invalid_lines,
        "invalid_horizon_rows": invalid_horizon_rows,
    }


def _json_safe(value: Any) -> Any:
    """Convert numpy scalars and non-finite floats for strict JSON output."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _generated_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_output(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Atomically write strict JSON so a reader never sees a partial scorecard."""
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(_json_safe(payload), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.replace(output)


def evaluate_predictions(
    predictions_path: str | Path,
    *,
    out_path: str | Path | None = None,
    cache: KlineCache | None = None,
    now_ms: int | None = None,
    existing: dict[tuple[str, int, str], EvaluatedRow] | None = None,
    fingerprints: dict[tuple[str, int, str], str] | None = None,
) -> tuple[dict[str, Any], list[EvaluatedRow], LoadedPredictions]:
    """Evaluate currently mature predictions and return payload plus new rows."""
    loaded = _read_predictions(predictions_path)
    cache = KlineCache() if cache is None else cache
    cache.begin_cycle()
    now_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
    ledger = {} if existing is None else existing
    seen_fingerprints = {} if fingerprints is None else fingerprints
    states: dict[tuple[str, int, str], str] = {}
    reasons: dict[str, int] = {}
    newly_evaluated: list[EvaluatedRow] = []
    for prediction in loaded.rows:
        previous = ledger.get(prediction.key)
        if previous is not None and seen_fingerprints.get(prediction.key) == prediction.fingerprint:
            states[prediction.key] = "evaluated"
            continue
        state, outcome, reason = _outcome_for_row(prediction, now_ms=now_ms, cache=cache)
        states[prediction.key] = state
        if reason:
            reasons[reason] = reasons.get(reason, 0) + 1
        if state == "evaluated" and outcome is not None:
            evaluated = EvaluatedRow(prediction, outcome)
            ledger[prediction.key] = evaluated
            seen_fingerprints[prediction.key] = prediction.fingerprint
            newly_evaluated.append(evaluated)
    all_rows = list(ledger.values())
    counts = _record_counts(
        loaded.rows,
        states,
        invalid_lines=loaded.invalid_lines,
        invalid_horizon_rows=loaded.invalid_horizon_rows,
    )
    scorecard = build_scorecard(all_rows)
    prediction_rows = {
        "deduplicated": len(loaded.rows),
        "evaluated": counts["evaluated_rows"],
        "pending": counts["pending_rows"],
        "skipped": counts["skipped_rows"],
    }
    run_metadata = {
        "generated_utc": _generated_utc(),
        "records_read": loaded.records_read,
        "evaluated": counts["evaluated"],
        "pending": counts["pending"],
        "skipped": counts["skipped"],
        "prediction_rows": prediction_rows,
        "invalid_lines": loaded.invalid_lines,
        "invalid_horizon_rows": loaded.invalid_horizon_rows,
        "skip_reasons": reasons,
    }
    payload: dict[str, Any] = {
        **run_metadata,
        "run_metadata": run_metadata,
        "scorecard": scorecard,
    }
    if out_path is not None:
        write_output(out_path, payload)
    return payload, newly_evaluated, loaded


def _metric_text(value: Any, *, percentage: bool = False) -> str:
    number = _finite(value)
    if number is None:
        return "-"
    return f"{number * 100:.2f}%" if percentage else f"{number:.6g}"


def print_scorecard(
    scorecard: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    only: Iterable[tuple[str, str]] | None = None,
    title: str = "Scorecard",
) -> None:
    """Print a compact table suitable for once and continuous operation."""
    selected = set(only) if only is not None else None
    print(title)
    print("symbol horizon arm n hit_rate persist momentum IC pred_MAE ewma_MAE", flush=True)
    printed = 0
    for symbol in ASSETS:
        horizons = scorecard.get(symbol, {})
        for horizon in (f"{h}m" for h in HORIZONS):
            if horizon not in horizons or (selected is not None and (symbol, horizon) not in selected):
                continue
            entry = horizons[horizon]
            baselines = entry.get("baselines", {})
            persistence = baselines.get("persistence_sign", {})
            momentum = baselines.get("momentum_sign", {})
            for arm_name, arm in entry.get("arms", {}).items():
                direction = arm.get("direction", {})
                volatility = arm.get("volatility", {})
                print(
                    f"{symbol:8} {horizon:7} {arm_name:11} "
                    f"{arm.get('n', 0):3d} "
                    f"{_metric_text(direction.get('hit_rate'), percentage=True):>9} "
                    f"{_metric_text(persistence.get('hit_rate'), percentage=True):>8} "
                    f"{_metric_text(momentum.get('hit_rate'), percentage=True):>8} "
                    f"{_metric_text(direction.get('ic_spearman')):>8} "
                    f"{_metric_text(volatility.get('mae')):>9} "
                    f"{_metric_text(volatility.get('ewma_mae')):>9}"
                )
                printed += 1
    if not printed:
        print("(no evaluated predictions yet)")
    print(flush=True)


def _make_synthetic_record(
    target_time_s: int,
    *,
    future_close: float,
    p_up: float,
    predicted_vol: float,
    ewma_vol: float,
    persistence_sign: int,
    momentum_sign: int,
    horizon_key: str = "1m",
    physics_p_up: float | None = None,
) -> dict[str, Any]:
    horizon: dict[str, Any] = {
        "market_only": {
            "direction_score": p_up - 0.5,
            "p_up": p_up,
            "predicted_vol": predicted_vol,
        },
        "baselines": {
            "persistence": float(persistence_sign) * 0.01,
            "persistence_sign": persistence_sign,
            "momentum": float(momentum_sign) * 0.01,
            "momentum_sign": momentum_sign,
            "ewma_vol": ewma_vol,
        },
    }
    if physics_p_up is not None:
        horizon["with_physics"] = {
            "direction_score": physics_p_up - 0.5,
            "p_up": physics_p_up,
            "predicted_vol": predicted_vol,
        }
    return {
        "timestamp": "2026-01-01T00:00:00+00:00",
        "target_time_s": target_time_s,
        "symbol": "BTCUSDT",
        "current_close": 100.0,
        "horizons": {horizon_key: horizon},
    }


def self_test() -> int:
    """Run deterministic, network-free scoring checks over canned klines."""
    base = 1_000_020
    records = [
        _make_synthetic_record(base, future_close=110.0, p_up=0.9, predicted_vol=0.1, ewma_vol=0.2, persistence_sign=1, momentum_sign=-1, physics_p_up=0.9),
        _make_synthetic_record(base + 600, future_close=90.0, p_up=0.1, predicted_vol=0.3, ewma_vol=0.2, persistence_sign=-1, momentum_sign=1, physics_p_up=0.6),
        _make_synthetic_record(base + 1_200, future_close=100.0, p_up=0.9, predicted_vol=0.0, ewma_vol=0.1, persistence_sign=1, momentum_sign=-1, physics_p_up=0.9),
    ]
    # A second horizon checks the multi-minute close-to-close and sum-of-
    # squared-returns formulas rather than only the one-minute case.
    five_target = base + 1_800
    records.append(
        _make_synthetic_record(
            five_target,
            future_close=110.0,
            p_up=0.9,
            predicted_vol=0.0,
            ewma_vol=0.2,
            persistence_sign=1,
            momentum_sign=1,
            horizon_key="5m",
        )
    )
    records[-1]["horizons"]["5m"]["market_only"]["predicted_vol"] = math.sqrt(
        (110.0 / 100.0 - 1.0) ** 2
        + (100.0 / 110.0 - 1.0) ** 2
        + (110.0 / 100.0 - 1.0) ** 2
        + (100.0 / 110.0 - 1.0) ** 2
        + (110.0 / 100.0 - 1.0) ** 2
    )

    canned: dict[tuple[str, int, int], list[list[Any]]] = {}
    for target, close in [(base, 110.0), (base + 600, 90.0), (base + 1_200, 100.0)]:
        canned[("BTCUSDT", target, 1)] = [
            [
                (target - 60) * 1000,
                "0",
                "0",
                "0",
                str(100.0),
                "1",
                (target - 60) * 1000 + CANDLE_CLOSE_OFFSET_MS,
                "0",
                "1",
            ],
            [
                target * 1000,
                "0",
                "0",
                "0",
                str(close),
                "1",
                target * 1000 + CANDLE_CLOSE_OFFSET_MS,
                "0",
                "1",
            ],
        ]
    closes = [100.0, 110.0, 100.0, 110.0, 100.0, 110.0]
    canned[("BTCUSDT", five_target, 5)] = [
        [
            (five_target - 60) * 1000,
            "0",
            "0",
            "0",
            str(100.0),
            "1",
            (five_target - 60) * 1000 + CANDLE_CLOSE_OFFSET_MS,
            "0",
            "1",
        ]
    ] + [
        [
            (five_target + offset * 60) * 1000,
            "0",
            "0",
            "0",
            str(close),
            "1",
            (five_target + offset * 60) * 1000 + CANDLE_CLOSE_OFFSET_MS,
            "0",
            "1",
        ]
        for offset, close in enumerate(closes[1:])
    ]

    def canned_fetcher(url: str, *, timeout: float = 10.0) -> Any:
        del timeout
        query = parse_qs(urlparse(url).query)
        symbol = query["symbol"][0]
        start = int(query["startTime"][0]) // 1000
        horizon = int(query["limit"][0]) - 2
        key = (symbol, start + 60, horizon)
        if key not in canned:
            raise AssertionError(f"unexpected network window {key}")
        return canned[key]

    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="price-prediction-live-eval-") as directory:
        predictions_path = Path(directory) / "live_predictions.jsonl"
        with predictions_path.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record) + "\n")
        cache = KlineCache(fetcher=canned_fetcher, sleep_fn=lambda _: None)
        payload, _, loaded = evaluate_predictions(
            predictions_path,
            cache=cache,
            now_ms=(five_target + 6 * 60) * 1000,
        )
        entry = payload["scorecard"]["BTCUSDT"]["1m"]
        market_direction = entry["arms"]["market_only"]["direction"]
        market_volatility = entry["arms"]["market_only"]["volatility"]
        persistence = entry["baselines"]["persistence_sign"]
        momentum = entry["baselines"]["momentum_sign"]
        five_entry = payload["scorecard"]["BTCUSDT"]["5m"]
        five_volatility = five_entry["arms"]["market_only"]["volatility"]
        physics_direction = entry["arms"]["with_physics"]["direction"]
        def assert_check(condition: bool, detail: Any = None) -> None:
            if not condition:
                raise AssertionError(detail)

        checks: list[tuple[str, Callable[[], None]]] = [
            ("deduplicated rows", lambda: assert_check(loaded.records_read == 4 and len(loaded.rows) == 4, loaded)),
            ("zero return excluded from directional N", lambda: assert_check(market_direction["n"] == 2, market_direction)),
            ("model directional hit-rate", lambda: assert_check(market_direction["hit_rate"] == 1.0, market_direction)),
            (
                "with_physics arm scored from its own values",
                lambda: assert_check(
                    physics_direction["n"] == 2 and physics_direction["hit_rate"] == 0.5,
                    physics_direction,
                ),
            ),
            (
                "record without the physics arm omits it",
                lambda: assert_check("with_physics" not in five_entry["arms"], sorted(five_entry["arms"])),
            ),
            ("persistence hit-rate", lambda: assert_check(persistence["hit_rate"] == 1.0, persistence)),
            ("momentum hit-rate", lambda: assert_check(momentum["hit_rate"] == 0.0, momentum)),
            ("IC skipped below ten rows", lambda: assert_check(market_direction["ic_spearman"] is None, market_direction)),
            (
                "one-minute model MAE",
                lambda: assert_check(
                    math.isclose(market_volatility["mae"], (0.0 + 0.2 + 0.0) / 3.0, rel_tol=0.0, abs_tol=1e-15),
                    market_volatility,
                ),
            ),
            (
                "one-minute EWMA MAE",
                lambda: assert_check(
                    math.isclose(market_volatility["ewma_mae"], (0.1 + 0.1 + 0.1) / 3.0, rel_tol=0.0, abs_tol=1e-15),
                    market_volatility,
                ),
            ),
            ("five-minute realized-vol formula", lambda: assert_check(five_volatility["mae"] == 0.0, five_volatility)),
        ]

        evaluation_now_ms = (five_target + 6 * 60) * 1000
        rows_by_open = {
            (symbol, int(row[0]) // 1000): row for (symbol, _, _), window_rows in canned.items() for row in window_rows
        }
        requested_ranges: list[tuple[int, int]] = []

        def range_fetcher(url: str, *, timeout: float = 10.0) -> Any:
            del timeout
            query = parse_qs(urlparse(url).query)
            symbol = query["symbol"][0]
            start_s = int(query["startTime"][0]) // 1000
            end_s = int(query["endTime"][0]) // 1000
            requested_ranges.append((start_s, end_s))
            return [
                row
                for (row_symbol, open_s), row in sorted(rows_by_open.items())
                if row_symbol == symbol and start_s <= open_s <= end_s
            ]

        def refuse_fetcher(url: str, *, timeout: float = 10.0) -> Any:
            raise AssertionError("candle source should have served this window without HTTP")

        def comparable(result: Mapping[str, Any]) -> str:
            projected = {key: result[key] for key in ("evaluated", "pending", "skipped", "prediction_rows", "skip_reasons", "scorecard")}
            return json.dumps(_json_safe(projected), sort_keys=True)

        def check_store_round_trip() -> None:
            store_path = Path(directory) / "kline_cache.sqlite"
            first = KlineCache(fetcher=canned_fetcher, sleep_fn=lambda _: None, store_path=store_path)
            first_payload, _, _ = evaluate_predictions(predictions_path, cache=first, now_ms=evaluation_now_ms)
            first.close()
            second = KlineCache(fetcher=refuse_fetcher, sleep_fn=lambda _: None, store_path=store_path)
            second_payload, _, _ = evaluate_predictions(predictions_path, cache=second, now_ms=evaluation_now_ms)
            second.close()
            assert_check(
                first.http_fetches == 4
                and second.http_fetches == 0
                and comparable(first_payload) == comparable(payload)
                and comparable(second_payload) == comparable(payload),
                (first.http_fetches, second.http_fetches),
            )

        def check_open_candle_not_persisted_and_missing_span_fetch() -> None:
            store_path = Path(directory) / "open_candle.sqlite"
            # Clock just past the anchor's close: the anchor is closed, the
            # target candle is still open and must stay out of the store.
            early = KlineCache(
                fetcher=range_fetcher,
                sleep_fn=lambda _: None,
                store_path=store_path,
                clock_ms=lambda: base * 1000 + PERSIST_CLOSE_MARGIN_MS,
            )
            early.ensure_window("BTCUSDT", base, 1)
            assert early.store is not None
            persisted = sorted(early.store.load("BTCUSDT", base - 60, base))
            early.close()
            requested_ranges.clear()
            later = KlineCache(fetcher=range_fetcher, sleep_fn=lambda _: None, store_path=store_path)
            later.ensure_window("BTCUSDT", base, 1)
            target_candle = later.get("BTCUSDT", base)
            later.close()
            assert_check(
                persisted == [base - 60]
                and requested_ranges == [(base, base + 60)]
                and target_candle is not None
                and target_candle.close == 110.0,
                (persisted, requested_ranges, target_candle),
            )

        def check_seed_db() -> None:
            seed_path = Path(directory) / "seed_klines_1m.db"
            connection = sqlite3.connect(seed_path)
            with connection:
                connection.execute(
                    "CREATE TABLE klines_1m (symbol TEXT NOT NULL, open_time_s INTEGER NOT NULL, "
                    "close REAL NOT NULL, PRIMARY KEY (symbol, open_time_s))"
                )
                connection.executemany(
                    "INSERT INTO klines_1m VALUES (?, ?, ?)",
                    [("BTCUSDT", base + 540, float("100.0")), ("BTCUSDT", base + 600, float("90.0"))],
                )
            connection.close()
            seeded = KlineCache(fetcher=refuse_fetcher, sleep_fn=lambda _: None, seed_path=seed_path)
            seeded.ensure_window("BTCUSDT", base + 600, 1)
            from_seed = [seeded.get("BTCUSDT", open_s) for open_s in (base + 540, base + 600)]
            seeded.close()
            from_binance = _parse_kline_payload(canned[("BTCUSDT", base + 600, 1)], "BTCUSDT")
            assert_check(
                seeded.http_fetches == 0 and from_seed == [from_binance[base + 540], from_binance[base + 600]],
                (from_seed, from_binance),
            )

        checks.extend(
            [
                ("persistent store serves a second run with zero HTTP and identical scores", check_store_round_trip),
                (
                    "open candle not persisted; only the missing open range is fetched",
                    check_open_candle_not_persisted_and_missing_span_fetch,
                ),
                ("seed DB candles equal parsed Binance candles without HTTP", check_seed_db),
            ]
        )
        for label, check in checks:
            try:
                check()
            except Exception as exc:
                failures.append(label)
                print(f"FAIL {label}: {exc}")
            else:
                print(f"PASS {label}")
    if failures:
        print(f"SELF-TEST FAILED ({len(failures)} assertion(s))")
        return 1
    print("SELF-TEST PASS")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=HERE / "live_predictions.jsonl",
        help="JSONL prediction log",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=HERE / "live_eval.json",
        help="JSON scorecard output",
    )
    parser.add_argument("--loop-interval", type=float, default=60.0, help="continuous polling interval in seconds")
    parser.add_argument("--once", action="store_true", help="evaluate currently mature predictions and exit")
    parser.add_argument("--self-test", action="store_true", help="run network-free scoring self-test and exit")
    parser.add_argument("--timeout", type=float, default=10.0, help="Binance request timeout in seconds")
    add_kline_cache_arguments(parser)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return self_test()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%SZ")
    if args.loop_interval <= 0.0:
        print("--loop-interval must be positive", file=sys.stderr)
        return 2
    cache = kline_cache_from_args(args, timeout=args.timeout)
    existing: dict[tuple[str, int, str], EvaluatedRow] = {}
    fingerprints: dict[tuple[str, int, str], str] = {}
    while True:
        try:
            fetches_before = cache.http_fetches
            payload, newly_evaluated, _ = evaluate_predictions(
                args.predictions,
                out_path=args.out,
                cache=cache,
                existing=existing,
                fingerprints=fingerprints,
            )
            LOGGER.info("Binance kline fetches this cycle: %d", cache.http_fetches - fetches_before)
            only = {(row.prediction.symbol, row.prediction.horizon_key) for row in newly_evaluated}
            if args.once:
                print_scorecard(payload["scorecard"], title="Live outcome scorecard")
            else:
                print_scorecard(payload["scorecard"], only=only if only else set(), title="Live outcome delta scorecard")
            print(
                f"records_read={payload['records_read']} evaluated={payload['evaluated']} "
                f"pending={payload['pending']} skipped={payload['skipped']}",
                flush=True,
            )
        except FileNotFoundError:
            LOGGER.error("prediction log does not exist: %s", args.predictions)
            if args.once:
                return 1
        except Exception:
            LOGGER.exception("live outcome evaluation cycle failed")
            if args.once:
                return 1
        if args.once:
            return 0
        try:
            time.sleep(args.loop_interval)
        except KeyboardInterrupt:
            print("Stopped.", flush=True)
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
