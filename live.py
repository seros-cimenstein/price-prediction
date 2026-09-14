#!/usr/bin/env python3
"""Live, closed-candle inference for the registered crypto models.

The runner intentionally predicts the next Binance one-minute open.  It only
uses klines whose close time has passed, aligns the four assets on their inner
open-time intersection, and delegates feature construction to ``features.py``.
GCP is an optional arm: an unavailable or malformed GCP response never blocks
market-only predictions.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from features import (  # noqa: E402
    ASSETS,
    FEATURES,
    GCP_FEATURES,
    HORIZONS,
    MARKET_FEATURES,
    build_live_feature_frame,
)

LOGGER = logging.getLogger("price_prediction.live")
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
GCP_NETVAR_URL = "https://gcp2.net/api/getcurrentnetvar"
KLINE_LIMIT = 130
MIN_COMMON_MINUTES = 61
GCP_HISTORY_LIMIT = 180
DEFAULT_GUARD_Z_THRESHOLD = 3.0
DEFAULT_MAX_GCP_STALENESS_MINUTES = 120
# Direction artifacts omit exact-zero targets, so their fitted GCP moments can
# differ slightly from the volatility artifacts.  Treat only larger drift as a
# contract mismatch.
GCP_SCALER_RTOL = 1e-3
GCP_SCALER_ATOL = 1e-3
_GCP_SCALER_MISMATCH_WARNED = False

# These are lookup locations only.  No credential is embedded in this file.
DEFAULT_TOKEN_FILES: tuple[Path, ...] = (
    HERE / ".gcp2_token",
    HERE / ".env",
    HERE / "gcp2-collector.env",
    PROJECT_ROOT / ".env",
    PROJECT_ROOT / "gcp_2.0" / ".env",
    PROJECT_ROOT / "gcp_2.0" / "config.py",
    PROJECT_ROOT / "gcp2-collector" / ".env",
    PROJECT_ROOT / "gcp2-collector" / "config.py",
    Path.home() / ".config" / "gcp2" / "token",
)
TOKEN_KEYS = re.compile(
    r"(?i)(?:GCP2?|NETVAR)[A-Z0-9_]*(?:TOKEN|AUTH|KEY)|(?:API|AUTH)[A-Z0-9_]*(?:TOKEN|KEY)"
)
BEARER_TOKEN = re.compile(r"(?i)\bbearer\s+([^\s\"'`},);]+)")
SKIP_PARTS = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "env",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
    }
)


class LiveDataError(RuntimeError):
    """Raised when required market data or model inputs are unusable."""


def _safe_error(exc: BaseException) -> str:
    if isinstance(exc, HTTPError):
        return f"HTTP {exc.code}"
    if isinstance(exc, URLError):
        return f"network error ({type(exc.reason).__name__})"
    return type(exc).__name__


def _http_json(
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    timeout: float = 10.0,
    opener: Callable[..., Any] = urlopen,
) -> Any:
    """Fetch and decode one JSON response without ever logging its body."""
    request_headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/126.0 Safari/537.36"}
    request_headers.update(headers or {})
    request = Request(url, headers=request_headers, method="GET")
    with opener(request, timeout=timeout) as response:
        body = response.read()
    try:
        return json.loads(body)
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LiveDataError("response was not valid JSON") from exc


def _candidate_token_files(explicit: str | Path | None = None) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()

    def add(path: str | Path) -> None:
        candidate = Path(path).expanduser()
        try:
            key = candidate.resolve()
        except OSError:
            key = candidate.absolute()
        if key not in seen:
            seen.add(key)
            paths.append(key)

    if explicit:
        add(explicit)
    env_path = os.environ.get("GCP2_TOKEN_FILE")
    if env_path:
        add(env_path)
    for path in DEFAULT_TOKEN_FILES:
        add(path)

    # Keep discovery narrow and deterministic.  In particular, never recurse
    # through virtualenvs or dependency trees looking for arbitrary secrets.
    roots = (
        PROJECT_ROOT / "gcp_2.0",
        PROJECT_ROOT / "gcp2-collector",
        PROJECT_ROOT / "gcp2-data",
        PROJECT_ROOT / "crypto-data",
    )
    patterns = (
        ".env",
        ".env.*",
        "*.env",
        "*.py",
        "*config*",
        "*settings*",
        "*.yaml",
        "*.yml",
        "*.json",
        "*.toml",
    )
    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        for pattern in patterns:
            for path in root.rglob(pattern):
                if path.is_file() and not (SKIP_PARTS & set(path.parts)):
                    add(path)
    return paths


def _usable_token(value: str | None) -> str | None:
    if value is None:
        return None
    token = value.strip().strip("\"'")
    if token.lower().startswith("bearer "):
        token = token[7:].strip()
    token = token.rstrip(",;)]}'\"}")
    if not token or token.startswith(("<", "${", "…")):
        return None
    if token.upper() in {"TOKEN", "YOUR_TOKEN", "REPLACE_ME", "CHANGEME", "NONE", "NULL"}:
        return None
    return token


def _token_from_text(text: str) -> str | None:
    for line in text.splitlines():
        bearer = BEARER_TOKEN.search(line)
        if bearer:
            token = _usable_token(bearer.group(1))
            if token:
                return token
        if "=" not in line and ":" not in line:
            continue
        key, value = re.split(r"\s*[=:]\s*", line, maxsplit=1)
        if TOKEN_KEYS.search(key):
            token = _usable_token(value.split("#", 1)[0])
            if token:
                return token
    return None


def load_gcp_token(token_file: str | Path | None = None) -> str | None:
    """Load GCP token from ``GCP2_TOKEN`` or a runtime config file.

    The token value is deliberately not returned to logs or persisted output.
    """
    token = _usable_token(os.environ.get("GCP2_TOKEN"))
    if token:
        return token
    for path in _candidate_token_files(token_file):
        try:
            if not path.is_file() or path.stat().st_size > 2_000_000:
                continue
            token = _token_from_text(path.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue
        if token:
            return token
    return None


def fetch_binance_closed_klines(
    symbol: str,
    *,
    now_ms: int | None = None,
    limit: int = KLINE_LIMIT,
    timeout: float = 10.0,
    fetcher: Callable[..., Any] = _http_json,
) -> pd.DataFrame:
    """Fetch Binance 1m klines and retain only candles already closed."""
    if symbol not in ASSETS:
        raise ValueError(f"unregistered symbol {symbol!r}")
    now_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
    query = urlencode({"symbol": symbol, "interval": "1m", "limit": int(limit)})
    try:
        payload = fetcher(f"{BINANCE_KLINES_URL}?{query}", timeout=timeout)
    except Exception as exc:
        raise LiveDataError(f"Binance {symbol} request failed: {_safe_error(exc)}") from exc
    if not isinstance(payload, list):
        raise LiveDataError(f"Binance {symbol} returned an unexpected payload")

    rows: list[tuple[int, float, float, float]] = []
    for raw in payload:
        if not isinstance(raw, (list, tuple)) or len(raw) < 9:
            continue
        try:
            open_ms = int(raw[0])
            close_ms = int(raw[6])
            close = float(raw[4])
            volume = float(raw[5])
            trades = float(raw[8])
        except (TypeError, ValueError, OverflowError):
            continue
        if close_ms >= now_ms:
            continue
        if not np.all(np.isfinite([close, volume, trades])):
            continue
        rows.append((open_ms // 1000, close, volume, trades))
    if not rows:
        raise LiveDataError(f"Binance {symbol} returned no closed candles")

    frame = pd.DataFrame(rows, columns=["open_time_s", "close", "volume", "trades"])
    frame = frame.drop_duplicates("open_time_s", keep="last").sort_values("open_time_s")
    frame.index = pd.to_datetime(frame["open_time_s"], unit="s", utc=True)
    frame.index.name = "time"
    return frame


def _latest_contiguous_run(times: Sequence[int]) -> list[int]:
    if not times:
        return []
    runs: list[tuple[int, int]] = []
    run_start = 0
    for pos in range(1, len(times)):
        if int(times[pos]) - int(times[pos - 1]) != 60:
            runs.append((run_start, pos))
            run_start = pos
    runs.append((run_start, len(times)))
    start, end = max(runs, key=lambda pair: (pair[1] - pair[0], pair[0]))
    return [int(value) for value in times[start:end]]


def fetch_market_candles(
    *,
    symbols: Sequence[str] = ASSETS,
    now_ms: int | None = None,
    timeout: float = 10.0,
    fetcher: Callable[..., Any] = _http_json,
) -> dict[str, pd.DataFrame]:
    """Fetch all registered assets and return their common closed-minute rows."""
    requested = tuple(symbols)
    if requested != ASSETS:
        raise ValueError(f"cross-asset live features require exactly {ASSETS}")
    now_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
    frames = {
        symbol: fetch_binance_closed_klines(
            symbol,
            now_ms=now_ms,
            timeout=timeout,
            fetcher=fetcher,
        )
        for symbol in requested
    }
    common = sorted(set.intersection(*(set(frame["open_time_s"]) for frame in frames.values())))
    common = _latest_contiguous_run(common)
    if len(common) < MIN_COMMON_MINUTES:
        raise LiveDataError(
            f"only {len(common)} common contiguous closed minutes; need {MIN_COMMON_MINUTES}"
        )
    # The API limit is intentionally bounded; keep the most recent contiguous
    # window so rolling feature formulas see the same recent context each cycle.
    common = common[-KLINE_LIMIT:]
    aligned: dict[str, pd.DataFrame] = {}
    for symbol, frame in frames.items():
        part = frame.loc[frame["open_time_s"].isin(common), ["open_time_s", "close", "volume", "trades"]].copy()
        part = part.sort_values("open_time_s")
        part.index = pd.to_datetime(part["open_time_s"], unit="s", utc=True)
        part.index.name = "time"
        aligned[symbol] = part
    return aligned


def load_gcp_history(cache_path: str | Path, limit: int = GCP_HISTORY_LIMIT) -> pd.Series | None:
    """Read recent cached GCP minute means in read-only mode, if available."""
    path = Path(cache_path).expanduser()
    if not path.is_file():
        return None
    try:
        uri = f"file:{path.resolve().as_posix()}?mode=ro"
        with sqlite3.connect(uri, uri=True) as connection:
            rows = pd.read_sql_query(
                "SELECT epoch_s, netvar FROM netvar_1m ORDER BY epoch_s DESC LIMIT ?",
                connection,
                params=(int(limit),),
            )
    except (OSError, sqlite3.Error, pd.errors.DatabaseError):
        return None
    if rows.empty:
        return None
    rows["epoch_s"] = pd.to_numeric(rows["epoch_s"], errors="coerce")
    rows["netvar"] = pd.to_numeric(rows["netvar"], errors="coerce")
    rows = rows.dropna(subset=["epoch_s", "netvar"])
    if rows.empty:
        return None
    index = pd.to_datetime(rows["epoch_s"].astype("int64"), unit="s", utc=True)
    series = pd.Series(rows["netvar"].to_numpy(dtype="float64"), index=index, name="netvar")
    series.index.name = "time"
    return series.sort_index()


def _append_gcp_observation(history: pd.Series | None, value: float, epoch_s: int) -> pd.Series:
    timestamp = pd.Timestamp(int(epoch_s), unit="s", tz="UTC")
    latest = pd.Series([float(value)], index=pd.DatetimeIndex([timestamp]), name="netvar")
    latest.index.name = "time"
    if history is None or history.empty:
        return latest
    merged = pd.concat([history.astype("float64"), latest])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    return merged.iloc[-GCP_HISTORY_LIMIT:]


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def fetch_current_netvar(
    token: str,
    *,
    timeout: float = 10.0,
    fetcher: Callable[..., Any] = _http_json,
) -> float | None:
    """Fetch one current GCP NetVar value, returning None on any API issue."""
    try:
        payload = fetcher(
            GCP_NETVAR_URL,
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout,
        )
    except Exception as exc:
        LOGGER.warning("GCP unavailable (%s); using market-only artifacts", _safe_error(exc))
        return None
    try:
        values = payload["netvar"] if isinstance(payload, dict) else payload
        if isinstance(values, list):
            value = values[0]
            value = value.get("netvar") if isinstance(value, dict) else value
        elif isinstance(values, dict):
            value = values.get("netvar")
        else:
            value = values
        result = _finite(value)
    except (KeyError, IndexError, TypeError, ValueError):
        result = None
    if result is None:
        LOGGER.warning("GCP unavailable (invalid netvar payload); using market-only artifacts")
    return result


def _artifact_path(model_dir: Path, symbol: str, horizon: int, task: str, arm: str) -> Path:
    return model_dir / f"model_{symbol}_{horizon}m_{task}_{arm}.joblib"


def load_artifact(
    model_dir: str | Path,
    symbol: str,
    horizon: int,
    task: str,
    arm: str,
    cache: dict[tuple[str, int, str, str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Load and minimally validate one persisted train.py artifact."""
    key = (symbol, int(horizon), task, arm)
    if cache is not None and key in cache:
        return cache[key]
    path = _artifact_path(Path(model_dir), symbol, horizon, task, arm)
    try:
        artifact = joblib.load(path)
    except (OSError, ValueError, EOFError) as exc:
        raise LiveDataError(f"cannot load model artifact {path.name}: {type(exc).__name__}") from exc
    if not isinstance(artifact, dict) or not isinstance(artifact.get("feature_names"), list):
        raise LiveDataError(f"invalid model artifact {path.name}")
    expected = MARKET_FEATURES if arm == "market_only" else FEATURES
    if tuple(artifact["feature_names"]) != tuple(expected):
        raise LiveDataError(f"feature contract mismatch in {path.name}")
    if artifact.get("asset") != symbol or int(artifact.get("horizon_minutes", -1)) != int(horizon):
        raise LiveDataError(f"model metadata mismatch in {path.name}")
    if artifact.get("task") != task or artifact.get("feature_arm") != arm:
        raise LiveDataError(f"model metadata mismatch in {path.name}")
    if "scaler" not in artifact or "model" not in artifact:
        raise LiveDataError(f"incomplete model artifact {path.name}")
    if cache is not None:
        cache[key] = artifact
    return artifact


def _empty_gcp_guard(*, gcp_available: bool = False) -> dict[str, Any]:
    return {
        "in_distribution": False,
        "cache_stale": False,
        "max_abs_z": None,
        "stale_minutes": None,
        "suppressed": bool(gcp_available),
    }


def _load_with_gcp_artifacts(
    model_dir: str | Path,
    model_cache: dict[tuple[str, int, str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Load every available with-GCP artifact for scaler consistency checks."""
    artifacts: list[dict[str, Any]] = []
    root = Path(model_dir)
    for symbol in ASSETS:
        for horizon in HORIZONS:
            for task in ("direction", "volatility"):
                path = _artifact_path(root, symbol, horizon, task, "with_gcp")
                if path.is_file():
                    artifacts.append(
                        load_artifact(
                            root,
                            symbol,
                            horizon,
                            task,
                            "with_gcp",
                            model_cache,
                        )
                    )
    if not artifacts:
        raise LiveDataError(f"no with-GCP model artifacts found in {root}")
    return artifacts


def _evaluate_gcp_guard(
    *,
    feature_frame: pd.DataFrame,
    target_time_s: int,
    current_netvar_time_s: int,
    gcp_history: pd.Series | None,
    artifacts: Sequence[Mapping[str, Any]],
    gcp_available: bool,
    guard_z_threshold: float,
    max_gcp_staleness_minutes: int,
) -> dict[str, Any]:
    """Evaluate fitted GCP scaler distance and cache age for one target row."""
    guard = _empty_gcp_guard(gcp_available=gcp_available)

    if gcp_history is not None and not gcp_history.empty:
        try:
            history_index = pd.to_datetime(gcp_history.index, utc=True, errors="coerce")
            history_index = history_index[~history_index.isna()]
            if len(history_index):
                newest_epoch_s = int(history_index.max().value // 1_000_000_000)
                guard["stale_minutes"] = max(
                    0,
                    int((int(current_netvar_time_s) - newest_epoch_s) // 60),
                )
                guard["cache_stale"] = bool(
                    guard["stale_minutes"] > int(max_gcp_staleness_minutes)
                )
        except (AttributeError, TypeError, ValueError, OverflowError):
            # An unreadable cache cannot establish an age, but the feature
            # finite-value check below still protects the with-GCP arm.
            guard["stale_minutes"] = None
            guard["cache_stale"] = False

    if not artifacts:
        guard["suppressed"] = bool(gcp_available)
        return guard

    first = artifacts[0]
    first_names = tuple(
        name for name in first.get("feature_names", ()) if name in GCP_FEATURES
    )
    if not first_names:
        guard["suppressed"] = bool(gcp_available)
        return guard
    try:
        first_indices = [first["feature_names"].index(name) for name in first_names]
        first_scaler = first["scaler"]
        reference_mean = np.asarray(first_scaler.mean_, dtype="float64")[first_indices]
        reference_scale = np.asarray(first_scaler.scale_, dtype="float64")[first_indices]
    except (AttributeError, KeyError, TypeError, ValueError, IndexError):
        guard["suppressed"] = bool(gcp_available)
        return guard

    global _GCP_SCALER_MISMATCH_WARNED
    scaler_mismatch = False
    if (
        reference_mean.shape != (len(first_names),)
        or reference_scale.shape != (len(first_names),)
        or not np.isfinite(reference_mean).all()
        or not np.isfinite(reference_scale).all()
        or np.any(reference_scale <= 0.0)
    ):
        scaler_mismatch = True
    for artifact in artifacts[1:]:
        names = tuple(
            name for name in artifact.get("feature_names", ()) if name in GCP_FEATURES
        )
        try:
            indices = [artifact["feature_names"].index(name) for name in first_names]
            scaler = artifact["scaler"]
            mean = np.asarray(scaler.mean_, dtype="float64")[indices]
            scale = np.asarray(scaler.scale_, dtype="float64")[indices]
        except (AttributeError, KeyError, TypeError, ValueError, IndexError):
            scaler_mismatch = True
            continue
        if names != first_names or not (
            np.allclose(
                reference_mean,
                mean,
                rtol=GCP_SCALER_RTOL,
                atol=GCP_SCALER_ATOL,
            )
            and np.allclose(
                reference_scale,
                scale,
                rtol=GCP_SCALER_RTOL,
                atol=GCP_SCALER_ATOL,
            )
        ):
            scaler_mismatch = True
    if scaler_mismatch and not _GCP_SCALER_MISMATCH_WARNED:
        LOGGER.warning(
            "with-GCP artifact scalers differ beyond tolerance; using the first scaler for the guard"
        )
        _GCP_SCALER_MISMATCH_WARNED = True

    target_rows = feature_frame.loc[
        feature_frame["target_time_s"].eq(int(target_time_s))
    ]
    if not target_rows.empty and set(first_names).issubset(target_rows.columns):
        try:
            values = target_rows.loc[:, list(first_names)].to_numpy(dtype="float64")
            if np.isfinite(values).all() and np.isfinite(reference_mean).all() and np.all(
                reference_scale > 0.0
            ):
                z = (values - reference_mean) / reference_scale
                max_abs_z = float(np.max(np.abs(z)))
                if math.isfinite(max_abs_z):
                    guard["max_abs_z"] = max_abs_z
                    guard["in_distribution"] = bool(
                        max_abs_z <= float(guard_z_threshold)
                    )
        except (TypeError, ValueError):
            pass
    guard["suppressed"] = bool(
        gcp_available and (not guard["in_distribution"] or guard["cache_stale"])
    )
    return guard


def _model_input(artifact: Mapping[str, Any], row: pd.Series) -> np.ndarray | None:
    names = artifact["feature_names"]
    try:
        values = row.loc[names].to_numpy(dtype="float64").reshape(1, -1)
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite(values).all():
        return None
    return artifact["scaler"].transform(values)


def _direction_prediction(artifact: Mapping[str, Any], row: pd.Series) -> dict[str, float] | None:
    x = _model_input(artifact, row)
    if x is None:
        return None
    model = artifact["model"]
    if hasattr(model, "decision_function"):
        score = _finite(np.asarray(model.decision_function(x)).reshape(-1)[0])
    else:
        score = None
    if hasattr(model, "predict_proba"):
        p_up = _finite(np.asarray(model.predict_proba(x)).reshape(1, -1)[0, -1])
    elif score is not None:
        p_up = 1.0 / (1.0 + math.exp(-max(-700.0, min(700.0, score))))
    else:
        p_up = None
    if score is None or p_up is None:
        return None
    return {"direction_score": score, "p_up": p_up}


def _volatility_prediction(artifact: Mapping[str, Any], row: pd.Series) -> float | None:
    x = _model_input(artifact, row)
    if x is None:
        return None
    try:
        prediction = np.asarray(artifact["model"].predict(x)).reshape(-1)[0]
    except (TypeError, ValueError):
        return None
    return _finite(prediction)


def _sign(value: Any) -> int | None:
    number = _finite(value)
    if number is None:
        return None
    return int(np.sign(number))


def _row_for_target(frame: pd.DataFrame, symbol: str, target_time_s: int) -> pd.Series:
    rows = frame.loc[(frame["symbol"].eq(symbol)) & (frame["target_time_s"].eq(int(target_time_s)))]
    if len(rows) != 1:
        raise LiveDataError(f"feature row missing for {symbol} at target {target_time_s}")
    row = rows.iloc[0]
    source = _finite(row.get("feature_source_max_time_s"))
    if source is None or not source < int(target_time_s):
        raise LiveDataError(f"non-causal live feature row for {symbol}")
    return row


def _record_predictions(
    *,
    frame: pd.DataFrame,
    market: Mapping[str, pd.DataFrame],
    target_time_s: int,
    timestamp: str,
    asof_time_s: int,
    model_dir: str | Path,
    output_symbols: Sequence[str],
    gcp_enabled: bool,
    model_cache: dict[tuple[str, int, str, str], dict[str, Any]],
    gcp_value: float | None,
    gcp_available: bool | None = None,
    gcp_guard: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    gcp_available = bool(gcp_enabled if gcp_available is None else gcp_available)
    if gcp_guard is None:
        guard_input: Mapping[str, Any] = _empty_gcp_guard(gcp_available=gcp_available)
        guard_input = {
            **guard_input,
            "in_distribution": bool(gcp_enabled),
            "suppressed": bool(gcp_available and not gcp_enabled),
        }
    else:
        guard_input = gcp_guard
    raw_stale_minutes = guard_input.get("stale_minutes")
    try:
        stale_minutes = None if raw_stale_minutes is None else int(raw_stale_minutes)
    except (TypeError, ValueError, OverflowError):
        stale_minutes = None
    guard_record = {
        "in_distribution": bool(guard_input.get("in_distribution", False)),
        "cache_stale": bool(guard_input.get("cache_stale", False)),
        "max_abs_z": _finite(guard_input.get("max_abs_z")),
        "stale_minutes": stale_minutes,
        "suppressed": bool(
            guard_input.get(
                "suppressed",
                gcp_available
                and (
                    not bool(guard_input.get("in_distribution", False))
                    or bool(guard_input.get("cache_stale", False))
                ),
            )
        ),
    }
    # Never allow a caller to emit the arm while the recorded guard says it is
    # suppressed, even if the two inputs were assembled inconsistently.
    with_gcp_enabled = bool(
        gcp_enabled
        and guard_record["in_distribution"]
        and not guard_record["cache_stale"]
        and not guard_record["suppressed"]
    )
    for symbol in output_symbols:
        row = _row_for_target(frame, symbol, target_time_s)
        current_close = _finite(market[symbol].iloc[-1]["close"])
        if current_close is None:
            raise LiveDataError(f"current close missing for {symbol}")
        record: dict[str, Any] = {
            "timestamp": timestamp,
            "asof_time_s": int(asof_time_s),
            "target_time_s": int(target_time_s),
            "symbol": symbol,
            "current_close": current_close,
            "gcp_available": gcp_available,
            "gcp_guard": dict(guard_record),
            "horizons": {},
        }
        if gcp_available and gcp_value is not None:
            record["gcp_netvar"] = gcp_value
        persistence = _finite(row.get("ret_1m"))
        momentum = _finite(row.get("ret_15m"))
        ewma_base = _finite(row.get("ewma_vol_60m"))
        for horizon in HORIZONS:
            direction_market = _direction_prediction(
                load_artifact(model_dir, symbol, horizon, "direction", "market_only", model_cache),
                row,
            )
            volatility_market = _volatility_prediction(
                load_artifact(model_dir, symbol, horizon, "volatility", "market_only", model_cache),
                row,
            )
            if direction_market is None or volatility_market is None:
                raise LiveDataError(f"market-only feature row is not finite for {symbol} {horizon}m")
            market_arm = {
                **direction_market,
                "predicted_vol": volatility_market,
            }
            horizon_record: dict[str, Any] = {
                "market_only": market_arm,
                "baselines": {
                    # Raw scores match backtest.baseline_scores; *_sign fields
                    # make the implied direction explicit for consumers.
                    "persistence": persistence,
                    "persistence_sign": _sign(persistence),
                    "momentum": momentum,
                    "momentum_sign": _sign(momentum),
                    "ewma_vol": ewma_base * math.sqrt(horizon)
                    if ewma_base is not None
                    else None,
                },
            }
            if with_gcp_enabled:
                direction_gcp = _direction_prediction(
                    load_artifact(model_dir, symbol, horizon, "direction", "with_gcp", model_cache),
                    row,
                )
                volatility_gcp = _volatility_prediction(
                    load_artifact(model_dir, symbol, horizon, "volatility", "with_gcp", model_cache),
                    row,
                )
                if direction_gcp is not None and volatility_gcp is not None:
                    horizon_record["with_gcp"] = {
                        **direction_gcp,
                        "predicted_vol": volatility_gcp,
                    }
            record["horizons"][f"{horizon}m"] = horizon_record
        records.append(record)
    if with_gcp_enabled:
        paired = [
            (
                record["horizons"][f"{horizon}m"]["market_only"],
                record["horizons"][f"{horizon}m"].get("with_gcp"),
            )
            for record in records
            for horizon in HORIZONS
        ]
        paired = [(market_arm, gcp_arm) for market_arm, gcp_arm in paired if gcp_arm is not None]
        if not paired:
            raise LiveDataError("GCP endpoint was live but no finite with-GCP predictions were produced")
        # A live GCP arm must be observably distinct from the ablation arm; this
        # catches accidentally loading market-only artifacts twice.
        if all(
            np.array_equal(
                np.asarray(
                    [market_arm["direction_score"], market_arm["predicted_vol"]],
                    dtype="float64",
                ),
                np.asarray(
                    [gcp_arm["direction_score"], gcp_arm["predicted_vol"]],
                    dtype="float64",
                ),
            )
            for market_arm, gcp_arm in paired
        ):
            raise AssertionError("with-GCP and market-only scores are bitwise-identical")
    return records


def append_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> None:
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")


def _print_records(records: Sequence[Mapping[str, Any]]) -> None:
    for record in records:
        print(
            f"{record['symbol']} close={record['current_close']:.8g} "
            f"target={record['target_time_s']} gcp={'on' if record['gcp_available'] else 'off'}"
        )
        for horizon, values in record["horizons"].items():
            market = values["market_only"]
            suffix = ""
            if "with_gcp" in values:
                gcp = values["with_gcp"]
                suffix = f" gcp(score={gcp['direction_score']:+.5f},vol={gcp['predicted_vol']:.6g})"
            print(
                f"  {horizon}: market(score={market['direction_score']:+.5f},"
                f"p_up={market['p_up']:.4f},vol={market['predicted_vol']:.6g}){suffix}"
            )


def run_cycle(
    *,
    model_dir: str | Path,
    log_file: str | Path,
    gcp_cache: str | Path,
    token_file: str | Path | None = None,
    output_symbol: str | None = None,
    timeout: float = 10.0,
    gcp_history: pd.Series | None = None,
    model_cache: dict[tuple[str, int, str, str], dict[str, Any]] | None = None,
    market_fetcher: Callable[..., Any] = _http_json,
    gcp_fetcher: Callable[..., Any] = _http_json,
    guard_z_threshold: float = DEFAULT_GUARD_Z_THRESHOLD,
    max_gcp_staleness_minutes: int = DEFAULT_MAX_GCP_STALENESS_MINUTES,
) -> tuple[list[dict[str, Any]], pd.Series | None]:
    """Fetch, score, print, and append one live cycle."""
    model_cache = {} if model_cache is None else model_cache
    now_ms = int(time.time() * 1000)
    market = fetch_market_candles(now_ms=now_ms, timeout=timeout, fetcher=market_fetcher)
    latest_open_s = max(int(frame["open_time_s"].iloc[-1]) for frame in market.values())
    target_time_s = latest_open_s + 60
    token = load_gcp_token(token_file)
    netvar: float | None = None
    if token:
        netvar = fetch_current_netvar(token, timeout=timeout, fetcher=gcp_fetcher)
    else:
        LOGGER.warning("GCP unavailable (no token configured); using market-only artifacts")

    # Keep the in-memory series (including observations fetched by this loop)
    # for feature construction, but inspect the read-only cache itself for the
    # staleness flag on every cycle.
    cache_history = load_gcp_history(gcp_cache)
    if gcp_history is None:
        gcp_history = cache_history
    gcp_series = None
    if netvar is not None:
        # Use the newest GCP observation as the last closed minute, so it is
        # strictly before the target open just like the newest Binance candle.
        gcp_series = _append_gcp_observation(gcp_history, netvar, target_time_s - 60)
    feature_frame = build_live_feature_frame(market, target_time_s, gcp=gcp_series)
    gcp_available = netvar is not None
    target_rows = feature_frame.loc[feature_frame["target_time_s"].eq(target_time_s)]
    gcp_features_finite = bool(
        gcp_available
        and gcp_series is not None
        and set(GCP_FEATURES).issubset(target_rows.columns)
        and not target_rows.empty
        and np.isfinite(target_rows.loc[:, list(GCP_FEATURES)].to_numpy(dtype="float64")).all()
    )
    all_features_finite = bool(
        gcp_features_finite
        and set(FEATURES).issubset(target_rows.columns)
        and np.isfinite(target_rows.loc[:, list(FEATURES)].to_numpy(dtype="float64")).all()
    )
    gcp_guard = _empty_gcp_guard(gcp_available=gcp_available)
    if gcp_available:
        with_gcp_artifacts = (
            _load_with_gcp_artifacts(model_dir, model_cache)
            if gcp_features_finite
            else []
        )
        gcp_guard = _evaluate_gcp_guard(
            feature_frame=feature_frame,
            target_time_s=target_time_s,
            current_netvar_time_s=target_time_s - 60,
            gcp_history=cache_history if cache_history is not None else gcp_history,
            artifacts=with_gcp_artifacts,
            gcp_available=gcp_available,
            guard_z_threshold=guard_z_threshold,
            max_gcp_staleness_minutes=max_gcp_staleness_minutes,
        )
        if gcp_guard["suppressed"]:
            LOGGER.warning(
                "GCP guard suppressed with-GCP arm (in_distribution=%s, cache_stale=%s, max_abs_z=%s, stale_minutes=%s)",
                gcp_guard["in_distribution"],
                gcp_guard["cache_stale"],
                gcp_guard["max_abs_z"],
                gcp_guard["stale_minutes"],
            )
    gcp_enabled = bool(
        gcp_available
        and all_features_finite
        and gcp_guard["in_distribution"]
        and not gcp_guard["cache_stale"]
    )
    output_symbols = (output_symbol,) if output_symbol else ASSETS
    timestamp = datetime.now(timezone.utc).isoformat()
    records = _record_predictions(
        frame=feature_frame,
        market=market,
        target_time_s=target_time_s,
        timestamp=timestamp,
        asof_time_s=int(now_ms // 1000),
        model_dir=model_dir,
        output_symbols=output_symbols,
        gcp_enabled=gcp_enabled,
        model_cache=model_cache,
        gcp_value=netvar,
        gcp_available=gcp_available,
        gcp_guard=gcp_guard,
    )
    if netvar is not None and not gcp_enabled and not gcp_guard["suppressed"]:
        LOGGER.warning("GCP value received but feature history is insufficient; omitted with-GCP arm")
    append_jsonl(log_file, records)
    _print_records(records)
    return records, (gcp_series if gcp_series is not None else gcp_history)


def _sleep_until_next_minute() -> None:
    now = time.time()
    next_minute = (math.floor(now / 60.0) + 1.0) * 60.0
    time.sleep(max(0.5, next_minute - now + 0.25))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="run one cycle and exit")
    parser.add_argument("--symbol", choices=ASSETS, help="emit one symbol (still fetch all four for dispersion)")
    parser.add_argument("--model-dir", type=Path, default=HERE, help="directory containing joblib artifacts")
    parser.add_argument(
        "--log-file",
        type=Path,
        default=HERE / "live_predictions.jsonl",
        help="append-only JSONL prediction log",
    )
    parser.add_argument(
        "--gcp-cache",
        type=Path,
        default=PROJECT_ROOT / "crypto-data" / "gcp2_1m.db",
        help="optional local netvar_1m SQLite cache for trailing GCP features",
    )
    parser.add_argument("--token-file", type=Path, help="optional runtime file containing the GCP bearer token")
    parser.add_argument("--timeout", type=float, default=10.0, help="HTTP timeout in seconds")
    parser.add_argument(
        "--guard-z-threshold",
        type=float,
        default=DEFAULT_GUARD_Z_THRESHOLD,
        help="maximum absolute fitted-scaler GCP feature z-score before suppression",
    )
    parser.add_argument(
        "--max-gcp-staleness-minutes",
        type=int,
        default=DEFAULT_MAX_GCP_STALENESS_MINUTES,
        help="maximum age of the cached GCP minute in minutes before suppression",
    )
    parser.add_argument("--verbose", action="store_true", help="enable debug logging")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    print("WARNING: outputs are probabilistic estimates, not guarantees.", flush=True)
    history = load_gcp_history(args.gcp_cache)
    model_cache: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    while True:
        try:
            _, history = run_cycle(
                model_dir=args.model_dir,
                log_file=args.log_file,
                gcp_cache=args.gcp_cache,
                token_file=args.token_file,
                output_symbol=args.symbol,
                timeout=args.timeout,
                gcp_history=history,
                model_cache=model_cache,
                guard_z_threshold=args.guard_z_threshold,
                max_gcp_staleness_minutes=args.max_gcp_staleness_minutes,
            )
        except KeyboardInterrupt:
            print("Stopped.", flush=True)
            return 0
        except Exception:
            LOGGER.exception("live prediction cycle failed")
            if args.once:
                return 1
            time.sleep(5.0)
            continue
        if args.once:
            return 0
        try:
            _sleep_until_next_minute()
        except KeyboardInterrupt:
            print("Stopped.", flush=True)
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
