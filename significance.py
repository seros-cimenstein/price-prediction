#!/usr/bin/env python3
"""Reconstruct and statistically analyse the live prediction edge."""
from __future__ import annotations

import argparse

import json
import math
import sqlite3
import sys
import time
import urllib.parse
import warnings
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import chi2_contingency, norm, spearmanr

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import live_eval  # noqa: E402

PREDICTIONS = HERE / "live_predictions.jsonl"
SUPPLEMENTAL = Path("/tmp/price-prediction-live-final.jsonl")
COMBINED = Path("/tmp/price-prediction-live-edge-input.jsonl")
OUTPUT = HERE / "live_edge_significance.json"
MARKET_DB = PROJECT_ROOT / "crypto-data" / "crypto_market.db"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
HORIZONS = [1, 5, 15, 60]
HORIZON_KEYS = {h: f"{h}m" for h in HORIZONS}
BOOTSTRAP_RESAMPLES = 5000
BOOTSTRAP_SEED = 20260914
BONFERRONI_ALPHA = 0.05 / 16.0
WINDOW_SECONDS = 6 * 60 * 60
WINDOW_MIN_ROWS_FOR_FRACTION = 10
WINDOW_MIN_ROWS_FOR_RELIABLE_STABILITY = 50
WINDOW_MIN_ROWS_FOR_NOISE = 20



def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def json_float(value: Any) -> float | None:
    number = finite(value)
    return number


def wilson_interval(hits: int, n: int, confidence: float = 0.95) -> list[float] | None:
    if n <= 0:
        return None
    z = float(norm.ppf(0.5 + confidence / 2.0))
    p = hits / n
    z2 = z * z
    denominator = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denominator
    radius = z * math.sqrt((p * (1.0 - p) / n) + z2 / (4.0 * n * n)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def bootstrap_indices(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    """Moving-block bootstrap indices, retaining the 1-minute row order."""
    if n <= 0:
        return np.empty((0, 0), dtype=np.int64)
    block_length = max(1, min(int(block_length), n))
    blocks = int(math.ceil(n / block_length))
    # Match price-prediction/backtest.py's registered circular moving-block
    # bootstrap: each block start is uniform over every row, and blocks wrap
    # around the chronological sequence.
    starts = rng.integers(0, n, size=(BOOTSTRAP_RESAMPLES, blocks))
    offsets = np.arange(block_length, dtype=np.int64)
    indices = (starts[:, :, None] + offsets[None, None, :]).reshape(BOOTSTRAP_RESAMPLES, -1) % n
    return indices[:, :n]


def bootstrap_mean(
    values: Sequence[float],
    *,
    block_length: int,
    seed_offset: int,
    null: float,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    n = int(array.size)
    if n == 0:
        return {
            "n": 0,
            "observed": None,
            "ci_95": None,
            "p_value": None,
            "valid_resamples": 0,
        }
    observed = float(array.mean())
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    indices = bootstrap_indices(n, block_length, rng)
    boot = array[indices].mean(axis=1)
    ci = [float(value) for value in np.quantile(boot, [0.025, 0.975])]
    # Recenter the observations to the null before resampling.  This makes the
    # two-sided tail probability a test of the null rather than of the observed
    # alternative, while preserving the observed serial-dependence pattern.
    null_values = array - observed + float(null)
    null_boot = null_values[indices].mean(axis=1)
    extreme = np.abs(null_boot - null) >= abs(observed - null) - 1e-15
    p_value = float((int(np.count_nonzero(extreme)) + 1) / (BOOTSTRAP_RESAMPLES + 1))
    return {
        "n": n,
        "observed": observed,
        "ci_95": ci,
        "p_value": p_value,
        "valid_resamples": BOOTSTRAP_RESAMPLES,
    }


def spearman_stat(scores: np.ndarray, returns: np.ndarray) -> float | None:
    if scores.size < 2:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = spearmanr(scores, returns)
    correlation = result.statistic if hasattr(result, "statistic") else result[0]
    return json_float(correlation)


def bootstrap_ic(scores: Sequence[float], returns: Sequence[float], *, block_length: int, seed_offset: int) -> dict[str, Any]:
    score_array = np.asarray(scores, dtype=float)
    return_array = np.asarray(returns, dtype=float)
    n = int(min(score_array.size, return_array.size))
    score_array = score_array[:n]
    return_array = return_array[:n]
    observed = spearman_stat(score_array, return_array) if n >= 10 else None
    if observed is None:
        return {
            "n": n,
            "spearman": None,
            "block_bootstrap_ci_95": None,
            "p_value": None,
            "p_value_method": "two-sided sign probability of paired moving-block bootstrap IC",
            "valid_resamples": 0,
        }
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    indices = bootstrap_indices(n, block_length, rng)
    boot_values: list[float] = []
    for index_row in indices:
        value = spearman_stat(score_array[index_row], return_array[index_row])
        if value is not None:
            boot_values.append(value)
    if not boot_values:
        return {
            "n": n,
            "spearman": observed,
            "block_bootstrap_ci_95": None,
            "p_value": None,
            "p_value_method": "two-sided sign probability of paired moving-block bootstrap IC",
            "valid_resamples": 0,
        }
    boot = np.asarray(boot_values, dtype=float)
    ci = [float(value) for value in np.quantile(boot, [0.025, 0.975])]
    # For a rank statistic, use the bootstrap probability that the association
    # has either sign.  This is deliberately labelled as a bootstrap sign p,
    # not as scipy's iid Spearman p-value.
    lower = (int(np.count_nonzero(boot <= 0.0)) + 1) / (boot.size + 1)
    upper = (int(np.count_nonzero(boot >= 0.0)) + 1) / (boot.size + 1)
    p_value = float(min(1.0, 2.0 * min(lower, upper)))
    return {
        "n": n,
        "spearman": observed,
        "block_bootstrap_ci_95": ci,
        "p_value": p_value,
        "p_value_method": "two-sided sign probability of paired moving-block bootstrap IC",
        "valid_resamples": int(boot.size),
    }


def broad_candles(symbol: str, start_s: int, end_s: int) -> dict[int, live_eval.Candle]:
    """Fetch one broad Binance window; fallback to the local verified candle DB."""
    query = urllib.parse.urlencode(
        {
            "symbol": symbol,
            "interval": "1m",
            "startTime": int(start_s * 1000),
            "endTime": int(end_s * 1000),
            "limit": 1000,
        }
    )
    url = f"{live_eval.BINANCE_KLINES_URL}?{query}"
    try:
        payload = live_eval._http_json(url, timeout=20.0)
        return live_eval._parse_kline_payload(payload, symbol)
    except Exception as exc:
        print(f"warning: Binance broad fetch failed for {symbol}: {type(exc).__name__}; using local DB", file=sys.stderr)
        candles: dict[int, live_eval.Candle] = {}
        with sqlite3.connect(MARKET_DB) as connection:
            rows = connection.execute(
                "SELECT open_time_s, close FROM klines_1m WHERE symbol = ? AND open_time_s BETWEEN ? AND ? ORDER BY open_time_s",
                (symbol, int(start_s), int(end_s)),
            )
            for open_time_s, close in rows:
                close_value = finite(close)
                if close_value is not None and close_value > 0.0:
                    candles[int(open_time_s)] = live_eval.Candle(
                        close=close_value,
                        close_time_ms=int(open_time_s) * 1000 + live_eval.CANDLE_CLOSE_OFFSET_MS,
                    )
        return candles


def snapshot_input() -> tuple[Path, int, int]:
    # Read the append-only live file once so a running generator can continue
    # independently without changing this analysis's population.
    main_text = PREDICTIONS.read_text(encoding="utf-8")
    supplemental_text = SUPPLEMENTAL.read_text(encoding="utf-8") if SUPPLEMENTAL.exists() else ""
    combined_text = main_text.rstrip("\n")
    if supplemental_text:
        combined_text += "\n" + supplemental_text.rstrip("\n")
    combined_text += "\n"
    COMBINED.write_text(combined_text, encoding="utf-8")
    records_read = sum(bool(line.strip()) for line in combined_text.splitlines())
    now_ms = int(time.time() * 1000)
    return COMBINED, records_read, now_ms


def baseline_hit(row: live_eval.EvaluatedRow, name: str) -> int | None:
    baselines = row.prediction.values.get("baselines")
    if not isinstance(baselines, Mapping):
        return None
    value = live_eval._baseline_value(baselines, name)
    return value


def direction_observations(rows: Sequence[live_eval.EvaluatedRow]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    model_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    ic_rows: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: item.prediction.target_time_s):
        realized = finite(row.outcome.forward_return)
        if realized is None or realized == 0.0:
            continue
        arm = row.prediction.values.get("market_only")
        if not isinstance(arm, Mapping):
            continue
        probability = finite(arm.get("p_up"))
        score = finite(arm.get("direction_score"))
        persistence = baseline_hit(row, "persistence")
        if probability is not None:
            model_hit = int((probability > 0.5) == (realized > 0.0))
            observation = {
                "target_time_s": int(row.prediction.target_time_s),
                "model_hit": model_hit,
                "persistence_hit": None if persistence is None else int((persistence > 0) == (realized > 0.0)),
                "realized_return": realized,
                "realized_vol": finite(row.outcome.realized_vol),
                "score": score,
            }
            model_rows.append(observation)
            if observation["persistence_hit"] is not None:
                paired_rows.append(observation)
        if score is not None:
            ic_rows.append(
                {
                    "target_time_s": int(row.prediction.target_time_s),
                    "score": score,
                    "realized_return": realized,
                    "realized_vol": finite(row.outcome.realized_vol),
                }
            )
    return model_rows, paired_rows, ic_rows


def summarize_half(observations: Sequence[dict[str, Any]], label: str) -> dict[str, Any]:
    n = len(observations)
    hits = int(sum(int(item["model_hit"]) for item in observations))
    persistence_values = [item["persistence_hit"] for item in observations if item["persistence_hit"] is not None]
    persistence_hits = int(sum(int(value) for value in persistence_values))
    vols = [item["realized_vol"] for item in observations if item["realized_vol"] is not None]
    return {
        "half": label,
        "n": n,
        "hits": hits,
        "hit_rate": float(hits / n) if n else None,
        "persistence_n": len(persistence_values),
        "persistence_hits": persistence_hits,
        "persistence_hit_rate": float(persistence_hits / len(persistence_values)) if persistence_values else None,
        "model_minus_persistence": float(hits / n - persistence_hits / len(persistence_values))
        if n and persistence_values
        else None,
        "mean_realized_vol": float(np.mean(vols)) if vols else None,
        "median_realized_vol": float(np.median(vols)) if vols else None,
    }


def classify_stability(rates: Sequence[float], chi_p: float | None) -> tuple[str, bool]:
    if not rates:
        return "unavailable", False
    first = float(rates[0])
    last = float(rates[-1])
    if first > 0.5 and last <= 0.5:
        label = "fades"
    elif first <= 0.5 and last > 0.5:
        label = "emerges"
    elif last - first >= 0.05:
        label = "strengthens"
    elif last - first <= -0.05:
        label = "weakens"
    elif max(rates) - min(rates) <= 0.05:
        label = "flat"
    else:
        label = "mixed"
    same_side = all(rate > 0.5 for rate in rates) or all(rate <= 0.5 for rate in rates)
    stable = bool(same_side and chi_p is not None and chi_p >= 0.05)
    return label, stable


def stability_summary(observations: Sequence[dict[str, Any]]) -> dict[str, Any]:
    n = len(observations)
    insufficient_reason = "insufficient data for chi-square"
    if n == 0:
        return {
            "thirds": [],
            "chi_square": {
                "statistic": None,
                "df": 2,
                "p_value": None,
                "reason": insufficient_reason,
            },
            "classification": "unavailable",
            "stable": False,
            "criterion": "all thirds on the same side of 50% and chi-square p >= 0.05",
        }
    sorted_rows = sorted(observations, key=lambda item: item["target_time_s"])
    parts = np.array_split(np.arange(n), 3)
    thirds: list[dict[str, Any]] = []
    table: list[list[int]] = []
    rates: list[float] = []
    for label, indices in zip(("first", "middle", "last"), parts):
        selected = [sorted_rows[int(index)] for index in indices]
        part_n = len(selected)
        hits = int(sum(item["model_hit"] for item in selected))
        rate = float(hits / part_n) if part_n else None
        if rate is not None:
            rates.append(rate)
        table.append([hits, part_n - hits])
        thirds.append(
            {
                "third": label,
                "target_time_start_s": int(selected[0]["target_time_s"]) if selected else None,
                "target_time_end_s": int(selected[-1]["target_time_s"]) if selected else None,
                "n": part_n,
                "hits": hits,
                "hit_rate": rate,
                "minus_chance": float(rate - 0.5) if rate is not None else None,
            }
        )
    chi_stat: float | None = None
    chi_p: float | None = None
    chi_reason: str | None = None
    observed = np.asarray(table, dtype=float)
    if observed.size == 0 or float(observed.sum()) <= 0.0:
        chi_reason = insufficient_reason
    else:
        row_totals = observed.sum(axis=1, keepdims=True)
        column_totals = observed.sum(axis=0, keepdims=True)
        expected = row_totals @ column_totals / float(observed.sum())
        if not np.all(np.isfinite(expected)) or bool(np.any(expected < 5.0)):
            chi_reason = insufficient_reason
        else:
            result = chi2_contingency(observed, correction=False)
            chi_stat = float(result.statistic)
            chi_p = float(result.pvalue)
    classification, stable = classify_stability(rates, chi_p)
    return {
        "thirds": thirds,
        "chi_square": {
            "statistic": chi_stat,
            "df": 2,
            "p_value": chi_p,
            "reason": chi_reason,
        },
        "classification": classification,
        "stable": stable,
        "criterion": "all thirds on the same side of 50% and chi-square p >= 0.05",
    }


def regime_summary(observations: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    valid = [item for item in observations if item["realized_vol"] is not None]
    if len(valid) < 2:
        return None
    median_vol = float(np.median([item["realized_vol"] for item in valid]))
    low = [item for item in valid if item["realized_vol"] <= median_vol]
    high = [item for item in valid if item["realized_vol"] > median_vol]
    low_summary = summarize_half(low, "low_realized_vol (<= median)")
    high_summary = summarize_half(high, "high_realized_vol (> median)")
    low_rate = low_summary["hit_rate"]
    high_rate = high_summary["hit_rate"]
    if low_rate is None or high_rate is None:
        concentration = "unavailable"
    elif high_rate - low_rate >= 0.05:
        concentration = "higher_in_high_vol"
    elif low_rate - high_rate >= 0.05:
        concentration = "higher_in_low_vol"
    else:
        concentration = "not_concentrated"
    persistence_delta = None
    if low_summary["persistence_hit_rate"] is not None and high_summary["persistence_hit_rate"] is not None:
        persistence_delta = float(high_summary["persistence_hit_rate"] - low_summary["persistence_hit_rate"])
    return {
        "proxy": "realized_vol (not a signed trend measure)",
        "median_realized_vol": median_vol,
        "low": low_summary,
        "high": high_summary,
        "high_minus_low_hit_rate": float(high_rate - low_rate) if low_rate is not None and high_rate is not None else None,
        "high_minus_low_persistence_rate": persistence_delta,
        "concentration": concentration,
        "interpretation": (
            "model and persistence rise together in the high-vol half, consistent with a broad regime effect"
            if concentration == "higher_in_high_vol" and persistence_delta is not None and persistence_delta >= 0.03
            else "descriptive volatility split; it is not evidence of a signed trend causal effect"
        ),
    }


def cell_summary(symbol: str, horizon: int, rows: Sequence[live_eval.EvaluatedRow], cell_index: int) -> dict[str, Any]:
    arm_rows = [row for row in rows if isinstance(row.prediction.values.get("market_only"), Mapping)]
    model_obs, paired_obs, ic_obs = direction_observations(arm_rows)
    model_hits = [item["model_hit"] for item in model_obs]
    persistence_hits = [item["persistence_hit"] for item in paired_obs]
    differences = [item["model_hit"] - item["persistence_hit"] for item in paired_obs]
    scores = [item["score"] for item in ic_obs]
    returns = [item["realized_return"] for item in ic_obs]
    block_length = min(horizon, max(1, len(model_obs)))
    chance = bootstrap_mean(model_hits, block_length=block_length, seed_offset=cell_index * 11 + 1, null=0.5)
    chance_hits = int(sum(model_hits))
    chance["hits"] = chance_hits
    chance["hit_rate"] = float(chance_hits / len(model_hits)) if model_hits else None
    chance["naive_wilson_ci_95"] = wilson_interval(chance_hits, len(model_hits))
    chance["null"] = 0.5
    chance["difference"] = float(chance["hit_rate"] - 0.5) if chance["hit_rate"] is not None else None
    persistence = bootstrap_mean(differences, block_length=min(horizon, max(1, len(differences))), seed_offset=cell_index * 11 + 2, null=0.0)
    persistence_hits_count = int(sum(persistence_hits))
    persistence["model_hits"] = chance_hits if len(persistence_hits) == len(model_obs) else int(sum(item["model_hit"] for item in paired_obs))
    persistence["persistence_hits"] = persistence_hits_count
    persistence["model_hit_rate"] = float(persistence["model_hits"] / len(paired_obs)) if paired_obs else None
    persistence["persistence_hit_rate"] = float(persistence_hits_count / len(paired_obs)) if paired_obs else None
    persistence["difference"] = float(persistence["observed"]) if persistence["observed"] is not None else None
    persistence["null"] = 0.0
    persistence["p_value_method"] = "two-sided null-centered paired moving-block bootstrap of hit difference"
    ic = bootstrap_ic(scores, returns, block_length=min(horizon, max(1, len(ic_obs))), seed_offset=cell_index * 11 + 3)
    persistence["block_length_rows"] = min(horizon, max(1, len(differences))) if differences else None
    chance["block_length_rows"] = block_length if model_obs else None
    chi = stability_summary(model_obs)
    regime = regime_summary(model_obs) if horizon == 60 else None
    chance_sig = bool(
        chance["p_value"] is not None
        and chance["p_value"] < BONFERRONI_ALPHA
        and chance["hit_rate"] is not None
        and chance["hit_rate"] > 0.5
    )
    persistence_sig = bool(
        persistence["p_value"] is not None
        and persistence["p_value"] < BONFERRONI_ALPHA
        and persistence["difference"] is not None
        and persistence["difference"] > 0.0
    )
    stable = bool(chi["stable"])
    if chance_sig and persistence_sig and stable:
        verdict_line = "supports a stable live edge beyond chance and persistence"
    elif chance_sig and persistence_sig:
        verdict_line = "clears chance and persistence after Bonferroni correction but is not stable across thirds"
    elif chance_sig:
        verdict_line = "above chance after Bonferroni correction, but not beyond persistence"
    elif persistence_sig:
        verdict_line = "beats persistence after Bonferroni correction, but not chance"
    else:
        verdict_line = "does not survive Bonferroni-corrected block-bootstrap testing"
    return {
        "symbol": symbol,
        "horizon_minutes": horizon,
        "horizon": HORIZON_KEYS[horizon],
        "rows_available": len(arm_rows),
        "direction": {
            "n": len(model_obs),
            "hits": chance_hits,
            "hit_rate": chance["hit_rate"],
            "vs_chance": chance,
            "vs_persistence": persistence,
            "ic": ic,
        },
        "stability": chi,
        "regime_60m": regime,
        "verdict": {
            "significant_vs_chance": chance_sig,
            "significant_vs_persistence": persistence_sig,
            "stable": stable,
            "overall": verdict_line,
        },
    }


def windowed_stability_summary(windows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Classify the descriptive fraction of per-window surviving tests."""
    counted = [window for window in windows if int(window["n"]) >= WINDOW_MIN_ROWS_FOR_FRACTION]
    surviving = int(sum(bool(window["survives_both"]) for window in counted))
    denominator = len(counted)
    fraction = float(surviving / denominator) if denominator else None
    has_too_small_window = any(int(window["n"]) < WINDOW_MIN_ROWS_FOR_NOISE for window in windows)
    all_counted_reliable = bool(counted) and all(
        int(window["n"]) >= WINDOW_MIN_ROWS_FOR_RELIABLE_STABILITY for window in counted
    )
    if not counted or has_too_small_window or fraction is None or fraction < 0.40:
        verdict = "noise"
    elif fraction >= 0.80 and all_counted_reliable:
        verdict = "consistent"
    else:
        verdict = "marginal"
    return {
        "fraction_surviving": fraction,
        "surviving_windows": surviving,
        "windows_in_fraction": denominator,
        "windows_skipped_n_lt_10": len(windows) - denominator,
        "minimum_window_n": min((int(window["n"]) for window in windows), default=None),
        "all_counted_windows_n_at_least_50": all_counted_reliable,
        "stability_verdict": verdict,
        "criterion": (
            "consistent requires fraction >= 80% and every counted window n >= 50; "
            "marginal is 40%-79%; noise is <40% or any window n < 20"
        ),
    }


def windowed_cell_summary(
    symbol: str,
    horizon: int,
    rows: Sequence[live_eval.EvaluatedRow],
    cell_index: int,
) -> dict[str, Any]:
    """Run the registered cell tests independently in each UTC six-hour window."""
    arm_rows = [row for row in rows if isinstance(row.prediction.values.get("market_only"), Mapping)]
    grouped: dict[int, list[live_eval.EvaluatedRow]] = {}
    for row in arm_rows:
        window_index = int(row.prediction.target_time_s) // WINDOW_SECONDS
        grouped.setdefault(window_index, []).append(row)

    windows: list[dict[str, Any]] = []
    cell_stride = len(SYMBOLS) * len(HORIZONS)
    for window_index, window_rows in sorted(grouped.items()):
        # Use a distinct deterministic bootstrap stream per absolute
        # UTC window/cell while leaving cumulative seeds unchanged.
        summary = cell_summary(
            symbol,
            horizon,
            window_rows,
            cell_index + window_index * cell_stride,
        )
        direction = summary["direction"]
        chance = dict(direction["vs_chance"])
        persistence = dict(direction["vs_persistence"])
        chance["significant"] = bool(summary["verdict"]["significant_vs_chance"])
        persistence["significant"] = bool(summary["verdict"]["significant_vs_persistence"])
        survives_both = bool(
            summary["verdict"]["significant_vs_chance"]
            and summary["verdict"]["significant_vs_persistence"]
        )
        windows.append(
            {
                "window_index": window_index,
                "window_start_s": window_index * WINDOW_SECONDS,
                "window_end_s": (window_index + 1) * WINDOW_SECONDS,
                "rows_available": len(window_rows),
                "n": direction["n"],
                "hits": direction["hits"],
                "hit_rate": direction["hit_rate"],
                "vs_chance": chance,
                "vs_persistence": persistence,
                "survives_both": survives_both,
                "counted_in_fraction": direction["n"] >= WINDOW_MIN_ROWS_FOR_FRACTION,
            }
        )
    return {
        "symbol": symbol,
        "horizon_minutes": horizon,
        "horizon": HORIZON_KEYS[horizon],
        "windows": windows,
        **windowed_stability_summary(windows),
    }


def windowed_overall_verdict(cells: Mapping[str, Mapping[str, Any]]) -> str:
    consistent = [key for key, cell in cells.items() if cell["stability_verdict"] == "consistent"]
    marginal = [key for key, cell in cells.items() if cell["stability_verdict"] == "marginal"]
    noise = [key for key, cell in cells.items() if cell["stability_verdict"] == "noise"]
    robust_60m = [
        sum(int(window["n"]) >= WINDOW_MIN_ROWS_FOR_RELIABLE_STABILITY for window in cell["windows"])
        for key, cell in cells.items()
        if key.endswith("_60m")
    ]
    if robust_60m:
        robust_text = f"{min(robust_60m)}-{max(robust_60m)}"
    else:
        robust_text = "0"
    return (
        "The six-hour windowed result is descriptive, not a confirmed trading edge: "
        f"{len(consistent)}/16 cells are consistent, {len(marginal)}/16 are marginal, and "
        f"{len(noise)}/16 are noise. The 60m cells provide only {robust_text} windows with n>=50, "
        "too few independent windows to conclude on 60m stability; the per-window Bonferroni "
        "tests are confirmatory within each window, while the survival fraction adds no alpha inflation."
    )


def build_windowed_summary(
    grouped: Mapping[tuple[str, int], Sequence[live_eval.EvaluatedRow]],
) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    for index, (symbol, horizon) in enumerate((item for item in ((s, h) for s in SYMBOLS for h in HORIZONS))):
        key = f"{symbol}_{HORIZON_KEYS[horizon]}"
        cells[key] = windowed_cell_summary(symbol, horizon, grouped.get((symbol, horizon), []), index)
    return {
        "window_seconds": WINDOW_SECONDS,
        "window_anchor": "UTC target_time_s // 21600; consecutive non-overlapping windows",
        "methods": {
            "tests": "same cell_summary direction tests as cumulative mode, run independently per window",
            "bootstrap": {
                "type": "moving-block bootstrap on chronological 1-minute row sequence",
                "block_length_rows": "horizon minutes (1, 5, 15, or 60)",
                "resamples": BOOTSTRAP_RESAMPLES,
                "seed": BOOTSTRAP_SEED,
                "bonferroni_alpha_per_window": BONFERRONI_ALPHA,
            },
            "fraction_surviving": (
                "descriptive count of windows surviving both direction tests; no new alpha inflation "
                "is applied to this fraction"
            ),
            "fraction_denominator": "windows with direction-valid n >= 10; smaller windows are listed but excluded",
            "stability_verdict": (
                "consistent >=80% with every counted window n>=50; marginal 40%-79%; "
                "noise <40% or any listed window n<20"
            ),
        },
        "cells": cells,
        "overall_verdict": windowed_overall_verdict(cells),
    }


def windowed_report_lines(windowed: Mapping[str, Any]) -> list[str]:
    lines = [
        "",
        "WINDOWED 6-HOUR STABILITY (market_only)",
        "=======================================",
        "Per-window n is shown as window_index:n; windows with n<10 are listed but excluded from the fraction.",
        "Cell                         windows (n)                          survive/count  fraction  verdict",
    ]
    for key in sorted(windowed["cells"]):
        cell = windowed["cells"][key]
        window_text = ", ".join(f"{item['window_index']}:{item['n']}" for item in cell["windows"]) or "none"
        fraction = cell["fraction_surviving"]
        fraction_text = "n/a" if fraction is None else f"{fraction * 100:.1f}%"
        lines.append(
            f"{key:<28} {window_text:<40} "
            f"{cell['surviving_windows']}/{cell['windows_in_fraction']:<11} "
            f"{fraction_text:<9} {cell['stability_verdict']}"
        )
    lines.extend(["", "OVERALL WINDOWED VERDICT: " + windowed["overall_verdict"]])
    return lines


def run_windowed_self_test() -> int:
    def indicators(hits: int, n: int = 100) -> list[int]:
        return [1] * hits + [0] * (n - hits)

    vectors = [indicators(60), indicators(40), indicators(65)]
    synthetic_windows = [
        {
            "n": len(vector),
            "hits": sum(vector),
            "hit_rate": sum(vector) / len(vector),
            "survives_both": survives,
        }
        for vector, survives in zip(vectors, (True, False, True))
    ]
    summary = windowed_stability_summary(synthetic_windows)
    zero_expected_observations = [
        {"target_time_s": index, "model_hit": 1}
        for index in range(6)
    ]
    zero_expected_summary = stability_summary(zero_expected_observations)
    try:
        assert [window["hit_rate"] for window in synthetic_windows] == [0.60, 0.40, 0.65]
        assert summary["surviving_windows"] == 2
        assert summary["windows_in_fraction"] == 3
        assert summary["fraction_surviving"] == 2 / 3
        assert summary["stability_verdict"] == "marginal"
        assert zero_expected_summary["chi_square"]["statistic"] is None
        assert zero_expected_summary["chi_square"]["p_value"] is None
        assert zero_expected_summary["chi_square"]["reason"] == "insufficient data for chi-square"
        assert zero_expected_summary["stable"] is False
    except (AssertionError, KeyError):
        print("WINDOWED SELF-TEST: FAIL")
        return 1
    print("WINDOWED SELF-TEST: PASS (windowed classifier and zero-expected chi-square skip)")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--windowed",
        action="store_true",
        help="also analyse consecutive non-overlapping six-hour UTC windows",
    )
    parser.add_argument(
        "--windowed-self-test",
        action="store_true",
        help="run the deterministic windowed stability classifier check and exit",
    )
    return parser.parse_args(argv)



def make_report(payload: Mapping[str, Any]) -> str:
    lines = [
        "LIVE EDGE SIGNIFICANCE (market_only)",
        "====================================",
        f"Input: {payload['input']['records_read']} nonblank records, {payload['input']['deduplicated_rows']} deduplicated rows, "
        f"{payload['input']['evaluated_rows']} evaluated rows, {payload['input']['pending_rows']} pending rows.",
        f"Bootstrap: {payload['methods']['bootstrap']['resamples']} moving-block resamples; block length equals horizon minutes; "
        f"Bonferroni alpha = {BONFERRONI_ALPHA:.6f} for each of the two pre-registered direction tests.",
        "CIs are percentile block-bootstrap intervals; hit p-values are null-centered two-sided tests; IC p-values are bootstrap sign probabilities.",
        "",
    ]
    for key in sorted(payload["cells"]):
        cell = payload["cells"][key]
        direction = cell["direction"]
        chance = direction["vs_chance"]
        persistence = direction["vs_persistence"]
        ci = chance["ci_95"]
        p_ci = persistence["ci_95"]
        ic = direction["ic"]
        thirds = cell["stability"]["thirds"]
        third_rates = ", ".join(
            f"{item['third']}={item['hit_rate'] * 100:.1f}% (n={item['n']})" for item in thirds
        )
        regime = cell["regime_60m"]
        regime_text = ""
        if regime:
            regime_text = (
                f"; vol low/high={regime['low']['hit_rate'] * 100:.1f}%/{regime['high']['hit_rate'] * 100:.1f}%"
            )
        lines.append(
            f"{key}: n={direction['n']}, hit={direction['hit_rate'] * 100:.2f}% [{ci[0] * 100:.2f}, {ci[1] * 100:.2f}]%, "
            f"p50={chance['p_value']:.4f}; vs persist Δ={persistence['difference'] * 100:.2f}pp "
            f"[{p_ci[0] * 100:.2f}, {p_ci[1] * 100:.2f}]pp p={persistence['p_value']:.4f}; "
            f"IC={ic['spearman']:.4f} p={ic['p_value']:.4f}"
            f"; thirds {third_rates} ({cell['stability']['classification']}, stable={'yes' if cell['verdict']['stable'] else 'no'})"
            f"{regime_text}."
        )
        lines.append(
            f"  verdict: chance={'yes' if cell['verdict']['significant_vs_chance'] else 'no'}, "
            f"persistence={'yes' if cell['verdict']['significant_vs_persistence'] else 'no'}; {cell['verdict']['overall']}."
        )
    lines.extend(
        [
            "",
            "Bonferroni-surviving cells vs chance: " + (", ".join(payload["surviving_cells"]["vs_chance"]) or "none"),
            "Bonferroni-surviving cells vs persistence: "
            + (", ".join(payload["surviving_cells"]["vs_persistence"]) or "none"),
            "Cells surviving both tests: " + (", ".join(payload["surviving_cells"]["both"]) or "none"),
            "",
            "OVERALL VERDICT: " + payload["overall_verdict"],
        ]
    )
    if "windowed" in payload:
        lines.extend(windowed_report_lines(payload["windowed"]))

    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.windowed_self_test:
        return run_windowed_self_test()
    input_path, records_read_snapshot, now_ms = snapshot_input()
    loaded = live_eval._read_predictions(input_path)
    if not loaded.rows:
        raise RuntimeError("no valid prediction rows in combined input")
    min_target = min(row.target_time_s for row in loaded.rows)
    max_target = max(row.target_time_s for row in loaded.rows)
    max_horizon = max(row.horizon for row in loaded.rows)
    cache = live_eval.KlineCache(timeout=20.0)
    # One broad request per symbol populates the evaluator's normal cache and
    # avoids one HTTP request per target window while preserving its formulas.
    for symbol in SYMBOLS:
        fetched = broad_candles(symbol, min_target - 60, max_target + max_horizon * 60)
        for open_time_s, candle in fetched.items():
            cache.candles[(symbol, int(open_time_s))] = candle
    payload, evaluated_rows, _ = live_eval.evaluate_predictions(
        input_path,
        out_path=None,
        cache=cache,
        now_ms=now_ms,
    )
    grouped: dict[tuple[str, int], list[live_eval.EvaluatedRow]] = {}
    for row in evaluated_rows:
        grouped.setdefault((row.prediction.symbol, row.prediction.horizon), []).append(row)
    cells: dict[str, Any] = {}
    for index, (symbol, horizon) in enumerate((item for item in ((s, h) for s in SYMBOLS for h in HORIZONS))):
        cells[f"{symbol}_{HORIZON_KEYS[horizon]}"] = cell_summary(
            symbol,
            horizon,
            grouped.get((symbol, horizon), []),
            index,
        )
    chance_cells = [key for key, cell in cells.items() if cell["verdict"]["significant_vs_chance"]]
    persistence_cells = [key for key, cell in cells.items() if cell["verdict"]["significant_vs_persistence"]]
    both_cells = [key for key, cell in cells.items() if cell["verdict"]["significant_vs_chance"] and cell["verdict"]["significant_vs_persistence"]]
    focus = [cells[f"{symbol}_{HORIZON_KEYS[horizon]}"] for symbol in SYMBOLS for horizon in (15, 60)]
    focus_rates = [cell["direction"]["hit_rate"] for cell in focus if cell["direction"]["hit_rate"] is not None]
    focus_persist = [
        cell["direction"]["vs_persistence"]["persistence_hit_rate"]
        for cell in focus
        if cell["direction"]["vs_persistence"]["persistence_hit_rate"] is not None
    ]
    focus_labels = "15m/60m"
    overall_verdict = (
        f"The live {focus_labels} hit-rate elevations are not a confirmed trading edge: after horizon-sized "
        f"block bootstrap and Bonferroni correction (alpha={BONFERRONI_ALPHA:.6f}), "
        f"{len(chance_cells)}/16 cells survive vs chance, {len(persistence_cells)}/16 survive vs persistence, "
        f"and {len(both_cells)}/16 survive both. The accumulated {focus_labels} market_only rates span "
        f"{min(focus_rates) * 100:.1f}%–{max(focus_rates) * 100:.1f}% versus persistence "
        f"{min(focus_persist) * 100:.1f}%–{max(focus_persist) * 100:.1f}%; the small, short live sample and "
        "third/regime variation are consistent with a trending/regime or sample artifact rather than a stable edge. "
        "The historical pooled walk-forward baseline was only about 51.7%–52.5% at 15m and 51.2%–51.7% at 60m, "
        "with naive intervals already known to be optimistic under serial dependence."
    )
    result: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input": {
            "prediction_paths": [str(PREDICTIONS), str(SUPPLEMENTAL)],
            "combined_snapshot_path": str(input_path),
            "records_read": int(records_read_snapshot),
            "invalid_lines": int(loaded.invalid_lines),
            "invalid_horizon_rows": int(loaded.invalid_horizon_rows),
            "deduplicated_rows": len(loaded.rows),
            "evaluated_rows": int(payload["prediction_rows"]["evaluated"]),
            "pending_rows": int(payload["prediction_rows"]["pending"]),
            "skipped_rows": int(payload["prediction_rows"]["skipped"]),
            "evaluation_now_ms": int(now_ms),
            "deduplication": "live_eval._read_predictions latest line per (symbol, target_time_s, horizon); supplemental rows added only when key absent",
            "note": "The live generator was left running; this analysis used one immutable read snapshot of each input log.",
        },
        "methods": {
            "evaluator": "price-prediction/live_eval.evaluate_predictions with its EvaluatedRow outcomes and build_scorecard semantics",
            "direction_denominator": "finite nonzero realized forward returns with finite market_only p_up; exact-zero returns excluded as in live_eval",
            "persistence_pair_denominator": "rows with finite nonzero realized forward returns, finite market_only p_up, and a persistence sign",
            "bootstrap": {
                "type": "moving-block bootstrap on chronological 1-minute row sequence",
                "block_length_rows": "horizon minutes (1, 5, 15, or 60)",
                "resamples": BOOTSTRAP_RESAMPLES,
                "seed": BOOTSTRAP_SEED,
                "ci": "percentile 2.5%/97.5%",
                "hit_p_value": "two-sided null-centered block bootstrap; null-centered observations preserve serial dependence",
                "paired_persistence": "same paired block resampling applied to model_hit - persistence_hit",
                "ic_p_value": "two-sided bootstrap sign probability from paired Spearman IC resamples; IID Spearman p-values not used",
            },
            "naive_wilson": "95% Wilson interval shown only for contrast; it treats rows as independent",
            "stability": {
                "split": "three chronological thirds by target_time_s using direction-valid rows",
                "test": "2x3 chi-square test of hit/failure proportions, descriptive only",
                "stable_definition": "all thirds on the same side of 50% and chi-square p >= 0.05",
            },
            "regime": "60m direction-valid rows split at median realized_vol; descriptive proxy, not a signed trend measure",
            "multiple_testing": {
                "pre_registered_cells": 16,
                "tests_with_adjustment": ["direction_vs_50_percent", "direction_vs_persistence"],
                "bonferroni_alpha": BONFERRONI_ALPHA,
                "thirds_and_regime": "descriptive; no additional confirmatory correction",
            },
        },
        "cells": cells,
        "surviving_cells": {
            "vs_chance": chance_cells,
            "vs_persistence": persistence_cells,
            "both": both_cells,
        },
        "overall_verdict": overall_verdict,
        "historical_contrast": {
            "walk_forward_15m_hit_rate_range": [0.517, 0.525],
            "walk_forward_60m_hit_rate_range": [0.512, 0.517],
            "historical_serial_dependence_note": "README says naive Wilson intervals are optimistic uncertainty intervals under minute-level serial dependence",
        },
    }
    if args.windowed:
        result["windowed"] = build_windowed_summary(grouped)

    report = make_report(result)
    result["printed_report"] = report
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
