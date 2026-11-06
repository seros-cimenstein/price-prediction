"""Leakage-aware walk-forward evaluation and statistical metrics.

This module deliberately keeps fold construction and score aggregation boring:
all transformations are fit inside each training fold, then frozen for its
embargoed test interval.  It is used by train.py for both the full feature set
and the market-only GCP ablation.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import erf
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from features import (
    FEATURE_ARMS,
    GCP_FEATURES,
    HORIZONS,
    MARKET_FEATURES,
)


@dataclass(frozen=True)
class Fold:
    """Integer positions for one expanding-window fold."""

    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    train_end_time_s: int
    test_start_time_s: int
    test_end_time_s: int


def make_walk_forward_splits(
    times_s: Sequence[int] | np.ndarray,
    horizon: int,
    n_folds: int = 4,
    initial_train_fraction: float = 0.50,
    embargo: int | None = None,
) -> list[Fold]:
    """Create expanding, time-ordered folds with an h-minute embargo.

    ``times_s`` may contain one row per asset.  Test starts are selected on
    unique timestamps, while train/test masks include all available assets at
    those timestamps.  Train targets therefore finish at least ``horizon``
    minutes before the first test target.
    """
    times = np.asarray(times_s, dtype="int64")
    if times.ndim != 1 or len(times) == 0:
        raise ValueError("times_s must be a non-empty one-dimensional array")
    if horizon <= 0 or n_folds <= 0:
        raise ValueError("horizon and n_folds must be positive")
    if not 0.0 < initial_train_fraction < 1.0:
        raise ValueError("initial_train_fraction must be between 0 and 1")
    unique = np.unique(times)
    initial = max(1, int(len(unique) * initial_train_fraction))
    if initial >= len(unique):
        raise ValueError("not enough unique timestamps for a test interval")
    remaining = len(unique) - initial
    test_size = max(horizon, remaining // n_folds)
    embargo_minutes = max(horizon, int(embargo or horizon))
    folds: list[Fold] = []
    for number in range(n_folds):
        start_pos = initial + number * test_size
        if start_pos >= len(unique):
            break
        stop_pos = min(len(unique), start_pos + test_size)
        test_start = int(unique[start_pos])
        test_end_exclusive = int(unique[stop_pos - 1]) + 60
        train_cutoff = test_start - embargo_minutes * 60
        train_idx = np.flatnonzero(times < train_cutoff)
        test_idx = np.flatnonzero((times >= test_start) & (times < test_end_exclusive))
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        train_end = int(times[train_idx].max())
        # This is an assertion rather than a comment: changing the fold logic
        # cannot silently remove the embargo guarantee.
        assert test_start - train_end >= embargo_minutes * 60
        folds.append(
            Fold(
                fold=number + 1,
                train_idx=train_idx,
                test_idx=test_idx,
                train_end_time_s=train_end,
                test_start_time_s=test_start,
                test_end_time_s=test_end_exclusive - 60,
            )
        )
    if not folds:
        raise ValueError("no valid walk-forward folds")
    return folds


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial hit rate."""
    if total <= 0:
        return (float("nan"), float("nan"))
    p = float(successes) / float(total)
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    radius = z * np.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denominator
    return (max(0.0, centre - radius), min(1.0, centre + radius))


def _spearman(y: np.ndarray, score: np.ndarray) -> float:
    good = np.isfinite(y) & np.isfinite(score)
    y = np.asarray(y, dtype="float64")[good]
    score = np.asarray(score, dtype="float64")[good]
    if len(y) < 3 or np.ptp(y) == 0.0 or np.ptp(score) == 0.0:
        return float("nan")
    return float(np.corrcoef(rankdata(y), rankdata(score))[0, 1])


def _ranked_correlation(rank_y: np.ndarray, rank_score: np.ndarray) -> float:
    """Pearson correlation of already-ranked series (Spearman building block).

    Identical to :func:`_spearman` when the ranks come from ``rankdata`` on the
    same samples; the paired IC bootstrap uses it to rank the resampled target
    once and share it between the two score arms.
    """
    if len(rank_y) < 3 or np.ptp(rank_y) == 0.0 or np.ptp(rank_score) == 0.0:
        return float("nan")
    return float(np.corrcoef(rank_y, rank_score)[0, 1])


def direction_metrics(y_return: Sequence[float], score: Sequence[float]) -> dict:
    """Hit rate, Wilson 95% CI, and Spearman IC for direction scores."""
    y = np.asarray(y_return, dtype="float64")
    s = np.asarray(score, dtype="float64")
    good = np.isfinite(y) & np.isfinite(s) & (y != 0.0)
    y = y[good]
    s = s[good]
    pred = np.sign(s)
    actual = np.sign(y)
    hits = pred == actual
    n = int(len(y))
    k = int(hits.sum())
    lo, hi = wilson_interval(k, n)
    # A normal approximation is only supplemental; CI is the primary chance
    # comparison in results.json.
    if n and 0.0 < 0.5 < 1.0:
        z = (k - n * 0.5) / np.sqrt(n * 0.25)
        chance_p = float(2.0 * (1.0 - _normal_cdf(abs(z))))
    else:
        chance_p = float("nan")
    return {
        "n": n,
        "hits": k,
        "hit_rate": float(k / n) if n else float("nan"),
        "hit_ci95": [float(lo), float(hi)],
        "hit_delta_vs_50": float(k / n - 0.5) if n else float("nan"),
        "chance_p_normal": chance_p,
        "ic_spearman": _spearman(y, s),
    }


def _normal_cdf(x: float) -> float:
    # scipy is already a dependency for rank/metrics, but this tiny expression
    # avoids importing another namespace in every metric call.
    return float(0.5 * (1.0 + erf(x / np.sqrt(2.0))))


def volatility_metrics(y_vol: Sequence[float], prediction: Sequence[float], ewma: Sequence[float] | None = None) -> dict:
    """Volatility MSE/MAE, optionally compared with the EWMA baseline."""
    y = np.asarray(y_vol, dtype="float64")
    p = np.asarray(prediction, dtype="float64")
    good = np.isfinite(y) & np.isfinite(p)
    y = y[good]
    p = p[good]
    mse = float(np.mean((p - y) ** 2)) if len(y) else float("nan")
    mae = float(np.mean(np.abs(p - y))) if len(y) else float("nan")
    result = {"n": int(len(y)), "mse": mse, "mae": mae}
    if ewma is not None:
        b = np.asarray(ewma, dtype="float64")[good]
        base_good = np.isfinite(b)
        if base_good.any():
            base_mse = float(np.mean((b[base_good] - y[base_good]) ** 2))
            base_mae = float(np.mean(np.abs(b[base_good] - y[base_good])))
            result.update(
                {
                    "ewma_mse": base_mse,
                    "ewma_mae": base_mae,
                    "mse_delta_vs_ewma": mse - base_mse,
                    "mae_delta_vs_ewma": mae - base_mae,
                }
            )
        else:
            result.update({"ewma_mse": float("nan"), "ewma_mae": float("nan")})
    return result


def _new_model(kind: str):
    if kind == "direction":
        return LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=300,
            random_state=0,
        )
    if kind == "volatility":
        return Ridge(alpha=1.0)
    raise ValueError(f"unknown model kind: {kind}")


def _fit_score(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, kind: str):
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(x_train)
    test_scaled = scaler.transform(x_test)
    model = _new_model(kind)
    if kind == "direction":
        y_binary = (y_train > 0.0).astype("int8")
        if np.unique(y_binary).size < 2:
            # This can occur only in a tiny custom range; preserve a truthful
            # constant predictor rather than leaking a test label into fitting.
            score = np.full(len(x_test), float(y_binary[0]) if len(y_binary) else 0.0)
            return scaler, model, score
        model.fit(train_scaled, y_binary)
        score = model.decision_function(test_scaled)
    else:
        model.fit(train_scaled, y_train)
        score = model.predict(test_scaled)
    return scaler, model, np.asarray(score, dtype="float64")


def _valid_frame(frame: pd.DataFrame, feature_cols: Sequence[str], target_col: str, kind: str) -> pd.DataFrame:
    cols = ["target_time_s", "symbol", "ewma_vol_60m", *feature_cols, target_col]
    missing = sorted(set(cols) - set(frame.columns))
    if missing:
        raise ValueError(f"frame missing evaluation columns: {missing}")
    valid = frame.loc[:, cols].copy()
    valid = valid.replace([np.inf, -np.inf], np.nan).dropna(subset=[*feature_cols, target_col])
    if kind == "direction":
        valid = valid.loc[valid[target_col].ne(0.0)]
    return valid.reset_index(drop=True)


def walk_forward_predict(
    frame: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    horizon: int,
    kind: str,
    n_folds: int = 4,
    initial_train_fraction: float = 0.50,
    embargo: int | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Fit frozen-scaled models on expanding folds and return OOS scores."""
    valid = _valid_frame(frame, feature_cols, target_col, kind)
    folds = make_walk_forward_splits(
        valid["target_time_s"].to_numpy(),
        horizon=horizon,
        n_folds=n_folds,
        initial_train_fraction=initial_train_fraction,
        embargo=embargo,
    )
    predictions: list[pd.DataFrame] = []
    fold_info: list[dict] = []
    x = valid.loc[:, feature_cols].to_numpy(dtype="float64")
    y = valid[target_col].to_numpy(dtype="float64")
    for fold in folds:
        scaler, model, score = _fit_score(x[fold.train_idx], y[fold.train_idx], x[fold.test_idx], kind)
        del scaler, model
        test = valid.iloc[fold.test_idx].copy()
        test["score"] = score
        test["fold"] = fold.fold
        test["prediction"] = score if kind == "volatility" else np.sign(score)
        predictions.append(test)
        fold_info.append(
            {
                "fold": fold.fold,
                "n_train": int(len(fold.train_idx)),
                "n_test": int(len(fold.test_idx)),
                "train_end_time_s": fold.train_end_time_s,
                "test_start_time_s": fold.test_start_time_s,
                "test_end_time_s": fold.test_end_time_s,
                "embargo_minutes": max(horizon, int(embargo or horizon)),
            }
        )
    if not predictions:
        raise ValueError("walk-forward produced no predictions")
    return pd.concat(predictions, ignore_index=True), fold_info


def baseline_scores(frame: pd.DataFrame, horizon: int, kind: str) -> dict[str, np.ndarray]:
    """Pre-registered baseline scores on a frame already aligned at t."""
    if kind == "direction":
        required = {"ret_1m", "ret_15m", f"forward_return_{horizon}m"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"baseline columns missing: {missing}")
        return {
            "persistence": frame["ret_1m"].to_numpy(dtype="float64"),
            "zero_change": np.zeros(len(frame), dtype="float64"),
            "momentum": frame["ret_15m"].to_numpy(dtype="float64"),
        }
    if kind == "volatility":
        if "ewma_vol_60m" not in frame:
            raise ValueError("EWMA baseline column missing")
        return {"ewma": frame["ewma_vol_60m"].to_numpy(dtype="float64") * np.sqrt(horizon)}
    raise ValueError(f"unknown model kind: {kind}")


def evaluate_predictions(predictions: pd.DataFrame, horizon: int, kind: str) -> dict:
    """Aggregate scores and baselines for one asset/horizon/model."""
    target_col = f"forward_return_{horizon}m" if kind == "direction" else f"forward_vol_{horizon}m"
    target = predictions[target_col].to_numpy(dtype="float64")
    result: dict = {"model": direction_metrics(target, predictions["score"]) if kind == "direction" else volatility_metrics(target, predictions["score"], predictions["ewma_vol_60m"] * np.sqrt(horizon))}
    result["folds"] = predictions.groupby("fold", sort=True).size().astype(int).to_dict()
    for name, score in baseline_scores(predictions, horizon, kind).items():
        result.setdefault("baselines", {})
        if kind == "direction":
            result["baselines"][name] = direction_metrics(target, score)
        else:
            result["baselines"][name] = volatility_metrics(target, score)
    if kind == "direction":
        model_hit = result["model"].get("hit_rate", float("nan"))
        for name, metrics in result["baselines"].items():
            result["model"][f"hit_delta_vs_{name}"] = model_hit - metrics.get("hit_rate", float("nan"))
            result["model"][f"ic_delta_vs_{name}"] = result["model"].get("ic_spearman", float("nan")) - metrics.get("ic_spearman", float("nan"))
    else:
        for name, metrics in result["baselines"].items():
            result["model"][f"mse_delta_vs_{name}"] = result["model"].get("mse", float("nan")) - metrics.get("mse", float("nan"))
    return result


def _block_sample_indices(n: int, rng: np.random.Generator, block_len: int) -> np.ndarray:
    if n <= 0:
        return np.empty(0, dtype="int64")
    block_len = max(1, min(int(block_len), n))
    starts = rng.integers(0, n, size=(n + block_len - 1) // block_len)
    offsets = np.arange(block_len, dtype="int64")
    indices = (starts[:, None] + offsets[None, :]) % n
    return indices.ravel()[:n]


def _block_mean_bootstrap(
    values: np.ndarray,
    n_boot: int,
    block_len: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Exact moving-block bootstrap of a mean from precomputed cyclic block sums.

    This reproduces ``values[_block_sample_indices(n, rng, block_len)].mean()``
    for every replicate (same block starts, same cyclic wrap, same truncation
    to ``n``) but sums precomputed 60-row block sums instead of gathering the
    index array, so high replicate counts stay cheap.  It is used only for
    metrics that are means of per-row paired differences (hit rate, MSE).
    """
    n = len(values)
    block_len = max(1, min(int(block_len), n))
    blocks = (n + block_len - 1) // block_len
    full_blocks = n // block_len
    remainder = n - full_blocks * block_len
    cumulative = np.concatenate([[0.0], np.cumsum(np.asarray(values, dtype="float64"))])
    starts = np.arange(n)
    block_sums = np.empty(n, dtype="float64")
    direct = starts <= n - block_len
    block_sums[direct] = cumulative[starts[direct] + block_len] - cumulative[starts[direct]]
    wrapped = starts[~direct]
    if wrapped.size:
        block_sums[wrapped] = (cumulative[n] - cumulative[wrapped]) + cumulative[wrapped + block_len - n]
    tail_offsets = np.arange(remainder)
    boot = np.empty(int(n_boot), dtype="float64")
    for i in range(int(n_boot)):
        chosen = rng.integers(0, n, size=blocks)
        total = float(block_sums[chosen[:full_blocks]].sum())
        if remainder:
            tail_start = int(chosen[full_blocks])
            total += float(values[(tail_start + tail_offsets) % n].sum())
        boot[i] = total / float(n)
    return boot


def paired_bootstrap_delta(
    y_return: Sequence[float],
    score_a: Sequence[float],
    score_b: Sequence[float],
    metric: str = "hit_rate",
    n_boot: int = 500,
    block_len: int = 60,
    seed: int = 20260914,
    max_points: int = 50_000,
) -> dict:
    """Paired moving-block bootstrap for candidate-vs-reference deltas.

    ``score_a`` and ``score_b`` are evaluated on exactly the same OOS rows.
    ``a_minus_b`` is the observed delta.  For very large test sets, a fixed
    deterministic prefix is used solely to bound the bootstrap computation;
    the reported point estimate still uses every row.

    The two-sided p-value has a resolution floor of ``2 / (n_boot + 1)``: with
    the default 500 replicates no p-value can fall below 0.00399, so callers
    applying a stricter threshold must raise ``n_boot``.  Hit-rate and MSE
    deltas are means of per-row paired differences and use the exact
    block-sum bootstrap (:func:`_block_mean_bootstrap`), which is cheap enough
    for several thousand replicates; the Spearman IC delta resamples ranks and
    remains O(n_boot * n log n).
    """
    y = np.asarray(y_return, dtype="float64")
    a = np.asarray(score_a, dtype="float64")
    b = np.asarray(score_b, dtype="float64")
    good = np.isfinite(y) & np.isfinite(a) & np.isfinite(b)
    if metric == "hit_rate":
        good &= y != 0.0
    y, a, b = y[good], a[good], b[good]
    if len(y) == 0:
        return {"delta": float("nan"), "p_value": float("nan"), "n": 0, "n_boot": 0}
    if metric == "hit_rate":
        observed_values = (np.sign(a) == np.sign(y)).astype("float64") - (np.sign(b) == np.sign(y)).astype("float64")
        observed = float(observed_values.mean())
    elif metric == "ic_spearman":
        observed = _spearman(y, a) - _spearman(y, b)
    elif metric == "mse":
        observed_values = (a - y) ** 2 - (b - y) ** 2
        observed = float(observed_values.mean())
    else:
        raise ValueError(f"unsupported paired metric: {metric}")
    rng = np.random.default_rng(seed)
    # Keep temporal ordering and paired rows.  The point estimate is not capped.
    if len(y) > max_points:
        yb, ab, bb = y[:max_points], a[:max_points], b[:max_points]
    else:
        yb, ab, bb = y, a, b
    n = len(yb)
    if metric == "hit_rate":
        vals = (np.sign(ab) == np.sign(yb)).astype("float64") - (np.sign(bb) == np.sign(yb)).astype("float64")
        boot = _block_mean_bootstrap(vals, n_boot, block_len, rng)
    elif metric == "mse":
        vals = (ab - yb) ** 2 - (bb - yb) ** 2
        boot = _block_mean_bootstrap(vals, n_boot, block_len, rng)
    else:
        boot = np.empty(n_boot, dtype="float64")
        for i in range(n_boot):
            idx = _block_sample_indices(n, rng, block_len)
            ranks_y = rankdata(yb[idx])
            boot[i] = _ranked_correlation(ranks_y, rankdata(ab[idx])) - _ranked_correlation(
                ranks_y, rankdata(bb[idx])
            )
    finite = boot[np.isfinite(boot)]
    if len(finite):
        p_value = float(min(1.0, 2.0 * min((1.0 + np.sum(finite <= 0.0)) / (len(finite) + 1.0), (1.0 + np.sum(finite >= 0.0)) / (len(finite) + 1.0))))
        ci = [float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))]
    else:
        p_value, ci = float("nan"), [float("nan"), float("nan")]
    return {
        "delta": float(observed),
        "p_value": p_value,
        "bootstrap_ci95": ci,
        "n": int(len(y)),
        "bootstrap_n": int(len(finite)),
        "bootstrap_points": int(n),
        "block_len": int(block_len),
    }


def shuffled_target_control(
    frame: pd.DataFrame,
    feature_cols: Sequence[str],
    horizon: int,
    kind: str = "direction",
    n_folds: int = 3,
    seed: int = 20260914,
) -> dict:
    """Run the pre-registered shuffled-target leakage control.

    Shuffling is performed before fold construction and the permutation is
    never used to select features or hyperparameters.  A valid implementation
    should produce direction near chance and no positive IC.
    """
    target_col = f"forward_return_{horizon}m" if kind == "direction" else f"forward_vol_{horizon}m"
    valid = _valid_frame(frame, feature_cols, target_col, kind)
    rng = np.random.default_rng(seed)
    shuffled = valid.copy()
    shuffled[target_col] = rng.permutation(shuffled[target_col].to_numpy())
    predictions, folds = walk_forward_predict(
        shuffled,
        feature_cols,
        target_col,
        horizon,
        kind,
        n_folds=n_folds,
        initial_train_fraction=0.50,
        embargo=horizon,
    )
    metrics = evaluate_predictions(predictions, horizon, kind)
    return {"metrics": metrics["model"], "folds": folds}


def registered_feature_columns(include_gcp: bool = True) -> tuple[str, ...]:
    """Return the exact pre-registered model columns for an ablation arm.

    Kept for existing callers; the arm registry itself lives in
    :func:`features.feature_columns_for_arm`.
    """
    return FEATURE_ARMS["with_gcp" if include_gcp else "market_only"]


__all__ = [
    "Fold",
    "HORIZONS",
    "MARKET_FEATURES",
    "GCP_FEATURES",
    "make_walk_forward_splits",
    "wilson_interval",
    "direction_metrics",
    "volatility_metrics",
    "walk_forward_predict",
    "evaluate_predictions",
    "paired_bootstrap_delta",
    "shuffled_target_control",
    "registered_feature_columns",
]
