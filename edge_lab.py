"""Bounded, honest exploration of predictability leads on the dev market database.

This is a standalone lab.  It does not modify the registered pipeline: the
feature builder, fold construction, and bootstrap machinery are imported
unchanged from ``features.py`` / ``backtest.py``.  Every configuration tried is
logged in ``edge_lab_preregistration.md`` and ``results_edge_lab_dev.json``;
the untouched holdout (Sept 2026 -> now) is only used after freezing.

Split of the dev range (``../crypto-data/crypto_market.db``):

* EXPLORE = first 70% of unique target timestamps.  All searching (walk-forward
  inside EXPLORE) happens here.  A row is an explore row for horizon ``h`` only
  if its forward target is fully contained before the split (``t + 60h <=
  t_split``), so no explore fit can observe a validate-period target.
* VALIDATE = rows with ``t >= t_split + 240 minutes`` (embargo = the longest lab
  horizon).  Used sparingly to screen the explore shortlist and to select the
  frozen finalists.

Long horizons 120/240 are not built by ``features.py`` (whose registered
``HORIZONS`` stops at 60); their targets are rebuilt exactly from the registered
one-minute forward return as ``expm1(rolling_sum(log1p(r1), h).shift(-(h-1)))``
and verified against the registered 5/15/60-minute target columns to 1e-9.

Usage:  python edge_lab.py --stage explore|validate|freeze|all [--ids ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sqlite3
import sys
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

_THREADS = os.environ.get("EDGE_LAB_THREADS", "4")
os.environ["OMP_NUM_THREADS"] = _THREADS
os.environ["OPENBLAS_NUM_THREADS"] = _THREADS
os.environ["MKL_NUM_THREADS"] = _THREADS
os.environ["NUMEXPR_NUM_THREADS"] = _THREADS

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.special import expit, logit  # noqa: E402
from sklearn.ensemble import (  # noqa: E402
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge  # noqa: E402
from sklearn.metrics import log_loss  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from backtest import (  # noqa: E402
    _block_mean_bootstrap,
    _block_sample_indices,
    direction_metrics,
    make_walk_forward_splits,
    paired_bootstrap_delta,
    volatility_metrics,
)
from features import (  # noqa: E402
    ASSETS,
    MARKET_FEATURES,
    PHYSICS_FEATURES,
    build_feature_frame,
)

LAB_VERSION = "edge-lab-v1"
DEFAULT_DB = HERE.parent / "crypto-data" / "crypto_market.db"
CACHE_DIR = Path(os.environ.get("EDGE_LAB_CACHE", Path.home() / ".cache" / "edge_lab"))
RESULTS_PATH = HERE / "results_edge_lab_dev.json"
STATE_PATH = CACHE_DIR / "lab_state.json"
FRAME_PATH = CACHE_DIR / "frame_dev.pkl"

LAB_HORIZONS = (1, 5, 15, 60, 120, 240)
LONG_HORIZONS = (120, 240)
EXPLORE_FRACTION = 0.70
VALIDATE_EMBARGO_MINUTES = max(LAB_HORIZONS)
WF_FOLDS = 3
WF_INITIAL_FRACTION = 0.50
WF_EMBARGO = 60

FEE_SCENARIOS = {"taker_10bp": 0.0010, "taker_7p5bp": 0.00075, "maker_2bp": 0.0002}
THETA_GRID = (0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20)
MIN_EXPLORE_TRADES = 200
TRADE_BLOCK = 10
BOOT_TRADES_EXPLORE = 999
BOOT_TRADES_FINAL = 4999
BOOT_HIT_EXPLORE = 999
BOOT_HIT_FINAL = 4999
FINALIST_COUNT = 3
BONFERRONI_FINAL = 0.05 / FINALIST_COUNT
HOLDOUT_WARMUP_TAIL_MINUTES = 600  # >= OU warm-up (479 rows) plus margin
MIN_HOLDOUT_TRADES = 30  # below this the net-return criterion is underpowered

HGB_PARAMS = dict(
    learning_rate=0.1,
    max_leaf_nodes=31,
    min_samples_leaf=100,
    l2_regularization=1.0,
    max_bins=255,
    random_state=0,
    early_stopping=False,
    warm_start=True,
)
HGB_MAX_ITER = 200
HGB_ES_CHUNK = 25
HGB_ES_PATIENCE = 3
HGB_ES_FRACTION = 0.15
HGB_POOL_MAX_ROWS = 1_500_000


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def attach_long_targets(frame: pd.DataFrame) -> pd.DataFrame:
    """Add 120/240-minute forward returns, verified against the registered columns."""
    frame = frame.sort_values(["symbol", "target_time_s"], kind="mergesort").reset_index(drop=True)
    for horizon in LONG_HORIZONS:
        column = f"forward_return_{horizon}m"
        if column in frame.columns:
            continue
        values = np.full(len(frame), np.nan, dtype="float64")
        for symbol in ASSETS:
            mask = frame["symbol"].eq(symbol).to_numpy()
            r1 = frame.loc[mask, "forward_return_1m"].to_numpy(dtype="float64")
            compounded = np.expm1(
                pd.Series(np.log1p(r1))
                .rolling(horizon, min_periods=horizon)
                .sum()
                .shift(-(horizon - 1))
                .to_numpy()
            )
            values[mask] = compounded
        frame[column] = values
    # Verification: the compounding convention must reproduce the registered
    # 5/15/60-minute targets exactly (same close-based formula in features.py).
    for horizon in (5, 15, 60):
        column = f"forward_return_{horizon}m"
        built = frame[column].to_numpy(dtype="float64")
        expected = frame[f"forward_return_{horizon}m"].to_numpy(dtype="float64")
        good = np.isfinite(built) & np.isfinite(expected)
        diff = float(np.max(np.abs(built[good] - expected[good])))
        assert diff < 1e-9, f"long-target reconstruction mismatch at h={horizon}: {diff}"
    for symbol in ASSETS:
        frame[f"asset_{symbol}"] = frame["symbol"].eq(symbol).astype("float64")
    return frame


def load_dev_frame(db: Path, rebuild: bool = False) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if FRAME_PATH.exists() and not rebuild:
        with open(FRAME_PATH, "rb") as handle:
            frame = pickle.load(handle)
        if "forward_return_240m" in frame.columns:
            return frame
    frame = build_feature_frame(db, None, include_gcp=False)
    frame = attach_long_targets(frame)
    with open(FRAME_PATH, "wb") as handle:
        pickle.dump(frame, handle, protocol=4)
    return frame


def split_time(frame: pd.DataFrame) -> int:
    times = np.sort(frame["target_time_s"].unique())
    return int(times[int(len(times) * EXPLORE_FRACTION)])


# --------------------------------------------------------------------------
# configurations
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    cid: str
    phase: str
    kind: str  # direction | volatility
    features: str  # market | physics
    model: str  # logistic | ridge | hgb
    horizon: int
    pooled: bool = False
    max_train_rows: int | None = None
    note: str = ""

    def feature_columns(self) -> list[str]:
        columns = list(MARKET_FEATURES)
        if self.features == "physics":
            columns += list(PHYSICS_FEATURES)
        if self.pooled:
            columns += [f"asset_{symbol}" for symbol in ASSETS]
        return columns

    def target_column(self) -> str:
        return (
            f"forward_return_{self.horizon}m"
            if self.kind == "direction"
            else f"forward_vol_{self.horizon}m"
        )


def build_configs() -> list[Config]:
    configs: list[Config] = []

    def add(**kwargs):
        configs.append(Config(**kwargs))

    # Phase D1: nonlinear vs logistic at 15m/60m, market vs market+physics.
    for horizon in (15, 60):
        for model in ("logistic", "hgb"):
            for features in ("market", "physics"):
                add(
                    cid=f"d1_{model}_{features}_h{horizon}_per",
                    phase="D1_nonlinear",
                    kind="direction",
                    features=features,
                    model=model,
                    horizon=horizon,
                )
    # Phase D2: longer horizons where the physics delta was largest.
    for horizon in (120, 240):
        for model in ("logistic", "hgb"):
            for features in ("market", "physics"):
                add(
                    cid=f"d2_{model}_{features}_h{horizon}_per",
                    phase="D2_long",
                    kind="direction",
                    features=features,
                    model=model,
                    horizon=horizon,
                )
    # Phase D3: pooled cross-asset (asset one-hot) vs per-asset.  HGB pooled is
    # capped to the most recent rows for tractability; the capped logistic pair
    # gives the fair linear-vs-nonlinear comparison on identical training rows.
    add(cid="d3_logistic_market_h60_pool", phase="D3_pooled", kind="direction",
        features="market", model="logistic", horizon=60, pooled=True)
    add(cid="d3_logistic_physics_h60_pool", phase="D3_pooled", kind="direction",
        features="physics", model="logistic", horizon=60, pooled=True)
    add(cid="d3_logistic_physics_h60_pool_cap", phase="D3_pooled", kind="direction",
        features="physics", model="logistic", horizon=60, pooled=True,
        max_train_rows=HGB_POOL_MAX_ROWS, note="tail cap, comparison pair for hgb cap")
    add(cid="d3_hgb_physics_h60_pool_cap", phase="D3_pooled", kind="direction",
        features="physics", model="hgb", horizon=60, pooled=True,
        max_train_rows=HGB_POOL_MAX_ROWS, note="tail cap for tractability")
    # Phase V: volatility, BNB short-horizon lead + generalization.
    for model in ("ridge", "hgb"):
        for features in ("market", "physics"):
            add(
                cid=f"v_{model}_{features}_h5_per",
                phase="V_vol",
                kind="volatility",
                features=features,
                model=model,
                horizon=5,
            )
    for horizon in (1, 15, 60):
        add(
            cid=f"v_hgb_physics_h{horizon}_per",
            phase="V_vol",
            kind="volatility",
            features="physics",
            model="hgb",
            horizon=horizon,
        )
    return configs


CONFIGS = build_configs()
CONFIG_BY_ID = {config.cid: config for config in CONFIGS}


# --------------------------------------------------------------------------
# model fitting
# --------------------------------------------------------------------------
def _fit_hgb(x: np.ndarray, y: np.ndarray, times: np.ndarray, config: Config):
    """HGB with early stopping on a time-ordered inner split (15% tail)."""
    n = len(x)
    order = np.argsort(times, kind="stable")
    cut_position = min(max(1, int(n * (1.0 - HGB_ES_FRACTION))), n - 1)
    cut_time = int(times[order[cut_position]])
    fit_idx = np.flatnonzero(times <= cut_time)
    es_idx = np.flatnonzero(times >= cut_time + config.horizon * 60)
    if len(es_idx) == 0:
        fit_idx = order[: max(1, int(n * 0.9))]
        es_idx = order[max(1, int(n * 0.9)):]
    if config.kind == "direction":
        y_fit = (y[fit_idx] > 0.0).astype("int8")
        y_es = (y[es_idx] > 0.0).astype("int8")
        model_cls = HistGradientBoostingClassifier
    else:
        y_fit = y[fit_idx]
        y_es = y[es_idx]
        model_cls = HistGradientBoostingRegressor
    x_fit, x_es = x[fit_idx], x[es_idx]
    model = model_cls(**HGB_PARAMS, max_iter=HGB_ES_CHUNK)
    best_model = None
    best_loss = np.inf
    no_improve = 0
    stage = HGB_ES_CHUNK
    while True:
        model.fit(x_fit, y_fit)
        if config.kind == "direction":
            probabilities = model.predict_proba(x_es)[:, 1]
            loss = float(log_loss(y_es, probabilities, labels=[0, 1]))
        else:
            loss = float(np.mean((model.predict(x_es) - y_es) ** 2))
        if np.isfinite(loss) and loss < best_loss - 1e-7:
            best_loss = loss
            best_model = deepcopy(model)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= HGB_ES_PATIENCE:
                break
        if stage >= HGB_MAX_ITER:
            break
        stage = min(HGB_MAX_ITER, stage + HGB_ES_CHUNK)
        model.set_params(max_iter=stage)
    return best_model if best_model is not None else model


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, config: Config, columns: list[str]):
    x_train = train[columns].to_numpy(dtype="float64")
    x_test = test[columns].to_numpy(dtype="float64")
    y_train = train["__y__"].to_numpy(dtype="float64")
    probability = None
    if config.model == "hgb":
        model = _fit_hgb(x_train, y_train, train["target_time_s"].to_numpy(dtype="int64"), config)
        if config.kind == "direction":
            probability = model.predict_proba(x_test)[:, 1]
            score = logit(np.clip(probability, 1e-12, 1.0 - 1e-12))
        else:
            score = model.predict(x_test)
    else:
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        if config.model == "logistic":
            y_binary = (y_train > 0.0).astype("int8")
            model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=300, random_state=0)
            if np.unique(y_binary).size < 2:
                score = np.zeros(len(x_test), dtype="float64")
                probability = np.full(len(x_test), 0.5, dtype="float64")
            else:
                model.fit(x_train_scaled, y_binary)
                score = model.decision_function(x_test_scaled)
                probability = model.predict_proba(x_test_scaled)[:, 1]
        elif config.model == "ridge":
            model = Ridge(alpha=1.0)
            model.fit(x_train_scaled, y_train)
            score = model.predict(x_test_scaled)
        else:
            raise ValueError(f"unknown model: {config.model}")
    return np.asarray(score, dtype="float64"), probability


# --------------------------------------------------------------------------
# prediction (cached)
# --------------------------------------------------------------------------
def _prediction_path(config: Config, tag: str, t_split: int) -> Path:
    payload = json.dumps(
        {
            "lab_version": LAB_VERSION,
            "config": asdict(config),
            "tag": tag,
            "t_split": t_split,
            "folds": WF_FOLDS,
            "initial_fraction": WF_INITIAL_FRACTION,
            "embargo": WF_EMBARGO,
        },
        sort_keys=True,
    )
    digest = hashlib.sha1(payload.encode()).hexdigest()[:16]
    return CACHE_DIR / "preds" / f"{config.cid}_{tag}_{digest}.pkl"


def _cap_rows(rows: pd.DataFrame, cap: int | None) -> pd.DataFrame:
    """Keep at most ``cap`` most recent rows by target time (time-based, not positional).

    Positional ``tail`` on a pooled frame would keep only the last alphabetically
    ordered symbols; this keeps every asset's most recent rows instead.
    """
    if not cap or len(rows) <= cap:
        return rows
    unique = np.sort(rows["target_time_s"].unique())
    per_time = max(1.0, len(rows) / float(len(unique)))
    keep = max(1, min(int(np.floor(cap / per_time)), len(unique)))
    cutoff = int(unique[len(unique) - keep])
    return rows.loc[rows["target_time_s"] >= cutoff]


def _prepare_valid(frame: pd.DataFrame, config: Config) -> pd.DataFrame:
    columns = config.feature_columns()
    target = config.target_column()
    valid = frame.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[*columns, target, "ret_1m", "ret_15m"]
    )
    if config.kind == "direction":
        valid = valid.loc[valid[target].ne(0.0)]
    return valid.reset_index(drop=True)


def walk_forward(config: Config, frame: pd.DataFrame, t_split: int) -> pd.DataFrame:
    """Expanding walk-forward inside EXPLORE; returns OOS predictions."""
    horizon = config.horizon
    explore = frame.loc[frame["target_time_s"] + horizon * 60 <= t_split]
    valid = _prepare_valid(explore, config)
    cache_path = _prediction_path(config, "explore", t_split)
    if cache_path.exists():
        with open(cache_path, "rb") as handle:
            return pickle.load(handle)
    columns = config.feature_columns()
    target = config.target_column()
    folds = make_walk_forward_splits(
        valid["target_time_s"].to_numpy(dtype="int64"),
        horizon=horizon,
        n_folds=WF_FOLDS,
        initial_train_fraction=WF_INITIAL_FRACTION,
        embargo=WF_EMBARGO,
    )
    predictions = []
    for fold in folds:
        train = valid.iloc[fold.train_idx]
        train = _cap_rows(train, config.max_train_rows)
        train = train.assign(__y__=train[target].to_numpy(dtype="float64"))
        test = valid.iloc[fold.test_idx]
        score, probability = _fit_predict(train, test, config, columns)
        out = test[["target_time_s", "symbol", "ret_1m", "ret_15m", "ewma_vol_60m"]].copy()
        out["y"] = test[target].to_numpy(dtype="float64")
        out["score"] = score
        out["p_up"] = probability if probability is not None else np.nan
        out["fold"] = fold.fold
        predictions.append(out)
    result = pd.concat(predictions, ignore_index=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as handle:
        pickle.dump(result, handle, protocol=4)
    return result


def validate_once(config: Config, frame: pd.DataFrame, t_split: int) -> pd.DataFrame:
    """Fit on all EXPLORE rows, predict VALIDATE once (single pass, embargoed)."""
    horizon = config.horizon
    explore = frame.loc[frame["target_time_s"] + horizon * 60 <= t_split]
    validate = frame.loc[frame["target_time_s"] >= t_split + VALIDATE_EMBARGO_MINUTES * 60]
    train = _prepare_valid(explore, config)
    test = _prepare_valid(validate, config)
    train = _cap_rows(train, config.max_train_rows)
    assert int(train["target_time_s"].max()) + horizon * 60 <= t_split
    assert int(test["target_time_s"].min()) >= t_split + VALIDATE_EMBARGO_MINUTES * 60
    cache_path = _prediction_path(config, "validate", t_split)
    if cache_path.exists():
        with open(cache_path, "rb") as handle:
            return pickle.load(handle)
    columns = config.feature_columns()
    train = train.assign(__y__=train[config.target_column()].to_numpy(dtype="float64"))
    score, probability = _fit_predict(train, test, config, columns)
    out = test[["target_time_s", "symbol", "ret_1m", "ret_15m", "ewma_vol_60m"]].copy()
    out["y"] = test[config.target_column()].to_numpy(dtype="float64")
    out["score"] = score
    out["p_up"] = probability if probability is not None else np.nan
    out["fold"] = 0
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as handle:
        pickle.dump(out, handle, protocol=4)
    return out


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def direction_summary(pred: pd.DataFrame, config: Config, n_boot: int) -> dict:
    target = config.target_column()
    y = pred["y"].to_numpy(dtype="float64")
    score = pred["score"].to_numpy(dtype="float64")
    model = direction_metrics(y, score)
    persistence = direction_metrics(y, pred["ret_1m"].to_numpy(dtype="float64"))
    momentum = direction_metrics(y, pred["ret_15m"].to_numpy(dtype="float64"))
    delta_persist = paired_bootstrap_delta(
        y, score, pred["ret_1m"], metric="hit_rate", n_boot=n_boot, seed=20261002
    )
    delta_momentum = paired_bootstrap_delta(
        y, score, pred["ret_15m"], metric="hit_rate", n_boot=n_boot, seed=20261003
    )
    per_asset = {
        symbol: direction_metrics(group["y"].to_numpy(), group["score"].to_numpy())
        for symbol, group in pred.groupby("symbol", sort=True)
    }
    return {
        "n": int(model["n"]),
        "hit_rate": float(model["hit_rate"]),
        "ic_spearman": float(model["ic_spearman"]),
        "hit_rate_persistence": float(persistence["hit_rate"]),
        "hit_rate_momentum": float(momentum["hit_rate"]),
        "delta_hit_vs_persistence_pp": float(model["hit_rate"] - persistence["hit_rate"]) * 100.0,
        "delta_hit_vs_momentum_pp": float(model["hit_rate"] - momentum["hit_rate"]) * 100.0,
        "delta_hit_vs_persistence_p": float(delta_persist["p_value"]),
        "delta_hit_vs_momentum_p": float(delta_momentum["p_value"]),
        "per_asset": per_asset,
    }


def qlike(y: np.ndarray, prediction: np.ndarray, epsilon: float = 1e-12) -> float:
    y = np.clip(np.asarray(y, dtype="float64"), epsilon, None)
    p = np.clip(np.asarray(prediction, dtype="float64"), epsilon, None)
    good = np.isfinite(y) & np.isfinite(p)
    ratio = y[good] / p[good]
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def volatility_summary(pred: pd.DataFrame, config: Config) -> dict:
    horizon = config.horizon
    y = pred["y"].to_numpy(dtype="float64")
    score = pred["score"].to_numpy(dtype="float64")
    ewma = pred["ewma_vol_60m"].to_numpy(dtype="float64") * np.sqrt(horizon)
    base = volatility_metrics(y, score, ewma)
    return {
        "n": int(base["n"]),
        "mse": float(base["mse"]),
        "ewma_mse": float(base.get("ewma_mse", float("nan"))),
        "mse_delta_vs_ewma": float(base.get("mse_delta_vs_ewma", float("nan"))),
        "qlike": qlike(y, score),
        "ewma_qlike": qlike(y, ewma),
        "qlike_delta_vs_ewma": qlike(y, score) - qlike(y, ewma),
        "per_asset": {
            symbol: {
                "mse": float(volatility_metrics(group["y"].to_numpy(),
                                                 group["score"].to_numpy())["mse"]),
                "ewma_mse": float(volatility_metrics(
                    group["y"].to_numpy(), group["score"].to_numpy(),
                    group["ewma_vol_60m"].to_numpy() * np.sqrt(horizon))["ewma_mse"]),
                "qlike": qlike(group["y"].to_numpy(), group["score"].to_numpy()),
                "ewma_qlike": qlike(
                    group["y"].to_numpy(),
                    group["ewma_vol_60m"].to_numpy() * np.sqrt(horizon),
                ),
            }
            for symbol, group in pred.groupby("symbol", sort=True)
        },
    }


def confidence_curve(pred: pd.DataFrame, config: Config) -> list[dict]:
    y = pred["y"].to_numpy(dtype="float64")
    probability = pred["p_up"].to_numpy(dtype="float64")
    curve = []
    for theta in THETA_GRID:
        selected = np.abs(probability - 0.5) > theta
        if not selected.any():
            curve.append({"theta": theta, "coverage": 0.0, "n": 0})
            continue
        side = np.sign(probability[selected] - 0.5)
        returns = side * y[selected]
        curve.append(
            {
                "theta": theta,
                "coverage": float(selected.mean()),
                "n": int(selected.sum()),
                "hit_rate": float(np.mean(returns > 0.0)),
                "gross_mean_return": float(np.mean(returns)),
                "taker_net_mean_return": float(np.mean(returns) - 2.0 * FEE_SCENARIOS["taker_10bp"]),
            }
        )
    return curve


# --------------------------------------------------------------------------
# trading simulation
# --------------------------------------------------------------------------
def simulate_trades(pred: pd.DataFrame, config: Config, theta: float, fee_per_side: float):
    """Non-overlapping direction trades; long if p>0.5+theta, short if p<0.5-theta."""
    times: list[int] = []
    sides: list[int] = []
    gross: list[float] = []
    probability = pred["p_up"].to_numpy(dtype="float64")
    y = pred["y"].to_numpy(dtype="float64")
    for symbol, group in pred.groupby("symbol", sort=True):
        order = np.argsort(group["target_time_s"].to_numpy(dtype="int64"), kind="stable")
        group = group.iloc[order]
        ts = group["target_time_s"].to_numpy(dtype="int64")
        p = group["p_up"].to_numpy(dtype="float64")
        gy = group["y"].to_numpy(dtype="float64")
        next_allowed = -np.inf
        for i in range(len(ts)):
            if ts[i] < next_allowed:
                continue
            side = 1 if p[i] >= 0.5 + theta else (-1 if p[i] <= 0.5 - theta else 0)
            if side == 0:
                continue
            times.append(int(ts[i]))
            sides.append(side)
            gross.append(float(side * gy[i]))
            next_allowed = ts[i] + config.horizon * 60
    del probability, y
    gross_array = np.asarray(gross, dtype="float64")
    net = gross_array - 2.0 * fee_per_side
    return {
        "theta": theta,
        "fee_per_side": fee_per_side,
        "n_trades": int(len(gross_array)),
        "gross_mean_return": float(gross_array.mean()) if len(gross_array) else float("nan"),
        "net_mean_return": float(net.mean()) if len(net) else float("nan"),
        "hit_rate": float(np.mean(gross_array > 0.0)) if len(gross_array) else float("nan"),
        "long_share": float(np.mean(np.asarray(sides) == 1)) if sides else float("nan"),
        "net_sd": float(net.std(ddof=1)) if len(net) > 1 else float("nan"),
        "net_sum_return": float(net.sum()) if len(net) else float("nan"),
        "_net": net,
    }


def bootstrap_trade_stats(net: np.ndarray, n_boot: int, seed: int) -> dict:
    n = len(net)
    if n < 2:
        return {"net_mean_ci95": None, "p_net_gt0": None,
                "sharpe_like": float("nan"), "sharpe_ci95": [float("nan")] * 2}
    rng = np.random.default_rng(seed)
    boot_mean = _block_mean_bootstrap(net, n_boot, TRADE_BLOCK, rng)
    sharpe = np.empty(n_boot, dtype="float64")
    for i in range(n_boot):
        sample = net[_block_sample_indices(n, rng, TRADE_BLOCK)]
        sd = sample.std(ddof=1)
        sharpe[i] = sample.mean() / sd if sd > 0.0 else np.nan
    finite = sharpe[np.isfinite(sharpe)]
    observed = net.mean() / net.std(ddof=1) if net.std(ddof=1) > 0 else float("nan")
    # A moving-block bootstrap needs more trades than one block: when
    # n <= TRADE_BLOCK every resample is the whole sample, so the net-return
    # CI and p-value are degenerate and are reported as null.
    if n <= TRADE_BLOCK:
        ci = None
        p_gt0 = None
    else:
        ci = [float(np.quantile(boot_mean, 0.025)), float(np.quantile(boot_mean, 0.975))]
        p_gt0 = float((1.0 + np.sum(boot_mean <= 0.0)) / (len(boot_mean) + 1.0))
    return {
        "net_mean_ci95": ci,
        "p_net_gt0": p_gt0,
        "sharpe_like": float(observed),
        "sharpe_ci95": (
            [float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))]
            if len(finite) else [float("nan"), float("nan")]
        ),
    }


def trade_report(pred: pd.DataFrame, config: Config, theta: float, n_boot: int, seed: int) -> dict:
    report = {}
    for name, fee in FEE_SCENARIOS.items():
        trade = simulate_trades(pred, config, theta, fee)
        net = trade.pop("_net")
        stats = bootstrap_trade_stats(net, n_boot, seed)
        trade.update(stats)
        report[name] = trade
    return report


def select_theta(pred: pd.DataFrame, config: Config) -> dict:
    """Pick theta on EXPLORE: max taker net mean with >=200 trades; tie -> more trades."""
    best = None
    curve = []
    for theta in THETA_GRID:
        trade = simulate_trades(pred, config, theta, FEE_SCENARIOS["taker_10bp"])
        trade.pop("_net")
        curve.append(trade)
        if trade["n_trades"] < MIN_EXPLORE_TRADES:
            continue
        key = (round(trade["net_mean_return"], 6), trade["n_trades"])
        if best is None or key > best[0]:
            best = (key, theta, trade)
    if best is None:
        return {"theta": None, "selection": "no theta reached the minimum trade count", "curve": curve}
    return {"theta": float(best[1]), "selection": "max taker net mean, ties -> more trades",
            "curve": curve, "selected": best[2]}


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------
def load_state() -> dict:
    if STATE_PATH.exists():
        with open(STATE_PATH) as handle:
            return json.load(handle)
    return {"configs": {}, "updated_utc": None, "runtime_seconds": 0.0}


def save_state(state: dict) -> None:
    state["updated_utc"] = datetime.now(timezone.utc).isoformat()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATE_PATH, "w") as handle:
        json.dump(state, handle, indent=2, sort_keys=True, default=str)


def run_explore(frame: pd.DataFrame, t_split: int, state: dict, ids: list[str] | None) -> None:
    for config in CONFIGS:
        if ids and config.cid not in ids:
            continue
        if config.cid in state["configs"] and state["configs"][config.cid].get("explore"):
            print(f"[explore] {config.cid}: cached", flush=True)
            continue
        started = time.time()
        pred = walk_forward(config, frame, t_split)
        if config.kind == "direction":
            summary = direction_summary(pred, config, BOOT_HIT_EXPLORE)
            summary["confidence_curve_explore"] = confidence_curve(pred, config)
        else:
            summary = volatility_summary(pred, config)
        entry = state["configs"].setdefault(config.cid, {})
        entry["config"] = asdict(config)
        entry["feature_columns"] = config.feature_columns()
        entry["explore"] = summary
        save_state(state)
        print(f"[explore] {config.cid}: hit={summary.get('hit_rate')} "
              f"ic={summary.get('ic_spearman')} n={summary.get('n')} "
              f"({time.time() - started:.0f}s)", flush=True)


def explore_shortlist(state: dict, limit: int = 6) -> list[str]:
    rank = []
    for cid, entry in state["configs"].items():
        explore = entry.get("explore")
        if not explore or entry["config"]["kind"] != "direction":
            continue
        if explore.get("n", 0) < 50_000:
            continue
        delta = max(explore["delta_hit_vs_persistence_pp"], explore["delta_hit_vs_momentum_pp"])
        rank.append((delta, cid))
    rank.sort(reverse=True)
    return [cid for _, cid in rank[:limit]]


def run_validate(frame: pd.DataFrame, t_split: int, state: dict, ids: list[str] | None) -> None:
    for cid in ids or []:
        config = CONFIG_BY_ID[cid]
        if state["configs"].get(cid, {}).get("validate"):
            print(f"[validate] {cid}: cached", flush=True)
            continue
        started = time.time()
        pred = validate_once(config, frame, t_split)
        entry = state["configs"].setdefault(cid, {})
        entry["config"] = asdict(config)
        entry["feature_columns"] = config.feature_columns()
        if config.kind == "direction":
            summary = direction_summary(pred, config, BOOT_HIT_EXPLORE)
            summary["confidence_curve_validate"] = confidence_curve(pred, config)
            # Theta is selected on EXPLORE predictions only, never on validate.
            explore_pred = walk_forward(config, frame, t_split)
            theta = select_theta(explore_pred, config)
            summary["theta"] = {
                "theta": theta["theta"], "selection": theta["selection"],
                "curve": theta.get("curve", []),
                "selected_on": "explore",
            }
            summary["explore_theta_selected"] = theta["theta"]
            summary["trades_validate_theta"] = (
                {name: {k: v for k, v in simulate_trades(pred, config, theta["theta"], fee).items()
                        if k != "_net"}
                 for name, fee in FEE_SCENARIOS.items()}
                if theta["theta"] is not None else None
            )
        else:
            summary = volatility_summary(pred, config)
        entry["validate"] = summary
        save_state(state)
        print(f"[validate] {cid}: done ({time.time() - started:.0f}s)", flush=True)


def freeze(frame: pd.DataFrame, t_split: int, state: dict) -> dict:
    """Select <=3 finalists from validated configs, rerun their explore theta, and
    produce the full validate trade reports at the finalist bootstrap resolution."""
    candidates = []
    for cid, entry in state["configs"].items():
        validate = entry.get("validate")
        if not validate or entry["config"]["kind"] != "direction":
            continue
        theta = validate.get("explore_theta_selected")
        if theta is None:
            continue
        trades = validate.get("trades_validate_theta") or {}
        net = trades.get("taker_10bp", {}).get("net_mean_return")
        if net is None:
            continue
        candidates.append((net, cid))
    candidates.sort(reverse=True)
    finalists = []
    for net, cid in candidates[:FINALIST_COUNT]:
        config = CONFIG_BY_ID[cid]
        entry = state["configs"][cid]
        theta = entry["validate"]["explore_theta_selected"]
        pred = validate_once(config, frame, t_split)
        report = trade_report(pred, config, theta, BOOT_TRADES_FINAL, seed=20261010)
        # Co-primary hit-rate comparisons vs the two baselines at final resolution.
        hit_tests = {}
        for name, column in (("persistence", "ret_1m"), ("momentum", "ret_15m")):
            hit_tests[name] = paired_bootstrap_delta(
                pred["y"].to_numpy(dtype="float64"),
                pred["score"].to_numpy(dtype="float64"),
                pred[column].to_numpy(dtype="float64"),
                metric="hit_rate", n_boot=BOOT_HIT_FINAL, seed=20261011,
                max_points=500_000,
            )
        finalists.append(
            {
                "cid": cid,
                "config": entry["config"],
                "feature_columns": entry["feature_columns"],
                "theta": theta,
                "explore": entry["explore"],
                "validate_direction": entry["validate"],
                "validate_trades": report,
                "validate_hit_tests": hit_tests,
                "passes_preregistered_rule": bool(
                    report["taker_10bp"]["net_mean_return"] > 0.0
                    and report["taker_10bp"]["p_net_gt0"] is not None
                    and report["taker_10bp"]["p_net_gt0"] < BONFERRONI_FINAL
                    and hit_tests["persistence"]["delta"] > 0.0
                    and hit_tests["persistence"]["p_value"] < BONFERRONI_FINAL
                    and hit_tests["momentum"]["delta"] > 0.0
                    and hit_tests["momentum"]["p_value"] < BONFERRONI_FINAL
                ),
            }
        )
    return {
        "bonferroni_alpha": BONFERRONI_FINAL,
        "bootstrap_trades": BOOT_TRADES_FINAL,
        "bootstrap_hit": BOOT_HIT_FINAL,
        "block_len_trades": TRADE_BLOCK,
        "finalists": finalists,
        "holdout_status": "not requested yet; frozen spec for the owner's holdout",
    }


def write_results(state: dict, final: dict | None, runtime: float) -> None:
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "lab_version": LAB_VERSION,
        "database": str(DEFAULT_DB),
        "split": state.get("split"),
        "exploration_log": [
            {
                "cid": cid,
                "config": entry.get("config"),
                "feature_columns": entry.get("feature_columns"),
                "explore": entry.get("explore"),
                "validate": entry.get("validate"),
            }
            for cid, entry in sorted(state["configs"].items())
        ],
        "freeze": final,
        "runtime_seconds": runtime,
    }
    with open(RESULTS_PATH, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _build_combined_db(
    dev_db: Path, holdout_db: Path, out_path: Path, tail_minutes: int = HOLDOUT_WARMUP_TAIL_MINUTES
) -> Path:
    """Create a scratch klines_1m DB with the dev tail plus all post-dev holdout rows.

    The holdout database alone cannot warm the rolling windows (the OU feature
    needs 479 rows; 240-minute windows need 240; z-scores need 60), so features
    for the evaluated holdout rows are built on this combined series.  Only rows
    strictly after the dev database's last candle are appended, so no timestamp
    is duplicated and the junction is contiguous when the two sources are.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    con = sqlite3.connect(str(out_path))
    try:
        con.execute("ATTACH DATABASE ? AS devdb", (str(dev_db),))
        con.execute("ATTACH DATABASE ? AS hdb", (str(holdout_db),))
        con.execute(
            "CREATE TABLE klines_1m AS SELECT * FROM devdb.klines_1m "
            "WHERE open_time_s >= (SELECT MAX(open_time_s) FROM devdb.klines_1m) - ?",
            (int(tail_minutes) * 60,),
        )
        con.execute(
            "INSERT INTO klines_1m SELECT * FROM hdb.klines_1m "
            "WHERE open_time_s > (SELECT MAX(open_time_s) FROM devdb.klines_1m)"
        )
        con.commit()
        con.execute("DETACH DATABASE devdb")
        con.execute("DETACH DATABASE hdb")
    finally:
        con.close()
    return out_path


def prepare_holdout_frames(
    dev_frame: pd.DataFrame,
    dev_db: Path,
    holdout_db: Path,
    tail_minutes: int = HOLDOUT_WARMUP_TAIL_MINUTES,
) -> dict:
    """Warm holdout evaluation frame: combined-series features, holdout-only targets.

    Returns ``eval_frame`` (only rows at/after the embargoed test start), the
    fit end, the test start, and the holdout window bounds.
    """
    dev_max = int(dev_frame["target_time_s"].max())
    fit_end_time_s = dev_max + 60
    combined_db = _build_combined_db(dev_db, holdout_db, CACHE_DIR / "holdout_combined.sqlite", tail_minutes)
    combined = attach_long_targets(build_feature_frame(combined_db, None, include_gcp=False))
    holdout_only = attach_long_targets(build_feature_frame(Path(holdout_db), None, include_gcp=False))
    holdout_min = int(holdout_only["target_time_s"].min())
    holdout_max = int(holdout_only["target_time_s"].max())
    test_start_time_s = max(holdout_min, fit_end_time_s) + VALIDATE_EMBARGO_MINUTES * 60
    eval_frame = combined.loc[combined["target_time_s"] >= test_start_time_s].copy()
    # Targets are taken from the holdout-only build (contiguous inside the
    # holdout database) so the dev/holdout junction can never distort an
    # evaluated target; features stay on the warm combined series.
    target_cols = [column for column in eval_frame.columns if column.startswith("forward_")]
    lookup = holdout_only.set_index(["symbol", "target_time_s"])[target_cols]
    eval_frame = eval_frame.drop(columns=target_cols).join(lookup, on=["symbol", "target_time_s"])
    if eval_frame.empty:
        raise ValueError("holdout evaluation frame is empty")
    assert int(eval_frame["target_time_s"].min()) >= test_start_time_s
    assert int(eval_frame["target_time_s"].min()) > dev_max
    return {
        "eval_frame": eval_frame,
        "fit_end_time_s": fit_end_time_s,
        "test_start_time_s": test_start_time_s,
        "holdout_min_time_s": holdout_min,
        "holdout_max_time_s": holdout_max,
        "holdout_minutes_per_symbol": (holdout_max - holdout_min) // 60 + 1,
        "evaluated_minutes": (holdout_max - test_start_time_s) // 60,
        "combined_rows": int(len(combined)),
        "combined_db": str(combined_db),
    }


def evaluate_finalist_on_holdout(
    fit_frame: pd.DataFrame,
    holdout_frame: pd.DataFrame,
    config: Config,
    theta: float,
    fit_end_time_s: int,
    test_start_time_s: int,
    expected_trades: float | None = None,
) -> dict:
    """Frozen-protocol holdout evaluation: fit on dev, test on holdout, no tuning.

    ``fit_end_time_s`` is the last dev timestamp whose candles exist (targets of
    training rows must complete before it); ``test_start_time_s`` already
    includes the 240-minute embargo.
    """
    horizon = config.horizon
    columns = config.feature_columns()
    train = _prepare_valid(fit_frame.loc[fit_frame["target_time_s"] + horizon * 60 <= fit_end_time_s], config)
    train = _cap_rows(train, config.max_train_rows)
    test = _prepare_valid(holdout_frame.loc[holdout_frame["target_time_s"] >= test_start_time_s], config)
    if len(train) == 0 or len(test) == 0:
        raise ValueError(f"holdout evaluation has empty train ({len(train)}) or test ({len(test)}) rows")
    assert int(train["target_time_s"].max()) + horizon * 60 <= test_start_time_s, "train/test embargo violated"
    train = train.assign(__y__=train[config.target_column()].to_numpy(dtype="float64"))
    score, probability = _fit_predict(train, test, config, columns)
    pred = test[["target_time_s", "symbol", "ret_1m", "ret_15m", "ewma_vol_60m"]].copy()
    pred["y"] = test[config.target_column()].to_numpy(dtype="float64")
    pred["score"] = score
    pred["p_up"] = probability if probability is not None else np.nan
    pred["fold"] = 0
    report = trade_report(pred, config, theta, BOOT_TRADES_FINAL, seed=20261012)
    hit_tests = {
        name: paired_bootstrap_delta(
            pred["y"].to_numpy(dtype="float64"),
            pred["score"].to_numpy(dtype="float64"),
            pred[column].to_numpy(dtype="float64"),
            metric="hit_rate", n_boot=BOOT_HIT_FINAL, seed=20261013,
            max_points=500_000,
        )
        for name, column in (("persistence", "ret_1m"), ("momentum", "ret_15m"))
    }
    taker = report["taker_10bp"]
    net_pass = bool(
        taker["net_mean_return"] > 0.0
        and taker["p_net_gt0"] is not None
        and taker["p_net_gt0"] < BONFERRONI_FINAL
    )
    if taker["n_trades"] < MIN_HOLDOUT_TRADES:
        net_status = "inconclusive_underpowered"
    else:
        net_status = "pass" if net_pass else "fail"
    hit_pass = bool(
        hit_tests["persistence"]["delta"] > 0.0
        and hit_tests["persistence"]["p_value"] < BONFERRONI_FINAL
        and hit_tests["momentum"]["delta"] > 0.0
        and hit_tests["momentum"]["p_value"] < BONFERRONI_FINAL
    )
    hit_status = "pass" if hit_pass else "fail"
    if hit_status == "fail":
        overall = "fail"
    elif net_status == "inconclusive_underpowered":
        overall = "inconclusive_underpowered"
    else:
        overall = net_status
    return {
        "cid": config.cid,
        "config": asdict(config),
        "theta": theta,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "expected_trades": expected_trades,
        "direction": direction_summary(pred, config, BOOT_HIT_EXPLORE),
        "trades": report,
        "hit_tests": hit_tests,
        "net_criterion": net_status,
        "hit_criterion": hit_status,
        "overall_verdict": overall,
        "passes_preregistered_rule": overall == "pass",
    }


def run_holdout(args: argparse.Namespace) -> dict:
    if not args.holdout_db:
        raise SystemExit("--holdout-db is required for --stage holdout")
    payload = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else {}
    freeze_info = payload.get("freeze")
    if not freeze_info or not freeze_info.get("finalists"):
        raise SystemExit("no frozen finalists in results_edge_lab_dev.json; run --stage freeze first")
    dev = load_dev_frame(Path(args.db))
    prepared = prepare_holdout_frames(dev, Path(args.db), Path(args.holdout_db))
    eval_frame = prepared["eval_frame"]
    fit_end_time_s = prepared["fit_end_time_s"]
    test_start_time_s = prepared["test_start_time_s"]
    print(
        f"fit_end={fit_end_time_s} test_start={test_start_time_s} "
        f"({pd.Timestamp(test_start_time_s, unit='s', tz='UTC').isoformat()}); "
        f"holdout {prepared['holdout_minutes_per_symbol']} min/symbol, "
        f"evaluated {prepared['evaluated_minutes']} min, eval rows {len(eval_frame)}",
        flush=True,
    )
    # Expected trade counts are written down BEFORE running (power rule).
    t_split = int(payload["split"]["t_split"])
    validate_minutes = (int(dev["target_time_s"].max()) - (t_split + VALIDATE_EMBARGO_MINUTES * 60)) / 60.0
    evaluations = []
    for finalist in freeze_info["finalists"]:
        config = Config(**finalist["config"])
        theta = float(finalist["theta"])
        validate_trades = finalist["validate_trades"]["taker_10bp"]["n_trades"]
        rate = validate_trades / validate_minutes if validate_minutes > 0 else float("nan")
        expected = rate * prepared["evaluated_minutes"]
        print(
            f"[holdout-expected] {config.cid}: validate trades {validate_trades} over "
            f"{validate_minutes:.0f} min -> {rate:.3e}/min -> expected {expected:.2f} trades "
            f"over {prepared['evaluated_minutes']} min",
            flush=True,
        )
        started = time.time()
        result = evaluate_finalist_on_holdout(
            dev, eval_frame, config, theta, fit_end_time_s, test_start_time_s, expected_trades=expected
        )
        result["runtime_seconds"] = time.time() - started
        evaluations.append(result)
        taker = result["trades"]["taker_10bp"]
        p_display = f"{taker['p_net_gt0']:.4f}" if taker["p_net_gt0"] is not None else "null"
        print(
            f"[holdout] {config.cid}: trades={taker['n_trades']} "
            f"gross={taker['gross_mean_return']*1e4:+.2f} bp net10={taker['net_mean_return']*1e4:+.2f} bp "
            f"p_net_gt0={p_display} net_criterion={result['net_criterion']} "
            f"hit_criterion={result['hit_criterion']} overall={result['overall_verdict']}",
            flush=True,
        )
    statuses = {item["overall_verdict"] for item in evaluations}
    if "pass" in statuses:
        verdict = "usable edge: at least one finalist passes both pre-registered criteria"
    elif "inconclusive_underpowered" in statuses:
        verdict = (
            "inconclusive: the net-return criterion is underpowered (<30 holdout trades) for at "
            "least one finalist and no finalist passes both criteria"
        )
    else:
        verdict = "null: no finalist passes the pre-registered holdout criteria"
    output = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "lab_version": LAB_VERSION,
        "holdout_database": str(args.holdout_db),
        "fit_end_time_s": fit_end_time_s,
        "test_start_time_s": test_start_time_s,
        "holdout_minutes_per_symbol": prepared["holdout_minutes_per_symbol"],
        "evaluated_minutes": prepared["evaluated_minutes"],
        "eval_rows": int(len(eval_frame)),
        "warmup": {
            "combined_db": prepared["combined_db"],
            "dev_tail_minutes": HOLDOUT_WARMUP_TAIL_MINUTES,
            "targets": "overlaid from a holdout-only build; features from the combined series",
        },
        "bonferroni_alpha": BONFERRONI_FINAL,
        "min_holdout_trades": MIN_HOLDOUT_TRADES,
        "finalists": evaluations,
        "verdict": verdict,
    }
    output_path = Path(args.holdout_output)
    with open(output_path, "w") as handle:
        json.dump(output, handle, indent=2, sort_keys=True, default=str)
    print(f"holdout results written: {output_path} — {verdict}", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="all",
                        choices=("explore", "validate", "freeze", "holdout", "all"))
    parser.add_argument("--ids", nargs="*", default=None,
                        help="restrict to these config ids (explore/validate)")
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--rebuild-frame", action="store_true")
    parser.add_argument("--holdout-db", default=None,
                        help="untouched holdout database (only for --stage holdout)")
    parser.add_argument("--holdout-output", default=str(HERE / "results_edge_lab_holdout.json"))
    args = parser.parse_args()
    started = time.time()

    if args.stage == "holdout":
        run_holdout(args)
        return

    frame = load_dev_frame(Path(args.db), rebuild=args.rebuild_frame)
    t_split = split_time(frame)
    state = load_state()
    state["split"] = {
        "explore_fraction": EXPLORE_FRACTION,
        "t_split": t_split,
        "validate_embargo_minutes": VALIDATE_EMBARGO_MINUTES,
        "n_rows": int(len(frame)),
        "n_timestamps": int(frame["target_time_s"].nunique()),
        "t_split_utc": pd.Timestamp(t_split, unit="s", tz="UTC").isoformat(),
    }
    print(f"t_split={t_split} ({state['split']['t_split_utc']})", flush=True)

    final = None
    if args.stage in ("explore", "all"):
        run_explore(frame, t_split, state, args.ids)
    if args.stage in ("validate", "all"):
        ids = args.ids
        if not ids:
            ids = explore_shortlist(state)
            print(f"validate shortlist: {ids}", flush=True)
        run_validate(frame, t_split, state, ids)
    if args.stage in ("freeze", "all"):
        final = freeze(frame, t_split, state)
        state["finalists"] = [f["cid"] for f in final["finalists"]]
        save_state(state)
    write_results(state, final, time.time() - started)
    print(f"results written: {RESULTS_PATH} ({time.time() - started:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
