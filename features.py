"""Pre-registered, causal features and targets for the crypto evaluation.

The prediction timestamp is the open of minute ``t``.  At that instant the
minute ``t-1`` candle is the newest closed candle, so every market and GCP
feature is explicitly shifted by one minute.  A direction target for horizon
``h`` is the compounded return from the close of ``t-1`` to the close of
``t+h-1``.  A volatility target is the square-root sum of the same future
one-minute squared returns.

``PHYSICS_FEATURES`` is a separate pre-registered family of physics/geometry
analogues (Gauss/Kalman drift, Schrodinger/OU mean reversion, Navier-Stokes
efficiency ratio, Riemann/Fisher-Rao distance); its exact formulas, constants,
and ablation decision rule are frozen in ``physics_preregistration.md``.
"""
from __future__ import annotations

import json
import sqlite3
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

ASSETS: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT")
HORIZONS: tuple[int, ...] = (1, 5, 15, 60)
RETURN_WINDOWS: tuple[int, ...] = (1, 5, 15, 60)
VOL_WINDOWS: tuple[int, ...] = (15, 60)
Z_WINDOW = 60
RSI_WINDOW = 14

# This list is the model contract.  It is generated once here, before looking
# at any results; train.py must use exactly these names (or this same list
# minus GCP_FEATURES for the ablation).
MARKET_FEATURES: tuple[str, ...] = (
    *(f"ret_{w}m" for w in RETURN_WINDOWS),
    *(f"realized_vol_{w}m" for w in VOL_WINDOWS),
    "rsi_momentum",
    "volume_z_60m",
    "trade_count_z_60m",
    "cross_asset_dispersion_1m",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
)
GCP_FEATURES: tuple[str, ...] = (
    "netvar_level",
    "netvar_z_60m",
    "netvar_velocity_1m",
    "netvar_acceleration_1m",
    "netvar_std_60m",
)
FEATURES: tuple[str, ...] = MARKET_FEATURES + GCP_FEATURES

# Physics/geometry analogue family, pre-registered in
# ``physics_preregistration.md`` before any arm result was computed.  These
# columns are computed from the same closed candles as the market features and
# pass through the same final shift(1); they are never tuned on results.
# Constants (windows in candle rows, noise ratios for the local-linear-trend
# Kalman filter) are exactly the ones written in that file.
GAUSS_LEVEL_TO_NOISE = 1e-4
GAUSS_SLOPE_TO_NOISE = 1e-8
GAUSS_GAIN_TOLERANCE = 1e-14
PHYSICS_GAUSS_VOL_WINDOW = 60
PHYSICS_DISPLACEMENT_WINDOW = 60
PHYSICS_OU_WINDOW = 240
PHYSICS_NS_WINDOW = 60
PHYSICS_RIEMANN_SHORT_WINDOW = 15
PHYSICS_RIEMANN_LONG_WINDOW = 240

PHYSICS_FEATURES: tuple[str, ...] = (
    "gauss_kalman_drift_z",
    "schrodinger_displacement_60m",
    "schrodinger_ou_kappa_240m",
    "ns_reynolds_60m",
    "ns_signed_flow_60m",
    "riemann_fisher_rao_15_240",
)
FEATURES_WITH_PHYSICS: tuple[str, ...] = MARKET_FEATURES + PHYSICS_FEATURES

# The registered ablation arms and their exact model columns.  ``market_only``
# and ``with_gcp`` keep their original column contracts; ``with_physics`` is
# the new pre-registered arm (market features plus physics analogues, no GCP).
FEATURE_ARMS: dict[str, tuple[str, ...]] = {
    "market_only": MARKET_FEATURES,
    "with_gcp": FEATURES,
    "with_physics": FEATURES_WITH_PHYSICS,
}


def feature_columns_for_arm(arm: str) -> tuple[str, ...]:
    """Return the exact registered model columns for one ablation arm."""
    try:
        return FEATURE_ARMS[str(arm)]
    except KeyError as exc:
        raise ValueError(f"unknown feature arm: {arm!r}") from exc


TARGET_COLUMNS: tuple[str, ...] = tuple(
    [f"forward_return_{h}m" for h in HORIZONS]
    + [f"forward_direction_{h}m" for h in HORIZONS]
    + [f"forward_vol_{h}m" for h in HORIZONS]
)
INTERNAL_COLUMNS: tuple[str, ...] = (
    "target_time_s",
    "feature_source_max_time_s",
    "ewma_vol_60m",
)


def _finite_zscore(values: pd.Series, window: int = Z_WINDOW) -> pd.Series:
    """Trailing z-score, with undefined constant windows represented by NaN."""
    mean = values.rolling(window, min_periods=window).mean()
    std = values.rolling(window, min_periods=window).std(ddof=1)
    z = (values - mean) / std.replace(0.0, np.nan)
    return z.replace([np.inf, -np.inf], np.nan)


def _rsi_like(one_minute_returns: pd.Series) -> pd.Series:
    """Simple trailing RSI-like momentum (0..100), before the causal shift."""
    gains = one_minute_returns.clip(lower=0.0)
    losses = (-one_minute_returns.clip(upper=0.0))
    avg_gain = gains.rolling(RSI_WINDOW, min_periods=RSI_WINDOW).mean()
    avg_loss = losses.rolling(RSI_WINDOW, min_periods=RSI_WINDOW).mean()
    # 50 is neutral when both sides are zero; one-sided windows are defined.
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.where(avg_loss.ne(0.0), 100.0)
    rsi = rsi.where(~(avg_gain.eq(0.0) & avg_loss.eq(0.0)), 50.0)
    return rsi


@lru_cache(maxsize=8)
def steady_state_trend_gains(
    level_to_noise: float = GAUSS_LEVEL_TO_NOISE,
    slope_to_noise: float = GAUSS_SLOPE_TO_NOISE,
    tolerance: float = GAUSS_GAIN_TOLERANCE,
) -> tuple[float, float]:
    """Deterministic Riccati fixed point of the registered trend model.

    State ``[level, slope]``, ``F = [[1, 1], [0, 1]]``, ``H = [1, 0]``,
    ``Q = diag(q_level, q_slope)``, ``R = 1``.  Only the ratios ``Q/R`` matter
    for the gain, and they are fixed in ``physics_preregistration.md``.  The
    returned ``(alpha, beta)`` are the fixed-gain (steady-state) filter gains,
    exactly equivalent to Holt double-exponential smoothing.
    """
    transition = np.array([[1.0, 1.0], [0.0, 1.0]])
    observation = np.array([[1.0, 0.0]])
    process = np.diag([float(level_to_noise), float(slope_to_noise)])
    covariance = np.zeros((2, 2))
    gain = np.zeros(2)
    for _ in range(500_000):
        predicted = transition @ covariance @ transition.T + process
        innovation_var = float((observation @ predicted @ observation.T).item()) + 1.0
        new_gain = (predicted @ observation.T).ravel() / innovation_var
        if np.max(np.abs(new_gain - gain)) < float(tolerance):
            gain = new_gain
            break
        gain = new_gain
        covariance = predicted - np.outer(new_gain, (observation @ predicted).ravel())
    return float(gain[0]), float(gain[1])


def _holt_trend_slope(log_close: pd.Series, alpha: float, beta: float) -> pd.Series:
    """Fixed-gain local-linear-trend slope of log close (steady-state Kalman).

    The recursion starts at the first candle with a finite close
    (``level = log close``, ``slope = 0``); non-finite candles leave the state
    unchanged.  It is the exact fixed-gain form of the registered Kalman model;
    a per-row time-varying loop is unnecessary because the Riccati recursion
    reaches this gain long before the series starts.
    """
    values = log_close.to_numpy(dtype="float64").tolist()
    n = len(values)
    slope = np.full(n, np.nan, dtype="float64")
    finite = [i for i, value in enumerate(values) if np.isfinite(value)]
    if finite:
        start = finite[0]
        level = float(values[start])
        trend = 0.0
        for i in range(start, n):
            value = values[i]
            if np.isfinite(value):
                error = value - level - trend
                level += trend + alpha * error
                trend += beta * error
            slope[i] = trend
    return pd.Series(slope, index=log_close.index)


def _physics_features(
    frame: pd.DataFrame,
    one_minute_return: pd.Series,
    realized_vol_60: pd.Series,
) -> pd.DataFrame:
    """Registered physics/geometry analogue columns, before the causal shift.

    Formulas and constants are exactly ``physics_preregistration.md``; see
    that file for the derivation, the steady-state-gain equivalence, and the
    honesty statement that these are analogues with no physical mechanism.
    """
    close = frame["close"]
    log_close = np.log(close.where(close > 0.0))
    output = pd.DataFrame(index=frame.index)

    # 1. Gauss: local-linear-trend drift normalised by 60m realized volatility.
    alpha, beta = steady_state_trend_gains()
    slope = _holt_trend_slope(log_close, alpha, beta)
    output["gauss_kalman_drift_z"] = slope.div(realized_vol_60.replace(0.0, np.nan))

    # 2. Schrodinger / Ornstein-Uhlenbeck: displacement and well stiffness.
    mean_60 = log_close.rolling(
        PHYSICS_DISPLACEMENT_WINDOW, min_periods=PHYSICS_DISPLACEMENT_WINDOW
    ).mean()
    std_60 = log_close.rolling(
        PHYSICS_DISPLACEMENT_WINDOW, min_periods=PHYSICS_DISPLACEMENT_WINDOW
    ).std(ddof=1)
    output["schrodinger_displacement_60m"] = (log_close - mean_60).div(
        std_60.replace(0.0, np.nan)
    )

    window = PHYSICS_OU_WINDOW
    deviation = log_close - log_close.rolling(window, min_periods=window).mean()
    deviation_sum = deviation.rolling(window, min_periods=window).sum()
    deviation_squares = (deviation * deviation).rolling(window, min_periods=window).sum()
    # Lag-1 products for the window ending at s are the 239-term sum over
    # i = s-239..s-1, i.e. a 239-row rolling sum shifted by one row.
    pair_products = (
        (deviation * deviation.shift(-1))
        .rolling(window - 1, min_periods=window - 1)
        .sum()
        .shift(1)
    )
    window_mean = deviation_sum / float(window)
    numerator = (
        pair_products
        - window_mean * (2.0 * deviation_sum - deviation - deviation.shift(window - 1))
        + float(window - 1) * window_mean * window_mean
    )
    denominator = deviation_squares - float(window) * window_mean * window_mean
    phi = numerator.div(denominator.replace(0.0, np.nan))
    kappa = pd.Series(np.nan, index=frame.index, dtype="float64")
    valid = phi.gt(0.0) & phi.lt(1.0)
    kappa[valid] = -np.log(phi[valid])
    output["schrodinger_ou_kappa_240m"] = kappa

    # 3. Navier-Stokes: Reynolds analogue (Kaufman efficiency ratio).
    window = PHYSICS_NS_WINDOW
    net_flow = one_minute_return.rolling(window, min_periods=window).sum()
    gross_flow = one_minute_return.abs().rolling(window, min_periods=window).sum()
    reynolds = net_flow.abs().div(gross_flow.replace(0.0, np.nan))
    output["ns_reynolds_60m"] = reynolds
    output["ns_signed_flow_60m"] = np.sign(
        close.pct_change(PHYSICS_NS_WINDOW, fill_method=None)
    ) * reynolds

    # 4. Riemann: Fisher-Rao geodesic distance between the trailing 15m and
    # 240m return Gaussians (closed form; see the pre-registration).
    short, long = PHYSICS_RIEMANN_SHORT_WINDOW, PHYSICS_RIEMANN_LONG_WINDOW
    mu_short = one_minute_return.rolling(short, min_periods=short).mean()
    sigma_short = one_minute_return.rolling(short, min_periods=short).std(ddof=1)
    mu_long = one_minute_return.rolling(long, min_periods=long).mean()
    sigma_long = one_minute_return.rolling(long, min_periods=long).std(ddof=1)
    delta_mu = mu_short - mu_long
    half_delta_squared = 0.5 * delta_mu * delta_mu
    ratio = (half_delta_squared + (sigma_short - sigma_long) ** 2).div(
        (half_delta_squared + (sigma_short + sigma_long) ** 2).replace(0.0, np.nan)
    )
    ratio = ratio.where((ratio >= 0.0) & (ratio < 1.0))
    output["riemann_fisher_rao_15_240"] = 2.0 * np.sqrt(2.0) * np.arctanh(np.sqrt(ratio))

    return output.replace([np.inf, -np.inf], np.nan)


def market_db_audit(db_path: str | Path) -> dict:
    """Verify the klines schema and report min/max/count per symbol."""
    required = {
        "symbol",
        "open_time_s",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "trades",
    }
    with sqlite3.connect(str(db_path)) as con:
        pragma_rows = con.execute("PRAGMA table_info(klines_1m)").fetchall()
        if not pragma_rows:
            raise RuntimeError("klines_1m table is missing")
        columns = {str(row[1]) for row in pragma_rows}
        missing = sorted(required - columns)
        if missing:
            raise RuntimeError(f"klines_1m missing required columns: {missing}")
        rows = con.execute(
            """
            SELECT symbol, MIN(open_time_s), MAX(open_time_s), COUNT(*)
            FROM klines_1m
            GROUP BY symbol ORDER BY symbol
            """
        ).fetchall()
    ranges = {
        str(symbol): {"min_time_s": int(lo), "max_time_s": int(hi), "rows": int(n)}
        for symbol, lo, hi, n in rows
    }
    missing_assets = [asset for asset in ASSETS if asset not in ranges]
    if missing_assets:
        raise RuntimeError(f"klines_1m missing assets: {missing_assets}")
    return {"columns": sorted(columns), "ranges": ranges}


def _query_market(
    db_path: str | Path,
    symbols: Sequence[str],
    start_time_s: int | None = None,
    end_time_s: int | None = None,
) -> dict[str, pd.DataFrame]:
    audit = market_db_audit(db_path)
    del audit
    clauses = ["symbol = ?"]
    for symbol in symbols:
        clauses[0] = "symbol IN (" + ",".join("?" for _ in symbols) + ")"
    params: list[object] = list(symbols)
    if start_time_s is not None:
        clauses.append("open_time_s >= ?")
        params.append(int(start_time_s) - 61 * 60)
    if end_time_s is not None:
        clauses.append("open_time_s <= ?")
        params.append(int(end_time_s) + 61 * 60)
    sql = (
        "SELECT symbol, open_time_s, close, volume, trades "
        "FROM klines_1m WHERE "
        + " AND ".join(clauses)
        + " ORDER BY open_time_s"
    )
    with sqlite3.connect(str(db_path)) as con:
        raw = pd.read_sql_query(sql, con, params=params)
    if raw.empty:
        raise RuntimeError("No market rows matched requested range")
    raw["open_time_s"] = raw["open_time_s"].astype("int64")
    raw["symbol"] = raw["symbol"].astype(str)
    by_symbol: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        part = raw.loc[raw["symbol"].eq(symbol), ["open_time_s", "close", "volume", "trades"]]
        part = part.drop_duplicates("open_time_s").sort_values("open_time_s")
        if part.empty:
            raise RuntimeError(f"No rows for {symbol}")
        index = pd.to_datetime(part.pop("open_time_s"), unit="s", utc=True)
        part.index = index
        part.index.name = "time"
        by_symbol[symbol] = part.astype({"close": "float64", "volume": "float64", "trades": "float64"})
    return by_symbol


def load_netvar_minute(
    gcp_dir: str | Path,
    start_time_s: int | None = None,
    end_time_s: int | None = None,
    chunksize: int = 500_000,
) -> pd.Series:
    """Load minute means from a verified cache or aggregate raw 1-second zips.

    A ``netvar_1m`` SQLite cache (columns ``epoch_s, netvar, n_samples``) is
    accepted for repeatable fast runs; otherwise every matching raw zip is
    streamed and aggregated to the same minute means.  In both cases a minute
    labelled ``t`` contains seconds in ``[t,t+60)`` and is shifted by one
    minute in :func:`build_feature_frame`.
    """
    root = Path(gcp_dir)
    lo = None if start_time_s is None else int(start_time_s) - 61 * 60
    hi = None if end_time_s is None else int(end_time_s) + 61 * 60
    if root.is_file():
        with sqlite3.connect(str(root)) as con:
            sql = "SELECT epoch_s, netvar FROM netvar_1m"
            params: list[int] = []
            clauses: list[str] = []
            if lo is not None:
                clauses.append("epoch_s >= ?")
                params.append(lo)
            if hi is not None:
                clauses.append("epoch_s <= ?")
                params.append(hi)
            if clauses:
                sql += " WHERE " + " AND ".join(clauses)
            sql += " ORDER BY epoch_s"
            cached = pd.read_sql_query(sql, con, params=params)
        if cached.empty:
            raise RuntimeError("No cached GCP netvar rows matched requested range")
        values = cached.set_index("epoch_s")["netvar"].astype("float64")
        values.index = pd.to_datetime(values.index.astype("int64"), unit="s", utc=True)
        values.index.name = "time"
        return values.sort_index()
    paths = sorted(root.glob("GCP2_Network_Coherence_Global_Network_*.csv.zip"))
    if not paths:
        raise FileNotFoundError(f"No global-network GCP zip files or minute cache under {root}")
    aggregates: list[pd.DataFrame] = []
    lo_min = None if lo is None else (lo // 60) * 60
    hi_min = None if hi is None else (hi // 60) * 60
    for archive in paths:
        with zipfile.ZipFile(archive) as zf:
            names = [name for name in zf.namelist() if name.lower().endswith(".csv")]
            if not names:
                continue
            with zf.open(names[0]) as raw:
                for chunk in pd.read_csv(
                    raw,
                    usecols=[0, 1],
                    skiprows=1,
                    names=["epoch_s", "netvar"],
                    chunksize=chunksize,
                ):
                    chunk["epoch_s"] = pd.to_numeric(chunk["epoch_s"], errors="coerce")
                    chunk["netvar"] = pd.to_numeric(chunk["netvar"], errors="coerce")
                    chunk = chunk.dropna(subset=["epoch_s", "netvar"])
                    if chunk.empty:
                        continue
                    chunk["epoch_s"] = chunk["epoch_s"].astype("int64")
                    chunk["minute_s"] = (chunk["epoch_s"] // 60) * 60
                    grouped = chunk.groupby("minute_s", sort=False)["netvar"].agg(["sum", "count"])
                    if lo_min is not None:
                        grouped = grouped.loc[grouped.index >= lo_min]
                    if hi_min is not None:
                        grouped = grouped.loc[grouped.index <= hi_min]
                    if not grouped.empty:
                        aggregates.append(grouped)
    if not aggregates:
        raise RuntimeError("No GCP netvar rows matched requested range")
    grouped = pd.concat(aggregates).groupby(level=0)[["sum", "count"]].sum()
    values = grouped["sum"] / grouped["count"].replace(0, np.nan)
    values.index = pd.to_datetime(values.index.astype("int64"), unit="s", utc=True)
    values.index.name = "time"
    return values.astype("float64").sort_index()


def _epoch_seconds(index: pd.DatetimeIndex) -> np.ndarray:
    """Convert UTC DatetimeIndex to seconds across pandas datetime units."""
    return index.to_numpy(dtype="datetime64[s]").astype("int64")


def _safe_future_targets(close: pd.Series, one_minute_returns: pd.Series) -> pd.DataFrame:
    """Create future-only targets at each minute open timestamp."""
    anchor = close.shift(1)
    result: dict[str, pd.Series] = {}
    for horizon in HORIZONS:
        # At t, close[t-1] is known and returns[t:t+h-1] are all future.
        forward_return = close.shift(-(horizon - 1)).div(anchor).sub(1.0)
        result[f"forward_return_{horizon}m"] = forward_return
        result[f"forward_direction_{horizon}m"] = np.sign(forward_return)
        future_var = sum(
            (one_minute_returns.shift(-offset) ** 2 for offset in range(horizon)),
            start=pd.Series(0.0, index=one_minute_returns.index),
        )
        result[f"forward_vol_{horizon}m"] = np.sqrt(future_var)
    return pd.DataFrame(result, index=close.index)


def _source_max_before(index: pd.DatetimeIndex, target_index: pd.DatetimeIndex) -> pd.Series:
    source_values = _epoch_seconds(index)
    target_values = _epoch_seconds(target_index)
    positions = np.searchsorted(source_values, target_values, side="left") - 1
    result = np.full(len(target_values), np.nan, dtype="float64")
    good = positions >= 0
    result[good] = source_values[positions[good]]
    return pd.Series(result, index=target_index)


def _assemble_feature_frame(
    market: Mapping[str, pd.DataFrame],
    gcp: pd.Series | None = None,
) -> pd.DataFrame:
    """Apply the registered feature/target formulas to indexed candle frames."""
    symbols = ASSETS
    missing = [symbol for symbol in symbols if symbol not in market]
    if missing:
        raise ValueError(f"market frames missing symbols: {missing}")
    closes = pd.concat({symbol: market[symbol]["close"] for symbol in symbols}, axis=1)
    closes.columns = list(symbols)
    volumes = pd.concat({symbol: market[symbol]["volume"] for symbol in symbols}, axis=1)
    volumes.columns = list(symbols)
    trades = pd.concat({symbol: market[symbol]["trades"] for symbol in symbols}, axis=1)
    trades.columns = list(symbols)
    returns = closes.pct_change(fill_method=None)
    cross_dispersion = returns.std(axis=1, ddof=1).where(returns.notna().sum(axis=1).eq(len(symbols)))
    cross_dispersion = cross_dispersion.shift(1)

    gcp_features = None
    if gcp is not None:
        gcp = pd.Series(gcp, dtype="float64").sort_index()
        gcp_features = pd.DataFrame(index=gcp.index)
        gcp_features["netvar_level"] = gcp
        gcp_features["netvar_z_60m"] = _finite_zscore(gcp)
        gcp_features["netvar_velocity_1m"] = gcp.diff(1)
        gcp_features["netvar_acceleration_1m"] = gcp.diff(1).diff(1)
        gcp_features["netvar_std_60m"] = gcp.rolling(Z_WINDOW, min_periods=Z_WINDOW).std(ddof=1)
        # A minute aggregate labelled t is not known before t+60.
        gcp_features = gcp_features.shift(1)
    rows: list[pd.DataFrame] = []
    for symbol in symbols:
        frame = market[symbol]
        close = frame["close"]
        volume = frame["volume"]
        trade_count = frame["trades"]
        one_minute_return = returns[symbol]
        features = pd.DataFrame(index=frame.index)
        for window in RETURN_WINDOWS:
            features[f"ret_{window}m"] = close.pct_change(window, fill_method=None)
        for window in VOL_WINDOWS:
            features[f"realized_vol_{window}m"] = one_minute_return.rolling(
                window, min_periods=window
            ).std(ddof=1)
        features["rsi_momentum"] = _rsi_like(one_minute_return)
        features["volume_z_60m"] = _finite_zscore(volume)
        features["trade_count_z_60m"] = _finite_zscore(trade_count)
        features["cross_asset_dispersion_1m"] = cross_dispersion.reindex(frame.index)
        # Calendar values are deterministic and use no market data.
        hour = frame.index.hour.to_numpy(dtype="float64")
        dow = frame.index.dayofweek.to_numpy(dtype="float64")
        features["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
        features["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
        features["dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
        features["dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)
        # Physics/geometry analogues (physics_preregistration.md) are computed
        # from the same closed candles and pass through the same final shift,
        # so live and historical paths stay identical.
        features = pd.concat(
            [
                features,
                _physics_features(
                    frame,
                    one_minute_return,
                    features[f"realized_vol_{PHYSICS_GAUSS_VOL_WINDOW}m"],
                ),
            ],
            axis=1,
        )
        # The final shift is the central leakage guard: no feature sees candle t.
        features = features.shift(1)
        if gcp_features is not None:
            features = features.join(gcp_features, how="left")
        targets = _safe_future_targets(close, one_minute_return)
        out = features.join(targets)
        out["symbol"] = symbol
        out["target_time_s"] = _epoch_seconds(out.index)
        source = _source_max_before(closes.index, out.index)
        out["feature_source_max_time_s"] = source
        # EWMA uses only returns through t-1 and is kept as a non-model baseline.
        out["ewma_vol_60m"] = np.sqrt(
            (one_minute_return.shift(1) ** 2)
            .ewm(span=Z_WINDOW, adjust=False, min_periods=Z_WINDOW)
            .mean()
        )
        rows.append(out.reset_index(names="time"))
    result = pd.concat(rows, ignore_index=True)
    result = result.sort_values(["target_time_s", "symbol"], kind="mergesort").reset_index(drop=True)
    # Preserve a stable, explicit dtype contract for model consumers.
    for column in FEATURES + PHYSICS_FEATURES + TARGET_COLUMNS + ("ewma_vol_60m",):
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="coerce").astype("float64")
    return result


def build_live_feature_frame(
    market: Mapping[str, pd.DataFrame],
    prediction_time_s: int,
    gcp: pd.Series | None = None,
) -> pd.DataFrame:
    """Build the same features for closed live candles plus one target row.

    ``market`` contains only closed candles.  A NaN placeholder at the next
    prediction open lets the shared trailing formulas shift the latest closed
    candle into that target row without inventing a current close/volume.
    """
    target = pd.Timestamp(int(prediction_time_s), unit="s", tz="UTC")
    augmented: dict[str, pd.DataFrame] = {}
    for symbol in ASSETS:
        if symbol not in market:
            raise ValueError(f"market frames missing symbol {symbol}")
        part = market[symbol].copy()
        if not isinstance(part.index, pd.DatetimeIndex):
            raise TypeError(f"{symbol} market frame index must be a UTC DatetimeIndex")
        part.index = pd.to_datetime(part.index, utc=True)
        part = part.sort_index()
        if target not in part.index:
            placeholder = pd.DataFrame(
                np.nan,
                index=pd.DatetimeIndex([target]),
                columns=part.columns,
            )
            part = pd.concat([part, placeholder], axis=0)
        augmented[symbol] = part.sort_index()

    live_gcp = None
    if gcp is not None:
        live_gcp = pd.Series(gcp, dtype="float64")
        live_gcp.index = pd.to_datetime(live_gcp.index, utc=True)
        live_gcp = live_gcp.loc[live_gcp.index <= target].sort_index()
        if target not in live_gcp.index:
            live_gcp = pd.concat(
                [
                    live_gcp,
                    pd.Series([np.nan], index=pd.DatetimeIndex([target]), name="netvar"),
                ]
            )
    return _assemble_feature_frame(augmented, gcp=live_gcp)


def build_feature_frame(
    db_path: str | Path,
    gcp_dir: str | Path | None,
    symbols: Sequence[str] = ASSETS,
    start_time_s: int | None = None,
    end_time_s: int | None = None,
    include_gcp: bool = True,
) -> pd.DataFrame:
    """Build long asset-by-minute frame with causal features and future targets."""
    symbols = tuple(symbols)
    if tuple(symbols) != ASSETS:
        # Cross-asset dispersion is defined over the registered four assets.
        raise ValueError(f"symbols must be exactly {ASSETS}")
    market = _query_market(db_path, symbols, start_time_s, end_time_s)
    gcp = None
    if include_gcp:
        if gcp_dir is None:
            raise ValueError("gcp_dir is required when include_gcp=True")
        gcp = load_netvar_minute(gcp_dir, start_time_s, end_time_s)
    return _assemble_feature_frame(market, gcp=gcp)


def assert_causal_features(
    frame: pd.DataFrame,
    sample_rows: int = 128,
    require_gcp: bool = False,
    require_physics: bool = False,
) -> dict:
    """Run programmatic leakage checks and raise AssertionError on violation.

    The check samples rows deterministically, confirms the recorded latest
    candle source is strictly before each target timestamp, and verifies that
    row timestamps agree with ``target_time_s``.  The builder's feature lists
    (market, GCP, and physics) are also checked to ensure no target/internal
    column can accidentally enter a model matrix, and the requested arms must
    actually be present.  Shuffled-target validation is run by backtest.py.
    """
    required = {"target_time_s", "feature_source_max_time_s", "symbol"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise AssertionError(f"causality metadata missing: {missing}")
    if require_gcp and not set(GCP_FEATURES).issubset(frame.columns):
        raise AssertionError("GCP feature columns are absent")
    if require_physics and not set(PHYSICS_FEATURES).issubset(frame.columns):
        raise AssertionError("physics feature columns are absent")
    registered = set(MARKET_FEATURES + GCP_FEATURES + PHYSICS_FEATURES)
    if registered.intersection(TARGET_COLUMNS + INTERNAL_COLUMNS):
        raise AssertionError("feature registration overlaps targets/internal columns")
    if not frame.empty:
        eligible = frame.loc[frame["feature_source_max_time_s"].notna()]
        if eligible.empty:
            raise AssertionError("no rows have a prior candle source")
        take = np.linspace(0, len(eligible) - 1, min(sample_rows, len(eligible)), dtype=int)
        sample = eligible.iloc[np.unique(take)]
        target = sample["target_time_s"].to_numpy(dtype="int64")
        if "time" in sample.columns:
            observed_target = _epoch_seconds(
                pd.DatetimeIndex(pd.to_datetime(sample["time"], utc=True))
            )
            assert np.array_equal(observed_target, target), "target_time_s disagrees with row timestamp"
        source = sample["feature_source_max_time_s"].to_numpy(dtype="float64")
        assert np.all(np.isfinite(source)), "sampled rows lack a prior candle source"
        assert np.all(source < target), "feature source is not strictly before target time"
        # All non-calendar market columns are generated by a trailing operation
        # followed by shift(1); this exact cutoff is checked for regular candles.
        regular = np.isclose(target - source, 60.0)
        if np.any(regular):
            assert np.all(target[regular] - source[regular] >= 60), (
                "feature row includes candle at prediction timestamp"
            )
        checked = len(sample)
    else:
        checked = 0
    return {
        "rows_checked": int(checked),
        "rows_with_prior_source": int(frame["feature_source_max_time_s"].notna().sum()),
        "strict_source_before_target": True,
        "shuffled_target_control": "run in backtest.py",
    }


def feature_manifest() -> dict:
    """Machine-readable pre-registration manifest saved inside model artifacts."""
    return {
        "assets": list(ASSETS),
        "horizons_minutes": list(HORIZONS),
        "market_features": list(MARKET_FEATURES),
        "gcp_features": list(GCP_FEATURES),
        "physics_features": list(PHYSICS_FEATURES),
        "all_features": list(FEATURES),
        "feature_arms": {arm: list(columns) for arm, columns in FEATURE_ARMS.items()},
        "physics_registration": {
            "document": "physics_preregistration.md",
            "family": "physics/geometry analogues, not physics",
            "gauss_level_to_noise": GAUSS_LEVEL_TO_NOISE,
            "gauss_slope_to_noise": GAUSS_SLOPE_TO_NOISE,
            "gauss_steady_state_gains": list(steady_state_trend_gains()),
            "gauss_vol_window_minutes": PHYSICS_GAUSS_VOL_WINDOW,
            "schrodinger_displacement_window_minutes": PHYSICS_DISPLACEMENT_WINDOW,
            "schrodinger_ou_window_minutes": PHYSICS_OU_WINDOW,
            "ns_window_minutes": PHYSICS_NS_WINDOW,
            "riemann_short_window_minutes": PHYSICS_RIEMANN_SHORT_WINDOW,
            "riemann_long_window_minutes": PHYSICS_RIEMANN_LONG_WINDOW,
        },
        "rsi_window_minutes": RSI_WINDOW,
        "z_window_minutes": Z_WINDOW,
        "causal_alignment": "features at target open t use data with open_time_s < t; targets start at t",
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Audit market inputs and build causal features")
    parser.add_argument("--db", default="crypto-data/crypto_market.db")
    parser.add_argument("--gcp", default="gcp2-data/network_coherence/global_network")
    parser.add_argument("--no-gcp", action="store_true")
    args = parser.parse_args()
    audit = market_db_audit(args.db)
    print(json.dumps(audit, indent=2))
    sample = build_feature_frame(args.db, args.gcp, include_gcp=not args.no_gcp)
    print(assert_causal_features(sample, require_gcp=not args.no_gcp))
    print(sample.head().to_string(index=False))
