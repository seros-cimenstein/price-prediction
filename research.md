# Short-horizon crypto predictability: time-boxed research

This is a short literature check, not a meta-analysis. I read the sources below directly on 2026-09-14. Their samples, venues, labels, and feature sets are not the same as this evaluation, so their reported numbers are context rather than evidence for this data.

## What the sources say

### 1. Binance candle semantics and live-data constraints

- [Binance Spot REST market-data documentation](https://developers.binance.com/en/docs/products/spot/rest-api#market-data-endpoints.md) says public market-data responses are chronological, timestamps are milliseconds by default, and `startTime`/`endTime` control the returned interval. It documents API limits and warns clients to back off on 429/418 responses.
- The live [Binance `/api/v3/klines` response](https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=5) returns one-minute records with open time, OHLC, base volume, close time, quote volume, trade count, and taker volumes. The historical SQLite table uses the relevant open time, OHLC, volume, and trade count fields. A live consumer must wait for a candle to close before using its close/volume/trade count; this pipeline therefore aligns a prediction at open `t` to candles with `open_time_s < t`.

### 2. Order flow can help volatility, but the comparison is demanding

- Lensky and Hao, [*Learning to Predict Short-Term Volatility with Order Flow Image Representation* (arXiv:2304.02472)](https://arxiv.org/abs/2304.02472), study Bitcoin order-flow snapshots (trade sizes/directions and LOB) and compare CNN/ResNet/ConvMixer models with GARCH, an MLP, and a naive current-volatility guess. Their abstract reports the image/CNN model's RMSPE as 0.85 +/- 1.1 with aggregated features versus 1.4 +/- 3.0 for the naive guess in a January 2021 experiment. The useful prior for this work is that volatility may contain short-lived microstructure information, but those authors have order flow and the present data has only one-minute OHLCV/trade counts. We therefore retain an EWMA baseline and do not imply that their result transfers here.

### 3. Simpler models can be competitive when inputs are good

- Wang, [*Exploring Microstructural Dynamics in Cryptocurrency Limit Order Books: Better Inputs Matter More Than Stacking Another Hidden Layer* (arXiv:2506.05764)](https://arxiv.org/abs/2506.05764), compares logistic regression, XGBoost, and deep LOB models on Bybit BTC/USDT snapshots from 100 ms to multi-second intervals. The abstract says preprocessing and feature engineering can let simpler models match or exceed more complex models, and that robustness/latency matter. This supports pre-registering a small regularized logistic/ridge model rather than searching a large model zoo; it does not establish predictability at 1/5/15/60 minutes from candles.

### 4. Cross-asset microstructure structure is plausible, but feature availability matters

- Bieganowski and Ślepaczuk, [*Explainable Patterns in Cryptocurrency Microstructure* (arXiv:2602.00776)](https://arxiv.org/abs/2602.00776), report stable engineered order-book/trade feature importance and SHAP shapes across Binance Futures assets, using one-second data, CatBoost, time-series CV, and conservative maker/taker checks. They connect order-flow imbalance, spread, and adverse selection to the observed patterns. This motivates the market-side volume/trade-count and cross-asset-dispersion controls here, while also highlighting that the present OHLCV feature set omits spread, queue imbalance, and signed order flow.

## Implications for this evaluation

1. Use causal, trailing-only features and a time-ordered walk-forward design; a random split would be especially misleading for overlapping short-horizon targets.
2. Treat any apparent gain over persistence/momentum or EWMA as provisional unless it survives the embargoed out-of-sample intervals and a shuffled-target control.
3. GCP2 netvar is an external candidate covariate, not a market-microstructure measure supported by these papers. The only defensible claim about it here is the paired ablation result on the registered sample; a positive in-sample association is not enough.
4. The literature's stronger results generally use LOB/order-flow data at finer resolution than this dataset. A null result from minute OHLCV plus netvar would therefore be unsurprising and is a valid result, not a failure of the evaluation.
