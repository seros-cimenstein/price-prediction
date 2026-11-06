# price-prediction

A pre-registered, leakage-controlled evaluation of short-horizon (1, 5, 15 and 60 minute)
direction and volatility prediction for BTC, ETH, SOL and BNB against USDT on Binance 1-minute
candles, plus a live loop that scores the frozen models on new candles.

**Result: no robust, cost-aware trading edge.** Market-data models beat naive baselines by a
statistically detectable but economically useless margin, and none of the extra signal families
tested (GCP 2.0 network-coherence data, physics analogues, a literal Schrödinger wave-function
model, determinism/"seed", folk technical analysis, LPPLS) added skill.

I built this to answer, with a protocol I could not fool myself with, whether any of these ideas
predicts crypto prices at minute scale. The answer is no. The full report, with every table and
amendment, is in [docs/full-report.md](docs/full-report.md).

## Method

- **Data.** Binance spot 1-minute klines for BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT from 2024-03-01
  (about 1.32 million clean rows per asset) as the development set; a separate holdout of new
  candles from 2026-09-14 onwards that no model or configuration saw before its single frozen
  evaluation.
- **Pre-registration.** Features, models, horizons, baselines, the ablation and the decision rule
  of each arm were written down before its results were computed (`*_preregistration.md`).
  Later changes are logged as numbered amendments, not silent edits.
- **Models.** L2 logistic regression for direction, ridge regression for volatility; scalers fit
  on training rows only.
- **Validation.** Expanding walk-forward folds with an embargo of at least the horizon,
  causal-feature assertions, shuffled-target controls (at chance: 49.94-50.41% hit rate),
  paired 60-minute moving-block bootstrap, Bonferroni correction over the 16 asset/horizon cells.
- **Baselines.** Persistence (`sign(ret_1m)`), 15-minute momentum, zero-change; EWMA for
  volatility.
- **Costs.** The edge lab checks results against a 20 bp round-trip taker fee (10 bp per side)
  and maker-fee scenarios.

## Results

| Arm | What it tested | Outcome |
| --- | --- | --- |
| Market-only (`train.py`) | trailing returns, volatility, RSI-like momentum, volume/trade z-scores, cross-asset dispersion, calendar | Beats persistence and momentum in all 16 cells, but hit rates are only 50.4-52.5% and Spearman IC 0.004-0.048; ridge beats EWMA volatility MSE by small margins |
| GCP (`--ablation gcp`) | 5 GCP 2.0 netvar features | No consistent gain (Δhit −0.18 to +0.03 pp); bootstrap resolution too coarse to show significance either way (noted in the report) |
| Physics (`--ablation physics`) | Kalman drift, OU, Reynolds-number and Fisher-Rao analogues | 0/16 Bonferroni survivors for direction; volatility mixed (3 better, 5 worse) |
| Schrödinger (`schrodinger.py`, `train_schrodinger.py`) | literal 1-D wave-function propagation of price | Null on dev and on the ~18.5-day holdout; Born-rule signal has unstable sign |
| Edge lab (`edge_lab.py`) | 27 cost-aware configurations, confidence filters, fee scenarios | Finalists hit 51.8-52.5% on the holdout vs ~48% baselines, but the gross edge is a few tenths of a bp per trade against 20 bp fees; filtered trades too few (0/3/1) to test; no edge claimed |
| Mixed arm (`mix_lab.py`) | GCP coupled into the Schrödinger equation, each coupling vs a time-shifted GCP control | 0/16 survivors in all five comparisons, dev and holdout |
| Seed lab (`seed_lab.py`) | low-dimensional determinism (simplex/S-map, compressibility, BDS), Gaussian processes, Fibonacci/Gann/lunar/Benford, FFT/Hurst, LPPLS | Determinism hypothesis not supported; every family null for direction on the 8.2-day holdout (0/16 survivors) |
| Live (`live.py`, `live_eval.py`, `significance.py`) | frozen models on new closed candles | A few cells beat chance at times but membership flips between runs; verdict "not a confirmed trading edge" |

Machine-readable numbers: `results.json`, `results_physics.json`, `results_*_dev.json`,
`results_*_holdout.json`, `live_eval.json`, `live_edge_significance.json`,
`track_record.jsonl`, `verdict_history.jsonl`. Literature context: `research.md`.

## Data (not in this repository)

The input databases are not committed (the development database alone is about 630 MB).
Paths in the scripts and in `docs/full-report.md` assume this folder sits in a workspace next to
`crypto-data/` and `gcp2-data/`:

| Input | Contents | How to rebuild |
| --- | --- | --- |
| `crypto-data/crypto_market.db` | SQLite table `klines_1m` (`symbol`, `open_time_s`, `open`, `high`, `low`, `close`, `volume`, `quote_volume`, `trades`) from 2024-03-01 | Download 1-minute klines for the four symbols from Binance's public `GET /api/v3/klines` and insert them into that schema |
| `crypto-data/holdout_klines_1m.db` | same schema, candles after the dev set's last minute (2026-09-14 05:08 UTC) | Same endpoint, starting one minute after the dev set ends |
| `crypto-data/gcp2_1m.db`, `gcp2_holdout_1m.db` | per-minute means of the 1-second GCP 2.0 netvar | `features.py` can rebuild the minute means by streaming the raw GCP 2.0 monthly zip archives (`--gcp gcp2-data/network_coherence/global_network`) |
| `gcp2-data/event_hypothesis_archives/events.json` | registered event list for the descriptive event study | GCP 2.0 event archive |

The small fetch helpers I used to fill the holdout databases are not part of this repository.
Model artifacts (`model_*.joblib`), the live prediction log (`live_predictions.jsonl`) and the
kline cache (`kline_cache.sqlite`) are generated by the scripts and ignored by git.

## Usage

Python 3 with `numpy`, `pandas`, `scipy`, `scikit-learn` and `joblib`:

```sh
python3 -m venv .venv
.venv/bin/pip install numpy pandas scipy scikit-learn joblib

# historical evaluation (writes results.json and the model_*.joblib artifacts)
.venv/bin/python train.py   # defaults to ../crypto-data/crypto_market.db

# live loop on public Binance candles; the GCP arm is optional and needs a gcp2.net
# token in GCP2_TOKEN or GCP2_TOKEN_FILE, otherwise output is market-only
.venv/bin/python live.py --once
.venv/bin/python live_eval.py --once
.venv/bin/python significance.py --windowed
```

Each lab script documents its stages in its module docstring and in its pre-registration file.
Tests: `test_schrodinger.py`, `test_mix_lab.py`, `test_seed_lab.py` (pytest).

## Limitations

- Minute OHLCV only; no order-book or trade-flow data, which the literature in `research.md`
  suggests is where short-horizon signal lives.
- Wilson intervals on hundreds of thousands of serially dependent minutes are optimistic; the
  block bootstrap is the test of record.
- Holdouts are 8-18 days long. The frozen tests can be re-run unchanged on longer holdouts.
- Nothing here is investment advice, and the code is research code, not a trading system.

## License

MIT, see [LICENSE](LICENSE).
