# Dividend Capture Spark (`dividend-capture-spark`)

Event-driven market-microstructure pipeline for analyzing ex-dividend price drops, post-ex-date recovery windows, sector-level directional classification, and transaction-cost-adjusted walk-forward dividend-capture backtesting.

## Core Modules

1. **`event_table.py`**:
   - Joins historical corporate dividend declarations (`declaration_date`, `ex_date`, `pay_date`, `dividend_amount`) with split-adjusted daily OHLCV bars across `[T-5, T+10]` trading sessions.
   - Computes drop ratios $\Delta P_{\text{ex}} / D$, recovery trading days, and sector-adjusted abnormal returns over `[T-1, T+k]`.
2. **`sector_classifier.py`**:
   - Benchmarks L2-regularized multinomial/binary Logistic Regression against Gaussian & Multinomial Naive Bayes across 11 GICS equity sectors (including Technology and Consumer Cyclical subsets).
3. **`gbdt_capture_model.py`**:
   - Point-in-time feature extraction (trailing volatility, drop-to-dividend history, dollar volume liquidity, yield regime) with strict declaration-to-ex-date causality (`no look-ahead leak`).
   - Gradient-boosted decision stump ensemble (`GBDTCaptureScorer`) and purged walk-forward backtester with round-trip transaction cost deduction (bps).

## Quickstart

```bash
uv run --with pytest pytest -v
```
