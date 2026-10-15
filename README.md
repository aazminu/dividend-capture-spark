# Dividend Capture Spark (`dividend-capture-spark`)

Event-driven market-microstructure pipeline for analyzing ex-dividend price drops, post-ex-date recovery windows, sector-level directional classification, and transaction-cost-adjusted walk-forward dividend-capture backtesting.

## Core Modules

1. **`event_analyzer.py`**:
   - Joins historical corporate dividend declarations with split-adjusted ex-dividend OHLCV bars (`DividendEventAnalyzer`).
   - Computes drop ratios $\Delta P_{\text{ex}} / D$, market-adjusted abnormal overnight returns, and transaction-cost-adjusted net capture (bps) across yield buckets.
2. **`pseudo_sector_classifier.py`**:
   - Z-score standardized $k$-NN feature-space sector imputation (`PseudoSectorClassifier`) for tickers missing SIC/GICS sector metadata.
3. **`gbdt_capture_model.py`**:
   - Point-in-time feature extraction with strict `ex-1 15:45 ET` temporal split verification (`no look-ahead leak`).
   - Gradient-boosted decision stump ensemble (`TemporalSplitGBDT`) with out-of-sample ROC-AUC evaluation and live minute-bar VWAP/order-imbalance streaming scorer (`LiveMinuteBarStreamingScorer`).


## Quickstart

```bash
uv run --with pytest pytest -v
```
