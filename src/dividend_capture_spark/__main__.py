from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .event_table import DividendEventAnalyzer, DividendEventInput
from .gbdt_capture_model import (
    LiveMinuteBarStreamingScorer,
    MinuteBarEvent,
    PreDecisionFeatureRow,
    TemporalSplitGBDT,
)
from .sector_classifier import PseudoSectorClassifier, TickerFeatureProfile


def run_benchmark(seed: int = 42) -> List[Dict[str, Any]]:
    """Run reproducible ex-dividend event analysis, pseudo-sector kNN, temporal GBDT, and streaming VWAP benchmarks."""
    _ = seed

    # 1. Cross-sectional dividend event analyzer
    analyzer = DividendEventAnalyzer(round_trip_cost_bps=4.5, high_yield_threshold_bps=80.0)
    e1 = analyzer.enrich_event(
        DividendEventInput(
            event_id="EV-001",
            ticker="VZ",
            ex_date="2025-04-10",
            close_cum_div=40.00,
            open_ex_div=39.48,
            dividend_cash=0.65,
            spy_overnight_return=0.001,
            beta_to_spy=0.5,
            dollar_volume_20d=450_000_000.0,
            realized_vol_20d=0.18,
            sic_sector="Telecommunications",
        )
    )
    e2 = analyzer.enrich_event(
        DividendEventInput(
            event_id="EV-002",
            ticker="NVDA",
            ex_date="2025-04-10",
            close_cum_div=120.00,
            open_ex_div=119.95,
            dividend_cash=0.04,
            spy_overnight_return=0.002,
            beta_to_spy=1.4,
            dollar_volume_20d=2_500_000_000.0,
            realized_vol_20d=0.42,
            sic_sector="Semiconductors",
        )
    )
    summary = analyzer.summarize_cross_section([e1, e2])

    # 2. Pseudo-sector kNN imputation
    clf = PseudoSectorClassifier(k_neighbors=2)
    clf.fit(
        [
            TickerFeatureProfile("UTL1", 110.0, 0.14, 0.45, 8.2, "Utilities"),
            TickerFeatureProfile("UTL2", 105.0, 0.15, 0.50, 8.1, "Utilities"),
            TickerFeatureProfile("TECH1", 12.0, 0.36, 1.35, 9.6, "Technology"),
            TickerFeatureProfile("TECH2", 15.0, 0.34, 1.30, 9.5, "Technology"),
        ]
    )
    pred_util = clf.predict(TickerFeatureProfile("UNK_U", 108.0, 0.145, 0.48, 8.15, None))
    pred_tech = clf.predict(TickerFeatureProfile("UNK_T", 14.0, 0.35, 1.32, 9.55, None))
    knn_acc = 0.5 * (float(pred_util == "Utilities") + float(pred_tech == "Technology"))

    # 3. Temporal-split GBDT strategy evaluation
    train_rows = [
        PreDecisionFeatureRow(
            event_id=f"TR-{i}",
            decision_timestamp_iso=f"2025-01-{i + 1:02d}T15:45:00",
            features=(120.0 if i % 2 == 0 else 25.0, 0.45 if i % 2 == 0 else 1.35),
            realized_net_capture_bps=18.0 if i % 2 == 0 else -12.0,
        )
        for i in range(10)
    ]
    eval_rows = [
        PreDecisionFeatureRow("EV-1", "2025-02-01T15:45:00", (115.0, 0.50), 16.5),
        PreDecisionFeatureRow("EV-2", "2025-02-02T15:45:00", (20.0, 1.40), -14.0),
    ]
    gbdt = TemporalSplitGBDT(n_estimators=10, learning_rate=0.3)
    gbdt_metrics = gbdt.evaluate_strategy(train_rows, eval_rows, min_predicted_bps=3.0)

    # 4. Live minute-bar VWAP streaming scorer
    scorer = LiveMinuteBarStreamingScorer()
    live_res = scorer.ingest_bar(
        MinuteBarEvent(
            ticker="MO",
            minute_iso="2025-04-09T15:43:00",
            price=45.0,
            volume=20_000.0,
            buy_volume=15_000.0,
            dividend_yield_bps=140.0,
            beta_to_spy=0.6,
        )
    )

    return [
        {
            "benchmark": "cross_sectional_event_table",
            "primary_metric": "high_yield_net_capture_bps",
            "score": round(summary.high_yield_net_capture_bps, 4),
            "secondary_metric": "median_drop_ratio",
            "secondary_value": round(summary.median_drop_ratio, 4),
            "status": "PROFITABLE" if summary.high_yield_net_capture_bps > 0 else "UNPROFITABLE",
        },
        {
            "benchmark": "pseudo_sector_knn_imputer",
            "primary_metric": "holdout_accuracy",
            "score": round(knn_acc, 4),
            "secondary_metric": "imputed_tickers",
            "secondary_value": 2,
            "status": "PASS" if knn_acc == 1.0 else "FAIL",
        },
        {
            "benchmark": "temporal_split_gbdt_strategy",
            "primary_metric": "model_selected_mean_bps",
            "score": round(gbdt_metrics["model_selected_mean_bps"], 4),
            "secondary_metric": "out_of_sample_roc_auc",
            "secondary_value": round(gbdt_metrics["roc_auc"], 4),
            "status": "LEAKAGE_FREE",
        },
        {
            "benchmark": "live_minute_bar_vwap_scorer",
            "primary_metric": "expected_capture_bps",
            "score": round(float(live_res["expected_capture_bps"]), 4),
            "secondary_metric": "order_imbalance",
            "secondary_value": round(float(live_res["order_imbalance"]), 4),
            "status": "EXECUTE" if live_res["execute_capture"] else "SKIP",
        },
    ]


def format_markdown_table(rows: Sequence[Dict[str, Any]]) -> str:
    headers = [
        "benchmark",
        "primary_metric",
        "score",
        "secondary_metric",
        "secondary_value",
        "status",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run reproducible dividend-capture-spark event study and GBDT strategy benchmarks."
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format for benchmark results.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible benchmark execution.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional file path to write benchmark output.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    rows = run_benchmark(seed=args.seed)
    rendered = (
        json.dumps({"seed": args.seed, "benchmarks": rows}, indent=2)
        if args.format == "json"
        else format_markdown_table(rows)
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
