from __future__ import annotations

import pytest

from dividend_capture_spark import (
    DividendEventAnalyzer,
    DividendEventInput,
    LiveMinuteBarStreamingScorer,
    MinuteBarEvent,
    PreDecisionFeatureRow,
    PseudoSectorClassifier,
    TemporalSplitGBDT,
    TickerFeatureProfile,
)


def test_event_analyzer_drop_ratio_and_cross_section() -> None:
    analyzer = DividendEventAnalyzer(round_trip_cost_bps=4.5, high_yield_threshold_bps=80.0)
    ev_high = DividendEventInput(
        event_id="EV-001",
        ticker="VZ",
        ex_date="2025-04-10",
        close_cum_div=40.00,
        open_ex_div=39.48,  # dropped 0.52 vs 0.65 cash dividend -> positive capture
        dividend_cash=0.65,
        spy_overnight_return=0.001,
        beta_to_spy=0.5,
        dollar_volume_20d=450_000_000.0,
        realized_vol_20d=0.18,
        sic_sector="Telecommunications",
    )
    ev_low = DividendEventInput(
        event_id="EV-002",
        ticker="NVDA",
        ex_date="2025-04-10",
        close_cum_div=120.00,
        open_ex_div=119.95,  # dropped 0.05 vs 0.04 cash dividend -> negative capture
        dividend_cash=0.04,
        spy_overnight_return=0.002,
        beta_to_spy=1.4,
        dollar_volume_20d=2_500_000_000.0,
        realized_vol_20d=0.42,
        sic_sector="Semiconductors",
    )

    e1 = analyzer.enrich_event(ev_high)
    e2 = analyzer.enrich_event(ev_low)
    assert e1.drop_ratio == pytest.approx(0.52 / 0.65, rel=1e-3)
    assert e1.yield_bucket == "HIGH_YIELD"
    assert e1.net_capture_after_costs_bps > 0.0
    assert e2.yield_bucket == "LOW_YIELD"
    assert e2.net_capture_after_costs_bps < 0.0

    summary = analyzer.summarize_cross_section([e1, e2])
    assert summary.event_count == 2
    assert summary.high_yield_net_capture_bps > summary.low_yield_net_capture_bps
    assert summary.profitable_fraction == 0.5


def test_pseudo_sector_classifier_imputes_missing_sector() -> None:
    clf = PseudoSectorClassifier(k_neighbors=2)
    labeled = [
        TickerFeatureProfile("UTL1", 110.0, 0.14, 0.45, 8.2, "Utilities"),
        TickerFeatureProfile("UTL2", 105.0, 0.15, 0.50, 8.1, "Utilities"),
        TickerFeatureProfile("TECH1", 12.0, 0.36, 1.35, 9.6, "Technology"),
        TickerFeatureProfile("TECH2", 15.0, 0.34, 1.30, 9.5, "Technology"),
    ]
    clf.fit(labeled)

    unknown_util = TickerFeatureProfile("UNK_U", 108.0, 0.145, 0.48, 8.15, None)
    unknown_tech = TickerFeatureProfile("UNK_T", 14.0, 0.35, 1.32, 9.55, None)
    assert clf.predict(unknown_util) == "Utilities"
    assert clf.predict(unknown_tech) == "Technology"


def test_temporal_split_gbdt_and_leakage_guard() -> None:
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
        PreDecisionFeatureRow(
            event_id="EV-1",
            decision_timestamp_iso="2025-02-01T15:45:00",
            features=(115.0, 0.50),
            realized_net_capture_bps=16.5,
        ),
        PreDecisionFeatureRow(
            event_id="EV-2",
            decision_timestamp_iso="2025-02-02T15:45:00",
            features=(20.0, 1.40),
            realized_net_capture_bps=-14.0,
        ),
    ]

    model = TemporalSplitGBDT(n_estimators=10, learning_rate=0.3)
    metrics = model.evaluate_strategy(train_rows, eval_rows, min_predicted_bps=3.0)
    assert metrics["model_selected_mean_bps"] > metrics["always_trade_mean_bps"]
    assert metrics["selected_event_count"] == 1.0

    # Leakage check: eval row earlier than train row must raise ValueError
    leaked_eval = [
        PreDecisionFeatureRow(
            event_id="LEAK",
            decision_timestamp_iso="2025-01-05T15:45:00",
            features=(100.0, 0.6),
            realized_net_capture_bps=10.0,
        )
    ]
    with pytest.raises(ValueError, match="Temporal lookahead leakage"):
        model.evaluate_strategy(train_rows, leaked_eval)


def test_live_minute_bar_streaming_scorer() -> None:
    scorer = LiveMinuteBarStreamingScorer()
    res1 = scorer.ingest_bar(
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
    assert res1["vwap"] == 45.0
    assert res1["order_imbalance"] == pytest.approx(0.5)
    assert res1["execute_capture"] is True
