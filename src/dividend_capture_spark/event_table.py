from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence


@dataclass(frozen=True)
class DividendEventInput:
    event_id: str
    ticker: str
    ex_date: str
    close_cum_div: float
    open_ex_div: float
    dividend_cash: float
    spy_overnight_return: float
    beta_to_spy: float
    dollar_volume_20d: float
    realized_vol_20d: float
    sic_sector: Optional[str] = None


@dataclass(frozen=True)
class EnrichedDividendEvent:
    event_id: str
    ticker: str
    ex_date: str
    drop_ratio: float
    dividend_yield_bps: float
    raw_capture_bps: float
    spy_adjusted_capture_bps: float
    net_capture_after_costs_bps: float
    yield_bucket: str
    sector: str


@dataclass(frozen=True)
class CrossSectionalSummary:
    event_count: int
    median_drop_ratio: float
    mean_unadjusted_capture_bps: float
    mean_net_adjusted_capture_bps: float
    high_yield_net_capture_bps: float
    low_yield_net_capture_bps: float
    profitable_fraction: float


class DividendEventAnalyzer:
    """Constructs event-grain dividend drop tables and evaluates SPY-adjusted overnight capture returns."""

    def __init__(self, round_trip_cost_bps: float = 4.5, high_yield_threshold_bps: float = 80.0) -> None:
        if round_trip_cost_bps < 0.0:
            raise ValueError("round_trip_cost_bps cannot be negative")
        if high_yield_threshold_bps <= 0.0:
            raise ValueError("high_yield_threshold_bps must be positive")
        self.round_trip_cost_bps = round_trip_cost_bps
        self.high_yield_threshold_bps = high_yield_threshold_bps

    def enrich_event(self, ev: DividendEventInput, fallback_sector: str = "Unclassified") -> EnrichedDividendEvent:
        if ev.close_cum_div <= 0.0 or ev.open_ex_div <= 0.0:
            raise ValueError("Prices must be strictly positive")
        if ev.dividend_cash <= 0.0:
            raise ValueError("dividend_cash must be strictly positive")

        price_drop = ev.close_cum_div - ev.open_ex_div
        drop_ratio = price_drop / ev.dividend_cash
        div_yield_bps = (ev.dividend_cash / ev.close_cum_div) * 10000.0
        raw_ret = (ev.open_ex_div + ev.dividend_cash - ev.close_cum_div) / ev.close_cum_div
        raw_capture_bps = raw_ret * 10000.0

        market_move_bps = ev.beta_to_spy * ev.spy_overnight_return * 10000.0
        spy_adj_bps = raw_capture_bps - market_move_bps
        net_bps = spy_adj_bps - self.round_trip_cost_bps

        bucket = "HIGH_YIELD" if div_yield_bps >= self.high_yield_threshold_bps else "LOW_YIELD"
        sector = ev.sic_sector if ev.sic_sector else fallback_sector

        return EnrichedDividendEvent(
            event_id=ev.event_id,
            ticker=ev.ticker,
            ex_date=ev.ex_date,
            drop_ratio=round(drop_ratio, 5),
            dividend_yield_bps=round(div_yield_bps, 3),
            raw_capture_bps=round(raw_capture_bps, 3),
            spy_adjusted_capture_bps=round(spy_adj_bps, 3),
            net_capture_after_costs_bps=round(net_bps, 3),
            yield_bucket=bucket,
            sector=sector,
        )

    def summarize_cross_section(self, events: Sequence[EnrichedDividendEvent]) -> CrossSectionalSummary:
        if not events:
            raise ValueError("events cannot be empty")
        n = len(events)
        median_drop = statistics.median(e.drop_ratio for e in events)
        mean_raw = sum(e.raw_capture_bps for e in events) / n
        mean_net = sum(e.net_capture_after_costs_bps for e in events) / n

        high_y = [e.net_capture_after_costs_bps for e in events if e.yield_bucket == "HIGH_YIELD"]
        low_y = [e.net_capture_after_costs_bps for e in events if e.yield_bucket == "LOW_YIELD"]
        high_mean = sum(high_y) / len(high_y) if high_y else 0.0
        low_mean = sum(low_y) / len(low_y) if low_y else 0.0
        prof_frac = sum(1 for e in events if e.net_capture_after_costs_bps > 0.0) / n

        return CrossSectionalSummary(
            event_count=n,
            median_drop_ratio=round(median_drop, 4),
            mean_unadjusted_capture_bps=round(mean_raw, 3),
            mean_net_adjusted_capture_bps=round(mean_net, 3),
            high_yield_net_capture_bps=round(high_mean, 3),
            low_yield_net_capture_bps=round(low_mean, 3),
            profitable_fraction=round(prof_frac, 4),
        )
