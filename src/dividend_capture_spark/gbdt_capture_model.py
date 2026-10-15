from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class PreDecisionFeatureRow:
    event_id: str
    decision_timestamp_iso: str
    features: Tuple[float, ...]
    realized_net_capture_bps: float


@dataclass(frozen=True)
class DecisionStump:
    feature_idx: int
    threshold: float
    left_value: float
    right_value: float


@dataclass(frozen=True)
class MinuteBarEvent:
    ticker: str
    minute_iso: str
    price: float
    volume: float
    buy_volume: float
    dividend_yield_bps: float
    beta_to_spy: float


class TemporalSplitGBDT:
    """Gradient-boosted regression stump ensemble with strict temporal split verification (ex-1 15:45 ET)."""

    def __init__(self, n_estimators: int = 12, learning_rate: float = 0.25) -> None:
        if n_estimators < 1:
            raise ValueError("n_estimators must be >= 1")
        if not (0.0 < learning_rate <= 1.0):
            raise ValueError("learning_rate must be in (0, 1]")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.base_prediction: float = 0.0
        self.stumps: List[DecisionStump] = []

    @staticmethod
    def verify_temporal_split(
        train_rows: Sequence[PreDecisionFeatureRow],
        eval_rows: Sequence[PreDecisionFeatureRow],
    ) -> None:
        if not train_rows or not eval_rows:
            raise ValueError("train_rows and eval_rows must both be non-empty")
        max_train_ts = max(r.decision_timestamp_iso for r in train_rows)
        min_eval_ts = min(r.decision_timestamp_iso for r in eval_rows)
        if max_train_ts >= min_eval_ts:
            raise ValueError(
                f"Temporal lookahead leakage detected: max train timestamp {max_train_ts} >= min eval timestamp {min_eval_ts}"
            )

    def fit(self, train_rows: Sequence[PreDecisionFeatureRow]) -> None:
        if not train_rows:
            raise ValueError("train_rows cannot be empty")
        dim = len(train_rows[0].features)
        if dim == 0 or any(len(r.features) != dim for r in train_rows):
            raise ValueError("All feature rows must have identical non-zero dimensionality")

        n = len(train_rows)
        targets = [r.realized_net_capture_bps for r in train_rows]
        self.base_prediction = sum(targets) / n
        preds = [self.base_prediction] * n
        self.stumps = []

        for _ in range(self.n_estimators):
            residuals = [targets[i] - preds[i] for i in range(n)]
            best_loss = float("inf")
            best_stump = DecisionStump(0, 0.0, 0.0, 0.0)

            for f_idx in range(dim):
                col_vals = sorted({r.features[f_idx] for r in train_rows})
                for thresh in col_vals:
                    left_res = [residuals[i] for i in range(n) if train_rows[i].features[f_idx] <= thresh]
                    right_res = [residuals[i] for i in range(n) if train_rows[i].features[f_idx] > thresh]
                    if not left_res or not right_res:
                        continue
                    l_mean = sum(left_res) / len(left_res)
                    r_mean = sum(right_res) / len(right_res)
                    loss = sum((x - l_mean) ** 2 for x in left_res) + sum(
                        (x - r_mean) ** 2 for x in right_res
                    )
                    if loss < best_loss:
                        best_loss = loss
                        best_stump = DecisionStump(f_idx, thresh, l_mean, r_mean)

            self.stumps.append(best_stump)
            for i in range(n):
                step_val = (
                    best_stump.left_value
                    if train_rows[i].features[best_stump.feature_idx] <= best_stump.threshold
                    else best_stump.right_value
                )
                preds[i] += self.learning_rate * step_val

    def predict_bps(self, features: Sequence[float]) -> float:
        pred = self.base_prediction
        for s in self.stumps:
            val = s.left_value if features[s.feature_idx] <= s.threshold else s.right_value
            pred += self.learning_rate * val
        return round(pred, 4)

    def evaluate_strategy(
        self,
        train_rows: Sequence[PreDecisionFeatureRow],
        eval_rows: Sequence[PreDecisionFeatureRow],
        min_predicted_bps: float = 2.0,
    ) -> Dict[str, float]:
        self.verify_temporal_split(train_rows, eval_rows)
        self.fit(train_rows)

        always_trade_mean = sum(r.realized_net_capture_bps for r in eval_rows) / len(eval_rows)
        selected = [
            r.realized_net_capture_bps
            for r in eval_rows
            if self.predict_bps(r.features) >= min_predicted_bps
        ]
        model_mean = sum(selected) / len(selected) if selected else 0.0
        return {
            "always_trade_mean_bps": round(always_trade_mean, 3),
            "model_selected_mean_bps": round(model_mean, 3),
            "selected_event_count": float(len(selected)),
        }


class LiveMinuteBarStreamingScorer:
    """Scores live-style pre-15:45 ET ex-1 minute bars with rolling VWAP and order-flow imbalance."""

    def __init__(self) -> None:
        self._pv_sum: Dict[str, float] = {}
        self._v_sum: Dict[str, float] = {}
        self._buy_v_sum: Dict[str, float] = {}

    def ingest_bar(self, bar: MinuteBarEvent) -> Dict[str, float | bool | str]:
        if bar.volume <= 0.0 or bar.price <= 0.0:
            raise ValueError("price and volume must be strictly positive")
        t = bar.ticker
        self._pv_sum[t] = self._pv_sum.get(t, 0.0) + bar.price * bar.volume
        self._v_sum[t] = self._v_sum.get(t, 0.0) + bar.volume
        self._buy_v_sum[t] = self._buy_v_sum.get(t, 0.0) + max(0.0, min(bar.buy_volume, bar.volume))

        vwap = self._pv_sum[t] / self._v_sum[t]
        order_imbalance = (2.0 * self._buy_v_sum[t] - self._v_sum[t]) / self._v_sum[t]
        expected_capture_bps = (
            0.13 * bar.dividend_yield_bps
            + 6.5 * order_imbalance
            - 3.2 * max(0.0, bar.beta_to_spy - 1.0)
            - 4.5
        )
        return {
            "ticker": t,
            "vwap": round(vwap, 4),
            "order_imbalance": round(order_imbalance, 4),
            "expected_capture_bps": round(expected_capture_bps, 3),
            "execute_capture": expected_capture_bps >= 5.0,
        }
