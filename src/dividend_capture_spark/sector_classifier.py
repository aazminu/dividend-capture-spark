from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class TickerFeatureProfile:
    ticker: str
    dividend_yield_bps: float
    realized_vol_20d: float
    beta_to_spy: float
    log_dollar_volume: float
    known_sector: str | None = None


class PseudoSectorClassifier:
    """Hybrid z-score standardized centroid + k-NN classifier imputing missing SIC sector labels."""

    def __init__(self, k_neighbors: int = 3) -> None:
        if k_neighbors < 1:
            raise ValueError("k_neighbors must be >= 1")
        self.k_neighbors = k_neighbors
        self._means: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
        self._stds: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
        self._labeled: List[Tuple[Tuple[float, float, float, float], str]] = []
        self._centroids: Dict[str, Tuple[float, float, float, float]] = {}

    @staticmethod
    def _raw_vec(p: TickerFeatureProfile) -> Tuple[float, float, float, float]:
        return (p.dividend_yield_bps, p.realized_vol_20d, p.beta_to_spy, p.log_dollar_volume)

    def _standardize(self, v: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        return (
            (v[0] - self._means[0]) / self._stds[0],
            (v[1] - self._means[1]) / self._stds[1],
            (v[2] - self._means[2]) / self._stds[2],
            (v[3] - self._means[3]) / self._stds[3],
        )

    def fit(self, profiles: Sequence[TickerFeatureProfile]) -> None:
        labeled = [p for p in profiles if p.known_sector]
        if not labeled:
            raise ValueError("At least one labeled TickerFeatureProfile is required")

        vecs = [self._raw_vec(p) for p in labeled]
        n = len(vecs)
        means = tuple(sum(v[i] for v in vecs) / n for i in range(4))
        stds = []
        for i in range(4):
            var = sum((v[i] - means[i]) ** 2 for v in vecs) / max(n - 1, 1)
            stds.append(max(math.sqrt(var), 1e-6))
        self._means = (means[0], means[1], means[2], means[3])
        self._stds = (stds[0], stds[1], stds[2], stds[3])

        self._labeled = [(self._standardize(self._raw_vec(p)), str(p.known_sector)) for p in labeled]
        by_sector: Dict[str, List[Tuple[float, float, float, float]]] = {}
        for zvec, sec in self._labeled:
            by_sector.setdefault(sec, []).append(zvec)
        self._centroids = {
            sec: (
                sum(v[0] for v in s_vecs) / len(s_vecs),
                sum(v[1] for v in s_vecs) / len(s_vecs),
                sum(v[2] for v in s_vecs) / len(s_vecs),
                sum(v[3] for v in s_vecs) / len(s_vecs),
            )
            for sec, s_vecs in by_sector.items()
        }

    def predict(self, profile: TickerFeatureProfile) -> str:
        if profile.known_sector:
            return profile.known_sector
        if not self._labeled:
            raise RuntimeError("PseudoSectorClassifier must be fit before calling predict")

        z = self._standardize(self._raw_vec(profile))
        dists = [
            (math.sqrt(sum((a - b) ** 2 for a, b in zip(z, train_z))), sec)
            for train_z, sec in self._labeled
        ]
        dists.sort(key=lambda item: item[0])
        top_k = dists[: min(self.k_neighbors, len(dists))]

        scores: Dict[str, float] = {}
        for d, sec in top_k:
            scores[sec] = scores.get(sec, 0.0) + 1.0 / (d + 0.1)

        for sec, cvec in self._centroids.items():
            c_dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(z, cvec)))
            scores[sec] = scores.get(sec, 0.0) + 0.65 / (c_dist + 0.25)

        return max(scores.items(), key=lambda kv: (kv[1], kv[0]))[0]
