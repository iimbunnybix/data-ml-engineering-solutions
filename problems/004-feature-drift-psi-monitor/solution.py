"""Training/serving feature drift monitor using the Population Stability Index.

PSI = Σ (actual_i − expected_i) · ln(actual_i / expected_i) over bins.
Widely used rule of thumb (credit-risk practice):
    < 0.1  no significant shift
    0.1–0.2 moderate shift, investigate
    > 0.2  significant shift, act
"""
from __future__ import annotations

import bisect
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Hashable, Iterable, Sequence

EPS = 1e-4          # floor for empty bins so ln() stays finite
WARN, ALERT = 0.1, 0.2


def psi(expected: Sequence[float], actual: Sequence[float], eps: float = EPS) -> float:
    """PSI between two proportion vectors of equal length."""
    if len(expected) != len(actual):
        raise ValueError("bin counts differ")
    total = 0.0
    for e, a in zip(expected, actual):
        e, a = max(e, eps), max(a, eps)
        total += (a - e) * math.log(a / e)
    return total


def _proportions(counts: Sequence[int]) -> list[float]:
    n = sum(counts)
    return [c / n for c in counts] if n else [0.0] * len(counts)


@dataclass
class NumericBaseline:
    """Quantile bin edges fitted on training data, plus missing-value rate."""
    edges: list[float]
    expected: list[float]
    missing_rate: float

    @classmethod
    def fit(cls, values: Iterable[float | None], bins: int = 10) -> "NumericBaseline":
        vals = [v for v in values]
        present = sorted(v for v in vals if v is not None and not math.isnan(v))
        if not present:
            raise ValueError("no non-missing training values")
        # interior quantile edges; dedupe so heavy point masses don't create empty bins
        edges = sorted({present[int(len(present) * q / bins)] for q in range(1, bins)})
        base = cls(edges, [], 1 - len(present) / len(vals))
        base.expected = _proportions(base._counts(present))
        return base

    def _counts(self, present: Iterable[float]) -> list[int]:
        counts = [0] * (len(self.edges) + 1)
        for v in present:
            counts[bisect.bisect_right(self.edges, v)] += 1
        return counts

    def score(self, values: Iterable[float | None]) -> tuple[float, float]:
        """Return (psi, missing_rate) for a serving window."""
        vals = list(values)
        present = [v for v in vals if v is not None and not math.isnan(v)]
        miss = 1 - len(present) / len(vals) if vals else 0.0
        return psi(self.expected, _proportions(self._counts(present))), miss


@dataclass
class CategoricalBaseline:
    expected: dict[Hashable, float]
    other_key: str = "__other__"

    @classmethod
    def fit(cls, values: Iterable[Hashable], min_share: float = 0.01) -> "CategoricalBaseline":
        c = Counter(values)
        n = sum(c.values())
        kept = {k: v / n for k, v in c.items() if v / n >= min_share}
        kept["__other__"] = 1 - sum(kept.values())
        return cls(kept)

    def score(self, values: Iterable[Hashable]) -> float:
        c = Counter(v if v in self.expected else self.other_key for v in values)
        n = sum(c.values()) or 1
        keys = list(self.expected)
        return psi([self.expected[k] for k in keys], [c.get(k, 0) / n for k in keys])


@dataclass
class DriftReport:
    feature: str
    psi: float
    missing_delta: float = 0.0
    level: str = field(init=False)

    def __post_init__(self):
        self.level = ("alert" if self.psi > ALERT or abs(self.missing_delta) > 0.05
                      else "warn" if self.psi > WARN else "ok")


class DriftMonitor:
    """Holds per-feature baselines and scores serving windows against them."""

    def __init__(self, min_window: int = 500):
        self.numeric: dict[str, NumericBaseline] = {}
        self.categorical: dict[str, CategoricalBaseline] = {}
        self.min_window = min_window

    def fit(self, rows: list[dict], numeric: list[str], categorical: list[str]) -> None:
        for f in numeric:
            self.numeric[f] = NumericBaseline.fit(r.get(f) for r in rows)
        for f in categorical:
            self.categorical[f] = CategoricalBaseline.fit(r.get(f) for r in rows)

    def check(self, window: list[dict]) -> list[DriftReport]:
        if len(window) < self.min_window:
            raise ValueError(f"window too small for a stable PSI (<{self.min_window} rows)")
        out = []
        for f, b in self.numeric.items():
            s, miss = b.score(r.get(f) for r in window)
            out.append(DriftReport(f, s, miss - b.missing_rate))
        for f, b in self.categorical.items():
            out.append(DriftReport(f, b.score(r.get(f) for r in window)))
        return sorted(out, key=lambda r: -r.psi)
