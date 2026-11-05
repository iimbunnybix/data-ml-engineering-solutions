"""Write-Audit-Publish (WAP) for a daily partition load.

The original job loaded each day's export straight into the published
``orders`` table. When the upstream export was truncated, the dashboards
showed the partial data as if it were real.

``WapLoader`` writes each day to a *staging* area, runs audits against it, and
only then publishes it with an atomic pointer swap. A failed audit leaves the
published table untouched (yesterday's state stays visible), quarantines the
staged data for inspection, and returns a report for alerting.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Callable, Optional

Row = dict


class Warehouse:
    """Tiny stand-in for a table with partitions plus a staging area."""

    def __init__(self) -> None:
        self.published: dict[str, list[Row]] = {}
        self.staging: dict[str, list[Row]] = {}
        self.quarantine: dict[str, list[Row]] = {}

    def revenue(self, day: str) -> float:
        return round(sum(r["amount"] for r in self.published.get(day, [])), 2)


def naive_load(wh: Warehouse, day: str, rows: list[Row]) -> None:
    """Original behaviour: overwrite the published partition with whatever arrived."""
    wh.published[day] = list(rows)


# ------------------------------------------------------------------- audits
@dataclass
class AuditResult:
    name: str
    passed: bool
    detail: str


@dataclass
class AuditReport:
    day: str
    results: list[AuditResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def failures(self) -> list[AuditResult]:
        return [r for r in self.results if not r.passed]


Audit = Callable[[list[Row], list[int]], AuditResult]


def row_count_audit(low: float = 0.5, high: float = 2.0, min_history: int = 3) -> Audit:
    """Row count must be within [low, high] x median of recent published partitions.

    The median (not mean) keeps one earlier bad day from skewing the baseline.
    With too little history the audit passes but says so.
    """
    def audit(rows: list[Row], history: list[int]) -> AuditResult:
        if len(history) < min_history:
            return AuditResult("row_count", True, f"skipped: only {len(history)} days of history")
        base = statistics.median(history)
        ratio = len(rows) / base if base else float("inf")
        ok = low <= ratio <= high
        return AuditResult("row_count", ok, f"{len(rows)} rows = {ratio:.2f}x median {base:g}")
    return audit


def null_rate_audit(column: str, max_rate: float) -> Audit:
    def audit(rows: list[Row], _history: list[int]) -> AuditResult:
        if not rows:
            return AuditResult(f"nulls:{column}", False, "no rows")
        nulls = sum(1 for r in rows if r.get(column) is None)
        rate = nulls / len(rows)
        return AuditResult(f"nulls:{column}", rate <= max_rate,
                           f"{nulls}/{len(rows)} null ({rate:.2%}, max {max_rate:.2%})")
    return audit


def unique_audit(column: str) -> Audit:
    def audit(rows: list[Row], _history: list[int]) -> AuditResult:
        seen, dupes = set(), 0
        for r in rows:
            key = r.get(column)
            if key in seen:
                dupes += 1
            seen.add(key)
        return AuditResult(f"unique:{column}", dupes == 0, f"{dupes} duplicate {column} values")
    return audit


def range_audit(column: str, minimum: float) -> Audit:
    """Values must be finite and >= ``minimum``.

    Every comparison with NaN is False, so ``nan < minimum`` alone would let a
    NaN through and turn the day's revenue into NaN; ``math.isfinite`` also
    rejects NaN and +/-inf.
    """
    def audit(rows: list[Row], _history: list[int]) -> AuditResult:
        bad = 0
        for r in rows:
            value = r.get(column)
            if value is None:
                continue                                 # null_rate_audit owns missing values
            if not math.isfinite(value) or value < minimum:
                bad += 1
        return AuditResult(f"range:{column}", bad == 0,
                           f"{bad} rows with {column} < {minimum} or not finite")
    return audit


DEFAULT_AUDITS: list[Audit] = [
    row_count_audit(),
    null_rate_audit("order_id", 0.0),
    null_rate_audit("customer_id", 0.01),
    # Every order needs an amount; range_audit skips None, and one None makes
    # Warehouse.revenue() raise TypeError on the published day.
    null_rate_audit("amount", 0.0),
    unique_audit("order_id"),
    range_audit("amount", 0.0),
]


# ------------------------------------------------------------------- loader
class WapLoader:
    """Write to staging, audit, then publish atomically or quarantine."""

    def __init__(self, wh: Warehouse, audits: Optional[list[Audit]] = None,
                 history_days: int = 7) -> None:
        self.wh = wh
        self.audits = audits if audits is not None else DEFAULT_AUDITS
        self.history_days = history_days
        self.alerts: list[AuditReport] = []

    def _history(self, day: str) -> list[int]:
        prior = sorted(d for d in self.wh.published if d < day)[-self.history_days:]
        return [len(self.wh.published[d]) for d in prior]

    def load(self, day: str, rows: list[Row], *, force: bool = False) -> AuditReport:
        """Stage, audit and (if clean, or ``force``) publish one day's rows."""
        self.wh.staging[day] = list(rows)                    # 1. write
        staged = self.wh.staging[day]
        report = AuditReport(day, [a(staged, self._history(day)) for a in self.audits])  # 2. audit
        if report.passed or force:                           # 3. publish (pointer swap)
            self.wh.published[day] = self.wh.staging.pop(day)
            self.wh.quarantine.pop(day, None)
        else:
            self.wh.quarantine[day] = self.wh.staging.pop(day)
            self.alerts.append(report)
        return report
