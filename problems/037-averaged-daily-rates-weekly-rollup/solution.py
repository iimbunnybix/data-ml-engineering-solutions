"""Roll up conversion rates by summing their parts, not by averaging rates.

A metrics job stored one conversion *rate* per (country, day) and built the
weekly and global numbers by averaging those rates. An average of ratios gives
a quiet day with 3 visits the same weight as a busy day with 30,000 visits, so
small, noisy groups dominate the result. This module keeps the additive parts
(conversions and visits) at every level, sums them on rollup, and divides once
at the end. Groups with no visits produce ``None`` instead of a fake 0% rate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterable


@dataclass(frozen=True)
class DailyFunnel:
    """One row of the daily funnel table: the additive counts for one group."""

    country: str
    day: str
    visits: int
    conversions: int

    def __post_init__(self) -> None:
        # A rate is only meaningful when 0 <= conversions <= visits.
        if self.visits < 0 or self.conversions < 0:
            raise ValueError("counts must be non-negative")
        if self.conversions > self.visits:
            raise ValueError(f"{self.country} {self.day}: more conversions than visits")


@dataclass
class RatioParts:
    """Numerator and denominator of a rate, kept separately so they can be summed."""

    conversions: int = 0
    visits: int = 0

    def add(self, other: "RatioParts") -> None:
        """Merge another group's counts into this one."""
        self.conversions += other.conversions
        self.visits += other.visits

    def rate(self) -> float | None:
        """Conversion rate, or None when there were no visits (rate is undefined)."""
        if self.visits == 0:
            return None
        return self.conversions / self.visits


# --- Original job: one rate per row, then averages of averages ----------------

def naive_daily_rates(rows: Iterable[DailyFunnel]) -> dict[tuple[str, str], float]:
    """What the old job stored: only the rate. Days with 0 visits became 0.0."""
    rates: dict[tuple[str, str], float] = {}
    for row in rows:
        rates[(row.country, row.day)] = row.conversions / row.visits if row.visits else 0.0
    return rates


def naive_rollup(rates: dict[tuple[str, str], float], level: str) -> dict[Hashable, float]:
    """Average the stored rates per country (level='country') or overall ('all')."""
    buckets: dict[Hashable, list[float]] = {}
    for (country, _day), rate in rates.items():
        key = country if level == "country" else "all"
        buckets.setdefault(key, []).append(rate)
    return {key: sum(values) / len(values) for key, values in buckets.items()}


# --- Fixed job: keep the parts, sum them, divide once -------------------------

def to_parts(rows: Iterable[DailyFunnel]) -> dict[tuple[str, str], RatioParts]:
    """Store the daily table as additive parts, which is safe to re-aggregate."""
    parts: dict[tuple[str, str], RatioParts] = {}
    for row in rows:
        key = (row.country, row.day)
        parts.setdefault(key, RatioParts()).add(RatioParts(row.conversions, row.visits))
    return parts


def rollup(parts: dict[tuple[str, str], RatioParts], level: str) -> dict[Hashable, RatioParts]:
    """Sum parts per country (level='country') or overall ('all')."""
    if level not in ("country", "all"):
        raise ValueError("level must be 'country' or 'all'")
    totals: dict[Hashable, RatioParts] = {}
    for (country, _day), group in parts.items():
        key = country if level == "country" else "all"
        totals.setdefault(key, RatioParts()).add(group)
    return totals


def rates(totals: dict[Hashable, RatioParts]) -> dict[Hashable, float | None]:
    """Turn summed parts into rates as the very last step."""
    return {key: group.rate() for key, group in totals.items()}


def weekly_conversion(rows: Iterable[DailyFunnel], level: str) -> dict[Hashable, float | None]:
    """The fixed pipeline end to end: daily parts -> rollup -> rates."""
    return rates(rollup(to_parts(rows), level))
