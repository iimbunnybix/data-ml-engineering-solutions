"""Turn "magic" missing-value sentinels into real nulls before aggregating.

A partner CRM exports missing values as sentinels instead of nulls: age 0 or
-1, annual income 9999999, tenure -1. The pipeline averaged these as real
numbers, so average customer age and income were wrong and the credit model
learned from fake values. This module applies a per-column registry of
declared sentinels (sentinel -> None) before any statistics are computed, and
flags undeclared sentinels: one value holding a share far above normal.
"""
from __future__ import annotations

import random
import statistics
from collections import Counter
from dataclasses import dataclass

# Declared with the data owner; each column lists the values that mean "unknown".
SENTINELS_BY_COLUMN: dict[str, frozenset[float]] = {
    "age": frozenset({0, -1}),
    "annual_income": frozenset({9_999_999}),
    "tenure_months": frozenset({-1}),
}
# A single value must hold at least this share of rows to be a suspect...
SPIKE_MIN_SHARE = 0.02
# ...and be this many times more common than a typical distinct value.
SPIKE_RATIO = 5.0
# Columns with fewer distinct values are categorical-like and spike naturally.
MIN_DISTINCT_VALUES = 20

Row = dict[str, float | None]


@dataclass(frozen=True)
class Spike:
    column: str
    value: float
    share: float
    typical_share: float


def naive_mean(rows: list[Row], column: str) -> float:
    """Original pipeline: average everything that is not None."""
    present = [row[column] for row in rows if row.get(column) is not None]
    return statistics.fmean(present)


def replace_sentinels(row: Row, sentinels: dict[str, frozenset[float]]) -> Row:
    """Copy of row with declared sentinel values replaced by None."""
    cleaned: Row = {}
    for column, value in row.items():
        if value is not None and value in sentinels.get(column, frozenset()):
            cleaned[column] = None
        else:
            cleaned[column] = value
    return cleaned


def clean_rows(rows: list[Row],
               sentinels: dict[str, frozenset[float]] = SENTINELS_BY_COLUMN) -> list[Row]:
    """Apply replace_sentinels to every row (inputs are not modified)."""
    return [replace_sentinels(row, sentinels) for row in rows]


def mean_and_missing_rate(rows: list[Row], column: str) -> tuple[float | None, float]:
    """Mean of non-null values, plus the share of rows that are null.

    Reporting the missing rate next to the mean keeps "unknown" visible
    instead of silently dropping it.
    """
    present = [row[column] for row in rows if row.get(column) is not None]
    missing_rate = 1 - len(present) / len(rows) if rows else 0.0
    if not present:
        return None, missing_rate
    return statistics.fmean(present), missing_rate


def find_spike(column: str, values: list[float | None]) -> Spike | None:
    """Most common value if it looks like an undeclared sentinel, else None."""
    present = [value for value in values if value is not None]
    counts = Counter(present)
    if len(counts) < MIN_DISTINCT_VALUES:
        return None
    top_value, top_count = counts.most_common(1)[0]
    # The median share of distinct values is what a "normal" value looks like;
    # a mean would be pulled up by the spike itself.
    typical_share = statistics.median(counts.values()) / len(present)
    top_share = top_count / len(present)
    if top_share >= SPIKE_MIN_SHARE and top_share >= SPIKE_RATIO * typical_share:
        return Spike(column, top_value, top_share, typical_share)
    return None


def find_spikes(rows: list[Row], columns: list[str]) -> list[Spike]:
    """Check every column; run after clean_rows to catch sentinels nobody declared."""
    spikes = []
    for column in columns:
        spike = find_spike(column, [row.get(column) for row in rows])
        if spike is not None:
            spikes.append(spike)
    return spikes


def simulate_crm_export(count: int, rng: random.Random) -> list[Row]:
    """Customers where the CRM writes sentinels for unknown fields."""
    rows = []
    for _ in range(count):
        age: float = rng.randint(18, 80)
        if rng.random() < 0.12:
            age = rng.choice([0, -1])
        income: float = round(rng.lognormvariate(13.3, 0.5))
        if rng.random() < 0.03:
            income = 9_999_999
        tenure: float = rng.randint(0, 120)
        if rng.random() < 0.10:
            # Undeclared: a newer CRM version writes 999 for unknown tenure.
            tenure = 999
        rows.append({"age": age, "annual_income": income, "tenure_months": tenure})
    return rows
