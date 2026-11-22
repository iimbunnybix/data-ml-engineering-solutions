"""Count values per band so that every value lands in exactly one band.

The delivery-pricing report bucketed trip distances into bands written as
(low, high) pairs and tested ``low <= distance <= high``. A trip of exactly
5.0 km matched both 0-5 and 5-10, so band totals exceeded the number of
trips. The first "fix" used bands 0-4.9, 5-9.9, which silently dropped trips
of 4.95 km that fall in the gap.

The fix defines bands by one sorted list of edges, so they are contiguous by
construction, uses half-open intervals [low, high) with the last band closed
on the right, reports values outside the range instead of dropping them, and
checks that counts reconcile with the number of input values.
"""
from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass, field


@dataclass
class BandReport:
    counts: dict[str, int]
    out_of_range: list[float] = field(default_factory=list)   # includes NaN

    @property
    def total(self) -> int:
        return sum(self.counts.values()) + len(self.out_of_range)


# --- the original versions --------------------------------------------------
def naive_band_counts(values: list[float], bands: list[tuple[float, float]]) -> dict[str, int]:
    """Original report: inclusive on both ends, so a shared edge counts twice."""
    counts = {f"{low:g}-{high:g}": 0 for low, high in bands}
    for value in values:
        for low, high in bands:
            if low <= value <= high:
                counts[f"{low:g}-{high:g}"] += 1
    return counts


# --- the fix ----------------------------------------------------------------
def validate_edges(edges: list[float]) -> None:
    """Edges must be finite and strictly increasing; at least one band."""
    if len(edges) < 2:
        raise ValueError("need at least two edges (one band)")
    for edge in edges:
        if not math.isfinite(edge):
            raise ValueError(f"edge {edge} is not finite")
    for index in range(1, len(edges)):
        if edges[index] <= edges[index - 1]:
            raise ValueError(f"edges must be strictly increasing: {edges[index - 1]} then {edges[index]}")


def band_labels(edges: list[float]) -> list[str]:
    """Labels like '[0, 5)' that state which end is included; last band is closed."""
    labels = []
    last_band = len(edges) - 2
    for index in range(len(edges) - 1):
        closing = "]" if index == last_band else ")"
        labels.append(f"[{edges[index]:g}, {edges[index + 1]:g}{closing}")
    return labels


def band_index(value: float, edges: list[float]) -> int | None:
    """Index of the band holding ``value``, or None if it is outside all bands."""
    # NaN compares False with everything; it must not fall into band 0.
    if math.isnan(value) or value < edges[0] or value > edges[-1]:
        return None
    # The top edge itself belongs to the last band, not to a band beyond it.
    if value == edges[-1]:
        return len(edges) - 2
    # bisect_right puts a value equal to an edge into the band starting there.
    return bisect_right(edges, value) - 1


def band_counts(values: list[float], edges: list[float]) -> BandReport:
    """Count values per band; every value is counted once or reported."""
    validate_edges(edges)
    labels = band_labels(edges)
    report = BandReport(counts={label: 0 for label in labels})
    for value in values:
        index = band_index(value, edges)
        if index is None:
            report.out_of_range.append(value)
        else:
            report.counts[labels[index]] += 1
    # Cheap invariant that would have caught both original bugs.
    if report.total != len(values):
        raise AssertionError(f"band counts total {report.total}, expected {len(values)}")
    return report
