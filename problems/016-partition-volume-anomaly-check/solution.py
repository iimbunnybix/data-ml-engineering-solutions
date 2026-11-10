"""Block publishing a daily partition whose row count is abnormal.

An upstream export was cut off at noon, so the daily ``events`` partition
held about half the usual rows. The original pipeline only checked that the
partition was non-empty, published it, and every dashboard showed a fake
50% drop. This module compares the new partition's row count with the same
weekday in previous weeks using a robust score (median and MAD), and blocks
publishing when the count is far outside that history, too low or too high.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from statistics import median

# Number of previous same-weekday partitions used as the baseline.
HISTORY_WEEKS = 8
# Fewer comparable days than this and the baseline is not trustworthy.
MIN_HISTORY_POINTS = 4
# Scales MAD so it estimates the standard deviation of normally distributed
# data; 0.6745 is the 75th percentile of the standard normal distribution.
MAD_TO_SIGMA = 0.6745
# Iglewicz & Hoaglin's commonly used cut-off for the modified z-score.
MAX_ABS_ROBUST_Z = 3.5
# If history is perfectly flat, MAD is 0; treat spread as at least 1% of the
# median so tiny normal changes are not flagged as infinite anomalies.
MIN_RELATIVE_SPREAD = 0.01
# When the median itself is 0 (a sparse table that is usually empty on this
# weekday), the relative floor is 0 too; never let the spread drop below one
# row, so the score stays finite.
MIN_ABSOLUTE_SPREAD = 1.0


@dataclass(frozen=True)
class VolumeDecision:
    """Outcome of the volume check for one partition."""

    publish: bool
    reason: str
    robust_z: float | None = None
    baseline_median: float | None = None


def naive_should_publish(row_count: int) -> bool:
    """Original check: anything non-empty is published."""
    return row_count > 0


def same_weekday_history(day: date, counts_by_day: dict[date, int],
                         weeks: int = HISTORY_WEEKS) -> list[int]:
    """Row counts of the same weekday in the previous ``weeks`` weeks.

    Comparing a Sunday with the previous Sundays (not with Friday) removes
    the weekly cycle, which is the largest normal swing in event volume.
    Missing days are skipped rather than treated as zero.
    """
    history = []
    for weeks_back in range(1, weeks + 1):
        earlier_day = day - timedelta(weeks=weeks_back)
        if earlier_day in counts_by_day:
            history.append(counts_by_day[earlier_day])
    return history


def median_absolute_deviation(values: list[int]) -> float:
    """Median distance from the median; unaffected by a few outliers."""
    center = median(values)
    distances = []
    for value in values:
        distances.append(abs(value - center))
    return median(distances)


def robust_z_score(row_count: int, history: list[int]) -> tuple[float, float]:
    """Return (modified z-score, baseline median) for ``row_count``.

    Median and MAD are used instead of mean and standard deviation because
    a past bad day (an earlier outage or a double load) would otherwise
    inflate the spread and hide today's anomaly.
    """
    center = median(history)
    spread = median_absolute_deviation(history)
    spread = max(spread, MIN_RELATIVE_SPREAD * center, MIN_ABSOLUTE_SPREAD)
    z_score = MAD_TO_SIGMA * (row_count - center) / spread
    return z_score, center


def check_partition_volume(day: date, row_count: int,
                           counts_by_day: dict[date, int]) -> VolumeDecision:
    """Decide whether the partition for ``day`` may be published."""
    if row_count <= 0:
        return VolumeDecision(False, "partition is empty")

    history = same_weekday_history(day, counts_by_day)
    if len(history) < MIN_HISTORY_POINTS:
        # Without a baseline we cannot judge volume; publish but say so,
        # so a brand-new table is not blocked forever.
        return VolumeDecision(True, "not enough history to check volume")

    z_score, center = robust_z_score(row_count, history)
    if abs(z_score) > MAX_ABS_ROBUST_Z:
        direction = "low" if z_score < 0 else "high"
        reason = (f"row count {row_count} is anomalously {direction} "
                  f"(robust z {z_score:.1f}, same-weekday median {center:.0f})")
        return VolumeDecision(False, reason, z_score, center)
    return VolumeDecision(True, "volume within normal range", z_score, center)
