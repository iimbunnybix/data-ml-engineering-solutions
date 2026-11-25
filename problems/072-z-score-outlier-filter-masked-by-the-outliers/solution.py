"""Find outliers with median and MAD, not with mean and standard deviation.

A cold-chain pipeline drops faulty fridge-sensor readings before they reach
the spoilage model, using "more than 3 standard deviations from the mean".
The outliers themselves pull the mean toward them and inflate the standard
deviation, so when a sensor gets stuck at a bad value for a while (or a batch
is small) no reading is ever 3 deviations away and nothing is removed.

Fix: measure distance from the median, scaled by the median absolute
deviation (MAD). Both ignore up to half of the data being bad, so the faulty
readings cannot hide themselves. Uses the Iglewicz-Hoaglin "modified z-score".
"""
from __future__ import annotations

import statistics

NAIVE_Z_LIMIT = 3.0
# Iglewicz & Hoaglin recommend flagging modified z-scores above 3.5.
MODIFIED_Z_LIMIT = 3.5
# 0.6745 is the 75th percentile of the standard normal: it makes MAD-based
# scores comparable to ordinary z-scores for normally distributed data.
MAD_TO_Z_FACTOR = 0.6745
# Fallback when MAD is 0: mean absolute deviation times 1.2533 (sqrt(pi/2))
# estimates the standard deviation for normal data.
MEAN_ABS_DEV_TO_SIGMA = 1.253314


# --------------------------------------------------------------- original bug
def naive_outlier_indexes(readings: list[float]) -> list[int]:
    """Original: flag readings more than 3 sample standard deviations from the mean."""
    if len(readings) < 2:
        return []
    mean = statistics.fmean(readings)
    # BUG: the outliers are part of the data these two numbers are computed from.
    spread = statistics.stdev(readings)
    if spread == 0:
        return []
    flagged = []
    for index, value in enumerate(readings):
        if abs(value - mean) / spread > NAIVE_Z_LIMIT:
            flagged.append(index)
    return flagged


# ----------------------------------------------------------------------- fix
def median_absolute_deviation(readings: list[float], median: float) -> float:
    """Median distance from the median: a spread measure outliers cannot inflate."""
    distances = [abs(value - median) for value in readings]
    return statistics.median(distances)


def robust_scores(readings: list[float]) -> list[float]:
    """Modified z-score of every reading (0.0 for all when the data has no spread)."""
    median = statistics.median(readings)
    mad = median_absolute_deviation(readings, median)
    if mad > 0:
        scale = mad / MAD_TO_Z_FACTOR
    else:
        # More than half the readings are identical, so MAD is 0. Fall back to
        # the mean absolute deviation, which is non-zero if anything differs.
        mean_abs_dev = statistics.fmean([abs(value - median) for value in readings])
        scale = mean_abs_dev * MEAN_ABS_DEV_TO_SIGMA
    if scale == 0:
        return [0.0] * len(readings)
    return [abs(value - median) / scale for value in readings]


def outlier_indexes(readings: list[float]) -> list[int]:
    """Indexes of readings whose modified z-score is above the limit."""
    if not readings:
        return []
    flagged = []
    for index, score in enumerate(robust_scores(readings)):
        if score > MODIFIED_Z_LIMIT:
            flagged.append(index)
    return flagged


# ------------------------------------------------------------------- helpers
def without_indexes(readings: list[float], indexes: list[int]) -> list[float]:
    """The readings that survive the filter, in their original order."""
    dropped = set(indexes)
    kept = []
    for index, value in enumerate(readings):
        if index not in dropped:
            kept.append(value)
    return kept
