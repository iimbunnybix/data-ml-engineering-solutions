"""Catch slow feature drift that a day-over-day comparison never sees.

The fraud model's drift monitor computed PSI between today's feature values
and yesterday's. When the average order value crept up a little every day
(inflation plus a new premium product line), each day looked almost the same
as the one before, so PSI stayed far below the alert threshold for weeks,
while the model's inputs drifted far away from what it was trained on.

Fix: compare every day against a fixed reference, the training-data snapshot
the model was fitted on, with bin edges frozen at training time. Day-over-day
PSI can still be reported to spot sudden jumps, but it is never the only check.
"""
from __future__ import annotations

import math
from bisect import bisect_right
from statistics import NormalDist, quantiles

BIN_COUNT = 10
PSI_ALERT_THRESHOLD = 0.2      # common rule of thumb: < 0.1 stable, > 0.2 significant shift
SMALLEST_SHARE = 1e-4          # stands in for an empty bin so the log stays finite
SAMPLES_PER_DAY = 2000


def daily_values(mean: float, spread: float, count: int = SAMPLES_PER_DAY) -> list[float]:
    """Deterministic sample of a normal distribution (evenly spaced quantiles)."""
    distribution = NormalDist(mean, spread)
    values = []
    for index in range(count):
        values.append(distribution.inv_cdf((index + 0.5) / count))
    return values


def bin_edges(reference: list[float], bin_count: int = BIN_COUNT) -> list[float]:
    """Inner bin edges at the reference's deciles, so each bin holds ~10% of it."""
    return quantiles(reference, n=bin_count)


def bin_shares(values: list[float], edges: list[float]) -> list[float]:
    """Fraction of ``values`` falling into each bin defined by ``edges``."""
    counts = [0] * (len(edges) + 1)
    for value in values:
        counts[bisect_right(edges, value)] += 1
    shares = []
    for count in counts:
        shares.append(max(count / len(values), SMALLEST_SHARE))
    return shares


def population_stability_index(reference: list[float], current: list[float],
                               edges: list[float] | None = None) -> float:
    """PSI = sum((current - reference) * ln(current / reference)) over bins."""
    if edges is None:
        edges = bin_edges(reference)
    reference_shares = bin_shares(reference, edges)
    current_shares = bin_shares(current, edges)
    psi = 0.0
    for expected, actual in zip(reference_shares, current_shares):
        psi += (actual - expected) * math.log(actual / expected)
    return psi


def original_drift_alerts(days: list[list[float]]) -> list[int]:
    """Original monitor: alert on day N if PSI(day N-1 -> day N) is high."""
    alerts = []
    for day_number in range(1, len(days)):
        psi = population_stability_index(days[day_number - 1], days[day_number])
        if psi > PSI_ALERT_THRESHOLD:
            alerts.append(day_number)
    return alerts


class DriftMonitor:
    """Compares each day against the frozen training snapshot."""

    def __init__(self, training_values: list[float]) -> None:
        self.training_values = training_values
        # WHY: edges are frozen with the reference; recomputing them from
        # each day's data would move the bins along with the drift.
        self.edges = bin_edges(training_values)

    def psi_against_training(self, values: list[float]) -> float:
        return population_stability_index(self.training_values, values, self.edges)

    def alerts(self, days: list[list[float]]) -> list[int]:
        """Day numbers whose PSI against training exceeds the threshold."""
        alerting_days = []
        for day_number, values in enumerate(days):
            if self.psi_against_training(values) > PSI_ALERT_THRESHOLD:
                alerting_days.append(day_number)
        return alerting_days


def slowly_drifting_days(day_count: int, start_mean: float, daily_growth: float,
                         spread: float) -> list[list[float]]:
    """Order values whose mean grows by ``daily_growth`` (e.g. 0.01 = 1%) per day."""
    days = []
    for day_number in range(day_count):
        mean = start_mean * (1 + daily_growth) ** day_number
        days.append(daily_values(mean, spread))
    return days
