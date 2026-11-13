"""Rolling-average features that only look backwards from the prediction time.

A demand-forecasting model predicted tomorrow's orders per store. Its main
feature was a 7-day rolling mean of orders computed with a *centered* window
(3 days before, the day itself, 3 days after), copied from an analysis
notebook where centering made charts smoother. That window contains the very
day being predicted plus three future days, so offline error looked tiny and
production, where the future is unknown, was far worse. The fix builds the
window from days strictly before the target day and adds a check that every
feature value only used data available at prediction time.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

WINDOW_DAYS = 7
CENTERED_HALF_WIDTH = WINDOW_DAYS // 2


@dataclass(frozen=True)
class FeatureRow:
    """One training example: feature for `target_day`, plus which days it read."""
    target_day: int
    rolling_mean: float
    days_used: tuple[int, ...]


def _mean_of_days(orders: list[float], days: list[int]) -> float:
    """Average of the given day indexes; plain loop so the inputs are obvious."""
    total = 0.0
    for day in days:
        total += orders[day]
    return total / len(days)


def naive_centered_features(orders: list[float]) -> list[FeatureRow]:
    """Original feature: centered window, i.e. it includes the target and future days."""
    rows = []
    for target_day in range(CENTERED_HALF_WIDTH, len(orders) - CENTERED_HALF_WIDTH):
        days = list(range(target_day - CENTERED_HALF_WIDTH, target_day + CENTERED_HALF_WIDTH + 1))
        rows.append(FeatureRow(target_day, _mean_of_days(orders, days), tuple(days)))
    return rows


def trailing_features(orders: list[float]) -> list[FeatureRow]:
    """Fixed feature: the 7 days strictly before the target day.

    This is exactly what production can compute the evening before the
    forecast, so offline and online values match.
    """
    rows = []
    for target_day in range(WINDOW_DAYS, len(orders)):
        days = list(range(target_day - WINDOW_DAYS, target_day))
        rows.append(FeatureRow(target_day, _mean_of_days(orders, days), tuple(days)))
    return rows


def leaking_rows(rows: list[FeatureRow]) -> list[int]:
    """Target days whose feature read the target day itself or a later day.

    Run this in the training pipeline: any result means the model is being
    shown information it will not have when it predicts.
    """
    leaks = []
    for row in rows:
        if max(row.days_used) >= row.target_day:
            leaks.append(row.target_day)
    return leaks


def mean_absolute_error(rows: list[FeatureRow], orders: list[float]) -> float:
    """Error of the simplest model, "prediction = feature", over the given rows."""
    total = 0.0
    for row in rows:
        total += abs(orders[row.target_day] - row.rolling_mean)
    return total / len(rows)


def simulate_daily_orders(days: int, seed: int) -> list[float]:
    """Orders with a slow random-walk trend plus day-to-day noise."""
    rng = random.Random(seed)
    level = 200.0
    orders = []
    for _ in range(days):
        level += rng.gauss(0, 4)
        orders.append(max(level + rng.gauss(0, 20), 0.0))
    return orders
