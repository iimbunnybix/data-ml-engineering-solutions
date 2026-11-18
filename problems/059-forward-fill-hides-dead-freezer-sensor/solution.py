"""Resample sensor readings without letting forward-fill hide a dead sensor.

A cold-storage company resamples every freezer's temperature readings onto a
5-minute grid and fills gaps with the last known value, so the anomaly rules
always get a complete series. When a sensor's battery died, forward-fill kept
repeating its last healthy reading for hours, and the "freezer too warm"
alarm never fired while the freezer itself was failing. This module only
carries a reading forward for a bounded time, marks older points as missing
together with their age, and raises a "sensor stale" alert for them.
"""
from __future__ import annotations

from dataclasses import dataclass

GRID_MINUTES = 5
# One or two lost radio packets are normal; three missed readings is not.
MAX_STALENESS_MINUTES = 15
# Food safety rule used by the warehouse: frozen stock must stay at or below -12 C.
FREEZER_ALARM_CELSIUS = -12.0


@dataclass(frozen=True)
class Reading:
    minute: int          # minutes since midnight, from the sensor gateway
    celsius: float


@dataclass(frozen=True)
class GridPoint:
    minute: int
    celsius: float | None        # None means "no trustworthy value"
    age_minutes: int | None      # how old the value is; None if no reading yet


@dataclass(frozen=True)
class Alert:
    minute: int
    kind: str                    # "too_warm" or "sensor_stale"


def grid_minutes(start: int, end: int) -> list[int]:
    """Grid timestamps from start to end inclusive, GRID_MINUTES apart."""
    minutes = []
    current = start
    while current <= end:
        minutes.append(current)
        current += GRID_MINUTES
    return minutes


def latest_reading_at(readings: list[Reading], minute: int) -> Reading | None:
    """Most recent reading taken at or before `minute` (readings sorted by time)."""
    latest = None
    for reading in readings:
        if reading.minute > minute:
            break
        latest = reading
    return latest


def naive_resample(readings: list[Reading], start: int, end: int) -> list[float | None]:
    """Original pipeline: forward-fill the last value with no limit at all."""
    ordered = sorted(readings, key=lambda reading: reading.minute)
    values = []
    for minute in grid_minutes(start, end):
        latest = latest_reading_at(ordered, minute)
        values.append(None if latest is None else latest.celsius)
    return values


def resample(readings: list[Reading], start: int, end: int,
             max_staleness: int = MAX_STALENESS_MINUTES) -> list[GridPoint]:
    """Forward-fill only while the last reading is at most `max_staleness` old."""
    # Gateways batch uploads, so readings can arrive out of order.
    ordered = sorted(readings, key=lambda reading: reading.minute)
    points = []
    for minute in grid_minutes(start, end):
        latest = latest_reading_at(ordered, minute)
        if latest is None:
            points.append(GridPoint(minute, None, None))
            continue
        age = minute - latest.minute
        # Keep the age even when the value is dropped: downstream models and
        # dashboards can use "how long since we heard from it" as a feature.
        value = latest.celsius if age <= max_staleness else None
        points.append(GridPoint(minute, value, age))
    return points


def naive_alerts(values: list[float | None], start: int) -> list[Alert]:
    """Original alarm rule: only looks at temperatures, never at freshness."""
    alerts = []
    for index, value in enumerate(values):
        if value is not None and value > FREEZER_ALARM_CELSIUS:
            alerts.append(Alert(start + index * GRID_MINUTES, "too_warm"))
    return alerts


def alerts_for(points: list[GridPoint]) -> list[Alert]:
    """Alert on warm freezers and, separately, on sensors we cannot trust.

    A stale alert is raised once when a sensor goes quiet (not on every grid
    point), and again only if it goes quiet a second time after recovering.
    """
    alerts = []
    was_stale = False
    for point in points:
        if point.celsius is None:
            if not was_stale:
                alerts.append(Alert(point.minute, "sensor_stale"))
            was_stale = True
            continue
        was_stale = False
        if point.celsius > FREEZER_ALARM_CELSIUS:
            alerts.append(Alert(point.minute, "too_warm"))
    return alerts
