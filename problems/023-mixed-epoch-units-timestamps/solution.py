"""Detect the unit of epoch timestamps before partitioning events by day.

Two upstream sources were merged into one events table. The mobile SDK sends
epoch seconds, the web tracker sends epoch milliseconds and a new partner feed
sends microseconds. The pipeline assumed milliseconds for everything, so
mobile events landed in January 1970 and partner events crashed the job
(their dates are tens of thousands of years in the future). This module tries
every common unit, keeps the one that gives a plausible date, and quarantines
rows where no unit (or more than one unit) is plausible instead of guessing.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone

# How many of each unit make one second; ordered from coarsest to finest.
UNITS_PER_SECOND: dict[str, int] = {
    "seconds": 1,
    "milliseconds": 1_000,
    "microseconds": 1_000_000,
    "nanoseconds": 1_000_000_000,
}
# No event in this product can be older than its launch...
EARLIEST_PLAUSIBLE = datetime(2015, 1, 1, tzinfo=timezone.utc)
# ...or more than a day ahead of the ingestion clock (allows for device clock skew).
MAX_FUTURE_SECONDS = 86_400


@dataclass(frozen=True)
class ParsedEvent:
    event_id: str
    timestamp: datetime
    unit: str


@dataclass(frozen=True)
class QuarantinedEvent:
    event_id: str
    raw_value: float
    reason: str


@dataclass
class ParseResult:
    events: list[ParsedEvent] = field(default_factory=list)
    quarantined: list[QuarantinedEvent] = field(default_factory=list)


def naive_parse_millis(raw_value: float) -> datetime:
    """Original code: every timestamp is assumed to be epoch milliseconds."""
    return datetime.fromtimestamp(raw_value / 1_000, tz=timezone.utc)


def plausible_units(raw_value: float, now: datetime,
                    earliest: datetime = EARLIEST_PLAUSIBLE) -> list[str]:
    """Return every unit under which raw_value is a date inside the plausible window."""
    earliest_seconds = earliest.timestamp()
    latest_seconds = now.timestamp() + MAX_FUTURE_SECONDS
    matches = []
    for unit, per_second in UNITS_PER_SECOND.items():
        # Compare in seconds with plain arithmetic: building a datetime for
        # a value in the year 50000 would itself raise an error.
        as_seconds = raw_value / per_second
        if earliest_seconds <= as_seconds <= latest_seconds:
            matches.append(unit)
    return matches


def parse_event_time(event_id: str, raw_value: float, now: datetime,
                     earliest: datetime = EARLIEST_PLAUSIBLE
                     ) -> ParsedEvent | QuarantinedEvent:
    """Parse one timestamp, or quarantine it when its unit cannot be decided."""
    units = plausible_units(raw_value, now, earliest)
    if not units:
        return QuarantinedEvent(event_id, raw_value, "no unit gives a plausible date")
    if len(units) > 1:
        # Only possible with a very wide window; guessing would silently
        # shift events by a factor of 1000, so a human has to decide.
        return QuarantinedEvent(event_id, raw_value, "ambiguous unit: " + ", ".join(units))
    unit = units[0]
    seconds = raw_value / UNITS_PER_SECOND[unit]
    return ParsedEvent(event_id, datetime.fromtimestamp(seconds, tz=timezone.utc), unit)


def parse_events(raw_events: list[tuple[str, float]], now: datetime,
                 earliest: datetime = EARLIEST_PLAUSIBLE) -> ParseResult:
    """Split raw (event_id, epoch value) pairs into parsed and quarantined events."""
    result = ParseResult()
    for event_id, raw_value in raw_events:
        parsed = parse_event_time(event_id, raw_value, now, earliest)
        if isinstance(parsed, ParsedEvent):
            result.events.append(parsed)
        else:
            result.quarantined.append(parsed)
    return result


def daily_counts(timestamps: list[datetime]) -> dict[str, int]:
    """Count events per UTC day, the partition key of the events table."""
    counts: Counter[str] = Counter()
    for timestamp in timestamps:
        counts[timestamp.date().isoformat()] += 1
    return dict(sorted(counts.items()))


def unit_mix(events: list[ParsedEvent]) -> dict[str, int]:
    """How many events used each unit; a sudden change per source is worth an alert."""
    return dict(Counter(event.unit for event in events))
