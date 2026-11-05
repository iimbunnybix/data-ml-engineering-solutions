"""Daily and hourly rollups in a business time zone that survive DST changes.

Events are stored with UTC timestamps; the business reports by *local* day in
America/New_York. The original job converted with a fixed offset
(``ts - 5h``), which is only right in winter, and grouped hourly charts by the
local wall-clock hour, which merges the two 01:00 hours on the autumn
"fall back" day.

The fix converts with the IANA time-zone database (``zoneinfo``), cuts
hourly buckets on local hours, keys them by their *UTC* start (unambiguous,
even for half-hour-offset zones), gives each a local label
that includes the offset, and computes which UTC range a local day covers so
UTC-partitioned tables can be queried correctly.
"""
from __future__ import annotations

from collections import Counter
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

UTC = timezone.utc
FIXED_EST = timedelta(hours=-5)


def _require_aware(ts: datetime) -> None:
    if ts.tzinfo is None or ts.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware (store UTC, never naive local time)")


# ------------------------------------------------------------- the old way
def naive_local_day(ts_utc: datetime) -> date:
    """Original bug: a fixed UTC-5 offset, wrong for the ~8 months of daylight time."""
    _require_aware(ts_utc)
    return (ts_utc.astimezone(UTC) + FIXED_EST).date()


def naive_hourly(events: list[datetime], tz: ZoneInfo) -> Counter:
    """Original hourly chart: key by local wall-clock 'YYYY-MM-DD HH'."""
    return Counter(e.astimezone(tz).strftime("%Y-%m-%d %H") for e in events)


# ------------------------------------------------------------------ the fix
def local_day(ts: datetime, tz: ZoneInfo) -> date:
    """Business day of an aware timestamp, using the real offset at that instant."""
    _require_aware(ts)
    return ts.astimezone(tz).date()


def daily_counts(events: list[datetime], tz: ZoneInfo) -> Counter:
    return Counter(local_day(e, tz) for e in events)


def local_day_bounds_utc(day: date, tz: ZoneInfo) -> tuple[datetime, datetime]:
    """UTC [start, end) of a local calendar day. 23, 24 or 25 hours long.

    Local midnight always exists in America/New_York (transitions happen at
    02:00). For zones that switch at midnight, ``datetime(..., tzinfo=tz)``
    resolves a non-existent midnight forward, which is still the first instant
    of that day.
    """
    start = datetime(day.year, day.month, day.day, tzinfo=tz).astimezone(UTC)
    nxt = day + timedelta(days=1)
    end = datetime(nxt.year, nxt.month, nxt.day, tzinfo=tz).astimezone(UTC)
    return start, end


def utc_partitions_for_local_day(day: date, tz: ZoneInfo) -> list[date]:
    """UTC date partitions that must be scanned to cover one local day."""
    start, end = local_day_bounds_utc(day, tz)
    last = (end - timedelta(microseconds=1)).date()
    out, d = [], start.date()
    while d <= last:
        out.append(d)
        d += timedelta(days=1)
    return out


def hourly_counts(events: list[datetime], tz: ZoneInfo) -> dict[datetime, int]:
    """Counts keyed by the UTC start of each *local* hour.

    The hour is cut in local time, so zones with a half- or quarter-hour
    offset (India UTC+5:30, Nepal UTC+5:45) get 00:00-01:00 local bars instead
    of 23:30-00:30 bars that straddle two business days. Converting the bucket
    start back to UTC keeps the key unambiguous: ``astimezone`` sets ``fold``,
    so the two 01:00 hours on a "fall back" day stay separate.
    """
    c: Counter = Counter()
    for e in events:
        _require_aware(e)
        local_hour_start = e.astimezone(tz).replace(minute=0, second=0, microsecond=0)
        c[local_hour_start.astimezone(UTC)] += 1
    return dict(sorted(c.items()))


def hour_label(bucket_utc: datetime, tz: ZoneInfo) -> str:
    """Human label for an hourly bucket, e.g. '2026-11-01 01:00 EDT' vs '... 01:00 EST'."""
    return bucket_utc.astimezone(tz).strftime("%Y-%m-%d %H:%M %Z")


def local_hours_in_day(day: date, tz: ZoneInfo) -> list[datetime]:
    """All hourly bucket starts (UTC) that belong to a local day."""
    start, end = local_day_bounds_utc(day, tz)
    out, t = [], start
    while t < end:
        out.append(t)
        t += timedelta(hours=1)
    return out
