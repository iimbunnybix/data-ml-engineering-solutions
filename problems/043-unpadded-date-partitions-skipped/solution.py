"""Pick up new daily partitions by their date, not by their name as a string.

An incremental loader remembered the last partition it processed (for example
"dt=2026-10-09") and on each run loaded every partition whose name sorted
after it. When the upstream exporter stopped zero-padding days, "dt=2026-10-10"
sorted BEFORE "dt=2026-10-9" as text, so October 10-31 were silently skipped.
This module parses every partition name into a real date, keeps the watermark
as a date, rejects names it cannot parse, refuses two spellings of the same
day, and reports gaps so a missing day is noticed instead of ignored.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta

# Accepts "dt=2026-10-09" and the unpadded "dt=2026-10-9"; anything else is
# rejected instead of guessed at.
PARTITION_PATTERN = re.compile(r"^dt=(\d{4})-(\d{1,2})-(\d{1,2})$")
ONE_DAY = timedelta(days=1)


class PartitionConflict(Exception):
    """Two partition folders claim the same date (e.g. padded and unpadded)."""


@dataclass
class LoadPlan:
    """What one run of the loader should do, and what looks wrong."""

    to_load: list[str] = field(default_factory=list)
    unparseable: list[str] = field(default_factory=list)
    missing_days: list[date] = field(default_factory=list)
    new_watermark: date | None = None


def naive_new_partitions(partition_names: list[str], last_loaded: str) -> list[str]:
    """Original logic: names that sort after the last loaded name, as text."""
    new_names = []
    for name in sorted(partition_names):
        if name > last_loaded:
            new_names.append(name)
    return new_names


def parse_partition_date(name: str) -> date | None:
    """Return the date a partition name stands for, or None if it is not one."""
    match = PARTITION_PATTERN.match(name)
    if match is None:
        return None
    year, month, day = (int(part) for part in match.groups())
    try:
        return date(year, month, day)
    except ValueError:
        # Matches the pattern but is not a real day, e.g. "dt=2026-02-30".
        return None


def partitions_by_date(partition_names: list[str]) -> tuple[dict[date, str], list[str]]:
    """Map each date to its partition name; also return names that do not parse."""
    by_date: dict[date, str] = {}
    unparseable: list[str] = []
    for name in partition_names:
        day = parse_partition_date(name)
        if day is None:
            unparseable.append(name)
            continue
        if day in by_date:
            # Loading both would double count the day; loading one would be a
            # guess. A person has to decide which folder is the real one.
            raise PartitionConflict(f"{by_date[day]!r} and {name!r} are both {day}")
        by_date[day] = name
    return by_date, sorted(unparseable)


def missing_days_between(start_exclusive: date, end_inclusive: date,
                         present: set[date]) -> list[date]:
    """Days after ``start_exclusive`` up to ``end_inclusive`` with no partition."""
    missing = []
    day = start_exclusive + ONE_DAY
    while day <= end_inclusive:
        if day not in present:
            missing.append(day)
        day += ONE_DAY
    return missing


def plan_incremental_load(partition_names: list[str], watermark: date) -> LoadPlan:
    """Choose partitions strictly after ``watermark``, in date order.

    The watermark advances only to the last day of the unbroken run of days
    after it, so a missing day is retried next run instead of being stepped
    over for good when it lands late.
    """
    by_date, unparseable = partitions_by_date(partition_names)
    plan = LoadPlan(unparseable=unparseable, new_watermark=watermark)
    new_days = sorted(day for day in by_date if day > watermark)
    if not new_days:
        return plan

    plan.missing_days = missing_days_between(watermark, new_days[-1], set(by_date))
    for day in new_days:
        if plan.missing_days and day > plan.missing_days[0]:
            # Stop at the first gap: later days wait until the gap is filled.
            break
        plan.to_load.append(by_date[day])
        plan.new_watermark = day
    return plan
