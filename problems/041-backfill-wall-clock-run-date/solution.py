"""Daily "7-day active users" job whose backfill wrote the same number to every day.

The job writes one partition per date. The original code took the end of the
7-day window from the wall clock ("yesterday") but wrote the result into the
partition for the date it was asked to build. Normal daily runs looked right,
but a 30-day backfill run in one morning filled all 30 partitions with
today's number, and a retried run on a later morning silently shifted the
window. This module derives the window only from the logical partition date
passed in, and uses the clock for one thing: refusing dates not yet complete.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Callable, Protocol

WINDOW_DAYS = 7  # "7-day active" = active on the partition date or the 6 days before it


class Clock(Protocol):
    """Anything that can say what today is (real clock or a test fake)."""

    def today(self) -> date: ...


class IncompleteDateError(Exception):
    """Raised when asked to build a partition whose day has not finished yet."""


@dataclass(frozen=True)
class Event:
    """One user activity event, already bucketed to its UTC calendar day."""

    user_id: str
    day: date


def window_for(end_day: date) -> tuple[date, date]:
    """Return the inclusive (first_day, last_day) of the 7-day window ending on end_day."""
    first_day = end_day - timedelta(days=WINDOW_DAYS - 1)
    return first_day, end_day


def count_active_users(events: list[Event], first_day: date, last_day: date) -> int:
    """Count distinct users with at least one event between the two days (inclusive)."""
    active_users = set()
    for event in events:
        if first_day <= event.day <= last_day:
            active_users.add(event.user_id)
    return len(active_users)


def naive_build_partition(events: list[Event], partition_date: date, clock: Clock) -> dict:
    """Original job: the window ends "yesterday" according to the wall clock.

    This matches partition_date only when the job runs on schedule, the
    morning after. A backfill or a late retry computes the wrong window but
    still labels the row with partition_date, so nothing looks wrong.
    """
    yesterday = clock.today() - timedelta(days=1)
    first_day, last_day = window_for(yesterday)
    return {"partition_date": partition_date,
            "active_users_7d": count_active_users(events, first_day, last_day)}


def build_partition(events: list[Event], partition_date: date, clock: Clock) -> dict:
    """Fixed job: the window comes only from partition_date, the logical date.

    The output is then a pure function of (input data, partition_date), so a
    backfill, a retry next week and the original scheduled run all agree.
    """
    # The clock is still needed, but only as a guard: a day that has not
    # ended yet has missing events, and publishing it would under-count.
    if partition_date >= clock.today():
        raise IncompleteDateError(
            f"{partition_date} is not complete yet (today is {clock.today()})")
    first_day, last_day = window_for(partition_date)
    return {"partition_date": partition_date,
            "active_users_7d": count_active_users(events, first_day, last_day)}


# Signature shared by the naive and fixed builders, so tests can backfill with either.
PartitionBuilder = Callable[[list[Event], date, Clock], dict]


def backfill(events: list[Event], first_date: date, last_date: date,
             clock: Clock, build: PartitionBuilder = build_partition) -> list[dict]:
    """Build one partition per date from first_date to last_date (inclusive), in order."""
    rows = []
    current = first_date
    while current <= last_date:
        rows.append(build(events, current, clock))
        current += timedelta(days=1)
    return rows
