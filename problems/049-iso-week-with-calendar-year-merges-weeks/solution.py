"""Weekly rollups keyed by ISO year + ISO week, never calendar year + ISO week.

The weekly orders job labelled each day with `strftime("%Y-W%V")`: calendar
year, ISO week number. Around New Year those disagree. 29-31 Dec 2025 belong
to ISO week 1 of *2026*, so they were labelled "2025-W01" and added to the
first week of January 2025, while the real first week of 2026 lost three days.
The fix takes the year and week from the same calendar (`isocalendar()`, or
`%G-W%V`) and adds a guard that no week can contain more than seven days.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date
from typing import Callable

DAYS_PER_WEEK = 7
MONDAY = 1

WeekKeyFunction = Callable[[date], str]


def naive_week_key(day: date) -> str:
    """Original label: calendar year (%Y) mixed with ISO week number (%V)."""
    return day.strftime("%Y-W%V")


def iso_week_key(day: date) -> str:
    """Correct label: ISO year and ISO week always come from the same calendar."""
    iso = day.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def week_start(week_key: str) -> date:
    """Monday of an ISO week key such as '2026-W01' (useful for charts and joins)."""
    year_text, week_text = week_key.split("-W")
    return date.fromisocalendar(int(year_text), int(week_text), MONDAY)


def weekly_totals(daily_orders: dict[date, int], key_function: WeekKeyFunction) -> dict[str, int]:
    """Sum daily order counts into weeks using the given labelling function."""
    totals: dict[str, int] = defaultdict(int)
    for day, orders in daily_orders.items():
        totals[key_function(day)] += orders
    return dict(totals)


def days_per_week(days: list[date], key_function: WeekKeyFunction) -> dict[str, int]:
    """How many distinct days landed in each week label."""
    days_by_week: dict[str, set[date]] = defaultdict(set)
    for day in days:
        days_by_week[key_function(day)].add(day)
    counts = {}
    for week, days_in_week in days_by_week.items():
        counts[week] = len(days_in_week)
    return counts


def overfull_weeks(days: list[date], key_function: WeekKeyFunction) -> list[str]:
    """Data-quality guard: week labels that collected more than seven distinct days.

    A real week can never have more; if one does, two different weeks were
    given the same label.
    """
    problems = []
    for week, count in sorted(days_per_week(days, key_function).items()):
        if count > DAYS_PER_WEEK:
            problems.append(week)
    return problems
