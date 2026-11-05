"""Compute "spend in the last 7 days" over 7 calendar days, not the last 7 rows.

The feature job ran ``SUM(spend) OVER (PARTITION BY user ORDER BY day ROWS
BETWEEN 6 PRECEDING AND CURRENT ROW)`` on a daily table that only has a row
for days on which the user spent money. For an active user 7 rows is 7 days,
but for an occasional user 7 rows can reach back months, so their "7-day
spend" was hugely inflated and the risk model flagged them.

The fix uses a time-range window: a row belongs to the window only if its
date is within the last 7 calendar days, however many rows that is. This is
the ``RANGE BETWEEN INTERVAL 6 DAYS PRECEDING AND CURRENT ROW`` form in SQL.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import date, timedelta

WINDOW_DAYS = 7


@dataclass(frozen=True)
class DailySpend:
    user_id: str
    day: date
    spend_cents: int


@dataclass(frozen=True)
class FeatureRow:
    user_id: str
    day: date
    spend_7d_cents: int


def group_by_user(rows: list[DailySpend]) -> dict[str, list[DailySpend]]:
    """Rows per user, sorted by day (the window's ORDER BY)."""
    by_user: dict[str, list[DailySpend]] = {}
    for row in rows:
        by_user.setdefault(row.user_id, []).append(row)
    for user_rows in by_user.values():
        user_rows.sort(key=lambda row: row.day)
    return by_user


def check_one_row_per_day(user_rows: list[DailySpend]) -> None:
    """The table is daily: two rows on one day means an upstream duplicate."""
    for previous, current in zip(user_rows, user_rows[1:]):
        if previous.day == current.day:
            raise ValueError(f"duplicate row for {current.user_id} on {current.day}")


def naive_rolling_spend(rows: list[DailySpend]) -> list[FeatureRow]:
    """Original job: sum of the current row and the 6 rows before it."""
    features: list[FeatureRow] = []
    for user_rows in group_by_user(rows).values():
        for position, row in enumerate(user_rows):
            first = max(0, position - (WINDOW_DAYS - 1))
            window = user_rows[first:position + 1]
            total = sum(item.spend_cents for item in window)
            features.append(FeatureRow(row.user_id, row.day, total))
    return features


def rolling_spend_for_user(user_rows: list[DailySpend]) -> list[FeatureRow]:
    """Calendar-day window for one user's sorted rows, in a single pass.

    The deque holds the rows still inside the window; its running total is
    updated as rows enter and leave, so the cost is O(rows) per user.
    """
    window: deque[DailySpend] = deque()
    running_total = 0
    features: list[FeatureRow] = []
    for row in user_rows:
        window.append(row)
        running_total += row.spend_cents
        # Days strictly before this date are outside "the last 7 days".
        oldest_allowed = row.day - timedelta(days=WINDOW_DAYS - 1)
        while window[0].day < oldest_allowed:
            running_total -= window.popleft().spend_cents
        features.append(FeatureRow(row.user_id, row.day, running_total))
    return features


def rolling_spend(rows: list[DailySpend]) -> list[FeatureRow]:
    """Fixed job: 7-day spend for every (user, day) row in the table."""
    features: list[FeatureRow] = []
    for user_rows in group_by_user(rows).values():
        check_one_row_per_day(user_rows)
        features.extend(rolling_spend_for_user(user_rows))
    return features


def spend_as_of(rows: list[DailySpend], user_id: str, as_of: date) -> int:
    """Reference definition used by the online service: spend in [as_of-6, as_of].

    Kept deliberately simple so tests can compare the batch job against it,
    including on days when the user had no row at all.
    """
    oldest_allowed = as_of - timedelta(days=WINDOW_DAYS - 1)
    total = 0
    for row in rows:
        if row.user_id == user_id and oldest_allowed <= row.day <= as_of:
            total += row.spend_cents
    return total
