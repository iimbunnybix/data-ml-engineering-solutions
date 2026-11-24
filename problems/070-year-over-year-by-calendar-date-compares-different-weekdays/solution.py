"""Compare each day with the same weekday a year earlier, not the same calendar date.

A retail sales dashboard showed "vs last year" per day and paged the
merchandising team when a day dropped more than 20%. It compared each date
with the same calendar date one year earlier. A year is 52 weeks plus one or
two days, so a Saturday was compared with a Friday, weekends looked like
booms and weekdays like crashes, Black Friday was compared with an ordinary
Saturday, and 29 February crashed the job.

Fix: compare with the day exactly 364 days (52 weeks) earlier, which is
always the same weekday and usually the same retail event such as Black Friday.
"""
from __future__ import annotations

from datetime import date, timedelta

DAYS_IN_52_WEEKS = 364
DROP_ALERT_THRESHOLD = -0.20  # page when a day is more than 20% below last year


def naive_last_year_date(day: date) -> date:
    """Original: same calendar date one year earlier (crashes on 29 Feb)."""
    return day.replace(year=day.year - 1)


def comparable_last_year_date(day: date) -> date:
    """Same weekday 52 weeks earlier: always a valid date with the same weekly pattern."""
    return day - timedelta(days=DAYS_IN_52_WEEKS)


def year_over_year_change(sales: dict[date, float], day: date, last_year_day: date) -> float | None:
    """Relative change versus the comparison day, or None if it cannot be computed."""
    previous = sales.get(last_year_day)
    current = sales.get(day)
    # WHY None: a missing or zero baseline has no meaningful percentage change.
    if previous is None or current is None or previous == 0:
        return None
    return (current - previous) / previous


def days_to_alert(sales: dict[date, float], days: list[date], use_fix: bool) -> list[date]:
    """Days whose year-over-year change is below the alert threshold."""
    alerts = []
    for day in days:
        if use_fix:
            last_year_day = comparable_last_year_date(day)
        else:
            last_year_day = naive_last_year_date(day)
        change = year_over_year_change(sales, day, last_year_day)
        if change is not None and change < DROP_ALERT_THRESHOLD:
            alerts.append(day)
    return alerts
