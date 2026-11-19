"""Convert Excel serial date numbers to real dates without a two-day shift.

The regional sales teams upload order sheets saved from Excel. When a date
cell loses its formatting, the CSV contains Excel's serial number (46112)
instead of "2026-03-31". The loader turned serials into dates by adding that
many days to 1 January 1900, which puts every modern date two days late:
orders from the last two days of March landed in April and Q1 revenue was
under-reported.

Excel counts 1 January 1900 as day 1 (not day 0) and also counts a
29 February 1900 that never existed. The fix uses the correct epochs for
both of Excel's date systems, rejects the impossible serials 0 and 60, and
keeps the fraction of a serial as the time of day.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Callable

# Excel for Windows ("1900 date system"). For serials from 61 upward the
# true epoch is 30 December 1899: one day for "1 Jan is day 1" and one for
# the fake leap day.
EPOCH_1900_FROM_MARCH = datetime(1899, 12, 30)
# Serials 1..59 (1 Jan - 28 Feb 1900) come before the fake leap day.
EPOCH_1900_BEFORE_LEAP_BUG = datetime(1899, 12, 31)
FAKE_LEAP_DAY_SERIAL = 60
LAST_SERIAL_BEFORE_FAKE_LEAP_DAY = 59
# Old Excel for Mac ("1904 date system"): day 0 is 1 January 1904.
EPOCH_1904 = datetime(1904, 1, 1)

DATE_SYSTEM_1900 = "1900"
DATE_SYSTEM_1904 = "1904"


class InvalidSerialDate(ValueError):
    """A serial number that does not correspond to a real date."""


def naive_serial_to_date(serial: float) -> date:
    """The original loader: add the serial as days to 1 January 1900."""
    return (datetime(1900, 1, 1) + timedelta(days=serial)).date()


def epoch_for_1900_system(serial: float) -> datetime:
    """Pick the epoch for a 1900-system serial, rejecting 0 and the fake leap day."""
    whole_days = int(serial)
    if whole_days < 1:
        # Excel shows serial 0 as "1900-01-00": an empty cell, not a date.
        raise InvalidSerialDate(f"serial {serial} is before 1 January 1900")
    if whole_days == FAKE_LEAP_DAY_SERIAL:
        raise InvalidSerialDate("serial 60 is Excel's non-existent 29 February 1900")
    if whole_days <= LAST_SERIAL_BEFORE_FAKE_LEAP_DAY:
        return EPOCH_1900_BEFORE_LEAP_BUG
    return EPOCH_1900_FROM_MARCH


def serial_to_datetime(serial: float, date_system: str = DATE_SYSTEM_1900) -> datetime:
    """Convert an Excel serial (days, with the time of day as the fraction)."""
    if date_system == DATE_SYSTEM_1900:
        epoch = epoch_for_1900_system(serial)
    elif date_system == DATE_SYSTEM_1904:
        if serial < 0:
            raise InvalidSerialDate(f"serial {serial} is before 1 January 1904")
        epoch = EPOCH_1904
    else:
        raise ValueError(f"unknown Excel date system {date_system!r}")
    # Round to whole seconds: 0.75 days is stored as a binary float and
    # would otherwise come out a few microseconds before 18:00.
    seconds = round(serial * 24 * 60 * 60)
    return epoch + timedelta(seconds=seconds)


def serial_to_date(serial: float, date_system: str = DATE_SYSTEM_1900) -> date:
    """The calendar date part of an Excel serial."""
    return serial_to_datetime(serial, date_system).date()


def quarter_of(day: date) -> str:
    """Label like "2026-Q1"."""
    quarter = (day.month - 1) // 3 + 1
    return f"{day.year}-Q{quarter}"


def revenue_by_quarter(orders: list[tuple[float, int]],
                       to_date: Callable[[float], date]) -> dict[str, int]:
    """Sum (serial, amount) orders per quarter using the given converter."""
    totals: dict[str, int] = defaultdict(int)
    for serial, amount in orders:
        totals[quarter_of(to_date(serial))] += amount
    return dict(totals)
