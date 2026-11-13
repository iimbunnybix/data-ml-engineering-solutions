import unittest
from datetime import date, timedelta

from solution import (
    iso_week_key,
    naive_week_key,
    overfull_weeks,
    days_per_week,
    week_start,
    weekly_totals,
)

ORDERS_PER_DAY = 100
# 30 Dec 2024 (Monday of ISO 2025-W01) to 4 Jan 2026 (Sunday of ISO 2026-W01):
# exactly 53 complete ISO weeks.
FIRST_DAY = date(2024, 12, 30)
LAST_DAY = date(2026, 1, 4)


def all_days() -> list[date]:
    days = []
    current = FIRST_DAY
    while current <= LAST_DAY:
        days.append(current)
        current += timedelta(days=1)
    return days


def flat_daily_orders() -> dict[date, int]:
    """Same order count every day, so every complete week should total 700."""
    return {day: ORDERS_PER_DAY for day in all_days()}


class ReproduceProblem(unittest.TestCase):
    def test_late_december_is_labelled_as_the_previous_years_first_week(self):
        self.assertEqual(naive_week_key(date(2025, 12, 30)), "2025-W01")

    def test_first_week_of_2025_absorbs_three_days_from_december_2025(self):
        totals = weekly_totals(flat_daily_orders(), naive_week_key)
        self.assertEqual(totals["2025-W01"], 8 * ORDERS_PER_DAY)

    def test_first_week_of_2026_loses_three_days(self):
        totals = weekly_totals(flat_daily_orders(), naive_week_key)
        self.assertEqual(totals["2026-W01"], 4 * ORDERS_PER_DAY)

    def test_naive_labels_produce_a_week_with_eight_days(self):
        self.assertEqual(overfull_weeks(all_days(), naive_week_key), ["2025-W01"])

    def test_new_years_day_2027_gets_a_week_53_that_does_not_exist_in_2027(self):
        self.assertEqual(naive_week_key(date(2027, 1, 1)), "2027-W53")


class FixIsoWeekKey(unittest.TestCase):
    def test_late_december_belongs_to_next_iso_year(self):
        self.assertEqual(iso_week_key(date(2025, 12, 30)), "2026-W01")

    def test_new_years_day_2027_is_week_53_of_2026(self):
        self.assertEqual(iso_week_key(date(2027, 1, 1)), "2026-W53")

    def test_every_week_has_exactly_seven_days(self):
        counts = days_per_week(all_days(), iso_week_key)
        self.assertEqual(len(counts), 53)
        self.assertEqual(set(counts.values()), {7})

    def test_every_week_totals_seven_days_of_orders(self):
        totals = weekly_totals(flat_daily_orders(), iso_week_key)
        self.assertEqual(set(totals.values()), {7 * ORDERS_PER_DAY})

    def test_matches_strftime_iso_year_directive(self):
        for day in all_days():
            self.assertEqual(iso_week_key(day), day.strftime("%G-W%V"))

    def test_guard_finds_nothing_with_iso_keys(self):
        self.assertEqual(overfull_weeks(all_days(), iso_week_key), [])

    def test_week_start_is_the_monday(self):
        self.assertEqual(week_start("2026-W01"), date(2025, 12, 29))
        self.assertEqual(week_start(iso_week_key(date(2027, 1, 3))), date(2026, 12, 28))


if __name__ == "__main__":
    unittest.main()
