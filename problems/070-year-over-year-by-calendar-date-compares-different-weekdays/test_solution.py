import unittest
from datetime import date, timedelta

from solution import (comparable_last_year_date, days_to_alert, naive_last_year_date,
                      year_over_year_change)

# Sales per weekday (Monday first): weekends sell about twice as much.
WEEKDAY_SALES = [100.0, 95.0, 100.0, 110.0, 150.0, 220.0, 200.0]
BLACK_FRIDAY_MULTIPLIER = 4.0
BLACK_FRIDAY_2025 = date(2025, 11, 28)
BLACK_FRIDAY_2026 = date(2026, 11, 27)


def all_days(first: date, last: date) -> list[date]:
    days, day = [], first
    while day <= last:
        days.append(day)
        day += timedelta(days=1)
    return days


def flat_sales(first: date, last: date) -> dict[date, float]:
    """A business with exactly zero growth: same weekly pattern and Black Friday every year."""
    sales = {}
    for day in all_days(first, last):
        amount = WEEKDAY_SALES[day.weekday()]
        if day in (BLACK_FRIDAY_2025, BLACK_FRIDAY_2026):
            amount *= BLACK_FRIDAY_MULTIPLIER
        sales[day] = amount
    return sales


SALES = flat_sales(date(2024, 12, 1), date(2026, 12, 31))
DAYS_2026 = all_days(date(2026, 1, 1), date(2026, 12, 31))


class ReproduceProblem(unittest.TestCase):
    def test_flat_business_pages_on_many_days(self):
        self.assertEqual(len(days_to_alert(SALES, DAYS_2026, use_fix=False)), 53)

    def test_saturday_is_compared_with_a_friday(self):
        saturday = date(2026, 11, 28)
        self.assertEqual(naive_last_year_date(saturday).strftime("%A"), "Friday")

    def test_day_after_black_friday_looks_like_a_crash(self):
        change = year_over_year_change(SALES, date(2026, 11, 28), naive_last_year_date(date(2026, 11, 28)))
        self.assertAlmostEqual(change, -0.6333, places=4)

    def test_leap_day_crashes_the_naive_job(self):
        with self.assertRaises(ValueError):
            naive_last_year_date(date(2028, 2, 29))


class FixSameWeekdayComparison(unittest.TestCase):
    def test_flat_business_never_pages(self):
        self.assertEqual(days_to_alert(SALES, DAYS_2026, use_fix=True), [])

    def test_comparison_day_has_the_same_weekday(self):
        for day in DAYS_2026:
            self.assertEqual(comparable_last_year_date(day).weekday(), day.weekday())

    def test_black_friday_is_compared_with_last_black_friday(self):
        self.assertEqual(comparable_last_year_date(BLACK_FRIDAY_2026), BLACK_FRIDAY_2025)
        self.assertEqual(year_over_year_change(SALES, BLACK_FRIDAY_2026, BLACK_FRIDAY_2025), 0.0)

    def test_leap_day_gets_a_valid_comparison_date(self):
        self.assertEqual(comparable_last_year_date(date(2028, 2, 29)), date(2027, 3, 2))

    def test_real_drop_still_pages(self):
        sales = dict(SALES)
        sales[date(2026, 6, 10)] = sales[date(2026, 6, 10)] * 0.5
        self.assertEqual(days_to_alert(sales, DAYS_2026, use_fix=True), [date(2026, 6, 10)])

    def test_missing_baseline_gives_no_change(self):
        self.assertIsNone(year_over_year_change(SALES, date(2026, 1, 1), date(2020, 1, 1)))

    def test_zero_baseline_gives_no_change(self):
        sales = {date(2025, 1, 1): 0.0, date(2025, 12, 31): 50.0}
        self.assertIsNone(year_over_year_change(sales, date(2025, 12, 31), date(2025, 1, 1)))


if __name__ == "__main__":
    unittest.main()
