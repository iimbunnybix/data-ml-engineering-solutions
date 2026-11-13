import unittest

from solution import (
    WINDOW_DAYS,
    leaking_rows,
    mean_absolute_error,
    naive_centered_features,
    simulate_daily_orders,
    trailing_features,
)

SIMULATED_DAYS = 365
SEED = 11


def year_of_orders() -> list[float]:
    return simulate_daily_orders(SIMULATED_DAYS, SEED)


class ReproduceProblem(unittest.TestCase):
    def test_centered_window_includes_the_target_day_and_future_days(self):
        row = naive_centered_features(year_of_orders())[0]
        self.assertIn(row.target_day, row.days_used)
        self.assertEqual(max(row.days_used), row.target_day + 3)

    def test_every_centered_row_leaks(self):
        rows = naive_centered_features(year_of_orders())
        self.assertEqual(len(leaking_rows(rows)), len(rows))

    def test_offline_error_with_leak_is_much_lower_than_honest_error(self):
        orders = year_of_orders()
        leaky_error = mean_absolute_error(naive_centered_features(orders), orders)
        honest_error = mean_absolute_error(trailing_features(orders), orders)
        self.assertLess(leaky_error, honest_error * 0.9)


class FixTrailingWindow(unittest.TestCase):
    def test_feature_uses_only_the_seven_previous_days(self):
        row = trailing_features(year_of_orders())[0]
        self.assertEqual(row.days_used, tuple(range(0, WINDOW_DAYS)))
        self.assertEqual(row.target_day, WINDOW_DAYS)

    def test_no_row_reads_the_target_day_or_later(self):
        self.assertEqual(leaking_rows(trailing_features(year_of_orders())), [])

    def test_value_is_the_plain_average_of_previous_week(self):
        orders = [float(day) for day in range(10)]
        row = trailing_features(orders)[0]
        self.assertEqual(row.rolling_mean, 3.0)

    def test_series_shorter_than_window_gives_no_rows(self):
        self.assertEqual(trailing_features([100.0] * (WINDOW_DAYS - 1)), [])

    def test_leak_check_catches_a_single_bad_row(self):
        rows = trailing_features(year_of_orders())
        bad = naive_centered_features(year_of_orders())[10]
        self.assertEqual(leaking_rows(rows + [bad]), [bad.target_day])


if __name__ == "__main__":
    unittest.main()
