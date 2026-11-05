import random
import unittest
from datetime import date, timedelta

from solution import (DailySpend, naive_rolling_spend, rolling_spend, spend_as_of)

START = date(2026, 1, 1)
DAYS = 180


def day(offset: int) -> date:
    return START + timedelta(days=offset)


def make_table(seed: int = 11) -> list[DailySpend]:
    """Daily spend for 100 daily users and 100 occasional users (about once a month)."""
    rng = random.Random(seed)
    rows: list[DailySpend] = []
    for number in range(100):
        for offset in range(DAYS):
            rows.append(DailySpend(f"daily-{number}", day(offset), rng.randint(1_000, 3_000)))
    for number in range(100):
        for offset in range(DAYS):
            if rng.random() < 1 / 30:
                rows.append(DailySpend(f"occasional-{number}", day(offset), rng.randint(1_000, 3_000)))
    return rows


def by_key(features) -> dict:
    return {(feature.user_id, feature.day): feature.spend_7d_cents for feature in features}


def reference_features(rows) -> dict:
    """Expected feature for every row, from the simple reference definition.

    Rows are grouped per user first so the reference stays fast enough for tests.
    """
    rows_by_user: dict = {}
    for row in rows:
        rows_by_user.setdefault(row.user_id, []).append(row)
    expected = {}
    for row in rows:
        expected[(row.user_id, row.day)] = spend_as_of(rows_by_user[row.user_id], row.user_id, row.day)
    return expected


def mean_overstatement(rows, features, prefix: str) -> float:
    """Average ratio of batch feature to the reference definition for a user group."""
    expected = reference_features(rows)
    ratios = []
    for key, value in by_key(features).items():
        if key[0].startswith(prefix):
            ratios.append(value / expected[key])
    return sum(ratios) / len(ratios)


class ReproduceProblem(unittest.TestCase):
    def test_row_window_is_correct_for_users_active_every_day(self):
        rows = make_table()
        features = naive_rolling_spend(rows)
        self.assertAlmostEqual(mean_overstatement(rows, features, "daily-"), 1.0, places=6)

    def test_row_window_inflates_spend_of_occasional_users(self):
        rows = make_table()
        features = naive_rolling_spend(rows)
        # Measured: 3.7x the true 7-day spend on average.
        self.assertGreater(mean_overstatement(rows, features, "occasional-"), 3.0)

    def test_purchases_months_apart_are_summed_as_one_week(self):
        rows = [DailySpend("u", day(0), 5_000), DailySpend("u", day(60), 5_000)]
        self.assertEqual(by_key(naive_rolling_spend(rows))[("u", day(60))], 10_000)


class FixCalendarWindow(unittest.TestCase):
    def test_batch_feature_matches_reference_for_every_row(self):
        rows = make_table()
        self.assertEqual(by_key(rolling_spend(rows)), reference_features(rows))

    def test_purchase_seven_days_back_is_outside_the_window(self):
        rows = [DailySpend("u", day(0), 5_000), DailySpend("u", day(7), 1_000)]
        self.assertEqual(by_key(rolling_spend(rows))[("u", day(7))], 1_000)

    def test_purchase_six_days_back_is_inside_the_window(self):
        rows = [DailySpend("u", day(0), 5_000), DailySpend("u", day(6), 1_000)]
        self.assertEqual(by_key(rolling_spend(rows))[("u", day(6))], 6_000)

    def test_unsorted_input_gives_the_same_result(self):
        rows = make_table()[:2_000]
        shuffled = list(rows)
        random.Random(3).shuffle(shuffled)
        self.assertEqual(by_key(rolling_spend(shuffled)), by_key(rolling_spend(rows)))

    def test_users_do_not_share_a_window(self):
        rows = [DailySpend("a", day(0), 100), DailySpend("b", day(1), 200)]
        self.assertEqual(by_key(rolling_spend(rows))[("b", day(1))], 200)

    def test_duplicate_day_for_a_user_is_rejected(self):
        rows = [DailySpend("u", day(0), 100), DailySpend("u", day(0), 100)]
        with self.assertRaises(ValueError):
            rolling_spend(rows)

    def test_empty_table_gives_no_features(self):
        self.assertEqual(rolling_spend([]), [])


if __name__ == "__main__":
    unittest.main()
