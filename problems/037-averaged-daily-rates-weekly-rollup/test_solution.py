import random
import unittest

from solution import (DailyFunnel, RatioParts, naive_daily_rates, naive_rollup,
                      rollup, to_parts, weekly_conversion)

DAYS = [f"2026-10-{number:02d}" for number in range(5, 12)]
TRUE_RATE = 0.031
DAILY_TRAFFIC = {"IN": 30_000, "US": 20_000, "DE": 8_000}
# A market launched this week: a handful of visits per day, one lucky conversion.
NEW_MARKET_DAYS = [(3, 0), (2, 0), (3, 0), (1, 0), (2, 0), (6, 0), (2, 1)]


def make_week(seed: int = 5) -> list[DailyFunnel]:
    """Three big markets converting at about 3.1% plus one tiny new market."""
    rng = random.Random(seed)
    rows = []
    for country, typical_visits in DAILY_TRAFFIC.items():
        for day in DAYS:
            visits = int(typical_visits * rng.uniform(0.9, 1.1))
            rows.append(DailyFunnel(country, day, visits, round(visits * TRUE_RATE)))
    for day, (visits, conversions) in zip(DAYS, NEW_MARKET_DAYS):
        rows.append(DailyFunnel("IS", day, visits, conversions))
    return rows


class ReproduceProblem(unittest.TestCase):
    def test_averaging_daily_rates_inflates_global_conversion(self):
        naive = naive_rollup(naive_daily_rates(make_week()), "all")["all"]
        self.assertEqual(round(naive, 4), 0.0411)

    def test_one_conversion_in_two_visits_counts_as_much_as_a_busy_day(self):
        rates = naive_daily_rates(make_week())
        self.assertEqual(rates[("IS", "2026-10-11")], 0.5)
        # 1 of the 28 country-days moves the global average by over a point.
        without_new_market = {key: rate for key, rate in rates.items() if key[0] != "IS"}
        self.assertEqual(round(naive_rollup(without_new_market, "all")["all"], 4), 0.031)

    def test_tracking_outage_day_is_stored_as_zero_percent(self):
        rows = [DailyFunnel("US", "d1", 1_000, 30), DailyFunnel("US", "d2", 0, 0)]
        self.assertEqual(naive_rollup(naive_daily_rates(rows), "country")["US"], 0.015)


class FixRollupBySummingParts(unittest.TestCase):
    def test_global_rate_matches_true_rate(self):
        fixed = weekly_conversion(make_week(), "all")["all"]
        self.assertEqual(round(fixed, 4), 0.031)

    def test_tiny_market_keeps_its_own_rate_without_moving_global(self):
        per_country = weekly_conversion(make_week(), "country")
        self.assertEqual(per_country["IS"], 1 / 19)
        self.assertEqual(round(per_country["IN"], 3), TRUE_RATE)

    def test_day_with_no_visits_does_not_change_the_rate(self):
        rows = [DailyFunnel("US", "d1", 1_000, 30), DailyFunnel("US", "d2", 0, 0)]
        self.assertEqual(weekly_conversion(rows, "country")["US"], 0.03)

    def test_group_with_no_visits_has_undefined_rate(self):
        rows = [DailyFunnel("NZ", "d1", 0, 0)]
        self.assertIsNone(weekly_conversion(rows, "country")["NZ"])

    def test_rolling_up_in_two_steps_equals_one_step(self):
        # Parts are additive, so country totals summed again give the global total.
        parts = to_parts(make_week())
        countries = rollup(parts, "country")
        combined = RatioParts()
        for group in countries.values():
            combined.add(group)
        self.assertEqual(combined, rollup(parts, "all")["all"])

    def test_duplicate_rows_for_same_day_are_summed(self):
        rows = [DailyFunnel("US", "d1", 100, 3), DailyFunnel("US", "d1", 300, 9)]
        self.assertEqual(to_parts(rows)[("US", "d1")], RatioParts(12, 400))


class InputValidation(unittest.TestCase):
    def test_more_conversions_than_visits_is_rejected(self):
        with self.assertRaises(ValueError):
            DailyFunnel("US", "d1", 5, 6)

    def test_negative_counts_are_rejected(self):
        with self.assertRaises(ValueError):
            DailyFunnel("US", "d1", -1, 0)

    def test_unknown_level_is_rejected(self):
        with self.assertRaises(ValueError):
            rollup({}, "region")


if __name__ == "__main__":
    unittest.main()
