import unittest
from datetime import date, timedelta

from solution import (check_partition_volume, median_absolute_deviation,
                      naive_should_publish, robust_z_score, same_weekday_history)

TUESDAY = date(2026, 11, 3)
SUNDAY = date(2026, 11, 8)
# Deterministic, slightly varying normal volumes for the last 8 weeks.
TUESDAY_COUNTS = [1_002_000, 987_000, 1_015_000, 996_000,
                  1_008_000, 979_000, 1_021_000, 993_000]
SUNDAY_COUNTS = [603_000, 588_000, 611_000, 597_000,
                 606_000, 592_000, 600_000, 609_000]


def build_history() -> dict[date, int]:
    """Eight weeks of row counts for Tuesdays and Sundays."""
    counts = {}
    for weeks_back, count in enumerate(TUESDAY_COUNTS, start=1):
        counts[TUESDAY - timedelta(weeks=weeks_back)] = count
    for weeks_back, count in enumerate(SUNDAY_COUNTS, start=1):
        counts[SUNDAY - timedelta(weeks=weeks_back)] = count
    return counts


class ReproduceProblem(unittest.TestCase):
    def test_half_day_partition_passes_the_non_empty_check(self):
        self.assertTrue(naive_should_publish(498_000))


class FixVolumeCheck(unittest.TestCase):
    def setUp(self):
        self.history = build_history()

    def test_half_day_partition_is_blocked(self):
        decision = check_partition_volume(TUESDAY, 498_000, self.history)
        self.assertFalse(decision.publish)
        self.assertIn("anomalously low", decision.reason)

    def test_normal_tuesday_is_published(self):
        decision = check_partition_volume(TUESDAY, 1_004_000, self.history)
        self.assertTrue(decision.publish)

    def test_double_load_is_blocked_as_too_high(self):
        decision = check_partition_volume(TUESDAY, 2_000_000, self.history)
        self.assertFalse(decision.publish)
        self.assertIn("anomalously high", decision.reason)

    def test_normal_sunday_is_judged_against_sundays_not_weekdays(self):
        decision = check_partition_volume(SUNDAY, 598_000, self.history)
        self.assertTrue(decision.publish)
        self.assertAlmostEqual(decision.baseline_median, 601_500)

    def test_empty_partition_is_blocked(self):
        self.assertFalse(check_partition_volume(TUESDAY, 0, self.history).publish)

    def test_new_table_without_history_is_published_with_warning(self):
        decision = check_partition_volume(TUESDAY, 10, {})
        self.assertTrue(decision.publish)
        self.assertIn("not enough history", decision.reason)


class FixRobustStatistics(unittest.TestCase):
    def test_history_only_contains_same_weekday(self):
        history = same_weekday_history(TUESDAY, build_history())
        self.assertEqual(history, TUESDAY_COUNTS)

    def test_one_past_outage_does_not_hide_a_new_one(self):
        history = list(TUESDAY_COUNTS)
        history[0] = 300_000  # an earlier bad day still in the baseline
        z_score, _ = robust_z_score(498_000, history)
        self.assertLess(z_score, -3.5)

    def test_flat_history_does_not_divide_by_zero(self):
        z_score, _ = robust_z_score(1_000_500, [1_000_000] * 6)
        self.assertLess(abs(z_score), 3.5)

    def test_mad_of_known_values(self):
        self.assertEqual(median_absolute_deviation([1, 2, 3, 4, 100]), 1)


if __name__ == "__main__":
    unittest.main()
