import unittest

from solution import (PSI_ALERT_THRESHOLD, DriftMonitor, daily_values, original_drift_alerts,
                      population_stability_index, slowly_drifting_days)

START_MEAN = 50.0         # average order value in euros at training time
SPREAD = 15.0
DAILY_GROWTH = 0.01       # +1% per day
DAYS = 60


def two_months_of_slow_drift() -> list[list[float]]:
    return slowly_drifting_days(DAYS, START_MEAN, DAILY_GROWTH, SPREAD)


class ReproduceProblem(unittest.TestCase):
    def test_day_over_day_check_never_alerts_on_slow_drift(self):
        self.assertEqual(original_drift_alerts(two_months_of_slow_drift()), [])

    def test_each_day_looks_almost_identical_to_the_day_before(self):
        days = two_months_of_slow_drift()
        for day_number in range(1, DAYS):
            psi = population_stability_index(days[day_number - 1], days[day_number])
            self.assertLess(psi, 0.01)

    def test_yet_the_last_day_is_far_from_training(self):
        days = two_months_of_slow_drift()
        self.assertGreater(population_stability_index(days[0], days[-1]), 5)


class FixFixedTrainingReference(unittest.TestCase):
    def test_slow_drift_alerts_once_it_has_built_up(self):
        days = two_months_of_slow_drift()
        alerts = DriftMonitor(days[0]).alerts(days)
        self.assertEqual(alerts[0], 13)
        self.assertEqual(alerts, list(range(13, DAYS)))

    def test_stable_feature_never_alerts(self):
        days = [daily_values(START_MEAN, SPREAD) for _ in range(DAYS)]
        self.assertEqual(DriftMonitor(days[0]).alerts(days), [])

    def test_sudden_jump_is_still_caught(self):
        training = daily_values(START_MEAN, SPREAD)
        days = [training, daily_values(80.0, SPREAD)]
        self.assertEqual(DriftMonitor(training).alerts(days), [1])

    def test_spread_change_with_same_mean_is_caught(self):
        training = daily_values(START_MEAN, SPREAD)
        wider = daily_values(START_MEAN, SPREAD * 3)
        monitor = DriftMonitor(training)
        self.assertGreater(monitor.psi_against_training(wider), PSI_ALERT_THRESHOLD)

    def test_bin_edges_stay_frozen_after_monitoring(self):
        days = two_months_of_slow_drift()
        monitor = DriftMonitor(days[0])
        edges_before = list(monitor.edges)
        monitor.alerts(days)
        self.assertEqual(monitor.edges, edges_before)

    def test_values_outside_training_range_do_not_crash(self):
        monitor = DriftMonitor(daily_values(START_MEAN, SPREAD))
        self.assertGreater(monitor.psi_against_training([1e9] * 100), PSI_ALERT_THRESHOLD)


if __name__ == "__main__":
    unittest.main()
