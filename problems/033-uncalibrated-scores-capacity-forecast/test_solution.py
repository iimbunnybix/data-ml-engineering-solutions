import math
import random
import unittest

from solution import (IsotonicCalibrator, brier_score, count_above, expected_calibration_error,
                      naive_expected_count)

CUSTOMERS_PER_MONTH = 50_000
CALIBRATION_SEED, NEXT_MONTH_SEED = 1, 2
# The overconfident model: its log-odds are 2.5x the true log-odds, plus 4.
MODEL_SHARPNESS = 2.5
MODEL_SHIFT = 4.0


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def simulate_month(seed: int) -> tuple[list[float], list[int]]:
    """(raw model scores, churned 0/1) for one month of customers.

    Each customer has a true churn log-odds z; the outcome is drawn from
    sigmoid(z) while the model reports sigmoid(2.5 z + 4): same ranking,
    wrong probabilities.
    """
    rng = random.Random(seed)
    scores, churned = [], []
    for _ in range(CUSTOMERS_PER_MONTH):
        true_log_odds = rng.gauss(-2.4, 1.1)
        scores.append(sigmoid(MODEL_SHARPNESS * true_log_odds + MODEL_SHIFT))
        churned.append(1 if rng.random() < sigmoid(true_log_odds) else 0)
    return scores, churned


def churned_among_selected(scores: list[float], churned: list[int], threshold: float) -> int:
    return sum(y for s, y in zip(scores, churned) if s > threshold)


class Months(unittest.TestCase):
    """A calibration month with known outcomes and the month being planned."""

    @classmethod
    def setUpClass(cls):
        cls.calibration_scores, cls.calibration_churned = simulate_month(CALIBRATION_SEED)
        cls.scores, cls.churned = simulate_month(NEXT_MONTH_SEED)
        cls.calibrator = IsotonicCalibrator(cls.calibration_scores, cls.calibration_churned)
        cls.calibrated = cls.calibrator.probabilities(cls.scores)


class ReproduceProblem(Months):
    def test_actual_churners_next_month(self):
        self.assertEqual(sum(self.churned), 6_016)

    def test_summed_raw_scores_forecast_more_than_twice_the_churners(self):
        self.assertEqual(round(naive_expected_count(self.scores)), 13_395)

    def test_most_customers_above_one_half_do_not_churn(self):
        self.assertEqual(count_above(self.scores, 0.5), 11_551)
        self.assertEqual(churned_among_selected(self.scores, self.churned, 0.5), 3_400)

    def test_raw_scores_are_badly_calibrated(self):
        self.assertAlmostEqual(expected_calibration_error(self.scores, self.churned), 0.156, places=3)
        self.assertAlmostEqual(brier_score(self.scores, self.churned), 0.154, places=3)


class FixIsotonicCalibration(Months):
    def test_calibrated_forecast_is_within_one_percent(self):
        forecast = sum(self.calibrated)
        self.assertEqual(round(forecast), 6_002)
        self.assertLess(abs(forecast - sum(self.churned)) / sum(self.churned), 0.01)

    def test_customers_above_one_half_now_mostly_churn(self):
        self.assertEqual(count_above(self.calibrated, 0.5), 838)
        self.assertEqual(churned_among_selected(self.calibrated, self.churned, 0.5), 502)

    def test_calibration_error_and_brier_score_drop(self):
        self.assertLess(expected_calibration_error(self.calibrated, self.churned), 0.007)
        self.assertAlmostEqual(brier_score(self.calibrated, self.churned), 0.092, places=3)

    def test_higher_raw_score_never_gets_lower_probability(self):
        ordered = sorted(self.scores)
        probabilities = self.calibrator.probabilities(ordered)
        for lower, higher in zip(probabilities, probabilities[1:]):
            self.assertLessEqual(lower, higher)


class CalibratorEdgeCases(unittest.TestCase):
    def test_pav_merges_a_decreasing_pair_into_its_mean(self):
        calibrator = IsotonicCalibrator([0.1, 0.2, 0.3, 0.4], [0, 1, 0, 1])
        self.assertEqual(calibrator.probabilities([0.1, 0.2, 0.3, 0.4]), [0.0, 0.5, 0.5, 1.0])

    def test_equal_scores_get_one_probability(self):
        calibrator = IsotonicCalibrator([0.7, 0.7, 0.7, 0.7], [1, 0, 0, 0])
        self.assertEqual(calibrator.probability(0.7), 0.25)

    def test_scores_outside_the_calibration_range_are_clamped(self):
        calibrator = IsotonicCalibrator([0.2, 0.8], [0, 1])
        self.assertEqual(calibrator.probability(0.0), 0.0)
        self.assertEqual(calibrator.probability(0.99), 1.0)

    def test_empty_or_mismatched_calibration_data_is_rejected(self):
        with self.assertRaises(ValueError):
            IsotonicCalibrator([], [])
        with self.assertRaises(ValueError):
            IsotonicCalibrator([0.1, 0.2], [1])

    def test_non_binary_labels_are_rejected(self):
        with self.assertRaises(ValueError):
            IsotonicCalibrator([0.1, 0.2], [0, 2])

    def test_perfect_probabilities_have_zero_calibration_error(self):
        self.assertEqual(expected_calibration_error([0.0, 0.0, 1.0], [0, 0, 1]), 0.0)
        self.assertEqual(brier_score([0.0, 1.0], [0, 1]), 0.0)


if __name__ == "__main__":
    unittest.main()
