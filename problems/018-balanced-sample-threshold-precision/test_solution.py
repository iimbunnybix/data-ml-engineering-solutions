import random
import unittest

from solution import (
    NAIVE_THRESHOLD,
    PRODUCTION_FRAUD_RATE,
    TRAINING_FRAUD_RATE,
    ScoredPayment,
    adjust_for_base_rate,
    choose_threshold_for_precision,
    corrected_alert_scores,
    naive_alert_scores,
    precision_and_recall,
    simulate_payments,
)

TARGET_PRECISION = 0.5


def balanced_sample() -> list[ScoredPayment]:
    """The 50/50 sample the model was originally tuned on."""
    return simulate_payments(random.Random(42), 20_000, TRAINING_FRAUD_RATE)


def production_sample(seed: int = 7) -> list[ScoredPayment]:
    """Labelled payments with the real 1% fraud rate."""
    return simulate_payments(random.Random(seed), 50_000, PRODUCTION_FRAUD_RATE)


class ReproduceProblem(unittest.TestCase):
    def test_threshold_looks_good_on_the_balanced_sample(self):
        payments = balanced_sample()
        precision, _ = precision_and_recall(payments, naive_alert_scores(payments), NAIVE_THRESHOLD)
        self.assertGreater(precision, 0.8)

    def test_same_threshold_collapses_precision_in_production(self):
        payments = production_sample()
        precision, _ = precision_and_recall(payments, naive_alert_scores(payments), NAIVE_THRESHOLD)
        self.assertLess(precision, 0.06)

    def test_balanced_probabilities_overstate_the_fraud_rate(self):
        payments = production_sample()
        average_probability = sum(naive_alert_scores(payments)) / len(payments)
        self.assertGreater(average_probability, 10 * PRODUCTION_FRAUD_RATE)


class FixBaseRateCorrection(unittest.TestCase):
    def test_no_change_when_rates_match(self):
        self.assertAlmostEqual(adjust_for_base_rate(0.7, 0.5, 0.5), 0.7)

    def test_even_odds_become_the_production_rate(self):
        # A 0.5 score on balanced data carries no evidence either way.
        self.assertAlmostEqual(adjust_for_base_rate(0.5, 0.5, 0.01), 0.01)

    def test_correction_keeps_the_ranking(self):
        raw = [0.1, 0.4, 0.9, 0.99]
        corrected = [adjust_for_base_rate(value, 0.5, 0.01) for value in raw]
        self.assertEqual(corrected, sorted(corrected))

    def test_extreme_probabilities_do_not_crash(self):
        self.assertLess(adjust_for_base_rate(0.0, 0.5, 0.01), 1e-9)
        self.assertGreater(adjust_for_base_rate(1.0, 0.5, 0.01), 0.999)

    def test_corrected_probabilities_match_the_real_fraud_rate(self):
        payments = production_sample()
        average_probability = sum(corrected_alert_scores(payments)) / len(payments)
        self.assertAlmostEqual(average_probability, PRODUCTION_FRAUD_RATE, delta=0.003)

    def test_corrected_scores_at_half_are_mostly_real_fraud(self):
        payments = production_sample()
        precision, _ = precision_and_recall(payments, corrected_alert_scores(payments), 0.5)
        self.assertGreater(precision, TARGET_PRECISION)


class FixThresholdChosenOnProductionRate(unittest.TestCase):
    def test_chosen_threshold_meets_target_on_tuning_data(self):
        payments = production_sample()
        scores = corrected_alert_scores(payments)
        threshold = choose_threshold_for_precision(payments, scores, TARGET_PRECISION)
        precision, _ = precision_and_recall(payments, scores, threshold)
        self.assertGreaterEqual(precision, TARGET_PRECISION)

    def test_chosen_threshold_catches_more_fraud_than_corrected_half(self):
        payments = production_sample()
        scores = corrected_alert_scores(payments)
        threshold = choose_threshold_for_precision(payments, scores, TARGET_PRECISION)
        _, recall_at_half = precision_and_recall(payments, scores, 0.5)
        _, recall_chosen = precision_and_recall(payments, scores, threshold)
        self.assertGreater(recall_chosen, recall_at_half)

    def test_unreachable_target_returns_none(self):
        payments = [ScoredPayment(0.9, False), ScoredPayment(0.8, False)]
        self.assertIsNone(choose_threshold_for_precision(payments, [0.9, 0.8], 0.5))

    def test_tied_scores_are_not_split(self):
        payments = [ScoredPayment(0.9, True), ScoredPayment(0.9, False), ScoredPayment(0.9, False)]
        # Alerting on 0.9 means alerting on all three, precision 1/3.
        self.assertIsNone(choose_threshold_for_precision(payments, [0.9, 0.9, 0.9], 0.5))


if __name__ == "__main__":
    unittest.main()
