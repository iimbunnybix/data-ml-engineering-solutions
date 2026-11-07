import random
import unittest

from solution import (FeatureStats, FrozenScaler, LinearRiskModel, naive_score_batch,
                      score_batch)

FEATURES = ["utilization", "income_k"]
MODEL = LinearRiskModel({"utilization": 1.2, "income_k": -0.6}, bias=-2.5)
CUSTOMER = {"utilization": 0.85, "income_k": 40.0}


def make_customers(count: int, seed: int, utilization_range: tuple[float, float]) -> list[dict]:
    """Synthetic customers; utilization drawn from the given range."""
    rng = random.Random(seed)
    low, high = utilization_range
    return [{"utilization": rng.uniform(low, high), "income_k": rng.uniform(20, 120)}
            for _ in range(count)]


def training_scaler() -> FrozenScaler:
    return FrozenScaler.fit(make_customers(5_000, seed=1, utilization_range=(0.0, 1.0)), FEATURES)


GENERAL_BATCH = make_customers(999, seed=2, utilization_range=(0.0, 1.0))
HIGH_RISK_BATCH = make_customers(999, seed=3, utilization_range=(0.7, 1.0))


class ReproduceProblem(unittest.TestCase):
    def test_same_customer_gets_different_scores_in_different_batches(self):
        in_general = naive_score_batch(MODEL, GENERAL_BATCH + [CUSTOMER])[-1]
        in_high_risk = naive_score_batch(MODEL, HIGH_RISK_BATCH + [CUSTOMER])[-1]
        self.assertEqual((round(in_general, 3), round(in_high_risk, 3)), (0.392, 0.129))

    def test_high_risk_batch_looks_average(self):
        # Batch z-scores always centre the batch on zero, so its mean score barely moves.
        general = naive_score_batch(MODEL, GENERAL_BATCH)
        high_risk = naive_score_batch(MODEL, HIGH_RISK_BATCH)
        self.assertLess(abs(sum(general) / 999 - sum(high_risk) / 999), 0.005)

    def test_single_row_batch_crashes(self):
        with self.assertRaises(ZeroDivisionError):
            naive_score_batch(MODEL, [CUSTOMER])


class FixFrozenTrainingStatistics(unittest.TestCase):
    def setUp(self):
        self.scaler = training_scaler()

    def test_score_does_not_depend_on_the_rest_of_the_batch(self):
        in_general = score_batch(MODEL, self.scaler, GENERAL_BATCH + [CUSTOMER])[-1]
        in_high_risk = score_batch(MODEL, self.scaler, HIGH_RISK_BATCH + [CUSTOMER])[-1]
        alone = score_batch(MODEL, self.scaler, [CUSTOMER])[0]
        self.assertEqual(in_general, in_high_risk)
        self.assertEqual(in_general, alone)
        self.assertEqual(round(alone, 3), 0.39)

    def test_high_risk_batch_scores_higher_than_general_batch(self):
        general = score_batch(MODEL, self.scaler, GENERAL_BATCH)
        high_risk = score_batch(MODEL, self.scaler, HIGH_RISK_BATCH)
        self.assertGreater(sum(high_risk) / 999 - sum(general) / 999, 0.05)

    def test_empty_batch_returns_no_scores(self):
        self.assertEqual(score_batch(MODEL, self.scaler, []), [])

    def test_constant_training_feature_does_not_divide_by_zero(self):
        scaler = FrozenScaler.fit([{"x": 5.0}, {"x": 5.0}], ["x"])
        self.assertEqual(scaler.stats["x"], FeatureStats(5.0, 1.0))
        self.assertEqual(scaler.transform({"x": 7.0}), {"x": 2.0})

    def test_missing_feature_in_row_is_an_error(self):
        with self.assertRaises(KeyError):
            score_batch(MODEL, self.scaler, [{"utilization": 0.5}])

    def test_scaler_without_a_model_feature_is_rejected(self):
        partial = FrozenScaler({"utilization": FeatureStats(0.5, 0.29)})
        with self.assertRaises(ValueError):
            score_batch(MODEL, partial, [CUSTOMER])

    def test_fit_on_empty_training_set_is_rejected(self):
        with self.assertRaises(ValueError):
            FrozenScaler.fit([], FEATURES)


if __name__ == "__main__":
    unittest.main()
