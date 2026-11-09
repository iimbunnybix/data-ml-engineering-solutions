import dataclasses
import random
import unittest

from solution import (TARGET_REVIEW_RATE, BundleError, FraudService, LinearModel,
                      NaiveFraudService, build_bundle, fingerprint, review_rate)

SAMPLE_SIZE = 20_000

# v7 was retrained with a new feature scaling, so its scores are higher overall.
MODEL_V6 = LinearModel(version="v6", weights=(1.2, 0.8, -0.5), bias=-3.0)
MODEL_V7 = LinearModel(version="v7", weights=(1.6, 1.1, -0.4), bias=-1.5)


def make_transactions(seed: int, size: int = SAMPLE_SIZE) -> list[tuple[float, ...]]:
    """Seeded transaction features: amount z-score, velocity, account age."""
    rng = random.Random(seed)
    rows = []
    for _ in range(size):
        rows.append((rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1)))
    return rows


def live_review_rate(service, transactions: list[tuple[float, ...]]) -> float:
    flagged = 0
    for features in transactions:
        if service.needs_review(features):
            flagged += 1
    return flagged / len(transactions)


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tuning = make_transactions(seed=1)
        self.reference = make_transactions(seed=2)
        self.live = make_transactions(seed=3)
        self.bundle_v6 = build_bundle(MODEL_V6, self.tuning)
        self.bundle_v7 = build_bundle(MODEL_V7, self.tuning)


class ReproduceProblem(Fixture):
    def test_v6_with_its_own_threshold_reviews_about_two_percent(self):
        service = NaiveFraudService(MODEL_V6, self.bundle_v6.threshold)
        self.assertAlmostEqual(live_review_rate(service, self.live), 0.02, delta=0.003)

    def test_rolling_out_v7_alone_grows_the_review_queue_tenfold(self):
        service = NaiveFraudService(MODEL_V6, self.bundle_v6.threshold)
        service.roll_out_model(MODEL_V7)
        self.assertGreater(live_review_rate(service, self.live), 9 * TARGET_REVIEW_RATE)

    def test_updating_threshold_before_model_reviews_almost_nothing(self):
        service = NaiveFraudService(MODEL_V6, self.bundle_v6.threshold)
        service.set_threshold(self.bundle_v7.threshold)
        self.assertLess(live_review_rate(service, self.live), 0.005)


class FixVersionedBundle(Fixture):
    def test_deploying_v7_bundle_keeps_review_rate_at_target(self):
        service = FraudService(self.bundle_v6, self.reference)
        service.deploy(self.bundle_v7)
        self.assertAlmostEqual(live_review_rate(service, self.live), 0.02, delta=0.003)

    def test_bundle_with_threshold_from_another_model_is_rejected(self):
        mixed = dataclasses.replace(self.bundle_v6, model=MODEL_V7)
        service = FraudService(self.bundle_v6, self.reference)
        with self.assertRaises(BundleError):
            service.deploy(mixed)

    def test_rejected_deploy_leaves_old_bundle_serving(self):
        mixed = dataclasses.replace(self.bundle_v6, model=MODEL_V7)
        service = FraudService(self.bundle_v6, self.reference)
        with self.assertRaises(BundleError):
            service.deploy(mixed)
        self.assertIs(service.bundle, self.bundle_v6)

    def test_threshold_tuned_on_unrepresentative_data_is_rejected(self):
        # Tuned only on high-amount transactions, so the threshold is too high.
        skewed = [row for row in self.tuning if row[0] > 1.0]
        bundle = build_bundle(MODEL_V7, skewed)
        service = FraudService(self.bundle_v6, self.reference)
        with self.assertRaises(BundleError):
            service.deploy(bundle)

    def test_fingerprint_changes_when_any_weight_changes(self):
        tweaked = dataclasses.replace(MODEL_V7, weights=(1.6, 1.1, -0.41))
        self.assertNotEqual(fingerprint(MODEL_V7), fingerprint(tweaked))

    def test_fingerprint_ignores_the_version_label(self):
        relabelled = dataclasses.replace(MODEL_V7, version="v7-hotfix")
        self.assertEqual(fingerprint(MODEL_V7), fingerprint(relabelled))

    def test_invalid_initial_bundle_is_refused_at_startup(self):
        mixed = dataclasses.replace(self.bundle_v7, model=MODEL_V6)
        with self.assertRaises(BundleError):
            FraudService(mixed, self.reference)

    def test_built_bundle_matches_target_on_its_tuning_sample(self):
        rate = review_rate(MODEL_V7, self.bundle_v7.threshold, self.tuning)
        self.assertAlmostEqual(rate, TARGET_REVIEW_RATE, delta=0.0005)


if __name__ == "__main__":
    unittest.main()
