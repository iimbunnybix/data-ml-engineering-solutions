import random
import unittest

from solution import (SampleRatioMismatchError, VariantStats, check_srm, chi_square_p_value_1dof,
                      chi_square_statistic, naive_read_results, read_results)

USERS = 100_000
TRUE_CONVERSION_RATE = 0.05          # identical in both variants: no real effect
TRACKING_LOSS_FOR_BOUNCERS = 0.08    # share of non-buying treatment users lost in the redirect
USER_SEED = 6
LOGGING_SEED = 99


def assign_users(seed: int = USER_SEED) -> list[tuple[str, bool]]:
    """(variant, converted) for every user, split 50/50 at random."""
    rng = random.Random(seed)
    users = []
    for _ in range(USERS):
        variant = "control" if rng.random() < 0.5 else "treatment"
        users.append((variant, rng.random() < TRUE_CONVERSION_RATE))
    return users


def logged_stats(users: list[tuple[str, bool]], tracking_loss: float) -> tuple[VariantStats, VariantStats]:
    """What the analytics pipeline sees after the treatment redirect drops some users.

    Only users who bounce (do not convert) can be lost: buyers reach the
    checkout page, which logs them again.
    """
    rng = random.Random(LOGGING_SEED)
    counts = {"control": [0, 0], "treatment": [0, 0]}
    for variant, converted in users:
        if variant == "treatment" and not converted and rng.random() < tracking_loss:
            continue
        counts[variant][0] += 1
        counts[variant][1] += int(converted)
    return (VariantStats("control", *counts["control"]), VariantStats("treatment", *counts["treatment"]))


def stats(control_users: int, treatment_users: int) -> tuple[VariantStats, VariantStats]:
    return VariantStats("control", control_users, 0), VariantStats("treatment", treatment_users, 0)


class ReproduceProblem(unittest.TestCase):
    def test_without_tracking_loss_there_is_no_significant_effect(self):
        control, treatment = logged_stats(assign_users(), tracking_loss=0.0)
        self.assertFalse(naive_read_results(control, treatment).significant)

    def test_tracking_loss_makes_a_variant_with_no_effect_look_like_a_winner(self):
        control, treatment = logged_stats(assign_users(), TRACKING_LOSS_FOR_BOUNCERS)
        readout = naive_read_results(control, treatment)
        self.assertEqual(readout.winner, "treatment")
        self.assertTrue(readout.significant)
        self.assertGreater(readout.lift, 0.09)

    def test_tracking_loss_skews_the_split_to_about_52_48(self):
        control, treatment = logged_stats(assign_users(), TRACKING_LOSS_FOR_BOUNCERS)
        share = control.users / (control.users + treatment.users)
        self.assertAlmostEqual(share, 0.52, delta=0.005)


class FixSrmCheck(unittest.TestCase):
    def test_biased_experiment_is_blocked(self):
        control, treatment = logged_stats(assign_users(), TRACKING_LOSS_FOR_BOUNCERS)
        with self.assertRaises(SampleRatioMismatchError):
            read_results(control, treatment)

    def test_biased_experiment_has_a_tiny_srm_p_value(self):
        control, treatment = logged_stats(assign_users(), TRACKING_LOSS_FOR_BOUNCERS)
        self.assertLess(check_srm(control, treatment).p_value, 1e-30)

    def test_healthy_experiment_can_be_read(self):
        control, treatment = logged_stats(assign_users(), tracking_loss=0.0)
        self.assertFalse(read_results(control, treatment).significant)

    def test_small_imbalance_is_flagged_with_enough_users(self):
        # 50.6% / 49.4% of 100,000 users: chi-square 14.4.
        self.assertTrue(check_srm(*stats(50_600, 49_400)).has_mismatch)

    def test_same_imbalance_is_not_flagged_with_few_users(self):
        # 50.6% / 49.4% of 10,000 users is ordinary noise: chi-square 1.44.
        self.assertFalse(check_srm(*stats(5_060, 4_940)).has_mismatch)

    def test_unequal_configured_split_is_respected(self):
        self.assertFalse(check_srm(*stats(90_000, 10_000), expected_control_share=0.9).has_mismatch)
        self.assertTrue(check_srm(*stats(50_000, 50_000), expected_control_share=0.9).has_mismatch)

    def test_false_alarm_rate_matches_the_threshold_on_fair_splits(self):
        rng = random.Random(5)
        experiments, users_per_experiment, alarms = 2_000, 1_000, 0
        for _ in range(experiments):
            control_users = sum(rng.random() < 0.5 for _ in range(users_per_experiment))
            result = check_srm(*stats(control_users, users_per_experiment - control_users))
            if result.p_value < 0.05:
                alarms += 1
        self.assertAlmostEqual(alarms / experiments, 0.05, delta=0.015)


class FixStatistics(unittest.TestCase):
    def test_chi_square_statistic_by_hand(self):
        # expected 50/50 of 100: (60-50)^2/50 + (40-50)^2/50 = 4.0
        self.assertAlmostEqual(chi_square_statistic([60, 40], [0.5, 0.5]), 4.0)

    def test_p_value_matches_known_critical_values(self):
        # Standard chi-square table, 1 degree of freedom.
        self.assertAlmostEqual(chi_square_p_value_1dof(3.841459), 0.05, places=6)
        self.assertAlmostEqual(chi_square_p_value_1dof(10.827566), 0.001, places=7)

    def test_perfect_split_has_p_value_one(self):
        self.assertEqual(check_srm(*stats(500, 500)).p_value, 1.0)

    def test_no_users_is_an_error(self):
        with self.assertRaises(ValueError):
            check_srm(*stats(0, 0))

    def test_invalid_expected_shares_are_rejected(self):
        with self.assertRaises(ValueError):
            chi_square_statistic([10, 10], [0.6, 0.6])


if __name__ == "__main__":
    unittest.main()
