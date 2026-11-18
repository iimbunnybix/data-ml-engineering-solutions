import unittest

from solution import (bucket_for, assignments_are_independent, chi_square_statistic,
                      contingency_table, in_treatment, measured_lift,
                      naive_in_treatment, simulate_conversions, treated_users)

USERS = list(range(40_000))
BUTTON = "button-colour"       # no real effect
BANNER = "free-shipping"       # real +3 percentage point effect
CONVERSION_SEED = 11


def run_both_experiments(assign) -> tuple[set[int], set[int], set[int]]:
    """Treated users of each experiment and the users who converted."""
    button_treated = treated_users(USERS, BUTTON, assign)
    banner_treated = treated_users(USERS, BANNER, assign)
    converted = simulate_conversions(USERS, banner_treated, seed=CONVERSION_SEED)
    return button_treated, banner_treated, converted


class ReproduceProblem(unittest.TestCase):
    def test_every_button_treatment_user_is_also_in_banner_treatment(self):
        button_treated, banner_treated, _ = run_both_experiments(naive_in_treatment)
        self.assertEqual(button_treated, banner_treated)

    def test_button_colour_shows_a_fake_lift_of_about_two_points(self):
        button_treated, _, converted = run_both_experiments(naive_in_treatment)
        lift = measured_lift(USERS, button_treated, converted)
        self.assertAlmostEqual(lift, 0.0239, places=4)


class FixSaltsTheHashPerExperiment(unittest.TestCase):
    def test_about_half_of_button_treatment_users_see_the_banner(self):
        button_treated, banner_treated, _ = run_both_experiments(in_treatment)
        overlap = len(button_treated & banner_treated) / len(button_treated)
        self.assertAlmostEqual(overlap, 0.5, delta=0.01)

    def test_button_colour_lift_is_close_to_zero(self):
        button_treated, _, converted = run_both_experiments(in_treatment)
        self.assertLess(abs(measured_lift(USERS, button_treated, converted)), 0.005)

    def test_real_banner_effect_is_still_detected(self):
        _, banner_treated, converted = run_both_experiments(in_treatment)
        self.assertGreater(measured_lift(USERS, banner_treated, converted), 0.025)

    def test_split_is_about_fifty_fifty(self):
        button_treated, _, _ = run_both_experiments(in_treatment)
        self.assertAlmostEqual(len(button_treated) / len(USERS), 0.5, delta=0.01)

    def test_a_user_always_gets_the_same_assignment(self):
        first = [in_treatment(BUTTON, user_id) for user_id in range(100)]
        second = [in_treatment(BUTTON, user_id) for user_id in range(100)]
        self.assertEqual(first, second)

    def test_buckets_differ_between_experiments(self):
        differing = 0
        for user_id in range(100):
            if bucket_for(BUTTON, user_id) != bucket_for(BANNER, user_id):
                differing += 1
        self.assertGreater(differing, 90)


class FixDetectsCorrelatedAssignments(unittest.TestCase):
    def test_check_flags_the_shared_hash(self):
        button_treated, banner_treated, _ = run_both_experiments(naive_in_treatment)
        self.assertFalse(assignments_are_independent(USERS, button_treated, banner_treated))

    def test_check_passes_salted_assignments(self):
        button_treated, banner_treated, _ = run_both_experiments(in_treatment)
        self.assertTrue(assignments_are_independent(USERS, button_treated, banner_treated))

    def test_everyone_on_one_side_counts_as_dependent(self):
        table = contingency_table([1, 2, 3], {1, 2, 3}, {1})
        self.assertEqual(chi_square_statistic(table), float("inf"))

    def test_empty_population_is_rejected(self):
        with self.assertRaises(ValueError):
            assignments_are_independent([], set(), set())


if __name__ == "__main__":
    unittest.main()
