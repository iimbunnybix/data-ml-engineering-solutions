import random
import unittest

from solution import (
    Customer,
    kaplan_meier,
    median_lifetime,
    naive_lifetime_from_churned,
    simulate_customers,
    true_median_lifetime,
)

MONTHLY_CHURN = 0.08
# Mean of a geometric lifetime with 8% monthly churn is 1 / 0.08 months.
TRUE_MEAN_LIFETIME = 12.5


def two_years_of_cohorts() -> list[Customer]:
    """24 monthly cohorts of 500 customers, observed until today."""
    return simulate_customers(24, 500, MONTHLY_CHURN, random.Random(56))


class ReproduceProblem(unittest.TestCase):
    def test_churned_only_average_is_about_half_the_true_mean(self):
        naive = naive_lifetime_from_churned(two_years_of_cohorts())
        self.assertLess(naive, TRUE_MEAN_LIFETIME * 0.55)

    def test_churned_only_average_is_below_even_the_true_median(self):
        naive = naive_lifetime_from_churned(two_years_of_cohorts())
        self.assertLess(naive, true_median_lifetime(MONTHLY_CHURN) - 2)


class FixUsesKaplanMeier(unittest.TestCase):
    def test_true_median_for_eight_percent_churn_is_nine_months(self):
        # 0.92 ** 8 = 0.513 is still above half, 0.92 ** 9 = 0.472 is not.
        self.assertEqual(true_median_lifetime(MONTHLY_CHURN), 9)

    def test_kaplan_meier_median_matches_the_true_median(self):
        curve = kaplan_meier(two_years_of_cohorts())
        self.assertEqual(median_lifetime(curve), 9)

    def test_survival_curve_tracks_the_true_curve(self):
        curve = kaplan_meier(two_years_of_cohorts())
        for month in (3, 6, 12):
            expected = (1 - MONTHLY_CHURN) ** month
            self.assertAlmostEqual(curve[month], expected, delta=0.02)

    def test_survival_never_increases(self):
        curve = kaplan_meier(two_years_of_cohorts())
        values = [curve[month] for month in sorted(curve)]
        for earlier, later in zip(values, values[1:]):
            self.assertLessEqual(later, earlier)

    def test_small_hand_checked_example(self):
        customers = [
            Customer(1, 1, True),    # cancelled in month 1
            Customer(2, 2, False),   # still active after 2 months
            Customer(3, 3, True),    # cancelled in month 3
            Customer(4, 3, False),   # still active after 3 months
        ]
        curve = kaplan_meier(customers)
        # Month 1: 1 of 4 cancels -> 3/4. Month 2: nobody cancels.
        # Month 3: 2 at risk, 1 cancels -> 3/4 * 1/2.
        self.assertEqual(curve, {1: 0.75, 2: 0.75, 3: 0.375})
        self.assertEqual(median_lifetime(curve), 3)

    def test_without_any_active_customers_it_matches_the_simple_answer(self):
        customers = [Customer(index, tenure, True) for index, tenure in enumerate([1, 2, 3, 4])]
        self.assertEqual(median_lifetime(kaplan_meier(customers)), 2)

    def test_median_is_unknown_when_most_customers_are_still_active(self):
        young_business = simulate_customers(6, 500, 0.05, random.Random(1))
        self.assertIsNone(median_lifetime(kaplan_meier(young_business)))

    def test_empty_input_is_rejected(self):
        with self.assertRaises(ValueError):
            kaplan_meier([])


if __name__ == "__main__":
    unittest.main()
