import math
import random
import unittest

from solution import (LOG_NOISE_STDDEV, Delivery, LogLinearModel, fit_log_linear,
                      make_deliveries, naive_predict_cost, smeared_predict_cost,
                      smearing_factor, total_forecast_error)

TRAINING_SEED = 47
NEXT_WEEK_SEED = 4700
TRAINING_COUNT = 5_000
NEXT_WEEK_COUNT = 2_000
# The finance team accepts a weekly total within 3% of the real spend.
ACCEPTABLE_TOTAL_ERROR = 0.03


def training_deliveries() -> list[Delivery]:
    """Last month's deliveries, the same on every run."""
    return make_deliveries(random.Random(TRAINING_SEED), TRAINING_COUNT)


def next_week_deliveries() -> list[Delivery]:
    """Unseen deliveries whose total cost we forecast."""
    return make_deliveries(random.Random(NEXT_WEEK_SEED), NEXT_WEEK_COUNT)


def actual_costs(deliveries: list[Delivery]) -> list[float]:
    return [delivery.cost for delivery in deliveries]


def naive_forecasts(model: LogLinearModel, deliveries: list[Delivery]) -> list[float]:
    return [naive_predict_cost(model, delivery.distance_km) for delivery in deliveries]


def smeared_forecasts(model: LogLinearModel, factor: float,
                      deliveries: list[Delivery]) -> list[float]:
    return [smeared_predict_cost(model, factor, delivery.distance_km) for delivery in deliveries]


def share_above_forecast(forecasts: list[float], actuals: list[float]) -> float:
    """Fraction of deliveries that cost more than forecast."""
    above = 0
    for forecast, actual in zip(forecasts, actuals):
        if actual > forecast:
            above += 1
    return above / len(actuals)


class ReproduceProblem(unittest.TestCase):
    def setUp(self):
        self.model = fit_log_linear(training_deliveries())
        self.deliveries = next_week_deliveries()

    def test_weekly_total_is_forecast_more_than_ten_percent_low(self):
        error = total_forecast_error(naive_forecasts(self.model, self.deliveries),
                                     actual_costs(self.deliveries))
        self.assertLess(error, -0.10)

    def test_naive_forecast_is_the_median_half_the_costs_are_above_it(self):
        share = share_above_forecast(naive_forecasts(self.model, self.deliveries),
                                     actual_costs(self.deliveries))
        self.assertAlmostEqual(share, 0.5, delta=0.03)

    def test_model_itself_is_fitted_correctly_on_the_log_scale(self):
        # The bias is not a bad fit: the slope matches the simulation's 0.08.
        self.assertAlmostEqual(self.model.slope, 0.08, delta=0.005)


class FixWithSmearingEstimator(unittest.TestCase):
    def setUp(self):
        training = training_deliveries()
        self.model = fit_log_linear(training)
        self.factor = smearing_factor(self.model, training)
        self.deliveries = next_week_deliveries()

    def test_weekly_total_is_within_three_percent(self):
        error = total_forecast_error(smeared_forecasts(self.model, self.factor, self.deliveries),
                                     actual_costs(self.deliveries))
        self.assertLess(abs(error), ACCEPTABLE_TOTAL_ERROR)

    def test_factor_matches_lognormal_theory(self):
        # For normal log-errors the mean/median ratio is exp(sigma^2 / 2).
        expected = math.exp(LOG_NOISE_STDDEV ** 2 / 2)
        self.assertAlmostEqual(self.factor, expected, delta=0.02)

    def test_factor_is_computed_from_training_data_only(self):
        # Same model, same training rows: the factor must not change with what we predict.
        again = smearing_factor(self.model, training_deliveries())
        self.assertEqual(again, self.factor)

    def test_perfect_fit_needs_no_correction(self):
        exact = [Delivery(distance, math.exp(1.0 + 0.1 * distance)) for distance in (1.0, 5.0, 9.0)]
        model = fit_log_linear(exact)
        self.assertAlmostEqual(smearing_factor(model, exact), 1.0, places=12)

    def test_factor_is_never_below_one(self):
        noisy = make_deliveries(random.Random(1), 50)
        model = fit_log_linear(noisy)
        self.assertGreaterEqual(smearing_factor(model, noisy), 1.0)

    def test_zero_cost_row_is_rejected(self):
        with self.assertRaises(ValueError):
            fit_log_linear([Delivery(1.0, 5.0), Delivery(2.0, 0.0)])

    def test_identical_distances_cannot_be_fitted(self):
        with self.assertRaises(ValueError):
            fit_log_linear([Delivery(3.0, 5.0), Delivery(3.0, 7.0)])

    def test_empty_training_set_has_no_factor(self):
        with self.assertRaises(ValueError):
            smearing_factor(self.model, [])


if __name__ == "__main__":
    unittest.main()
