"""Turn a log-scale model's predictions back into costs without under-forecasting the total.

The delivery-cost model is trained on log(cost) because costs are skewed.
Serving simply took exp(prediction). exp of the mean of log(cost) is the
*median* cost, not the mean, so every prediction is a little low and the
summed forecast used for budgeting is systematically short. This module keeps
that naive back-transform and fixes it with Duan's smearing estimator: multiply
exp(prediction) by the average of exp(residual) measured on the training data.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

# Simulated delivery network: log(cost) grows linearly with distance.
BASE_LOG_COST = math.log(4.0)       # about $4 for a delivery next door
LOG_COST_PER_KM = 0.08              # each km adds about 8% to the cost
LOG_NOISE_STDDEV = 0.6              # traffic, waiting time, surcharges
MIN_DISTANCE_KM = 1.0
MAX_DISTANCE_KM = 20.0


@dataclass(frozen=True)
class Delivery:
    """One completed delivery: the feature and the cost we want to forecast."""

    distance_km: float
    cost: float


@dataclass(frozen=True)
class LogLinearModel:
    """log(cost) = intercept + slope * distance_km, fitted by least squares."""

    intercept: float
    slope: float

    def predict_log_cost(self, distance_km: float) -> float:
        """Prediction on the log scale, the scale the model was trained on."""
        return self.intercept + self.slope * distance_km


def log_costs_of(deliveries: list[Delivery]) -> list[float]:
    """log(cost) of every delivery, rejecting costs the log cannot handle."""
    log_costs = []
    for delivery in deliveries:
        if delivery.cost <= 0:
            # log(0) is undefined; a zero-cost row is a data error, not a delivery.
            raise ValueError(f"cost must be positive, got {delivery.cost}")
        log_costs.append(math.log(delivery.cost))
    return log_costs


def fit_log_linear(deliveries: list[Delivery]) -> LogLinearModel:
    """Ordinary least squares of log(cost) on distance (one feature, closed form)."""
    if len(deliveries) < 2:
        raise ValueError("need at least two deliveries to fit a line")
    distances = [delivery.distance_km for delivery in deliveries]
    log_costs = log_costs_of(deliveries)
    mean_distance = sum(distances) / len(distances)
    mean_log_cost = sum(log_costs) / len(log_costs)
    covariance = 0.0
    variance = 0.0
    for distance, log_cost in zip(distances, log_costs):
        covariance += (distance - mean_distance) * (log_cost - mean_log_cost)
        variance += (distance - mean_distance) ** 2
    if variance == 0:
        raise ValueError("all distances are equal; the slope is undefined")
    slope = covariance / variance
    return LogLinearModel(intercept=mean_log_cost - slope * mean_distance, slope=slope)


def naive_predict_cost(model: LogLinearModel, distance_km: float) -> float:
    """Original serving code: undo the log with exp(). This is the median cost."""
    return math.exp(model.predict_log_cost(distance_km))


def smearing_factor(model: LogLinearModel, training: list[Delivery]) -> float:
    """Duan's smearing estimate: mean of exp(residual) over the training data.

    exp(prediction) misses the spread of the errors. Averaging exp(residual)
    puts that spread back without assuming the errors are normal. The result
    is always >= 1 (Jensen's inequality), because mean residual is zero.
    """
    if not training:
        raise ValueError("need training deliveries to estimate the smearing factor")
    total = 0.0
    for delivery in training:
        residual = math.log(delivery.cost) - model.predict_log_cost(delivery.distance_km)
        total += math.exp(residual)
    return total / len(training)


def smeared_predict_cost(model: LogLinearModel, factor: float, distance_km: float) -> float:
    """Fixed serving code: the median prediction scaled up to the mean."""
    return naive_predict_cost(model, distance_km) * factor


def total_forecast_error(forecasts: list[float], actuals: list[float]) -> float:
    """(sum of forecasts - sum of actuals) / sum of actuals; negative means too low."""
    actual_total = sum(actuals)
    return (sum(forecasts) - actual_total) / actual_total


def make_deliveries(rng: random.Random, count: int) -> list[Delivery]:
    """Deterministic deliveries whose log(cost) is linear in distance plus normal noise."""
    deliveries = []
    for _ in range(count):
        distance = rng.uniform(MIN_DISTANCE_KM, MAX_DISTANCE_KM)
        log_cost = BASE_LOG_COST + LOG_COST_PER_KM * distance + rng.gauss(0.0, LOG_NOISE_STDDEV)
        deliveries.append(Delivery(distance, math.exp(log_cost)))
    return deliveries
