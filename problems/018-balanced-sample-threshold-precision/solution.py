"""Re-calibrate a fraud model trained on a class-balanced sample.

The fraud model was trained and its 0.5 threshold tuned on a 50/50 sample of
fraud and normal payments. In production only about 1% of payments are fraud,
so the model's probabilities are far too high and most alerts are false
positives. This module corrects the probabilities for the real base rate
(prior-shift correction) and picks the alert threshold on data that has the
production fraud rate, so precision on live traffic matches what was promised.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

TRAINING_FRAUD_RATE = 0.5      # the balanced training sample
PRODUCTION_FRAUD_RATE = 0.01   # observed share of fraud in live payments
NAIVE_THRESHOLD = 0.5
# Synthetic model: the risk feature is N(0, 1) for normal payments and
# N(FRAUD_FEATURE_MEAN, 1) for fraud.
FRAUD_FEATURE_MEAN = 2.0


@dataclass(frozen=True)
class ScoredPayment:
    probability: float   # model output, calibrated for the TRAINING fraud rate
    is_fraud: bool


def sigmoid(log_odds: float) -> float:
    """Convert log-odds to a probability."""
    return 1.0 / (1.0 + math.exp(-log_odds))


def logit(probability: float) -> float:
    """Convert a probability (0 < p < 1) to log-odds."""
    return math.log(probability / (1.0 - probability))


def balanced_model_probability(risk_feature: float) -> float:
    """What a model fitted on 50/50 data outputs for one payment.

    For two unit-variance normals the exact log-likelihood ratio is
    mean * x - mean^2 / 2; with equal priors that is also the log-odds.
    """
    log_likelihood_ratio = FRAUD_FEATURE_MEAN * risk_feature - FRAUD_FEATURE_MEAN ** 2 / 2
    return sigmoid(log_likelihood_ratio)


def simulate_payments(rng: random.Random, count: int, fraud_rate: float) -> list[ScoredPayment]:
    """Draw scored payments with the given fraud share (seeded, so deterministic)."""
    payments = []
    for _ in range(count):
        is_fraud = rng.random() < fraud_rate
        mean = FRAUD_FEATURE_MEAN if is_fraud else 0.0
        feature = rng.gauss(mean, 1.0)
        payments.append(ScoredPayment(balanced_model_probability(feature), is_fraud))
    return payments


def adjust_for_base_rate(probability: float, training_rate: float, production_rate: float) -> float:
    """Prior-shift correction: rescale the odds by the change in class odds.

    The model learned P(fraud | x) under the training prior. Bayes' rule says
    the likelihood ratio is unchanged in production, only the prior odds
    differ, so we swap the training prior odds for the production ones.
    """
    training_prior_log_odds = logit(training_rate)
    production_prior_log_odds = logit(production_rate)
    # Clamp so a model output of exactly 0 or 1 does not break logit().
    clamped = min(max(probability, 1e-12), 1.0 - 1e-12)
    return sigmoid(logit(clamped) - training_prior_log_odds + production_prior_log_odds)


def precision_and_recall(payments: list[ScoredPayment], scores: list[float],
                         threshold: float) -> tuple[float, float]:
    """Precision and recall of alerting on score >= threshold."""
    true_alerts = 0
    false_alerts = 0
    total_fraud = 0
    for payment, score in zip(payments, scores):
        if payment.is_fraud:
            total_fraud += 1
        if score >= threshold:
            if payment.is_fraud:
                true_alerts += 1
            else:
                false_alerts += 1
    alerts = true_alerts + false_alerts
    precision = true_alerts / alerts if alerts else 0.0
    recall = true_alerts / total_fraud if total_fraud else 0.0
    return precision, recall


def naive_alert_scores(payments: list[ScoredPayment]) -> list[float]:
    """Original behaviour: use the balanced-sample probabilities as-is."""
    scores = []
    for payment in payments:
        scores.append(payment.probability)
    return scores


def corrected_alert_scores(payments: list[ScoredPayment]) -> list[float]:
    """Probabilities corrected to the production fraud rate."""
    scores = []
    for payment in payments:
        scores.append(adjust_for_base_rate(payment.probability, TRAINING_FRAUD_RATE,
                                           PRODUCTION_FRAUD_RATE))
    return scores


def choose_threshold_for_precision(payments: list[ScoredPayment], scores: list[float],
                                   target_precision: float) -> float | None:
    """Lowest threshold whose precision meets the target, for maximum recall.

    `payments` must have the production fraud rate (e.g. last month's labelled
    traffic); tuning on a balanced sample is exactly the original mistake.
    Returns None if no threshold reaches the target.
    """
    ranked = sorted(zip(scores, payments), key=_score_of, reverse=True)
    best_threshold = None
    true_alerts = 0
    alerts = 0
    # One pass from the highest score down: lowering the threshold to the
    # current score alerts on everything seen so far.
    for position, (score, payment) in enumerate(ranked):
        alerts += 1
        if payment.is_fraud:
            true_alerts += 1
        next_is_tie = position + 1 < len(ranked) and ranked[position + 1][0] == score
        # A threshold cannot split equal scores, so only judge at the end of a tie.
        if not next_is_tie and true_alerts / alerts >= target_precision:
            best_threshold = score
    return best_threshold


def _score_of(pair: tuple[float, ScoredPayment]) -> float:
    """Sort key: the score part of a (score, payment) pair."""
    return pair[0]
