"""Correct click-through predictions for negatives that were downsampled in training.

An ads team trained its click model on all clicks but only 10% of the
non-click impressions, to make training ten times cheaper. Nobody undid the
sampling, so the model's probabilities described the sampled data, where
clicks are ten times more common, and the bidder paid for clicks that would
never happen. This module keeps the cheap sampling but either weights each
kept negative by 1 / sampling rate or applies the equivalent closed-form
correction to the predicted probability.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

NEGATIVE_SAMPLING_RATE = 0.10
# True click-through rate per ad slot in the simulated traffic.
TRUE_CTR_BY_SLOT: dict[str, float] = {"top": 0.05, "side": 0.02, "footer": 0.005}


@dataclass(frozen=True)
class Impression:
    slot: str
    clicked: bool


def simulate_impressions(count_per_slot: int, seed: int) -> list[Impression]:
    """Generate traffic with a known CTR per slot (seeded, so tests are repeatable)."""
    generator = random.Random(seed)
    impressions = []
    for slot, true_ctr in TRUE_CTR_BY_SLOT.items():
        for _ in range(count_per_slot):
            impressions.append(Impression(slot, generator.random() < true_ctr))
    return impressions


def downsample_negatives(impressions: list[Impression], rate: float,
                         seed: int) -> list[Impression]:
    """Keep every click and a random `rate` share of non-clicks."""
    if not 0 < rate <= 1:
        raise ValueError("sampling rate must be in (0, 1]")
    generator = random.Random(seed)
    kept = []
    for impression in impressions:
        if impression.clicked or generator.random() < rate:
            kept.append(impression)
    return kept


def naive_train(training_rows: list[Impression]) -> dict[str, float]:
    """Original model: click rate per slot on the training rows, as if unsampled."""
    clicks: dict[str, int] = {}
    totals: dict[str, int] = {}
    for row in training_rows:
        totals[row.slot] = totals.get(row.slot, 0) + 1
        clicks[row.slot] = clicks.get(row.slot, 0) + int(row.clicked)
    return {slot: clicks[slot] / totals[slot] for slot in totals}


def weighted_train(training_rows: list[Impression], rate: float) -> dict[str, float]:
    """Fix A: each kept negative stands for 1 / rate real negatives."""
    negative_weight = 1 / rate
    clicks: dict[str, float] = {}
    weighted_totals: dict[str, float] = {}
    for row in training_rows:
        weight = 1.0 if row.clicked else negative_weight
        weighted_totals[row.slot] = weighted_totals.get(row.slot, 0.0) + weight
        clicks[row.slot] = clicks.get(row.slot, 0.0) + (1.0 if row.clicked else 0.0)
    return {slot: clicks[slot] / weighted_totals[slot] for slot in weighted_totals}


def correct_for_sampling(sampled_probability: float, rate: float) -> float:
    """Fix B: map a probability learned on sampled data back to real traffic.

    Downsampling multiplies the odds of a click by 1 / rate, so dividing the
    odds by the same factor undoes it:  p = q / (q + (1 - q) / rate).
    """
    if not 0 < rate <= 1:
        raise ValueError("sampling rate must be in (0, 1]")
    if not 0 <= sampled_probability <= 1:
        raise ValueError("probability must be in [0, 1]")
    return sampled_probability / (sampled_probability + (1 - sampled_probability) / rate)


def corrected_model(sampled_model: dict[str, float], rate: float) -> dict[str, float]:
    """Apply correct_for_sampling to every prediction of a model trained on sampled rows."""
    return {slot: correct_for_sampling(probability, rate)
            for slot, probability in sampled_model.items()}


def bid_per_impression(predicted_ctr: float, value_per_click: float) -> float:
    """What the bidder will pay for one impression: expected value of a click."""
    return predicted_ctr * value_per_click
