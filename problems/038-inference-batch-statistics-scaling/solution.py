"""Scale serving features with statistics frozen at training time.

A default-risk model was trained on z-scored features, ``(value - mean) / std``.
The batch scoring job recomputed the mean and std from *each scoring batch*,
so a customer's score depended on who else was in the file: a batch of mostly
high-utilization applicants made everyone look average, and a one-row batch
crashed with a zero std. This module fits the scaler once on training data,
stores its statistics with the model, and applies the same numbers to every
row at serving time, so a row's score never depends on its neighbours.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean, pstdev
from typing import Sequence

Row = dict[str, float]

# A feature with no spread in training carries no information; scaling it by 1
# instead of 0 keeps the value finite and its contribution constant.
CONSTANT_FEATURE_SCALE = 1.0


@dataclass(frozen=True)
class FeatureStats:
    """Mean and standard deviation of one feature, measured on training data."""

    mean: float
    std: float


class LinearRiskModel:
    """A tiny logistic model with fixed weights over z-scored features."""

    def __init__(self, weights: dict[str, float], bias: float) -> None:
        self.weights = weights
        self.bias = bias

    def score(self, scaled_row: Row) -> float:
        """Probability of default for one already-scaled row."""
        logit = self.bias
        for feature, weight in self.weights.items():
            logit += weight * scaled_row[feature]
        return 1.0 / (1.0 + math.exp(-logit))


# --- Original job: statistics recomputed from the scoring batch ---------------

def naive_score_batch(model: LinearRiskModel, batch: Sequence[Row]) -> list[float]:
    """Z-score each feature with the batch's own mean/std, then score."""
    stats = {}
    for feature in model.weights:
        values = [row[feature] for row in batch]
        stats[feature] = FeatureStats(fmean(values), pstdev(values))
    scores = []
    for row in batch:
        # Dividing by the batch std raises ZeroDivisionError for 1-row batches.
        scaled = {name: (row[name] - s.mean) / s.std for name, s in stats.items()}
        scores.append(model.score(scaled))
    return scores


# --- Fixed job: statistics fitted once, shipped with the model ----------------

class FrozenScaler:
    """Standardizes features with statistics captured at training time."""

    def __init__(self, stats: dict[str, FeatureStats]) -> None:
        self.stats = stats

    @classmethod
    def fit(cls, training_rows: Sequence[Row], features: Sequence[str]) -> "FrozenScaler":
        """Measure mean and std of each feature on the training set only."""
        if not training_rows:
            raise ValueError("cannot fit a scaler on an empty training set")
        stats = {}
        for feature in features:
            values = [row[feature] for row in training_rows]
            std = pstdev(values)
            if std == 0:
                std = CONSTANT_FEATURE_SCALE
            stats[feature] = FeatureStats(fmean(values), std)
        return cls(stats)

    def transform(self, row: Row) -> Row:
        """Scale one row. Missing features are an error, not a silent 0."""
        scaled = {}
        for feature, feature_stats in self.stats.items():
            if feature not in row:
                raise KeyError(f"row is missing feature {feature!r}")
            scaled[feature] = (row[feature] - feature_stats.mean) / feature_stats.std
        return scaled


def score_batch(model: LinearRiskModel, scaler: FrozenScaler, batch: Sequence[Row]) -> list[float]:
    """Score every row independently with the frozen training statistics."""
    missing = set(model.weights) - set(scaler.stats)
    if missing:
        raise ValueError(f"scaler has no statistics for {sorted(missing)}")
    return [model.score(scaler.transform(row)) for row in batch]
