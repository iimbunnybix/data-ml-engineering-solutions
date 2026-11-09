"""Fraud review queue grew tenfold after a model rollout because the threshold was not shipped with it.

The fraud service loaded the model file from one place and the "send to
manual review" threshold from a separate config store. Retraining changed
the score distribution, but the model was rolled out on its own and the old
threshold stayed, so the review rate jumped. This module ships the model and
its threshold as one versioned bundle with a fingerprint of the model
weights, verifies the bundle on deploy (fingerprint and review rate on a
reference sample) and swaps the whole bundle at once.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

TARGET_REVIEW_RATE = 0.02      # the review team can handle 2% of transactions
REVIEW_RATE_TOLERANCE = 0.005  # deploy fails if the sample rate is off by more than this


class BundleError(Exception):
    """Raised when a model bundle is inconsistent and must not serve traffic."""


@dataclass(frozen=True)
class LinearModel:
    """A tiny logistic-regression model: score = sigmoid(weights . features + bias)."""

    version: str
    weights: tuple[float, ...]
    bias: float

    def score(self, features: tuple[float, ...]) -> float:
        """Return a fraud score between 0 and 1."""
        total = self.bias
        for weight, value in zip(self.weights, features):
            total += weight * value
        return 1.0 / (1.0 + math.exp(-total))


@dataclass(frozen=True)
class ModelBundle:
    """Everything that must change together when a model is deployed."""

    model: LinearModel
    threshold: float
    expected_review_rate: float
    model_fingerprint: str


def fingerprint(model: LinearModel) -> str:
    """Hash the model's parameters so a bundle can prove which weights it was tuned for."""
    payload = json.dumps({"weights": list(model.weights), "bias": model.bias})
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def threshold_for_review_rate(scores: list[float], review_rate: float) -> float:
    """Pick the score above which roughly review_rate of the sample falls."""
    ranked = sorted(scores, reverse=True)
    flagged_count = max(1, int(len(ranked) * review_rate))
    # Halfway between the last flagged score and the first unflagged one, so
    # the sample gives the requested rate with ">=" or ">".
    return (ranked[flagged_count - 1] + ranked[flagged_count]) / 2


def review_rate(model: LinearModel, threshold: float, sample: list[tuple[float, ...]]) -> float:
    """Share of the sample this model and threshold would send to review."""
    flagged = 0
    for features in sample:
        if model.score(features) >= threshold:
            flagged += 1
    return flagged / len(sample)


class NaiveFraudService:
    """Original service: model and threshold are loaded and updated independently."""

    def __init__(self, model: LinearModel, threshold: float):
        self.model = model
        self.threshold = threshold

    def roll_out_model(self, model: LinearModel) -> None:
        """Model deploy pipeline: swaps the model file, never touches the config."""
        self.model = model

    def set_threshold(self, threshold: float) -> None:
        """Config store update, done by a different team on a different day."""
        self.threshold = threshold

    def needs_review(self, features: tuple[float, ...]) -> bool:
        return self.model.score(features) >= self.threshold


def build_bundle(model: LinearModel, tuning_sample: list[tuple[float, ...]],
                 target_rate: float = TARGET_REVIEW_RATE) -> ModelBundle:
    """Tune the threshold for this exact model and package them together."""
    scores = [model.score(features) for features in tuning_sample]
    threshold = threshold_for_review_rate(scores, target_rate)
    return ModelBundle(model=model, threshold=threshold,
                       expected_review_rate=target_rate,
                       model_fingerprint=fingerprint(model))


def verify_bundle(bundle: ModelBundle, reference_sample: list[tuple[float, ...]]) -> None:
    """Refuse a bundle whose threshold was not tuned for its model.

    The fingerprint catches a threshold copied from another model version;
    the review-rate check catches one tuned on unrepresentative data.
    """
    if fingerprint(bundle.model) != bundle.model_fingerprint:
        raise BundleError(f"threshold was tuned for different weights than "
                          f"model {bundle.model.version}")
    rate = review_rate(bundle.model, bundle.threshold, reference_sample)
    if abs(rate - bundle.expected_review_rate) > REVIEW_RATE_TOLERANCE:
        raise BundleError(f"model {bundle.model.version} would review {rate:.2%} "
                          f"of the reference sample, expected "
                          f"{bundle.expected_review_rate:.2%}")


class FraudService:
    """Fixed service: serves exactly one verified bundle at a time."""

    def __init__(self, bundle: ModelBundle, reference_sample: list[tuple[float, ...]]):
        self.reference_sample = reference_sample
        verify_bundle(bundle, reference_sample)
        self.bundle = bundle

    def deploy(self, bundle: ModelBundle) -> None:
        """Verify first, then swap the whole bundle in one assignment.

        A failed check leaves the old bundle serving, so a bad deploy is
        rejected instead of half-applied.
        """
        verify_bundle(bundle, self.reference_sample)
        self.bundle = bundle

    def needs_review(self, features: tuple[float, ...]) -> bool:
        # Read the bundle once so the model and threshold come from the same version.
        bundle = self.bundle
        return bundle.model.score(features) >= bundle.threshold
