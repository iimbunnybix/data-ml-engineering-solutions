"""Report a model's accuracy on data that was not used to choose it.

A churn team ran a 100-configuration hyperparameter search and, for each
candidate, computed accuracy on the test set; they shipped the best one and
put its test accuracy in the launch review. Picking the maximum of 100 noisy
scores on the same data selects for luck, so the reported number was
optimistic and production accuracy came in clearly lower.

Fix: split three ways. Candidates are compared only on a validation set; the
single chosen model is scored once on a locked test set that refuses a second
evaluation, so the reported number is an unbiased estimate.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

FEATURE_COUNT = 5
SIGNAL_FEATURE = 0
LABEL_NOISE = 2.0  # std-dev of noise added to the signal feature before thresholding


@dataclass(frozen=True)
class Example:
    features: tuple[float, ...]
    label: int


@dataclass(frozen=True)
class LinearModel:
    """Predicts 1 when the weighted sum of features is positive."""
    weights: tuple[float, ...]

    def predict(self, features: tuple[float, ...]) -> int:
        score = 0.0
        for weight, value in zip(self.weights, features):
            score += weight * value
        return 1 if score > 0 else 0


@dataclass(frozen=True)
class SelectionReport:
    chosen_index: int
    selection_accuracy: float  # score on the data used to pick the model
    reported_accuracy: float   # the number that goes in the launch review


class TestSetReusedError(Exception):
    """The locked test set was asked to score a second model."""


def accuracy(model: LinearModel, examples: list[Example]) -> float:
    """Share of examples the model labels correctly."""
    if not examples:
        raise ValueError("cannot score an empty dataset")
    correct = 0
    for example in examples:
        if model.predict(example.features) == example.label:
            correct += 1
    return correct / len(examples)


def best_candidate(candidates: list[LinearModel], examples: list[Example]) -> tuple[int, float]:
    """Index and accuracy of the highest-scoring candidate (first one wins ties)."""
    if not candidates:
        raise ValueError("no candidates to choose from")
    best_index, best_accuracy = 0, accuracy(candidates[0], examples)
    for index in range(1, len(candidates)):
        score = accuracy(candidates[index], examples)
        if score > best_accuracy:
            best_index, best_accuracy = index, score
    return best_index, best_accuracy


# --- the original process ---------------------------------------------------
def naive_select_and_report(candidates: list[LinearModel], test: list[Example]) -> SelectionReport:
    """Original: choose on the test set and report that same (now biased) score."""
    index, test_accuracy = best_candidate(candidates, test)
    return SelectionReport(index, selection_accuracy=test_accuracy, reported_accuracy=test_accuracy)


# --- the fix ------------------------------------------------------------------
class LockedTestSet:
    """A test set that can score exactly one model.

    WHY: every extra look at the test set turns it into a selection set and
    biases it upward. Making reuse an error keeps that rule from being
    broken by accident in a notebook or a sweep script.
    """

    def __init__(self, examples: list[Example]) -> None:
        self._examples = list(examples)
        self._used = False

    def evaluate_once(self, model: LinearModel) -> float:
        """Score ``model``; any further call raises ``TestSetReusedError``."""
        if self._used:
            raise TestSetReusedError("test set already used; collect fresh data for another look")
        self._used = True
        return accuracy(model, self._examples)


def select_and_report(candidates: list[LinearModel], validation: list[Example],
                      locked_test: LockedTestSet) -> SelectionReport:
    """Choose on validation, then report the chosen model's single test score."""
    index, validation_accuracy = best_candidate(candidates, validation)
    test_accuracy = locked_test.evaluate_once(candidates[index])
    return SelectionReport(index, selection_accuracy=validation_accuracy, reported_accuracy=test_accuracy)


# --- simulation helpers used by the tests and README -------------------------
def make_examples(count: int, rng: random.Random) -> list[Example]:
    """Customers with 5 features; churn depends weakly on feature 0 only."""
    examples = []
    for _ in range(count):
        features = tuple(rng.gauss(0.0, 1.0) for _ in range(FEATURE_COUNT))
        noisy_signal = features[SIGNAL_FEATURE] + rng.gauss(0.0, LABEL_NOISE)
        examples.append(Example(features, 1 if noisy_signal > 0 else 0))
    return examples


def random_search_candidates(count: int, rng: random.Random) -> list[LinearModel]:
    """Stand-ins for the configurations of a random hyperparameter search."""
    candidates = []
    for _ in range(count):
        weights = tuple(rng.gauss(0.0, 1.0) for _ in range(FEATURE_COUNT))
        candidates.append(LinearModel(weights))
    return candidates
