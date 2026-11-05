"""Encode categorical features with the vocabulary saved at training time.

The serving code one-hot encoded each request batch from the categories that
happened to be in that batch. A batch with no Android users produced columns
[ios, web] instead of [android, ios, web], so the model read "ios" as
"android" and every score was silently wrong; a new category added an extra
column that the model ignored. The fix saves the training vocabulary (with an
explicit unknown bucket) next to the model, encodes every batch with it, and
refuses to score when the column names or order do not match the model.
"""
from __future__ import annotations

from dataclasses import dataclass

UNKNOWN = "__unknown__"


def naive_batch_columns(values: list[str]) -> list[str]:
    """Original serving code: columns are whatever categories this batch contains."""
    return sorted(set(values))


def one_hot(values: list[str], columns: list[str]) -> list[list[float]]:
    """One row per value with a 1.0 in the column of its category."""
    rows = []
    for value in values:
        rows.append([1.0 if column == value else 0.0 for column in columns])
    return rows


@dataclass(frozen=True)
class Vocabulary:
    """Category list frozen at training time, saved alongside the model."""
    feature: str
    categories: tuple[str, ...]

    @classmethod
    def fit(cls, feature: str, training_values: list[str]) -> "Vocabulary":
        # Sorting makes the column order reproducible across training runs.
        return cls(feature, tuple(sorted(set(training_values))))

    def columns(self) -> list[str]:
        """Column names in model order; the unknown bucket is always last."""
        names = [f"{self.feature}={category}" for category in self.categories]
        names.append(f"{self.feature}={UNKNOWN}")
        return names

    def encode(self, values: list[str]) -> list[list[float]]:
        """One-hot encode with a fixed width; unseen values go to the unknown bucket."""
        known = set(self.categories)
        mapped = []
        for value in values:
            mapped.append(value if value in known else UNKNOWN)
        return one_hot(mapped, list(self.categories) + [UNKNOWN])

    def unknown_share(self, values: list[str]) -> float:
        """Share of values outside the vocabulary; a spike means retrain or fix upstream."""
        if not values:
            return 0.0
        known = set(self.categories)
        unknown_count = 0
        for value in values:
            if value not in known:
                unknown_count += 1
        return unknown_count / len(values)


class ColumnMismatch(ValueError):
    """Raised instead of scoring when features do not line up with the model."""


@dataclass(frozen=True)
class LinearModel:
    """A tiny trained model: one weight per named input column."""
    column_names: tuple[str, ...]
    weights: tuple[float, ...]
    bias: float

    def naive_score(self, row: list[float]) -> float:
        """Original scoring: zip() silently drops extra or missing columns."""
        total = self.bias
        for weight, value in zip(self.weights, row):
            total += weight * value
        return total

    def score(self, row: list[float], column_names: list[str]) -> float:
        """Score only when the caller's columns match the model's, name by name."""
        check_columns(list(self.column_names), column_names)
        if len(row) != len(self.weights):
            raise ColumnMismatch(f"row has {len(row)} values, model expects {len(self.weights)}")
        total = self.bias
        for weight, value in zip(self.weights, row):
            total += weight * value
        return total


def check_columns(expected: list[str], actual: list[str]) -> None:
    """Fail loudly on any difference in column count, names or order."""
    if len(expected) != len(actual):
        raise ColumnMismatch(f"expected {len(expected)} columns, got {len(actual)}")
    for position, (want, got) in enumerate(zip(expected, actual)):
        if want != got:
            raise ColumnMismatch(f"column {position}: expected {want!r}, got {got!r}")


def score_batch(model: LinearModel, vocabulary: Vocabulary, values: list[str]) -> list[float]:
    """Fixed serving path: encode with the saved vocabulary, then score with a column check."""
    columns = vocabulary.columns()
    scores = []
    for row in vocabulary.encode(values):
        scores.append(model.score(row, columns))
    return scores


def naive_score_batch(model: LinearModel, values: list[str]) -> list[float]:
    """Original serving path, kept for the before/after comparison."""
    columns = naive_batch_columns(values)
    scores = []
    for row in one_hot(values, columns):
        scores.append(model.naive_score(row))
    return scores
