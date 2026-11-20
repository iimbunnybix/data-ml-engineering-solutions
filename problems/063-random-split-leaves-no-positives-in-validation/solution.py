"""Split rare-event data so validation always contains positives.

A fraud model for a newly launched market had 2,000 labelled transactions and
only 10 frauds (0.5%). A plain random 80/20 split often left the validation
set with zero or one fraud, so recall was undefined or swung between 0% and
100% from run to run. The old metric code turned "no positives" into recall 0
for every candidate, the tie-break on false positives then picked a model that
never flags anything, and that model was promoted.

The fix splits each label separately (a seeded, stratified split), guarantees
a minimum number of positives on both sides, refuses to split when there are
too few positives, and reports recall as undefined instead of 0 or NaN, so
model selection stops rather than choosing on a meaningless number.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

# Fewer positives than this on either side makes recall too noisy to trust.
MIN_POSITIVES_PER_SPLIT = 2
DEFAULT_VALIDATION_FRACTION = 0.2


@dataclass(frozen=True)
class Transaction:
    transaction_id: int
    risk_score: float
    is_fraud: bool


@dataclass(frozen=True)
class MetricResult:
    """A metric value, or None with the reason it cannot be computed."""
    name: str
    value: float | None
    reason: str = ""

    @property
    def is_defined(self) -> bool:
        return self.value is not None


class InsufficientPositivesError(ValueError):
    """Too few positives to give both splits the required minimum."""


class UndefinedMetricError(ValueError):
    """Model selection was asked to compare models on an undefined metric."""


Model = Callable[[Transaction], bool]


def make_transactions(rows: int, frauds: int, seed: int) -> list[Transaction]:
    """Synthetic data: frauds tend to have high risk scores, but overlap with legit rows."""
    rng = random.Random(seed)
    transactions = []
    for transaction_id in range(rows):
        is_fraud = transaction_id < frauds
        low, high = (0.6, 1.0) if is_fraud else (0.0, 0.8)
        transactions.append(Transaction(transaction_id, rng.uniform(low, high), is_fraud))
    rng.shuffle(transactions)
    return transactions


def naive_random_split(rows: list[Transaction], validation_fraction: float,
                       seed: int) -> tuple[list[Transaction], list[Transaction]]:
    """The original split: shuffle all rows and cut, ignoring the label."""
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    validation_size = round(len(shuffled) * validation_fraction)
    return shuffled[validation_size:], shuffled[:validation_size]


def naive_recall(model: Model, rows: list[Transaction]) -> float:
    """The original metric: max(positives, 1) hides the zero-positive case as recall 0."""
    positives = [row for row in rows if row.is_fraud]
    caught = sum(1 for row in positives if model(row))
    return caught / max(len(positives), 1)


def false_positive_rate(model: Model, rows: list[Transaction]) -> float:
    """Share of legitimate transactions the model flags."""
    negatives = [row for row in rows if not row.is_fraud]
    if not negatives:
        return 0.0
    return sum(1 for row in negatives if model(row)) / len(negatives)


def naive_choose_model(models: dict[str, Model], validation: list[Transaction]) -> str:
    """Original promotion rule: highest recall, ties broken by lowest false-positive rate."""
    def sort_key(name: str) -> tuple[float, float, str]:
        model = models[name]
        return (-naive_recall(model, validation), false_positive_rate(model, validation), name)
    return sorted(models, key=sort_key)[0]


def validation_positive_count(total_positives: int, validation_fraction: float,
                              min_positives: int) -> int:
    """Positives to put in validation; raises if either side would get too few."""
    validation_positives = max(round(total_positives * validation_fraction), min_positives)
    train_positives = total_positives - validation_positives
    if train_positives < min_positives:
        raise InsufficientPositivesError(
            f"{total_positives} positives cannot give both splits at least {min_positives}; "
            "collect more labels or use cross-validation")
    return validation_positives


def stratified_split(rows: list[Transaction], validation_fraction: float, seed: int,
                     min_positives: int = MIN_POSITIVES_PER_SPLIT
                     ) -> tuple[list[Transaction], list[Transaction]]:
    """Split frauds and non-frauds separately so both sides get their share of positives."""
    # Sort by id first so the result depends only on the seed, not on input order.
    ordered = sorted(rows, key=lambda row: row.transaction_id)
    positives = [row for row in ordered if row.is_fraud]
    negatives = [row for row in ordered if not row.is_fraud]
    validation_positives = validation_positive_count(len(positives), validation_fraction,
                                                     min_positives)
    validation_negatives = round(len(negatives) * validation_fraction)
    rng = random.Random(seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)
    validation = positives[:validation_positives] + negatives[:validation_negatives]
    train = positives[validation_positives:] + negatives[validation_negatives:]
    return train, validation


def recall(model: Model, rows: list[Transaction]) -> MetricResult:
    """Recall, or an undefined result when there are no positives to recall."""
    positives = [row for row in rows if row.is_fraud]
    if not positives:
        return MetricResult("recall", None, "no positive rows in this split")
    caught = sum(1 for row in positives if model(row))
    return MetricResult("recall", caught / len(positives))


def choose_model(models: dict[str, Model], validation: list[Transaction]) -> str:
    """Promotion rule from before, but it refuses to rank models on an undefined recall."""
    recalls: dict[str, float] = {}
    for name, model in models.items():
        result = recall(model, validation)
        if not result.is_defined:
            raise UndefinedMetricError(f"cannot choose a model: recall for {name}: {result.reason}")
        recalls[name] = result.value
    def sort_key(name: str) -> tuple[float, float, str]:
        return (-recalls[name], false_positive_rate(models[name], validation), name)
    return sorted(models, key=sort_key)[0]


def flags_above(threshold: float) -> Model:
    """A simple model: flag transactions whose risk score reaches the threshold."""
    def model(row: Transaction) -> bool:
        return row.risk_score >= threshold
    return model


def never_flag(row: Transaction) -> bool:
    """A useless model that calls every transaction legitimate."""
    return False
