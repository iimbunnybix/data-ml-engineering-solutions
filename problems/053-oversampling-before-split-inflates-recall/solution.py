"""Oversample the minority class only after the train/test split.

A fraud team balanced its data by duplicating fraud rows (random
oversampling) and only then split it into train and test. Copies of the same
fraud row landed on both sides, so a model that memorises training rows
"recognised" most test fraud and offline recall looked excellent. On fresh
production traffic recall was far lower. This module keeps the leaky
pipeline for comparison and fixes it: split first, oversample only the
training fold, and assert that no test row also appears in train.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

FRAUD = 1
LEGIT = 0
# Share of transactions that are fraud in the simulated traffic.
FRAUD_RATE = 0.03
# Fraud and legit feature centres overlap, so a perfect model is impossible.
FRAUD_CENTRE = (1.0, 1.0)
LEGIT_CENTRE = (0.0, 0.0)
FEATURE_SPREAD = 0.6
TEST_FRACTION = 0.25


@dataclass(frozen=True)
class Transaction:
    """One labelled transaction. row_id identifies the original source row."""
    row_id: int
    features: tuple[float, float]
    label: int


def simulate_transactions(count: int, rng: random.Random, first_id: int = 0) -> list[Transaction]:
    """Generate transactions where about FRAUD_RATE of them are fraud."""
    rows = []
    for offset in range(count):
        label = FRAUD if rng.random() < FRAUD_RATE else LEGIT
        centre = FRAUD_CENTRE if label == FRAUD else LEGIT_CENTRE
        features = (rng.gauss(centre[0], FEATURE_SPREAD), rng.gauss(centre[1], FEATURE_SPREAD))
        rows.append(Transaction(first_id + offset, features, label))
    return rows


def oversample_minority(rows: list[Transaction], rng: random.Random) -> list[Transaction]:
    """Duplicate random fraud rows until fraud and legit counts are equal."""
    fraud_rows = [row for row in rows if row.label == FRAUD]
    legit_count = len(rows) - len(fraud_rows)
    if not fraud_rows:
        return list(rows)
    balanced = list(rows)
    copies_needed = legit_count - len(fraud_rows)
    for _ in range(copies_needed):
        balanced.append(rng.choice(fraud_rows))
    rng.shuffle(balanced)
    return balanced


def random_split(rows: list[Transaction], test_fraction: float,
                 rng: random.Random) -> tuple[list[Transaction], list[Transaction]]:
    """Shuffle a copy of rows and cut it into (train, test)."""
    shuffled = list(rows)
    rng.shuffle(shuffled)
    test_size = math.ceil(len(shuffled) * test_fraction)
    return shuffled[test_size:], shuffled[:test_size]


def naive_prepare(rows: list[Transaction], rng: random.Random) -> tuple[list[Transaction], list[Transaction]]:
    """Original pipeline: balance the whole dataset, then split it (leaky)."""
    balanced = oversample_minority(rows, rng)
    return random_split(balanced, TEST_FRACTION, rng)


def fixed_prepare(rows: list[Transaction], rng: random.Random) -> tuple[list[Transaction], list[Transaction]]:
    """Fixed pipeline: split the original rows, then balance only the train fold.

    The test fold keeps the real class mix, so its metrics describe the
    traffic the model will actually see.
    """
    train, test = random_split(rows, TEST_FRACTION, rng)
    assert_no_shared_rows(train, test)
    return oversample_minority(train, rng), test


def shared_row_ids(train: list[Transaction], test: list[Transaction]) -> set[int]:
    """Source rows that appear in both folds (any copy counts)."""
    train_ids = {row.row_id for row in train}
    return {row.row_id for row in test if row.row_id in train_ids}


def assert_no_shared_rows(train: list[Transaction], test: list[Transaction]) -> None:
    """Fail the training job if a source row is on both sides of the split."""
    shared = shared_row_ids(train, test)
    if shared:
        raise ValueError(f"{len(shared)} source rows appear in both train and test")


class NearestNeighbourModel:
    """1-nearest-neighbour classifier: it memorises training rows exactly.

    Flexible models (deep trees, boosted ensembles) behave similarly on rows
    they have seen, which is why duplicated rows leak so badly.
    """

    def __init__(self, train: list[Transaction]) -> None:
        self.train = train

    def predict(self, features: tuple[float, float]) -> int:
        """Label of the closest training row."""
        best_label = LEGIT
        best_distance = math.inf
        for row in self.train:
            distance = math.dist(features, row.features)
            if distance < best_distance:
                best_distance = distance
                best_label = row.label
        return best_label


def fraud_recall(model: NearestNeighbourModel, rows: list[Transaction]) -> float:
    """Share of fraud rows in `rows` that the model flags as fraud."""
    fraud_rows = [row for row in rows if row.label == FRAUD]
    if not fraud_rows:
        raise ValueError("recall is undefined without any fraud rows")
    caught = 0
    for row in fraud_rows:
        if model.predict(row.features) == FRAUD:
            caught += 1
    return caught / len(fraud_rows)
