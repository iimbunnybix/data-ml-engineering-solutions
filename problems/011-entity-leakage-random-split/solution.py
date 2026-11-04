"""Split training data by entity so offline metrics predict production.

When one entity (a customer, patient, device, merchant) contributes many rows,
a *row-level* random split puts some of that entity's rows in train and the
rest in test. A model that merely recognises the entity (from features that
are nearly constant per entity) then scores well offline and collapses on
entities it has never seen, which is exactly what production looks like.

This module provides:

* ``random_row_split``: the original split (kept to reproduce the problem);
* ``group_split``: deterministic split by a hash of the entity key, so all of
  an entity's rows land on one side, and the assignment is stable when new
  data arrives;
* ``group_kfold``: k folds with disjoint entities, for cross-validation;
* ``leakage_report`` / ``assert_no_entity_leakage``: a guard to run in the
  training pipeline before any metric is computed;
* ``make_ticket_dataset`` and ``NearestNeighbour``: a small synthetic dataset
  and model used to demonstrate the effect.
"""
from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Dict, Hashable, Iterator, List, Sequence, Tuple


@dataclass(frozen=True)
class Row:
    entity: Hashable
    features: Tuple[float, ...]
    label: int


Split = Tuple[List[Row], List[Row]]


def random_row_split(rows: Sequence[Row], test_fraction: float, seed: int) -> Split:
    """Shuffle rows and cut: entities end up on both sides."""
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    cut = int(len(rows) * (1 - test_fraction))
    return rows[:cut], rows[cut:]


def _bucket(entity: Hashable, salt: str) -> float:
    """Map an entity to [0, 1) with a stable hash (not Python's salted hash())."""
    digest = hashlib.sha256(f"{salt}:{entity}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2 ** 64


def group_split(rows: Sequence[Row], test_fraction: float, salt: str = "v1") -> Split:
    """All rows of an entity go to the same side; stable across runs and data growth."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    train, test = [], []
    for r in rows:
        (test if _bucket(r.entity, salt) < test_fraction else train).append(r)
    return train, test


def group_kfold(rows: Sequence[Row], k: int, salt: str = "v1") -> Iterator[Split]:
    """Yield ``k`` (train, validation) splits with disjoint entities."""
    if k < 2:
        raise ValueError("k must be at least 2")
    fold_of = lambda r: min(int(_bucket(r.entity, salt) * k), k - 1)
    for f in range(k):
        yield ([r for r in rows if fold_of(r) != f], [r for r in rows if fold_of(r) == f])


@dataclass(frozen=True)
class LeakageReport:
    shared_entities: int
    test_entities: int
    test_rows_with_seen_entity: float   # fraction of test rows


def leakage_report(train: Sequence[Row], test: Sequence[Row]) -> LeakageReport:
    seen = {r.entity for r in train}
    test_entities = {r.entity for r in test}
    hits = sum(1 for r in test if r.entity in seen)
    return LeakageReport(len(seen & test_entities), len(test_entities),
                         hits / len(test) if test else 0.0)


class EntityLeakageError(AssertionError):
    pass


def assert_no_entity_leakage(train: Sequence[Row], test: Sequence[Row]) -> None:
    rep = leakage_report(train, test)
    if rep.shared_entities:
        raise EntityLeakageError(
            f"{rep.shared_entities} entities appear in both train and test "
            f"({rep.test_rows_with_seen_entity:.0%} of test rows)")


class NearestNeighbour:
    """1-nearest-neighbour classifier: a model that can memorise entities."""

    def fit(self, rows: Sequence[Row]) -> "NearestNeighbour":
        self.rows = list(rows)
        return self

    def predict(self, features: Sequence[float]) -> int:
        best = min(self.rows, key=lambda r: math.dist(r.features, features))
        return best.label


def accuracy(model: NearestNeighbour, rows: Sequence[Row]) -> float:
    return sum(model.predict(r.features) == r.label for r in rows) / len(rows)


def make_ticket_dataset(customers: int = 200, tickets_per_customer: int = 10,
                        seed: int = 0) -> List[Row]:
    """Synthetic support tickets whose label is a property of the *customer*.

    Each customer has a fixed "fingerprint" (account size, plan, region...)
    that shows up, slightly noisy, in every ticket, and an escalation tendency
    drawn at random. Nothing in the features generalises to a new customer, so
    the honest accuracy on unseen customers is about 50%.
    """
    rng = random.Random(seed)
    rows: List[Row] = []
    for c in range(customers):
        fingerprint = [rng.gauss(0, 1) for _ in range(4)]
        escalates = rng.random() < 0.5
        for _ in range(tickets_per_customer):
            feats = tuple(x + rng.gauss(0, 0.05) for x in fingerprint)
            label = int(escalates) if rng.random() < 0.9 else int(not escalates)
            rows.append(Row(f"cust-{c}", feats, label))
    return rows
