"""Compute "previous transaction" features per customer, not per table row.

A fraud model uses two lag features: the amount of the customer's previous
transaction and the days since it. The original job sorted all transactions
by customer and time and took "the row above", so every customer's FIRST
transaction borrowed the last transaction of the customer sorted before them.
First purchases (the riskiest ones) looked like routine repeat purchases.

Fix: walk each customer's transactions separately. The first one gets no
previous value (None) plus an explicit is_first_transaction flag, so the model
can learn that "no history" is a signal instead of being fed a stranger's data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

SECONDS_PER_DAY = 86_400


@dataclass(frozen=True)
class Transaction:
    transaction_id: str
    customer_id: str
    timestamp: int  # seconds since epoch
    amount: float


@dataclass(frozen=True)
class LagFeatures:
    transaction_id: str
    previous_amount: Optional[float]      # None when there is no earlier transaction
    days_since_previous: Optional[float]  # None when there is no earlier transaction
    is_first_transaction: bool


def sort_key(transaction: Transaction) -> tuple[str, int, str]:
    """Customer, then time, then id so equal timestamps are ordered the same way every run."""
    return (transaction.customer_id, transaction.timestamp, transaction.transaction_id)


def days_between(earlier: Transaction, later: Transaction) -> float:
    """Gap between two transactions in days."""
    return (later.timestamp - earlier.timestamp) / SECONDS_PER_DAY


# --- the original job -----------------------------------------------------------
def naive_lag_features(transactions: list[Transaction]) -> list[LagFeatures]:
    """Original: 'previous' is simply the row above in the sorted table (the bug)."""
    ordered = sorted(transactions, key=sort_key)
    features = []
    for index, transaction in enumerate(ordered):
        if index == 0:
            features.append(LagFeatures(transaction.transaction_id, None, None, True))
            continue
        # BUG: the row above may belong to a different customer.
        previous = ordered[index - 1]
        features.append(LagFeatures(transaction.transaction_id, previous.amount,
                                    days_between(previous, transaction), False))
    return features


# --- the fix ---------------------------------------------------------------------
def group_by_customer(transactions: list[Transaction]) -> dict[str, list[Transaction]]:
    """Each customer's transactions, sorted by time (ties broken by id)."""
    groups: dict[str, list[Transaction]] = {}
    for transaction in transactions:
        groups.setdefault(transaction.customer_id, []).append(transaction)
    for customer_transactions in groups.values():
        customer_transactions.sort(key=sort_key)
    return groups


def lag_features_for_customer(customer_transactions: list[Transaction]) -> list[LagFeatures]:
    """Lag features inside one customer's history; the first row has no previous value."""
    features = []
    previous: Optional[Transaction] = None
    for transaction in customer_transactions:
        if previous is None:
            # WHY None and a flag, not 0: zero would look like a real (tiny) amount.
            features.append(LagFeatures(transaction.transaction_id, None, None, True))
        else:
            features.append(LagFeatures(transaction.transaction_id, previous.amount,
                                        days_between(previous, transaction), False))
        previous = transaction
    return features


def lag_features(transactions: list[Transaction]) -> list[LagFeatures]:
    """Lag features with the window partitioned by customer, in (customer, time) order."""
    features = []
    groups = group_by_customer(transactions)
    for customer_id in sorted(groups):
        features.extend(lag_features_for_customer(groups[customer_id]))
    return features


def count_borrowed_history(features: list[LagFeatures], first_ids: set[str]) -> int:
    """How many true first transactions were given a previous value anyway."""
    borrowed = 0
    for feature in features:
        if feature.transaction_id in first_ids and feature.previous_amount is not None:
            borrowed += 1
    return borrowed
