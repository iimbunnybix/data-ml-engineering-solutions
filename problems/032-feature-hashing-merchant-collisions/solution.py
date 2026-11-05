"""Stop the hashing trick from merging trusted merchants with fraudulent ones.

The fraud model turned ``merchant_id`` into a feature with the hashing trick:
``bucket = hash(merchant_id) % 1024``, sized when there were a few hundred
merchants. After thousands of merchants were onboarded, many buckets held
several merchants, and the model learned one risk per *bucket*. Large, honest
merchants that shared a bucket with a fraud-ring merchant inherited its fraud
rate, and their customers' cards were declined.

The fix gives every merchant with enough training volume its own exact index
(a vocabulary), hashes only the long tail into a separate bucket range, and
reports collisions between high-volume merchants so the table size can be
checked whenever the merchant count grows.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass

HASH_BUCKETS = 1024
MIN_TRANSACTIONS_FOR_OWN_INDEX = 200
DECLINE_THRESHOLD = 0.05
PRIOR_STRENGTH = 100            # pseudo-transactions at the global fraud rate per index


@dataclass(frozen=True)
class Transaction:
    merchant_id: str
    is_fraud: bool


def stable_bucket(key: str, buckets: int) -> int:
    """Deterministic hash bucket.

    Python's built-in ``hash`` of a string is salted per process, so training
    and serving would disagree; a fixed digest is the same everywhere.
    """
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % buckets


# --------------------------------------------------------------- original bug
class NaiveHashedEncoder:
    """Original feature: every merchant is hashed into ``buckets`` slots."""

    def __init__(self, buckets: int = HASH_BUCKETS) -> None:
        self.buckets = buckets

    def index(self, merchant_id: str) -> int:
        """Feature index for a merchant (collisions are silent)."""
        return stable_bucket(merchant_id, self.buckets)


# ----------------------------------------------------------------------- fix
class VocabularyWithHashedTail:
    """Exact index for frequent merchants, hash buckets only for the rare tail."""

    def __init__(self, training: list[Transaction], tail_buckets: int = HASH_BUCKETS,
                 min_count: int = MIN_TRANSACTIONS_FOR_OWN_INDEX) -> None:
        counts = count_by_merchant(training)
        frequent = sorted(m for m, n in counts.items() if n >= min_count)
        self.vocabulary = {merchant: i for i, merchant in enumerate(frequent)}
        self.tail_buckets = tail_buckets

    def index(self, merchant_id: str) -> int:
        """Own index if the merchant is in the vocabulary, else a tail bucket.

        Tail buckets start after the vocabulary so the two ranges never overlap.
        """
        if merchant_id in self.vocabulary:
            return self.vocabulary[merchant_id]
        return len(self.vocabulary) + stable_bucket(merchant_id, self.tail_buckets)


def count_by_merchant(transactions: list[Transaction]) -> dict[str, int]:
    """Training transactions per merchant."""
    counts: dict[str, int] = defaultdict(int)
    for tx in transactions:
        counts[tx.merchant_id] += 1
    return dict(counts)


def heavy_merchant_collisions(merchant_counts: dict[str, int], encoder,
                              min_count: int = MIN_TRANSACTIONS_FOR_OWN_INDEX) -> dict[int, list[str]]:
    """Indices shared by two or more merchants that each have ``min_count`` transactions.

    Run this when the vocabulary is built or the merchant count grows; with the
    fixed encoder it must be empty.
    """
    by_index: dict[int, list[str]] = defaultdict(list)
    for merchant, count in merchant_counts.items():
        if count >= min_count:
            by_index[encoder.index(merchant)].append(merchant)
    shared = {}
    for index, merchants in by_index.items():
        if len(merchants) > 1:
            shared[index] = sorted(merchants)
    return shared


def expected_colliding_keys(keys: int, buckets: int) -> float:
    """Expected number of keys that share their bucket with at least one other key.

    Uniform hashing: a key is alone with probability (1 - 1/buckets)^(keys - 1).
    """
    if keys == 0:
        return 0.0
    alone_probability = (1 - 1 / buckets) ** (keys - 1)
    return keys * (1 - alone_probability)


# ------------------------------------------------------------- model and use
def train_index_fraud_rates(training: list[Transaction], encoder) -> dict[int, float]:
    """Learned risk per feature index: the smoothed fraud rate of its training traffic.

    This stands in for one model weight per index; any learner sees the same
    thing, because merchants that share an index are indistinguishable to it.
    Smoothing towards the global rate keeps one fraud at a tiny merchant from
    looking like a 100% fraud rate.
    """
    totals: dict[int, int] = defaultdict(int)
    frauds: dict[int, int] = defaultdict(int)
    for tx in training:
        index = encoder.index(tx.merchant_id)
        totals[index] += 1
        if tx.is_fraud:
            frauds[index] += 1
    global_rate = sum(frauds.values()) / max(1, sum(totals.values()))
    rates = {}
    for index, total in totals.items():
        rates[index] = (frauds[index] + PRIOR_STRENGTH * global_rate) / (total + PRIOR_STRENGTH)
    return rates


def is_declined(merchant_id: str, encoder, rates: dict[int, float]) -> bool:
    """Decline when the merchant's learned risk is above the threshold."""
    return rates.get(encoder.index(merchant_id), 0.0) > DECLINE_THRESHOLD


def declined_share(transactions: list[Transaction], encoder, rates: dict[int, float]) -> float:
    """Share of ``transactions`` the model declines (callers pick the slice)."""
    if not transactions:
        return 0.0
    declined = 0
    for tx in transactions:
        if is_declined(tx.merchant_id, encoder, rates):
            declined += 1
    return declined / len(transactions)
