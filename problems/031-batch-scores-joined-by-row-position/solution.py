"""Attach batch-scoring results to customers by key, not by row position.

The nightly churn-scoring job split the customer table into chunks, scored the
chunks in parallel and then attached the scores back with
``zip(customer_ids, scores)``. Parallel workers finish in any order, so the
score list came back in completion order, not input order, and most customers
silently received another customer's churn score.

The fix carries the entity key through the parallel step: every worker
returns ``(customer_id, score)`` pairs, results are joined by key, and the
join is checked (every input customer scored exactly once, no unknown keys)
before anything is written.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable

CHUNK_SIZE = 1_000
CHURN_WEIGHT_DAYS_INACTIVE = 0.08
CHURN_WEIGHT_TICKETS = 0.6
CHURN_BIAS = -4.0


@dataclass(frozen=True)
class Customer:
    customer_id: str
    days_inactive: int
    support_tickets: int


@dataclass(frozen=True)
class ScoredCustomer:
    customer_id: str
    churn_score: float


class JoinMismatchError(ValueError):
    """Raised when scores do not cover the input customers exactly once."""


def churn_score(customer: Customer) -> float:
    """Toy logistic churn model: more inactivity and tickets mean higher risk."""
    logit = (CHURN_BIAS + CHURN_WEIGHT_DAYS_INACTIVE * customer.days_inactive
             + CHURN_WEIGHT_TICKETS * customer.support_tickets)
    return 1.0 / (1.0 + math.exp(-logit))


def split_into_chunks(customers: list[Customer], chunk_size: int) -> list[list[Customer]]:
    """Cut the table into fixed-size chunks, one per worker task."""
    chunks = []
    for start in range(0, len(customers), chunk_size):
        chunks.append(customers[start:start + chunk_size])
    return chunks


def run_in_completion_order(chunks: list[list[Customer]], task: Callable[[list[Customer]], list],
                            rng: random.Random) -> list[list]:
    """Simulate a worker pool (``as_completed`` / ``imap_unordered``).

    Results come back in the order tasks *finish*, which depends on load and
    is effectively random. A seeded shuffle stands in for that here.
    """
    results = [task(chunk) for chunk in chunks]
    rng.shuffle(results)
    return results


# --------------------------------------------------------------- original bug
def naive_score_batch(customers: list[Customer], rng: random.Random,
                      chunk_size: int = CHUNK_SIZE) -> dict[str, float]:
    """Original job: workers return bare scores; zip them back by position."""
    def score_chunk(chunk: list[Customer]) -> list[float]:
        return [churn_score(c) for c in chunk]

    chunk_results = run_in_completion_order(split_into_chunks(customers, chunk_size), score_chunk, rng)
    scores = []
    for chunk_scores in chunk_results:
        scores.extend(chunk_scores)
    customer_ids = [c.customer_id for c in customers]
    # BUG: assumes scores are still in input order after the parallel step.
    return dict(zip(customer_ids, scores))


# ----------------------------------------------------------------------- fix
def score_chunk_with_keys(chunk: list[Customer]) -> list[ScoredCustomer]:
    """Worker task: return each score together with the key it belongs to."""
    return [ScoredCustomer(c.customer_id, churn_score(c)) for c in chunk]


def join_scores_by_key(customer_ids: list[str],
                       chunk_results: list[list[ScoredCustomer]]) -> dict[str, float]:
    """Join scores to customers by id and refuse a join that is not one-to-one."""
    expected = set(customer_ids)
    if len(expected) != len(customer_ids):
        raise JoinMismatchError("input has duplicate customer ids")
    scores: dict[str, float] = {}
    for chunk in chunk_results:
        for row in chunk:
            if row.customer_id not in expected:
                raise JoinMismatchError(f"score for unknown customer {row.customer_id!r}")
            if row.customer_id in scores:
                raise JoinMismatchError(f"customer {row.customer_id!r} scored twice")
            scores[row.customer_id] = row.churn_score
    missing = len(expected) - len(scores)
    if missing:
        raise JoinMismatchError(f"{missing} customers were not scored")
    return scores


def score_batch(customers: list[Customer], rng: random.Random,
                chunk_size: int = CHUNK_SIZE) -> dict[str, float]:
    """Fixed job: keyed results from the pool, validated join by customer id."""
    chunks = split_into_chunks(customers, chunk_size)
    chunk_results = run_in_completion_order(chunks, score_chunk_with_keys, rng)
    return join_scores_by_key([c.customer_id for c in customers], chunk_results)


# ------------------------------------------------------------------- helpers
def top_risk_ids(scores: dict[str, float], count: int) -> set[str]:
    """Customers the retention team would call: the ``count`` highest scores."""
    # Tie-break on id so the list does not depend on the order scores arrived in.
    ranked = sorted(scores, key=lambda customer_id: (-scores[customer_id], customer_id))
    return set(ranked[:count])


def wrong_score_share(scores: dict[str, float], customers: list[Customer]) -> float:
    """Share of customers whose attached score is not their own model score."""
    wrong = 0
    for customer in customers:
        if scores[customer.customer_id] != churn_score(customer):
            wrong += 1
    return wrong / len(customers)
