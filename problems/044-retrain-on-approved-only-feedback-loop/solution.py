"""Stop a fraud model from "forgetting" the traffic it blocks.

A fraud model declined a risky segment, so those transactions never got a
chargeback label. The nightly retrain only had labels for approved traffic,
saw no examples from that segment, fell back to the average fraud rate, and
the next model approved the segment again, which then let the fraud through.
The model flipped between blocking and approving it on every retrain. This
module keeps a small, random exploration slice of would-be declines, logs
each approval's probability (propensity), and retrains with inverse-propensity
weights so estimates reflect all traffic, not just what the last model let in.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

# Decline a segment when its estimated fraud rate is above this.
DECLINE_THRESHOLD = 0.10
# Share of would-be declines approved anyway, so they get labels.
EXPLORATION_RATE = 0.05
# Policy-approved transactions are approved with certainty.
CERTAIN_APPROVAL = 1.0


@dataclass(frozen=True)
class Transaction:
    """One scored transaction and what the policy did with it."""

    segment: str
    is_fraud: bool
    approved: bool
    # Probability that the policy approves a transaction like this one.
    propensity: float


def generate_traffic(rng: random.Random, fraud_rates: dict[str, float],
                     count_per_segment: int) -> list[tuple[str, bool]]:
    """Simulated incoming traffic as (segment, is_fraud) pairs."""
    traffic = []
    for segment, fraud_rate in fraud_rates.items():
        for _ in range(count_per_segment):
            traffic.append((segment, rng.random() < fraud_rate))
    return traffic


def apply_policy(traffic: list[tuple[str, bool]], model: dict[str, float],
                 rng: random.Random, exploration_rate: float) -> list[Transaction]:
    """Approve or decline each transaction and record the approval propensity."""
    decided = []
    for segment, is_fraud in traffic:
        if model[segment] <= DECLINE_THRESHOLD:
            decided.append(Transaction(segment, is_fraud, True, CERTAIN_APPROVAL))
            continue
        # The policy wants to decline. Approve a random slice anyway so the
        # segment keeps producing labels; log how likely that approval was.
        approved = rng.random() < exploration_rate
        decided.append(Transaction(segment, is_fraud, approved, exploration_rate))
    return decided


def labelled_only(transactions: list[Transaction]) -> list[Transaction]:
    """Chargeback labels exist only for transactions that were approved."""
    labelled = []
    for transaction in transactions:
        if transaction.approved:
            labelled.append(transaction)
    return labelled


def naive_retrain(labelled: list[Transaction], segments: list[str]) -> dict[str, float]:
    """Original retrain: plain fraud rate per segment on labelled data.

    A segment with no labels gets the overall labelled fraud rate, which is
    low precisely because the risky traffic was declined.
    """
    overall_frauds = sum(1 for item in labelled if item.is_fraud)
    overall_rate = overall_frauds / len(labelled)
    model = {}
    for segment in segments:
        rows = [item for item in labelled if item.segment == segment]
        if not rows:
            model[segment] = overall_rate
            continue
        model[segment] = sum(1 for item in rows if item.is_fraud) / len(rows)
    return model


def weighted_fraud_rate(rows: list[Transaction]) -> float | None:
    """Inverse-propensity weighted fraud rate, or None when there are no rows.

    An explored approval with propensity 0.05 stands in for 20 transactions
    like it, most of which were declined and never labelled.
    """
    total_weight = 0.0
    fraud_weight = 0.0
    for item in rows:
        weight = 1.0 / item.propensity
        total_weight += weight
        if item.is_fraud:
            fraud_weight += weight
    if total_weight == 0.0:
        return None
    return fraud_weight / total_weight


def weighted_retrain(labelled: list[Transaction], segments: list[str],
                     previous_model: dict[str, float]) -> dict[str, float]:
    """Retrain per segment with propensity weights.

    If a segment still has no labels, keep the previous estimate: "no data"
    must never be read as "average risk".
    """
    model = {}
    for segment in segments:
        rows = [item for item in labelled if item.segment == segment]
        estimate = weighted_fraud_rate(rows)
        model[segment] = previous_model[segment] if estimate is None else estimate
    return model


def approved_fraud_count(transactions: list[Transaction]) -> int:
    """Fraudulent transactions that were let through (the business loss)."""
    return sum(1 for item in transactions if item.approved and item.is_fraud)


def run_retrain_cycles(fraud_rates: dict[str, float], count_per_segment: int,
                       cycles: int, seed: int, use_fix: bool) -> list[dict]:
    """Simulate score -> label -> retrain cycles and report each one.

    The first model is trained on a launch period in which everything was
    approved, so it starts with unbiased labels for every segment.
    """
    rng = random.Random(seed)
    segments = list(fraud_rates)
    exploration_rate = EXPLORATION_RATE if use_fix else 0.0
    launch = generate_traffic(rng, fraud_rates, count_per_segment)
    model = naive_retrain(apply_policy(launch, dict.fromkeys(segments, 0.0), rng, 0.0),
                          segments)

    history = []
    for _ in range(cycles):
        traffic = generate_traffic(rng, fraud_rates, count_per_segment)
        decided = apply_policy(traffic, model, rng, exploration_rate)
        history.append({"model": model, "approved_fraud": approved_fraud_count(decided)})
        labelled = labelled_only(decided)
        if use_fix:
            model = weighted_retrain(labelled, segments, model)
        else:
            model = naive_retrain(labelled, segments)
    return history
