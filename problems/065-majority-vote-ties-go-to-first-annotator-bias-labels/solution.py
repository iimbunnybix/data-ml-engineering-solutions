"""Aggregate crowd labels without letting ties silently pick the first annotator.

Every comment in the moderation training set is labelled by two annotators and
the "majority" label is taken with ``Counter(labels).most_common(1)``. On a tie
``Counter`` keeps insertion order, so the first vote wins. The senior
annotator always labels first and misses some toxic comments, so every
disagreement became her label and the training set lost toxic examples.

The fix counts at most one vote per annotator, accepts a label only when it
has a strict plurality and enough votes, sends ties and under-labelled items
to an adjudication queue, and reports the agreement rate so a drop is visible.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

MIN_VOTES = 2
TIE = "tie"
TOO_FEW_VOTES = "too_few_votes"


@dataclass(frozen=True)
class Vote:
    item_id: str
    annotator_id: str
    label: str


@dataclass
class AggregationResult:
    labels: dict[str, str] = field(default_factory=dict)            # item -> accepted label
    adjudication: dict[str, str] = field(default_factory=dict)      # item -> reason
    unanimous_items: int = 0
    items_with_enough_votes: int = 0

    @property
    def agreement_rate(self) -> float:
        """Share of sufficiently labelled items on which every annotator agreed."""
        if self.items_with_enough_votes == 0:
            return 0.0
        return self.unanimous_items / self.items_with_enough_votes


# --- the original version ---------------------------------------------------
def naive_majority_labels(votes: list[Vote]) -> dict[str, str]:
    """Original job: most_common(1) per item; ties go to whoever voted first."""
    labels_per_item: dict[str, list[str]] = {}
    for vote in votes:
        labels_per_item.setdefault(vote.item_id, []).append(vote.label)
    result = {}
    for item_id, labels in labels_per_item.items():
        result[item_id] = Counter(labels).most_common(1)[0][0]
    return result


# --- the fix ----------------------------------------------------------------
def latest_vote_per_annotator(votes: list[Vote]) -> dict[str, dict[str, str]]:
    """item -> {annotator -> label}, keeping each annotator's latest vote only.

    Without this, an annotator who submits twice gets two votes and can break
    a tie on her own.
    """
    per_item: dict[str, dict[str, str]] = {}
    for vote in votes:
        per_item.setdefault(vote.item_id, {})[vote.annotator_id] = vote.label
    return per_item


def plurality_label(labels: list[str]) -> str | None:
    """The label with strictly more votes than any other, or None on a tie."""
    ranked = Counter(labels).most_common()
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return None
    return ranked[0][0]


def aggregate_labels(votes: list[Vote], min_votes: int = MIN_VOTES) -> AggregationResult:
    """Accepted labels plus an adjudication queue for ties and thin items."""
    result = AggregationResult()
    for item_id, by_annotator in latest_vote_per_annotator(votes).items():
        labels = list(by_annotator.values())
        if len(labels) < min_votes:
            result.adjudication[item_id] = TOO_FEW_VOTES
            continue
        result.items_with_enough_votes += 1
        if len(set(labels)) == 1:
            result.unanimous_items += 1
        winner = plurality_label(labels)
        if winner is None:
            # A tie carries no information about the true label; a human decides.
            result.adjudication[item_id] = TIE
        else:
            result.labels[item_id] = winner
    return result


def label_share(labels: dict[str, str], label: str) -> float:
    """Fraction of labelled items that carry ``label``."""
    if not labels:
        return 0.0
    matching = 0
    for value in labels.values():
        if value == label:
            matching += 1
    return matching / len(labels)
