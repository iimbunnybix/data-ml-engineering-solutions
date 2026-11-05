"""Turn a churn model's raw scores into probabilities before planning with them.

The retention team sized next month's call-centre staffing by adding up the
churn model's scores ("expected churners = sum of p") and offered a discount
to everyone with a score above 0.5 ("more likely than not to leave"). The
scores ranked customers well but were not probabilities: the model was
overconfident, so both the forecast and the 0.5 cut-off were badly off.

The fix fits an isotonic calibration map (pool-adjacent-violators) on a
held-out calibration month with known outcomes, applies it to new scores, and
measures calibration with expected calibration error (ECE) and Brier score.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass

ECE_BINS = 10


# ------------------------------------------------------- how scores were used
def naive_expected_count(scores: list[float]) -> float:
    """Original forecast: treat every raw score as a probability and add them up."""
    return sum(scores)


def count_above(probabilities: list[float], threshold: float) -> int:
    """How many customers a "probability above threshold" rule would select."""
    selected = 0
    for p in probabilities:
        if p > threshold:
            selected += 1
    return selected


# ---------------------------------------------------------------- calibration
@dataclass
class _Block:
    """A run of sorted scores that PAV has merged into one predicted rate."""
    lowest_score: float
    positives: float
    count: int

    @property
    def rate(self) -> float:
        return self.positives / self.count


def _blocks_of_equal_scores(scores: list[float], labels: list[int]) -> list[_Block]:
    """One block per distinct score, in increasing score order.

    Equal scores must get the same calibrated value, so they start merged.
    """
    pairs = sorted(zip(scores, labels))
    blocks: list[_Block] = []
    for score, label in pairs:
        if blocks and blocks[-1].lowest_score == score:
            blocks[-1].positives += label
            blocks[-1].count += 1
        else:
            blocks.append(_Block(score, label, 1))
    return blocks


def _pool_adjacent_violators(blocks: list[_Block]) -> list[_Block]:
    """Merge neighbouring blocks until the rates never decrease with the score.

    This is the least-squares non-decreasing fit: a higher raw score can never
    get a lower probability, so the model's ranking is kept.
    """
    merged: list[_Block] = []
    for block in blocks:
        merged.append(block)
        # WHY a loop: merging can create a new violation with the block before.
        while len(merged) >= 2 and merged[-2].rate >= merged[-1].rate:
            last = merged.pop()
            merged[-1].positives += last.positives
            merged[-1].count += last.count
    return merged


class IsotonicCalibrator:
    """Step-function map from raw score to observed outcome rate."""

    def __init__(self, scores: list[float], labels: list[int]) -> None:
        if not scores or len(scores) != len(labels):
            raise ValueError("need the same, non-zero number of scores and labels")
        if any(label not in (0, 1) for label in labels):
            raise ValueError("labels must be 0 or 1")
        blocks = _pool_adjacent_violators(_blocks_of_equal_scores(scores, labels))
        self.block_starts = [b.lowest_score for b in blocks]
        self.block_rates = [b.rate for b in blocks]

    def probability(self, score: float) -> float:
        """Calibrated probability of a raw score.

        Scores below the lowest calibration score use the first block, and
        scores above the highest use the last one (no extrapolation).
        """
        position = bisect_right(self.block_starts, score) - 1
        return self.block_rates[max(position, 0)]

    def probabilities(self, scores: list[float]) -> list[float]:
        return [self.probability(s) for s in scores]


# -------------------------------------------------------------------- metrics
def brier_score(probabilities: list[float], labels: list[int]) -> float:
    """Mean squared error between predicted probability and outcome (lower is better)."""
    total = 0.0
    for p, y in zip(probabilities, labels):
        total += (p - y) ** 2
    return total / len(labels)


def expected_calibration_error(probabilities: list[float], labels: list[int],
                               bins: int = ECE_BINS) -> float:
    """Weighted average gap between predicted and observed rate per score bin.

    Equal-width bins over [0, 1]; each bin's gap is weighted by its share of
    customers, so a few extreme scores cannot dominate.
    """
    predicted_sum = [0.0] * bins
    observed_sum = [0] * bins
    counts = [0] * bins
    for p, y in zip(probabilities, labels):
        index = min(int(p * bins), bins - 1)        # p == 1.0 goes in the top bin
        predicted_sum[index] += p
        observed_sum[index] += y
        counts[index] += 1
    error = 0.0
    for index in range(bins):
        if counts[index]:
            gap = abs(predicted_sum[index] - observed_sum[index]) / counts[index]
            error += gap * counts[index] / len(labels)
    return error
