"""An accuracy-based release gate shipped a fraud model that flagged nothing.

Only 0.8% of transactions are fraud, so a model that predicts "legit" for
everything is 99.2% accurate. A retrain that collapsed to that constant
answer passed the "accuracy >= 99%" gate and went to production. The fix
gates on what the business needs from the rare class: recall (share of fraud
caught) and precision (share of flags that are real), and refuses to judge a
model on an evaluation set that has no fraud in it.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

EVAL_SIZE = 10_000
FRAUD_COUNT = 80                     # 0.8% fraud, typical for card payments
NAIVE_MIN_ACCURACY = 0.99
MIN_RECALL = 0.60                    # catch at least 60% of fraud
MIN_PRECISION = 0.30                 # at most ~2 false alarms per real catch
SEED = 7

FRAUD = 1
LEGIT = 0


@dataclass(frozen=True)
class Confusion:
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int


def make_labels(seed: int = SEED) -> list[int]:
    """Evaluation labels with exactly FRAUD_COUNT fraud cases, shuffled."""
    labels = [FRAUD] * FRAUD_COUNT + [LEGIT] * (EVAL_SIZE - FRAUD_COUNT)
    random.Random(seed).shuffle(labels)
    return labels


def collapsed_model(labels: list[int]) -> list[int]:
    """The broken retrain: predicts legit for every transaction."""
    return [LEGIT] * len(labels)


def useful_model(labels: list[int], caught: int = 60, false_alarms: int = 40) -> list[int]:
    """A working model: flags ``caught`` fraud cases and ``false_alarms`` legit ones."""
    predictions = [LEGIT] * len(labels)
    fraud_seen = 0
    legit_flagged = 0
    for index, label in enumerate(labels):
        if label == FRAUD and fraud_seen < caught:
            predictions[index] = FRAUD
            fraud_seen += 1
        elif label == LEGIT and legit_flagged < false_alarms:
            predictions[index] = FRAUD
            legit_flagged += 1
    return predictions


def confusion(labels: list[int], predictions: list[int]) -> Confusion:
    """Count the four outcomes, treating fraud as the positive class."""
    if len(labels) != len(predictions):
        raise ValueError("labels and predictions must be the same length")
    caught = false_alarms = missed = correct_legit = 0
    for label, predicted in zip(labels, predictions):
        if predicted == FRAUD and label == FRAUD:
            caught += 1
        elif predicted == FRAUD:
            false_alarms += 1
        elif label == FRAUD:
            missed += 1
        else:
            correct_legit += 1
    return Confusion(caught, false_alarms, missed, correct_legit)


def accuracy(counts: Confusion) -> float:
    """Share of all predictions that were right."""
    total = counts.true_positives + counts.false_positives + counts.false_negatives + counts.true_negatives
    return (counts.true_positives + counts.true_negatives) / total


def recall(counts: Confusion) -> float:
    """Share of real fraud the model caught."""
    actual_fraud = counts.true_positives + counts.false_negatives
    if actual_fraud == 0:
        raise ValueError("evaluation set has no fraud; recall is undefined")
    return counts.true_positives / actual_fraud


def precision(counts: Confusion) -> float:
    """Share of flags that were real fraud; 0.0 when nothing was flagged.

    A model that flags nothing has no precision to speak of. Treating it as
    0.0 (not 1.0 or an error) makes the gate reject it, which is the point.
    """
    flagged = counts.true_positives + counts.false_positives
    if flagged == 0:
        return 0.0
    return counts.true_positives / flagged


# ---------------------------------------------------------------- original
def naive_release_gate(labels: list[int], predictions: list[int]) -> bool:
    """Original gate: ship if accuracy is at least 99%."""
    return accuracy(confusion(labels, predictions)) >= NAIVE_MIN_ACCURACY


# ---------------------------------------------------------------- the fix
def release_gate(labels: list[int], predictions: list[int]) -> tuple[bool, str]:
    """Ship only if the model catches enough fraud without too many false alarms.

    Returns the decision and a reason that goes into the release log, so a
    rejected model says *why* it was rejected.
    """
    counts = confusion(labels, predictions)
    model_recall = recall(counts)            # raises if there is no fraud to judge
    model_precision = precision(counts)
    if model_recall < MIN_RECALL:
        return False, f"recall {model_recall:.2f} below {MIN_RECALL:.2f}"
    if model_precision < MIN_PRECISION:
        return False, f"precision {model_precision:.2f} below {MIN_PRECISION:.2f}"
    return True, f"recall {model_recall:.2f}, precision {model_precision:.2f}"
