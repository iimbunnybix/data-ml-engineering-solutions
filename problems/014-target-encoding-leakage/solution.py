"""Leak-free target encoding for high-cardinality categorical features.

Target (mean) encoding replaces a category such as ``merchant_id`` with the
average label of that category. If the averages are computed over the same
rows the model is trained or validated on, each row's own label is part of its
feature value. For rare categories the feature *is* the label (a merchant seen
once with a fraud gets encoding 1.0), so offline metrics look excellent and
production performance does not.

This module provides:

* ``TargetEncoder`` - smoothed mean encoding fitted on one set of rows and
  applied to others (validation, test, serving).
* ``out_of_fold_encode`` - training-set encodings where each row's value is
  computed only from the other folds, so no row sees its own label.
* ``auc`` - ROC AUC (Mann-Whitney U with tie handling) for evaluation.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Hashable, Sequence


class TargetEncoder:
    """Smoothed mean encoding: ``(sum_y + m * prior) / (count + m)``.

    ``m`` (pseudo-count) pulls rare categories toward the global prior, so a
    category seen once cannot get an extreme value. Unseen categories get the
    prior.
    """

    def __init__(self, smoothing: float = 20.0):
        if smoothing < 0:
            raise ValueError("smoothing must be >= 0")
        self.m = smoothing
        # None until fit: an unfitted encoder must not quietly return 0.0
        # ("zero risk") for every category.
        self.prior: float | None = None
        self.table: dict[Hashable, float] = {}

    def fit(self, cats: Sequence[Hashable], labels: Sequence[int]) -> "TargetEncoder":
        if len(cats) != len(labels) or not cats:
            raise ValueError("cats and labels must be non-empty and the same length")
        self.prior = sum(labels) / len(labels)
        sums: dict[Hashable, float] = defaultdict(float)
        counts: dict[Hashable, int] = defaultdict(int)
        for c, y in zip(cats, labels):
            sums[c] += y
            counts[c] += 1
        self.table = {c: (sums[c] + self.m * self.prior) / (counts[c] + self.m)
                      for c in counts}
        return self

    def transform(self, cats: Sequence[Hashable]) -> list[float]:
        """Encode categories; unseen ones get the prior. Requires fit first."""
        if self.prior is None:
            raise RuntimeError("TargetEncoder must be fitted before transform")
        return [self.table.get(c, self.prior) for c in cats]


def kfold_indices(n: int, k: int, seed: int) -> list[list[int]]:
    if not 2 <= k <= n:
        raise ValueError("need 2 <= k <= n")
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    return [idx[i::k] for i in range(k)]


def out_of_fold_encode(cats: Sequence[Hashable], labels: Sequence[int], k: int = 5,
                       smoothing: float = 20.0, seed: int = 0) -> list[float]:
    """Encode each training row using an encoder fitted on the other k-1 folds."""
    # Check here, not only in fit: fold subsets are built from len(cats), so
    # extra labels would be silently ignored and rows encoded misaligned.
    if len(cats) != len(labels) or not cats:
        raise ValueError("cats and labels must be non-empty and the same length")
    out = [0.0] * len(cats)
    for fold in kfold_indices(len(cats), k, seed):
        held = set(fold)
        rest = [i for i in range(len(cats)) if i not in held]
        enc = TargetEncoder(smoothing).fit([cats[i] for i in rest], [labels[i] for i in rest])
        for i, v in zip(fold, enc.transform([cats[i] for i in fold])):
            out[i] = v
    return out


def auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """ROC AUC via average ranks (ties count half)."""
    pairs = sorted(zip(scores, labels))
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        for t in range(i, j + 1):
            ranks[t] = (i + j) / 2 + 1
        i = j + 1
    pos = sum(y for _, y in pairs)
    neg = len(pairs) - pos
    if pos == 0 or neg == 0:
        raise ValueError("AUC needs both classes")
    rank_sum = sum(r for r, (_, y) in zip(ranks, pairs) if y == 1)
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)
