"""Take a uniform random sample from a stream that is too big to load.

The training-data job needed 50,000 rows from a daily export of tens of
millions of transactions. It took the *first* 50,000 rows. The export is
written region by region (and by time within each region), so the "sample"
contained only the first region's early-morning traffic, and the model
trained on it performed badly everywhere else.

The fix is reservoir sampling (Vitter's Algorithm R): one pass, memory for
only ``sample_size`` rows, and every row in the stream has the same chance
``sample_size / total_rows`` of being chosen, whatever order the file is in.
"""
from __future__ import annotations

import random
from collections import Counter
from typing import Iterable, TypeVar

Row = TypeVar("Row")


def naive_head_sample(rows: Iterable[Row], sample_size: int) -> list[Row]:
    """The original job: keep the first ``sample_size`` rows of the file."""
    sample = []
    for row in rows:
        if len(sample) == sample_size:
            break
        sample.append(row)
    return sample


def reservoir_sample(rows: Iterable[Row], sample_size: int, rng: random.Random) -> list[Row]:
    """Uniform sample of ``sample_size`` rows in a single pass.

    Keep the first ``sample_size`` rows. After that, row number ``seen``
    (1-based) replaces a random reservoir slot with probability
    ``sample_size / seen``. By induction every row seen so far is in the
    reservoir with exactly that probability, so the final sample is uniform.
    """
    if sample_size < 0:
        raise ValueError("sample_size must be >= 0")
    reservoir: list[Row] = []
    seen = 0
    for row in rows:
        seen += 1
        if len(reservoir) < sample_size:
            reservoir.append(row)
            continue
        # randrange(seen) is uniform over 0..seen-1; it lands inside the
        # reservoir (0..sample_size-1) with probability sample_size / seen.
        slot = rng.randrange(seen)
        if slot < sample_size:
            reservoir[slot] = row
    return reservoir


def share_by_key(rows: Iterable[dict], key: str) -> dict[str, float]:
    """Fraction of rows per value of ``key``, e.g. per region."""
    counts = Counter(row[key] for row in rows)
    total = sum(counts.values())
    shares = {}
    for value, count in counts.items():
        shares[value] = count / total
    return shares
