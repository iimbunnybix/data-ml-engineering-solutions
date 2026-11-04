"""Fix a straggling shuffle join caused by one hot key, using key salting.

A distributed join hashes the join key to pick a partition (task). If one
key holds a large share of the rows (one marketplace merchant, a null/
"unknown" id, a bot user), every one of its rows lands on the same task.
That task runs many times longer than the rest and the whole stage waits.

This module simulates the shuffle so the effect can be measured, and
implements a salted join:

1. Detect hot keys from (sampled) key counts on the big side.
2. Big side: a hot key's rows get a salt ``0..S-1`` (round-robin, so it is
   deterministic), turning one key into S keys.
3. Small side: rows for hot keys are replicated once per salt.
4. Join on (key, salt) for hot keys; non-hot keys keep their plain key and
   are neither salted nor replicated.

The output is identical to the plain join; only the work distribution changes.
"""
from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Hashable, Iterable, Sequence

Row = tuple  # (key, value)


def stable_hash(key: Hashable) -> int:
    """Process-independent hash (Python's hash() is salted per process)."""
    return int.from_bytes(hashlib.md5(repr(key).encode()).digest()[:8], "big")


def partition_of(key: Hashable, partitions: int) -> int:
    return stable_hash(key) % partitions


@dataclass
class ShuffleStats:
    rows_per_task: list[int]

    @property
    def max(self) -> int:
        return max(self.rows_per_task)

    @property
    def median(self) -> float:
        s = sorted(self.rows_per_task)
        n = len(s)
        return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2

    @property
    def skew_ratio(self) -> float:
        """Slowest task / median task: stage time is set by the slowest."""
        return self.max / self.median if self.median else float("inf")


def find_hot_keys(keys: Iterable[Hashable], partitions: int,
                  factor: float = 2.0) -> set:
    """Keys holding more than ``factor`` × the fair share of one task."""
    counts = Counter(keys)
    total = sum(counts.values())
    threshold = factor * total / partitions
    return {k for k, c in counts.items() if c > threshold}


def _join_partitioned(big: Iterable[tuple], small: Iterable[tuple],
                      partitions: int) -> tuple[list[tuple], ShuffleStats]:
    """Hash-partition both sides on their join key, join each task locally.

    Rows are ``(join_key, payload)``. Task load = big rows + small rows.
    """
    big_parts: list[list] = [[] for _ in range(partitions)]
    small_parts: list[dict] = [defaultdict(list) for _ in range(partitions)]
    load = [0] * partitions
    for jk, payload in small:
        p = partition_of(jk, partitions)
        small_parts[p][jk].append(payload)
        load[p] += 1
    for jk, payload in big:
        p = partition_of(jk, partitions)
        big_parts[p].append((jk, payload))
        load[p] += 1
    out = []
    for p in range(partitions):
        lookup = small_parts[p]
        for jk, payload in big_parts[p]:
            for s in lookup.get(jk, ()):
                out.append((jk, payload, s))
    return out, ShuffleStats(load)


def plain_join(big: Sequence[Row], small: Sequence[Row],
               partitions: int) -> tuple[list[tuple], ShuffleStats]:
    """Inner join on key. Returns ((key, big_val, small_val) rows, stats)."""
    return _join_partitioned(big, small, partitions)


def salted_join(big: Sequence[Row], small: Sequence[Row], partitions: int,
                salts: int | None = None, hot_keys: set | None = None,
                ) -> tuple[list[tuple], ShuffleStats]:
    """Inner join with hot keys spread over ``salts`` sub-keys.

    ``salts`` defaults to 4 × partitions: salted sub-keys are themselves
    hash-partitioned, so more salts smooth out hash collisions. ``hot_keys`` defaults to
    ``find_hot_keys`` on the big side (pass a sample-based set in production).
    """
    if salts is None:
        salts = 4 * partitions
    if salts < 1:
        raise ValueError("salts must be >= 1")
    if hot_keys is None:
        hot_keys = find_hot_keys((k for k, _ in big), partitions)

    rr: dict = defaultdict(int)

    # Non-hot rows keep their plain key, so they shuffle exactly as before.
    # The payload carries the original key so results can be un-salted.
    def salted_big():
        for k, v in big:
            if k in hot_keys:
                salt = rr[k] % salts
                rr[k] += 1
                yield (k, salt), (k, v)
            else:
                yield k, (k, v)

    def salted_small():
        for k, v in small:
            if k in hot_keys:
                for salt in range(salts):
                    yield (k, salt), v
            else:
                yield k, v

    rows, stats = _join_partitioned(salted_big(), salted_small(), partitions)
    return [(k, bv, sv) for _jk, (k, bv), sv in rows], stats
