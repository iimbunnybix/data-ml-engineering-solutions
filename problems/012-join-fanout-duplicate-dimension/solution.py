"""Catch row fan-out when a "lookup" table unexpectedly has duplicate keys.

A fact table (orders) is enriched by joining a dimension table (customers) on
``customer_id``. The pipeline assumes the dimension has one row per key. When
an upstream re-ingest leaves two rows for some customers, every order of those
customers is duplicated by the join and every downstream SUM is inflated, with
no error anywhere.

``naive_left_join`` is the original join. The fix:

* ``validated_join`` declares the expected cardinality (``many_to_one``,
  ``one_to_one``...) and raises ``CardinalityError`` with sample offending
  keys if the data violates it (the same idea as pandas'
  ``merge(validate="m:1")`` or dbt's ``unique`` test);
* ``dedupe_latest`` is an *explicit* resolution policy (keep the most recently
  updated row per key) for when duplicates are expected;
* ``assert_measure_preserved`` reconciles a measure before and after the join,
  a cheap guard that catches fan-out regardless of its cause.
"""
from __future__ import annotations

from collections import Counter
from decimal import Decimal
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

Record = Dict[str, object]

CARDINALITIES = {"one_to_one", "one_to_many", "many_to_one", "many_to_many"}


class CardinalityError(ValueError):
    def __init__(self, side: str, key: str, duplicates: Mapping[object, int]) -> None:
        sample = dict(list(duplicates.items())[:5])
        super().__init__(f"{side} side has {len(duplicates)} duplicate value(s) of "
                         f"{key!r}; sample {{key: count}} = {sample}")
        self.side, self.key, self.duplicates = side, key, dict(duplicates)


class ReconciliationError(AssertionError):
    pass


def _index(rows: Iterable[Record], key: str) -> Dict[object, List[Record]]:
    idx: Dict[object, List[Record]] = {}
    for r in rows:
        idx.setdefault(r[key], []).append(r)
    return idx


def naive_left_join(left: Sequence[Record], right: Sequence[Record], key: str) -> List[Record]:
    """Left join exactly as SQL does it: one output row per matching pair.

    As in SQL, a NULL (``None``) key never matches anything.
    """
    idx = _index((r for r in right if r.get(key) is not None), key)
    out: List[Record] = []
    for l in left:
        matches = idx.get(l[key]) if l.get(key) is not None else None
        if not matches:
            out.append(dict(l))
        for m in matches or []:
            out.append({**m, **l})
    return out


def duplicate_keys(rows: Iterable[Record], key: str) -> Dict[object, int]:
    """Key values that appear more than once (NULL keys never match, so they are skipped)."""
    counts = Counter(r[key] for r in rows if r.get(key) is not None)
    return {k: c for k, c in counts.items() if c > 1}


def validated_join(left: Sequence[Record], right: Sequence[Record], key: str,
                   cardinality: str = "many_to_one") -> List[Record]:
    """Left join that refuses to run when the data breaks the declared cardinality."""
    if cardinality not in CARDINALITIES:
        raise ValueError(f"cardinality must be one of {sorted(CARDINALITIES)}")
    left_unique = cardinality.startswith("one_")
    right_unique = cardinality.endswith("_one")
    if left_unique and (d := duplicate_keys(left, key)):
        raise CardinalityError("left", key, d)
    if right_unique and (d := duplicate_keys(right, key)):
        raise CardinalityError("right", key, d)
    return naive_left_join(left, right, key)


def dedupe_latest(rows: Sequence[Record], key: str, order_by: str) -> List[Record]:
    """Keep one row per key: the one with the greatest ``order_by`` value.

    Exact ties on ``order_by`` are ambiguous, so they raise instead of
    silently picking one.
    """
    best: Dict[object, Record] = {}
    tied: set = set()
    for r in rows:
        k = r[key]
        cur = best.get(k)
        if cur is None or r[order_by] > cur[order_by]:
            best[k] = r
            tied.discard(k)
        elif r[order_by] == cur[order_by] and r != cur:
            tied.add(k)
    if tied:
        raise CardinalityError("right", f"{key} (tied on {order_by})", {k: 2 for k in tied})
    return list(best.values())


def total(rows: Iterable[Record], measure: str) -> Decimal:
    return sum((Decimal(str(r[measure])) for r in rows), Decimal(0))


def assert_measure_preserved(before: Sequence[Record], after: Sequence[Record],
                             measure: str, tolerance: Decimal = Decimal(0)) -> None:
    """A left join to a many-to-one dimension must not change row count or sums."""
    if len(after) != len(before):
        raise ReconciliationError(f"row count changed {len(before)} -> {len(after)}")
    b, a = total(before, measure), total(after, measure)
    if abs(a - b) > tolerance:
        raise ReconciliationError(f"sum({measure}) changed {b} -> {a}")


def revenue_by(rows: Iterable[Record], dim: str, measure: str = "amount") -> Dict[Optional[object], Decimal]:
    out: Dict[Optional[object], Decimal] = {}
    for r in rows:
        out[r.get(dim)] = out.get(r.get(dim), Decimal(0)) + Decimal(str(r[measure]))
    return out
