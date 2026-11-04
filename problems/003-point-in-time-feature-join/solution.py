"""Point-in-time correct ("as-of") feature joins for building training sets.

Joining each training label to the *latest* value of a feature leaks
information from the future: the feature may have been updated *because of*
the outcome we are trying to predict. A point-in-time join instead picks,
for each label row, the most recent feature value that was actually
*available* at prediction time.
"""
from __future__ import annotations

import bisect
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class LabelRow:
    entity_id: str
    event_ts: int          # when the prediction would have been made (seconds)
    label: int


@dataclass(frozen=True)
class FeatureRow:
    entity_id: str
    feature_ts: int        # event time the value describes
    value: float
    ingested_ts: int | None = None   # when it landed in the feature store (None = feature_ts)

    @property
    def available_ts(self) -> int:
        return self.feature_ts if self.ingested_ts is None else self.ingested_ts


def latest_join(labels: Iterable[LabelRow], features: Iterable[FeatureRow],
                default: float | None = None) -> list[tuple[LabelRow, float | None]]:
    """The buggy approach: join every label to the entity's latest feature value."""
    latest: dict[str, FeatureRow] = {}
    for f in features:
        cur = latest.get(f.entity_id)
        if cur is None or f.feature_ts >= cur.feature_ts:
            latest[f.entity_id] = f
    return [(lbl, latest[lbl.entity_id].value if lbl.entity_id in latest else default)
            for lbl in labels]


class PointInTimeJoiner:
    """As-of join keyed by entity using binary search over availability time.

    Args:
        ttl_s: ignore feature values older than this at prediction time
            (stale values are worse than a known "missing" default).
        default: value used when no eligible feature value exists.

    Equality ``available_ts == event_ts`` counts as available; pass
    ``strict=True`` if your serving path cannot see same-second writes.
    """

    def __init__(self, features: Iterable[FeatureRow], ttl_s: int | None = None,
                 default: float | None = None, strict: bool = False):
        self.ttl_s = ttl_s
        self.default = default
        self.strict = strict
        by_entity: dict[str, list[FeatureRow]] = defaultdict(list)
        for f in features:
            by_entity[f.entity_id].append(f)
        self._ts: dict[str, list[int]] = {}
        self._rows: dict[str, list[FeatureRow]] = {}
        self._out_of_order: dict[str, bool] = {}
        for ent, rows in by_entity.items():
            # Sort by availability, then by feature_ts so the newest event-time wins ties.
            rows.sort(key=lambda r: (r.available_ts, r.feature_ts))
            self._rows[ent] = rows
            self._ts[ent] = [r.available_ts for r in rows]
            # True if some older value became available after a newer one (late backfill).
            self._out_of_order[ent] = any(
                rows[i].feature_ts > rows[i + 1].feature_ts for i in range(len(rows) - 1))

    def lookup(self, entity_id: str, as_of: int) -> float | None:
        ts = self._ts.get(entity_id)
        if not ts:
            return self.default
        idx = (bisect.bisect_left(ts, as_of) if self.strict
               else bisect.bisect_right(ts, as_of)) - 1
        if idx < 0:
            return self.default
        # A late-arriving older value may be "available" after a newer one; pick
        # the max feature_ts among everything available by as_of.
        best = max(self._rows[entity_id][: idx + 1], key=lambda r: r.feature_ts) \
            if self._out_of_order[entity_id] else self._rows[entity_id][idx]
        if self.ttl_s is not None and as_of - best.feature_ts > self.ttl_s:
            return self.default
        return best.value

    def join(self, labels: Iterable[LabelRow]) -> list[tuple[LabelRow, float | None]]:
        return [(lbl, self.lookup(lbl.entity_id, lbl.event_ts)) for lbl in labels]


def best_threshold_accuracy(pairs: list[tuple[LabelRow, float | None]]) -> float:
    """Accuracy of the best single 'value > t → 1' rule; a quick leakage smell test.

    If one feature alone almost perfectly predicts the label, suspect leakage.
    """
    xs = sorted({(v if v is not None else float("-inf")) for _, v in pairs})
    best = 0.0
    for t in [float("-inf"), *xs]:
        correct = sum(((v if v is not None else float("-inf")) > t) == bool(l.label)
                      for l, v in pairs)
        best = max(best, correct / len(pairs))
    return best
