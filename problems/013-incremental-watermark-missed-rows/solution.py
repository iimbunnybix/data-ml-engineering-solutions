"""Incremental extraction that does not lose rows committed "in the past".

Typical incremental job: every few minutes, copy rows where
``updated_at > last_watermark`` from an OLTP database into the warehouse, then
set the watermark to the largest ``updated_at`` it saw. Two things break it:

1. ``updated_at`` is set when the write happens inside a transaction (in
   PostgreSQL ``now()`` is the transaction's start time), but the row only
   becomes visible when the transaction commits. A row with an older
   timestamp can appear *after* the job has moved its watermark past it.
2. Paging with ``updated_at > :ts LIMIT n`` drops rows that share the last
   timestamp of a page (second-precision timestamps tie constantly).

This module models the source table with explicit start and commit times and
provides the naive extractor, the fixed one and a reconciliation check.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Write:
    id: int
    version: int
    updated_at: int        # seconds; set at transaction start
    committed_at: int      # when the row becomes visible to readers


@dataclass(frozen=True)
class Row:
    id: int
    version: int
    updated_at: int


class SourceTable:
    """Current state of each row as of a given time (MVCC snapshot)."""

    def __init__(self, writes: list[Write]):
        self.writes = sorted(writes, key=lambda w: w.committed_at)
        self._rows: dict[int, Row] = {}
        self._pos = 0
        self._as_of = -1

    def snapshot(self, as_of: int) -> dict[int, Row]:
        """Rows visible at ``as_of``. Built incrementally when time moves
        forward (the common case in a replay), rebuilt if it moves back."""
        if as_of < self._as_of:
            self._rows, self._pos = {}, 0
        self._as_of = as_of
        while self._pos < len(self.writes) and self.writes[self._pos].committed_at <= as_of:
            w = self.writes[self._pos]
            cur = self._rows.get(w.id)
            if cur is None or w.version > cur.version:
                self._rows[w.id] = Row(w.id, w.version, w.updated_at)
            self._pos += 1
        return dict(self._rows)

    def query(self, as_of: int, after: tuple[int, int], upper_ts: int | None,
              limit: int, tie_safe: bool) -> list[Row]:
        """``WHERE (updated_at, id) > after [AND updated_at <= upper]
        ORDER BY updated_at, id LIMIT n``. With ``tie_safe=False`` only the
        timestamp is compared (``updated_at > :ts``), like the original job."""
        out = []
        for r in self.snapshot(as_of).values():
            key = (r.updated_at, r.id)
            if tie_safe:
                if key <= after:
                    continue
            elif r.updated_at <= after[0]:
                continue
            if upper_ts is not None and r.updated_at > upper_ts:
                continue
            out.append(r)
        out.sort(key=lambda r: (r.updated_at, r.id))
        return out[:limit]


class Target:
    """Warehouse table loaded by idempotent upsert (highest version wins)."""

    def __init__(self) -> None:
        self.rows: dict[int, Row] = {}

    def upsert(self, rows: list[Row]) -> None:
        for r in rows:
            cur = self.rows.get(r.id)
            if cur is None or r.version >= cur.version:
                self.rows[r.id] = r


class NaiveExtractor:
    """``updated_at > watermark``, watermark = max(updated_at) seen."""

    def __init__(self, source: SourceTable, target: Target, batch_size: int = 500):
        self.source, self.target, self.batch_size = source, target, batch_size
        self.watermark = -1

    def run(self, now: int) -> int:
        copied = 0
        while True:
            page = self.source.query(now, (self.watermark, 0), None, self.batch_size,
                                     tie_safe=False)
            if not page:
                return copied
            self.target.upsert(page)
            copied += len(page)
            self.watermark = page[-1].updated_at
            if len(page) < self.batch_size:
                return copied


class SafeExtractor:
    """Keyset cursor on (updated_at, id) and a settled upper bound.

    Only rows with ``updated_at <= now - settle_seconds`` are read. If no
    transaction runs longer than ``settle_seconds``, every row with such a
    timestamp has already committed, so the cursor never passes a row that is
    still invisible. Set ``settle_seconds`` above a bound on *total*
    transaction duration (``transaction_timeout`` on PostgreSQL 17+, or a
    monitor on ``pg_stat_activity.xact_start``). Per-statement and
    idle-in-transaction timeouts do not bound a transaction's length.
    """

    def __init__(self, source: SourceTable, target: Target, settle_seconds: int,
                 batch_size: int = 500):
        self.source, self.target = source, target
        self.settle, self.batch_size = settle_seconds, batch_size
        self.cursor: tuple[int, int] = (-1, -1)

    def run(self, now: int) -> int:
        upper = now - self.settle
        copied = 0
        while True:
            page = self.source.query(now, self.cursor, upper, self.batch_size, tie_safe=True)
            if not page:
                return copied
            self.target.upsert(page)          # idempotent: re-runs are safe
            copied += len(page)
            self.cursor = (page[-1].updated_at, page[-1].id)
            if len(page) < self.batch_size:
                return copied


@dataclass
class ReconcileReport:
    missing: list[int] = field(default_factory=list)
    stale: list[int] = field(default_factory=list)
    by_hour: dict[int, int] = field(default_factory=dict)   # hour -> problems

    @property
    def ok(self) -> bool:
        return not self.missing and not self.stale


def reconcile(source: SourceTable, target: Target, as_of: int,
              upper_ts: int) -> ReconcileReport:
    """Compare source and target for rows with ``updated_at <= upper_ts``.

    In production compare per-hour counts and checksums first, then drill down
    to ids only for the hours that differ.
    """
    rep = ReconcileReport()
    hours: dict[int, int] = defaultdict(int)
    for r in source.snapshot(as_of).values():
        if r.updated_at > upper_ts:
            continue
        t = target.rows.get(r.id)
        if t is None:
            rep.missing.append(r.id)
        elif t.version < r.version:
            rep.stale.append(r.id)
        else:
            continue
        hours[r.updated_at // 3600] += 1
    rep.missing.sort()
    rep.stale.sort()
    rep.by_hour = dict(sorted(hours.items()))
    return rep


def replay(extractor, start: int, end: int, every: int) -> None:
    """Run the extractor on a schedule from ``start`` to ``end`` inclusive."""
    for t in range(start, end + 1, every):
        extractor.run(t)
