"""Apply change-data-capture (CDC) events correctly when they arrive out of order.

The original bug: a CDC connector streamed row changes from the orders
database into the analytics warehouse. The sink simply applied each event in
the order it arrived: ``insert``/``update`` -> upsert, ``delete`` -> delete.
Retries after a sink timeout and a consumer-group rebalance re-delivered older
batches *after* newer ones. An old update then overwrote a newer one (orders
shown as "pending" long after they shipped), and an old update arriving after
a delete brought cancelled orders back to life.

Fix: every change carries its log sequence number (LSN), which is strictly
increasing per source database. The sink stores the LSN of the last change it
applied to each key, ignores any event whose LSN is not newer, and keeps
*tombstones* (deleted keys with their delete LSN) so a late update cannot
resurrect a deleted row. Tombstones are purged once no older event can still
be in flight.
"""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class ChangeEvent:
    op: str                 # "c" create, "u" update, "d" delete
    key: str
    lsn: int                # position in the source write-ahead log
    after: dict | None      # new row image (None for deletes)


class NaiveSink:
    """Arrival-order apply. Correct only if delivery is exactly-once and in order."""

    def __init__(self):
        self.rows: dict[str, dict] = {}

    def apply(self, ev: ChangeEvent) -> None:
        if ev.op == "d":
            self.rows.pop(ev.key, None)
        else:
            self.rows[ev.key] = dict(ev.after)

    def snapshot(self) -> dict[str, dict]:
        return dict(self.rows)


@dataclass
class _Entry:
    lsn: int
    row: dict | None        # None = tombstone


class VersionedSink:
    """LSN-guarded upserts and deletes; idempotent and order-insensitive per key."""

    def __init__(self):
        self.entries: dict[str, _Entry] = {}
        self.applied = 0
        self.skipped_stale = 0

    def apply(self, ev: ChangeEvent) -> bool:
        """Apply ``ev`` if it is newer than what we hold for its key. Returns True if applied.

        In SQL this is ``MERGE ... WHEN MATCHED AND src.lsn > tgt.lsn THEN UPDATE``,
        with deletes written as ``is_deleted = true`` rows instead of real deletes.
        """
        if ev.op not in ("c", "u", "d"):
            raise ValueError(f"unknown op {ev.op!r}")
        if ev.op != "d" and ev.after is None:
            raise ValueError("create/update without row image")
        cur = self.entries.get(ev.key)
        if cur is not None and ev.lsn <= cur.lsn:
            self.skipped_stale += 1           # duplicate or older than current state
            return False
        self.entries[ev.key] = _Entry(ev.lsn, None if ev.op == "d" else dict(ev.after))
        self.applied += 1
        return True

    def snapshot(self) -> dict[str, dict]:
        return {k: dict(e.row) for k, e in self.entries.items() if e.row is not None}

    def tombstones(self) -> int:
        return sum(1 for e in self.entries.values() if e.row is None)

    def purge_tombstones(self, safe_lsn: int) -> int:
        """Drop tombstones with lsn < ``safe_lsn``.

        ``safe_lsn`` must be a low watermark: the smallest LSN that could still
        be redelivered (e.g. the oldest uncommitted consumer offset). Purging
        above it reopens the resurrection bug.
        """
        dead = [k for k, e in self.entries.items() if e.row is None and e.lsn < safe_lsn]
        for k in dead:
            del self.entries[k]
        return len(dead)


# --- simulation helpers used by the tests and README ------------------------
def generate_changes(n_keys: int, n_events: int, seed: int) -> list[ChangeEvent]:
    """A realistic source history: creates, several updates, some deletes, re-creates."""
    rng = random.Random(seed)
    live: dict[str, int] = {}
    events = []
    for lsn in range(1, n_events + 1):
        key = f"order-{rng.randrange(n_keys)}"
        if key not in live:
            live[key] = 0
            events.append(ChangeEvent("c", key, lsn, {"status": "pending", "rev": 0}))
        elif rng.random() < 0.1:
            del live[key]
            events.append(ChangeEvent("d", key, lsn, None))
        else:
            live[key] += 1
            status = rng.choice(["paid", "packed", "shipped", "delivered"])
            events.append(ChangeEvent("u", key, lsn, {"status": status, "rev": live[key]}))
    return events


def source_state(events: list[ChangeEvent]) -> dict[str, dict]:
    """Ground truth: replay in LSN order exactly once."""
    sink = NaiveSink()
    for ev in sorted(events, key=lambda e: e.lsn):
        sink.apply(ev)
    return sink.snapshot()


def at_least_once_delivery(events: list[ChangeEvent], seed: int, batch: int = 50,
                           redeliver_prob: float = 0.3) -> list[ChangeEvent]:
    """Batches are delivered in order, but some earlier batches are re-sent later
    (sink timeout -> retry, rebalance -> replay from last committed offset)."""
    rng = random.Random(seed)
    batches = [events[i:i + batch] for i in range(0, len(events), batch)]
    out = []
    for i, b in enumerate(batches):
        out.extend(b)
        if i and rng.random() < redeliver_prob:
            out.extend(batches[rng.randrange(i)])   # replay an older batch
    return out
