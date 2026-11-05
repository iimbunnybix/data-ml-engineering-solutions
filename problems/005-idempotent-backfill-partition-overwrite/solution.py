"""Idempotent, restartable backfills via atomic partition overwrite.

The original job appended aggregated rows into ``daily_revenue``. When a
30-day backfill failed on day 18 and was simply re-run, days 1-17 were
written a second time and finance dashboards doubled for those days.

The fix makes every partition write a pure function of its inputs and
replaces the partition atomically (DELETE + INSERT in one transaction). A
run manifest records which partitions completed with which input
fingerprint, so a re-run skips finished days, recomputes days whose
source data changed, and can never double-count.
"""
from __future__ import annotations

import hashlib
import sqlite3
from datetime import date, timedelta
from typing import Callable, Iterable

SCHEMA = """
CREATE TABLE IF NOT EXISTS payments (
    payment_id TEXT PRIMARY KEY, day TEXT NOT NULL, merchant TEXT NOT NULL,
    amount_cents INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_payments_day ON payments(day);
CREATE TABLE IF NOT EXISTS daily_revenue (
    day TEXT NOT NULL, merchant TEXT NOT NULL, revenue_cents INTEGER NOT NULL,
    payments INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS backfill_manifest (
    job TEXT NOT NULL, day TEXT NOT NULL, input_fingerprint TEXT NOT NULL,
    rows_written INTEGER NOT NULL, PRIMARY KEY (job, day)
);
"""

JOB = "daily_revenue"


def connect(path: str = ":memory:") -> sqlite3.Connection:
    db = sqlite3.connect(path, isolation_level=None)
    db.executescript(SCHEMA)
    return db


def days(start: date, end: date) -> Iterable[date]:
    """Inclusive date range."""
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def _aggregate(db: sqlite3.Connection, day: str) -> list[tuple]:
    return db.execute(
        "SELECT ?, merchant, SUM(amount_cents), COUNT(*) FROM payments "
        "WHERE day = ? GROUP BY merchant ORDER BY merchant", (day, day)
    ).fetchall()


def input_fingerprint(db: sqlite3.Connection, day: str) -> str:
    """Order-independent hash of the source rows for one partition."""
    h = hashlib.sha256()
    for row in db.execute(
        "SELECT payment_id, merchant, amount_cents FROM payments WHERE day = ? "
        "ORDER BY payment_id", (day,)
    ):
        h.update(repr(row).encode())
    return h.hexdigest()


# --- original job ---------------------------------------------------------
def backfill_append(db, start: date, end: date,
                    fail_on: Callable[[date], bool] = lambda d: False) -> None:
    """The bug: INSERT-only. Any re-run of a day duplicates its rows."""
    for d in days(start, end):
        if fail_on(d):
            raise RuntimeError(f"executor lost while processing {d}")
        db.executemany("INSERT INTO daily_revenue VALUES (?, ?, ?, ?)",
                       _aggregate(db, d.isoformat()))


# --- fixed job ------------------------------------------------------------
def write_partition(db: sqlite3.Connection, day: str,
                    fail_mid_write: bool = False) -> int:
    """Atomically replace one day's partition; return rows written."""
    # Start the transaction BEFORE reading the source. The aggregate and the
    # fingerprint must see the same rows: if a late payment landed between
    # two autocommit reads, the manifest would claim it was counted and every
    # later run would skip the day, losing the payment for good.
    db.execute("BEGIN IMMEDIATE")
    try:
        rows = _aggregate(db, day)
        fp = input_fingerprint(db, day)
        db.execute("DELETE FROM daily_revenue WHERE day = ?", (day,))
        db.executemany("INSERT INTO daily_revenue VALUES (?, ?, ?, ?)", rows)
        if fail_mid_write:
            raise RuntimeError("executor lost mid-write")
        db.execute(
            "INSERT OR REPLACE INTO backfill_manifest VALUES (?, ?, ?, ?)",
            (JOB, day, fp, len(rows)),
        )
        db.execute("COMMIT")
    except BaseException:
        db.execute("ROLLBACK")
        raise
    return len(rows)


def backfill(db: sqlite3.Connection, start: date, end: date, *, force: bool = False,
             fail_on: Callable[[date], bool] = lambda d: False) -> dict[str, list[str]]:
    """Idempotent backfill over [start, end].

    A day is skipped when the manifest says it was written from identical
    input; it is (re)written when new, changed, or ``force`` is set.
    ``fail_on`` simulates a crash in the middle of that day's write.
    """
    report: dict[str, list[str]] = {"written": [], "skipped": []}
    for d in days(start, end):
        day = d.isoformat()
        prev = db.execute(
            "SELECT input_fingerprint FROM backfill_manifest WHERE job = ? AND day = ?",
            (JOB, day),
        ).fetchone()
        if not force and prev and prev[0] == input_fingerprint(db, day):
            report["skipped"].append(day)
            continue
        write_partition(db, day, fail_mid_write=fail_on(d))
        report["written"].append(day)
    return report


def total_revenue(db: sqlite3.Connection, start: date, end: date) -> int:
    return db.execute(
        "SELECT COALESCE(SUM(revenue_cents), 0) FROM daily_revenue WHERE day BETWEEN ? AND ?",
        (start.isoformat(), end.isoformat()),
    ).fetchone()[0]
