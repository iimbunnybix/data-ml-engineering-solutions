import os
import random
import sqlite3
import tempfile
import unittest
from datetime import date

from solution import (backfill, backfill_append, connect, days, total_revenue,
                      write_partition)

START, END = date(2026, 9, 1), date(2026, 9, 30)
FAIL_DAY = date(2026, 9, 18)


def seeded_db(seed=7):
    db = connect()
    rng = random.Random(seed)
    rows, n = [], 0
    for d in days(START, END):
        for _ in range(rng.randint(20, 40)):
            n += 1
            rows.append((f"p{n}", d.isoformat(), rng.choice("ABCDE"), rng.randint(100, 50_000)))
    db.executemany("INSERT INTO payments VALUES (?, ?, ?, ?)", rows)
    true_total = sum(r[3] for r in rows)
    return db, true_total


def source_total(db, d):
    return db.execute("SELECT SUM(amount_cents) FROM payments WHERE day = ?",
                      (d.isoformat(),)).fetchone()[0]


class ReproduceAppendRerun(unittest.TestCase):
    def test_rerun_after_failure_double_counts_completed_days(self):
        db, true_total = seeded_db()
        with self.assertRaises(RuntimeError):
            backfill_append(db, START, END, fail_on=lambda d: d == FAIL_DAY)
        backfill_append(db, START, END)                       # "just re-run it"
        done_before_failure = sum(source_total(db, d) for d in days(START, date(2026, 9, 17)))
        self.assertEqual(total_revenue(db, START, END), true_total + done_before_failure)


class IdempotentBackfill(unittest.TestCase):
    def setUp(self):
        self.db, self.true_total = seeded_db()

    def test_rerun_after_mid_write_crash_is_exact(self):
        with self.assertRaises(RuntimeError):
            backfill(self.db, START, END, fail_on=lambda d: d == FAIL_DAY)
        # The crashed day left nothing half-written.
        self.assertEqual(total_revenue(self.db, FAIL_DAY, FAIL_DAY), 0)
        report = backfill(self.db, START, END)
        self.assertEqual(len(report["skipped"]), 17)          # days 1-17 already done
        self.assertEqual(len(report["written"]), 13)          # days 18-30
        self.assertEqual(total_revenue(self.db, START, END), self.true_total)

    def test_full_rerun_is_a_noop(self):
        backfill(self.db, START, END)
        snapshot = self.db.execute("SELECT * FROM daily_revenue ORDER BY day, merchant").fetchall()
        report = backfill(self.db, START, END)
        self.assertEqual(report["written"], [])
        self.assertEqual(
            self.db.execute("SELECT * FROM daily_revenue ORDER BY day, merchant").fetchall(),
            snapshot)

    def test_late_arriving_source_data_recomputes_only_that_day(self):
        backfill(self.db, START, END)
        self.db.execute("INSERT INTO payments VALUES ('late1', '2026-09-05', 'A', 12345)")
        report = backfill(self.db, START, END)
        self.assertEqual(report["written"], ["2026-09-05"])
        self.assertEqual(total_revenue(self.db, START, END), self.true_total + 12345)

    def test_deleted_source_rows_are_removed_from_output(self):
        backfill(self.db, START, END)
        self.db.execute("DELETE FROM payments WHERE day = '2026-09-10'")
        backfill(self.db, START, END)
        self.assertEqual(total_revenue(self.db, date(2026, 9, 10), date(2026, 9, 10)), 0)

    def test_force_rewrites_but_stays_exact(self):
        backfill(self.db, START, END)
        report = backfill(self.db, START, END, force=True)
        self.assertEqual(len(report["written"]), 30)
        self.assertEqual(total_revenue(self.db, START, END), self.true_total)

    def test_overlapping_ranges_do_not_double_count(self):
        backfill(self.db, START, date(2026, 9, 20))
        backfill(self.db, date(2026, 9, 10), END, force=True)
        self.assertEqual(total_revenue(self.db, START, END), self.true_total)

    def test_partition_write_is_atomic(self):
        write_partition(self.db, "2026-09-03")
        before = total_revenue(self.db, date(2026, 9, 3), date(2026, 9, 3))
        with self.assertRaises(RuntimeError):
            write_partition(self.db, "2026-09-03", fail_mid_write=True)
        self.assertEqual(total_revenue(self.db, date(2026, 9, 3), date(2026, 9, 3)), before)

    def test_empty_day_produces_empty_partition(self):
        report = backfill(self.db, date(2026, 10, 1), date(2026, 10, 1))
        self.assertEqual(report["written"], ["2026-10-01"])
        self.assertEqual(backfill(self.db, date(2026, 10, 1), date(2026, 10, 1))["skipped"],
                         ["2026-10-01"])


class LatePaymentDuringWrite:
    """Connection wrapper that simulates another writer inserting a late
    payment right after the job has read the day's aggregate.

    The insert comes from a second connection with no busy timeout, so it
    fails with "database is locked" if the job holds a write transaction. In
    that case the payment is inserted after the job commits instead, which is
    what a real retrying writer would do.
    """

    def __init__(self, path: str, day: str) -> None:
        self.db = connect(path)
        self.other_writer = sqlite3.connect(path, isolation_level=None, timeout=0)
        self.day = day
        self.pending = True
        self.aggregate_was_read = False
        self.blocked = False

    def execute(self, sql, params=()):
        # The late payment arrives just before the job's next statement after
        # the aggregate read (once that read's cursor has been consumed).
        if self.aggregate_was_read:
            self.aggregate_was_read = False
            self._try_insert_late_payment()
        cursor = self.db.execute(sql, params)
        if self.pending and "SUM(amount_cents)" in sql:
            self.pending = False
            self.aggregate_was_read = True
        if self.blocked and sql == "COMMIT":
            self.blocked = False
            self._insert_late_payment()
        return cursor

    def executemany(self, sql, rows):
        return self.db.executemany(sql, rows)

    def _try_insert_late_payment(self):
        try:
            self._insert_late_payment()
        except sqlite3.OperationalError:
            self.blocked = True

    def _insert_late_payment(self):
        self.other_writer.execute(
            "INSERT INTO payments VALUES ('late', ?, 'A', 777)", (self.day,))


class LatePaymentDuringPartitionWrite(unittest.TestCase):
    def test_payment_inserted_mid_write_is_counted_on_the_next_run(self):
        day = date(2026, 9, 5)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "warehouse.db")
            db = LatePaymentDuringWrite(path, day.isoformat())
            db.execute("INSERT INTO payments VALUES ('p1', ?, 'A', 1000)", (day.isoformat(),))
            backfill(db, day, day)
            second = backfill(db, day, day)
            self.assertEqual(second["written"], [day.isoformat()])
            self.assertEqual(total_revenue(db, day, day), 1777)
            db.db.close()
            db.other_writer.close()


class LogicVersionInManifest(unittest.TestCase):
    def test_rerun_with_new_logic_version_rewrites_every_day(self):
        db, _ = seeded_db()
        backfill(db, START, END, logic_version="v1")
        report = backfill(db, START, END, logic_version="v2")
        self.assertEqual(len(report["written"]), 30)
        self.assertEqual(report["skipped"], [])

    def test_rerun_with_same_logic_version_still_skips(self):
        db, _ = seeded_db()
        backfill(db, START, END, logic_version="v2")
        report = backfill(db, START, END, logic_version="v2")
        self.assertEqual(len(report["skipped"]), 30)

    def test_manifest_records_the_logic_version_used(self):
        db, _ = seeded_db()
        backfill(db, START, START, logic_version="v7")
        stored = db.execute("SELECT logic_version FROM backfill_manifest").fetchall()
        self.assertEqual(stored, [("v7",)])


if __name__ == "__main__":
    unittest.main()
