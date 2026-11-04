import random
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


if __name__ == "__main__":
    unittest.main()
