import random
import unittest

from solution import (NaiveExtractor, SafeExtractor, SourceTable, Target, Write,
                      reconcile, replay)

DAY = 86_400


def make_writes(seed, n=20_000, ids=5_000, long_share=0.02, long_max=90):
    rng = random.Random(seed)
    versions = {}
    writes = []
    for _ in range(n):
        i = rng.randrange(ids)
        start = rng.randrange(DAY)
        if rng.random() < long_share:
            dur = rng.randint(10, long_max)    # batch jobs / slow transactions
        else:
            dur = rng.randint(0, 2)
        versions[i] = versions.get(i, 0) + 1
        writes.append(Write(i, versions[i], start, start + dur))
    # Versions must increase with commit order for the same id.
    by_id = {}
    for w in sorted(writes, key=lambda w: w.committed_at):
        by_id.setdefault(w.id, []).append(w)
    fixed = []
    for i, ws in by_id.items():
        for v, w in enumerate(ws, 1):
            fixed.append(Write(i, v, w.updated_at, w.committed_at))
    return fixed


def run_naive(writes, every=300, batch=500):
    src, tgt = SourceTable(writes), Target()
    replay(NaiveExtractor(src, tgt, batch), 0, DAY + 600, every)
    return src, tgt


def run_safe(writes, settle=120, every=300, batch=500):
    src, tgt = SourceTable(writes), Target()
    replay(SafeExtractor(src, tgt, settle, batch), 0, DAY + 600, every)
    return src, tgt


END = DAY + 600


class ReproduceMissedRows(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.writes = make_writes(1)
        cls.src, cls.tgt = run_naive(cls.writes)
        cls.rep = reconcile(cls.src, cls.tgt, END, END)

    def test_naive_loses_rows_permanently(self):
        self.assertFalse(self.rep.ok)
        # 12 of 4,907 rows wrong after a day: 2 never copied, 10 left at an
        # older version. A re-run of the naive job never fixes them.
        self.assertEqual((len(self.rep.missing), len(self.rep.stale)), (2, 10))
        self.assertEqual(len(self.src.snapshot(END)), 4_907)

    def test_error_rate_across_seeds(self):
        wrong = total = 0
        for seed in range(5):
            src, tgt = run_naive(make_writes(seed))
            rep = reconcile(src, tgt, END, END)
            wrong += len(rep.missing) + len(rep.stale)
            total += len(src.snapshot(END))
        self.assertEqual((wrong, total), (73, 24_513))     # 0.30%

    def test_late_commit_skipped(self):
        # A transaction that started before an extract run and committed after.
        writes = [Write(1, 1, 100, 100), Write(2, 1, 150, 400), Write(3, 1, 200, 200)]
        src, tgt = SourceTable(writes), Target()
        ex = NaiveExtractor(src, tgt)
        ex.run(300)                     # sees 1 and 3; watermark -> 200
        ex.run(600)                     # row 2 (updated_at 150) is now visible
        self.assertEqual(sorted(tgt.rows), [1, 3])

    def test_page_boundary_ties_dropped(self):
        writes = [Write(i, 1, 10, 10) for i in range(5)]   # same second
        src, tgt = SourceTable(writes), Target()
        NaiveExtractor(src, tgt, batch_size=2).run(100)
        self.assertEqual(len(tgt.rows), 2)


class SafeExtraction(unittest.TestCase):
    def test_no_rows_lost_when_settle_exceeds_longest_transaction(self):
        for seed in range(5):
            writes = make_writes(seed)
            src, tgt = run_safe(writes, settle=120)
            rep = reconcile(src, tgt, END, END - 120)
            self.assertTrue(rep.ok, (seed, len(rep.missing), len(rep.stale)))
            self.assertEqual(len(tgt.rows), len(src.snapshot(END)))

    def test_ties_across_pages_kept(self):
        writes = [Write(i, 1, 10, 10) for i in range(5)]
        src, tgt = SourceTable(writes), Target()
        SafeExtractor(src, tgt, settle_seconds=0, batch_size=2).run(100)
        self.assertEqual(sorted(tgt.rows), [0, 1, 2, 3, 4])

    def test_late_commit_within_settle_window_is_caught(self):
        writes = [Write(1, 1, 100, 100), Write(2, 1, 150, 400), Write(3, 1, 200, 200)]
        src, tgt = SourceTable(writes), Target()
        ex = SafeExtractor(src, tgt, settle_seconds=300)
        ex.run(300)                     # upper = 0: nothing settled yet
        self.assertEqual(tgt.rows, {})
        ex.run(600)                     # upper = 300: all three visible
        self.assertEqual(sorted(tgt.rows), [1, 2, 3])

    def test_rerun_is_idempotent(self):
        writes = make_writes(2, n=2_000, ids=500)
        src, tgt = run_safe(writes)
        before = dict(tgt.rows)
        ex = SafeExtractor(src, tgt, 120)
        ex.run(END)                     # full re-extract from scratch
        self.assertEqual(tgt.rows, before)

    def test_updates_reach_latest_version(self):
        writes = [Write(1, 1, 10, 10), Write(1, 2, 500, 501)]
        src, tgt = SourceTable(writes), Target()
        replay(SafeExtractor(src, tgt, 60), 0, 1200, 300)
        self.assertEqual(tgt.rows[1].version, 2)

    def test_reconcile_detects_transactions_longer_than_settle(self):
        writes = make_writes(1, long_max=900)   # some run up to 15 minutes
        src, tgt = run_safe(writes, settle=120)
        rep = reconcile(src, tgt, END, END - 120)
        self.assertFalse(rep.ok)
        self.assertTrue(rep.by_hour)
        self.assertEqual(sum(rep.by_hour.values()), len(rep.missing) + len(rep.stale))

    def test_long_transaction_of_short_statements_breaks_settle_and_reconcile_catches_it(self):
        # A batch transaction of 300 one-second UPDATEs stamps rows at t=1000
        # but commits at t=1300. statement_timeout / idle_in_transaction
        # timeouts of 60 s never fire, so only a total-transaction bound
        # (transaction_timeout) would have protected the 120 s settle window.
        writes = [Write(1, 1, 900, 900), Write(2, 1, 1000, 1300), Write(3, 1, 1050, 1050)]
        src, tgt = SourceTable(writes), Target()
        replay(SafeExtractor(src, tgt, 120), 0, 2000, 300)
        self.assertEqual(sorted(tgt.rows), [1, 3])
        self.assertEqual(reconcile(src, tgt, 2000, 1880).missing, [2])


if __name__ == "__main__":
    unittest.main()
