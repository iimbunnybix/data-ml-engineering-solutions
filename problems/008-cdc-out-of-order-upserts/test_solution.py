import random
import unittest

from solution import (ChangeEvent, NaiveSink, VersionedSink, at_least_once_delivery,
                      generate_changes, parse_pg_lsn, source_state)


def mismatched_keys(snapshot, truth):
    keys = set(snapshot) | set(truth)
    return sorted(k for k in keys if snapshot.get(k) != truth.get(k))


class ReproduceStaleOverwrites(unittest.TestCase):
    def test_redelivered_batch_rolls_rows_back(self):
        events = generate_changes(n_keys=200, n_events=3000, seed=1)
        delivered = at_least_once_delivery(events, seed=2)
        sink = NaiveSink()
        for ev in delivered:
            sink.apply(ev)
        bad = mismatched_keys(sink.snapshot(), source_state(events))
        self.assertGreater(len(bad), 20)

    def test_late_update_resurrects_deleted_row(self):
        sink = NaiveSink()
        sink.apply(ChangeEvent("c", "o1", 1, {"status": "pending"}))
        sink.apply(ChangeEvent("d", "o1", 3, None))
        sink.apply(ChangeEvent("u", "o1", 2, {"status": "paid"}))   # retried, arrives last
        self.assertIn("o1", sink.snapshot())                         # cancelled order is back


class LsnGuardedApply(unittest.TestCase):
    def test_matches_source_under_redelivery(self):
        events = generate_changes(n_keys=200, n_events=3000, seed=1)
        delivered = at_least_once_delivery(events, seed=2)
        sink = VersionedSink()
        for ev in delivered:
            sink.apply(ev)
        self.assertEqual(mismatched_keys(sink.snapshot(), source_state(events)), [])
        self.assertEqual(sink.applied, len(events))
        self.assertEqual(sink.skipped_stale, len(delivered) - len(events))

    def test_any_permutation_converges(self):
        events = generate_changes(n_keys=30, n_events=500, seed=5)
        truth = source_state(events)
        for seed in range(5):
            shuffled = events + random.Random(seed).sample(events, 100)
            random.Random(seed + 100).shuffle(shuffled)
            sink = VersionedSink()
            for ev in shuffled:
                sink.apply(ev)
            self.assertEqual(sink.snapshot(), truth, seed)

    def test_tombstone_blocks_resurrection(self):
        sink = VersionedSink()
        sink.apply(ChangeEvent("c", "o1", 1, {"status": "pending"}))
        sink.apply(ChangeEvent("d", "o1", 3, None))
        self.assertFalse(sink.apply(ChangeEvent("u", "o1", 2, {"status": "paid"})))
        self.assertEqual(sink.snapshot(), {})
        self.assertEqual(sink.tombstones(), 1)

    def test_recreate_after_delete(self):
        sink = VersionedSink()
        sink.apply(ChangeEvent("d", "o1", 3, None))           # delete seen before create
        sink.apply(ChangeEvent("c", "o1", 1, {"v": 1}))
        sink.apply(ChangeEvent("c", "o1", 7, {"v": 2}))       # key reused later
        self.assertEqual(sink.snapshot(), {"o1": {"v": 2}})

    def test_duplicate_is_noop(self):
        sink = VersionedSink()
        ev = ChangeEvent("u", "o1", 4, {"status": "paid"})
        self.assertTrue(sink.apply(ev))
        self.assertFalse(sink.apply(ev))

    def test_row_image_is_copied(self):
        row = {"status": "paid"}
        sink = VersionedSink()
        sink.apply(ChangeEvent("c", "o1", 1, row))
        row["status"] = "mutated"
        self.assertEqual(sink.snapshot()["o1"]["status"], "paid")

    def test_invalid_events(self):
        sink = VersionedSink()
        with self.assertRaises(ValueError):
            sink.apply(ChangeEvent("x", "o1", 1, {}))
        with self.assertRaises(ValueError):
            sink.apply(ChangeEvent("u", "o1", 1, None))

    def test_textual_lsn_is_rejected_instead_of_compared_as_string(self):
        sink = VersionedSink()
        sink.apply(ChangeEvent("u", "o", 16, {"s": "shipped"}))
        with self.assertRaises(ValueError):
            sink.apply(ChangeEvent("u", "o", "0/9", {"s": "pending"}))
        self.assertEqual(sink.snapshot(), {"o": {"s": "shipped"}})

    def test_bool_and_negative_lsns_are_rejected(self):
        sink = VersionedSink()
        for bad_lsn in (True, -1):
            with self.assertRaises(ValueError):
                sink.apply(ChangeEvent("u", "o", bad_lsn, {"s": "paid"}))

    def test_parsed_pg_lsns_compare_in_wal_order(self):
        self.assertLess(parse_pg_lsn("0/9"), parse_pg_lsn("0/10"))
        self.assertLess(parse_pg_lsn("0/FFFFFFFF"), parse_pg_lsn("1/0"))
        self.assertEqual(parse_pg_lsn("1/2A"), (1 << 32) | 0x2A)
        sink = VersionedSink()
        sink.apply(ChangeEvent("u", "o", parse_pg_lsn("0/10"), {"s": "shipped"}))
        self.assertFalse(sink.apply(ChangeEvent("u", "o", parse_pg_lsn("0/9"), {"s": "pending"})))
        self.assertEqual(sink.snapshot(), {"o": {"s": "shipped"}})

    def test_malformed_pg_lsn_is_rejected(self):
        for text in ("09", "0/", "0/xyz", "-1/0"):
            with self.assertRaises(ValueError):
                parse_pg_lsn(text)


class TombstonePurge(unittest.TestCase):
    def test_purge_below_watermark_only(self):
        sink = VersionedSink()
        sink.apply(ChangeEvent("d", "a", 10, None))
        sink.apply(ChangeEvent("d", "b", 50, None))
        sink.apply(ChangeEvent("c", "c", 5, {"v": 1}))
        self.assertEqual(sink.purge_tombstones(safe_lsn=20), 1)
        self.assertEqual(sink.tombstones(), 1)
        self.assertEqual(sink.snapshot(), {"c": {"v": 1}})     # live rows untouched

    def test_purging_too_early_reopens_the_bug(self):
        sink = VersionedSink()
        sink.apply(ChangeEvent("d", "o1", 3, None))
        sink.purge_tombstones(safe_lsn=100)                     # wrong watermark
        sink.apply(ChangeEvent("u", "o1", 2, {"status": "paid"}))
        self.assertIn("o1", sink.snapshot())


if __name__ == "__main__":
    unittest.main()
