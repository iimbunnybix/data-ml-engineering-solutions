import random
import unittest

from solution import WapLoader, Warehouse, naive_load


def day_rows(rng, day, n):
    return [{"order_id": f"{day}-{i}", "customer_id": f"c{rng.randrange(50_000)}",
             "amount": round(rng.uniform(5, 200), 2)} for i in range(n)]


def seeded_history(loader_or_wh, rng, naive=False):
    days = [f"2026-03-0{d}" for d in range(1, 8)]
    for d in days:
        rows = day_rows(rng, d, rng.randint(9_500, 10_500))
        if naive:
            naive_load(loader_or_wh, d, rows)
        else:
            loader_or_wh.load(d, rows)
    return days


TODAY = "2026-03-08"


class ReproducePartialLoad(unittest.TestCase):
    def test_truncated_export_is_published_as_truth(self):
        rng = random.Random(7)
        wh = Warehouse()
        seeded_history(wh, rng, naive=True)
        full = day_rows(rng, TODAY, 10_000)
        true_revenue = round(sum(r["amount"] for r in full), 2)
        truncated = full[: int(len(full) * 0.41)]       # export stopped at 41%
        naive_load(wh, TODAY, truncated)
        shown = wh.revenue(TODAY) / true_revenue
        self.assertAlmostEqual(shown, 0.41, delta=0.01)  # dashboard: revenue "down 59%"
        self.assertEqual(len(wh.published[TODAY]), 4_100)


class WriteAuditPublish(unittest.TestCase):
    def setUp(self):
        self.rng = random.Random(7)
        self.wh = Warehouse()
        self.loader = WapLoader(self.wh)
        seeded_history(self.loader, self.rng)
        self.assertEqual(self.loader.alerts, [])

    def test_truncated_load_is_quarantined_not_published(self):
        full = day_rows(self.rng, TODAY, 10_000)
        report = self.loader.load(TODAY, full[:4_100])
        self.assertFalse(report.passed)
        self.assertEqual([f.name for f in report.failures], ["row_count"])
        self.assertIn("4100 rows = 0.42x median 9816", report.failures[0].detail)
        self.assertNotIn(TODAY, self.wh.published)      # dashboard keeps showing last good day
        self.assertEqual(len(self.wh.quarantine[TODAY]), 4_100)
        self.assertEqual(len(self.loader.alerts), 1)

    def test_rerun_with_full_export_publishes_and_clears_quarantine(self):
        full = day_rows(self.rng, TODAY, 10_000)
        self.loader.load(TODAY, full[:4_100])
        report = self.loader.load(TODAY, full)
        self.assertTrue(report.passed)
        self.assertEqual(len(self.wh.published[TODAY]), 10_000)
        self.assertNotIn(TODAY, self.wh.quarantine)

    def test_legitimate_spike_passes(self):
        report = self.loader.load(TODAY, day_rows(self.rng, TODAY, 18_000))  # sale day, 1.8x
        self.assertTrue(report.passed, report.failures)

    def test_double_loaded_file_caught_by_uniqueness(self):
        rows = day_rows(self.rng, TODAY, 5_000)
        report = self.loader.load(TODAY, rows + rows)
        self.assertEqual([f.name for f in report.failures], ["unique:order_id"])

    def test_broken_join_nulls_caught(self):
        rows = day_rows(self.rng, TODAY, 10_000)
        for r in rows[:500]:
            r["customer_id"] = None                     # 5% nulls > 1% allowed
        report = self.loader.load(TODAY, rows)
        self.assertEqual([f.name for f in report.failures], ["nulls:customer_id"])

    def test_negative_amounts_caught(self):
        rows = day_rows(self.rng, TODAY, 10_000)
        rows[3]["amount"] = -12.5
        self.assertEqual([f.name for f in self.loader.load(TODAY, rows).failures],
                         ["range:amount"])

    def test_missing_amount_is_quarantined_not_published(self):
        rows = day_rows(self.rng, TODAY, 10_000)
        rows[3]["amount"] = None                        # broken price join / empty CSV cell
        report = self.loader.load(TODAY, rows)
        self.assertEqual([f.name for f in report.failures], ["nulls:amount"])
        self.assertNotIn(TODAY, self.wh.published)
        self.assertEqual(len(self.wh.quarantine[TODAY]), 10_000)

    def test_empty_partition_fails(self):
        report = self.loader.load(TODAY, [])
        self.assertIn("row_count", [f.name for f in report.failures])

    def test_force_publish_override(self):
        report = self.loader.load(TODAY, day_rows(self.rng, TODAY, 100), force=True)
        self.assertFalse(report.passed)
        self.assertEqual(len(self.wh.published[TODAY]), 100)

    def test_baseline_ignores_one_bad_day(self):
        # A forced tiny day in history should not drag the median down.
        self.loader.load(TODAY, day_rows(self.rng, TODAY, 100), force=True)
        report = self.loader.load("2026-03-09", day_rows(self.rng, "2026-03-09", 4_000))
        self.assertFalse(report.passed)


class ColdStart(unittest.TestCase):
    def test_row_count_skipped_without_history(self):
        loader = WapLoader(Warehouse())
        report = loader.load("2026-01-01", day_rows(random.Random(1), "2026-01-01", 10))
        self.assertTrue(report.passed)
        self.assertIn("skipped", report.results[0].detail)


if __name__ == "__main__":
    unittest.main()
