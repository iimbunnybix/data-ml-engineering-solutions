import random
import unittest

from solution import (FeatureRow, LabelRow, PointInTimeJoiner, best_threshold_accuracy,
                      latest_join)

DAY = 86_400


def chargeback_dataset(n_users=400, fraud_rate=0.1, seed=42):
    """Transactions labeled fraud/not; `chargebacks` feature is updated days later
    *because* a fraudulent transaction got charged back (the leak)."""
    rng = random.Random(seed)
    labels, feats = [], []
    for u in range(n_users):
        uid = f"u{u}"
        base_cb = rng.choice([0, 0, 0, 1])                  # history unrelated to label
        feats.append(FeatureRow(uid, 0, float(base_cb)))
        ts = rng.randint(10, 60) * DAY
        is_fraud = int(rng.random() < fraud_rate)
        labels.append(LabelRow(uid, ts, is_fraud))
        if is_fraud:                                        # chargeback lands 5-30 days later
            feats.append(FeatureRow(uid, ts + rng.randint(5, 30) * DAY, float(base_cb + 3)))
    return labels, feats


class PointInTimeTests(unittest.TestCase):
    def test_latest_join_leaks_label(self):
        labels, feats = chargeback_dataset()
        acc = best_threshold_accuracy(latest_join(labels, feats))
        self.assertEqual(acc, 1.0)          # "perfect" model offline == leakage

    def test_point_in_time_removes_leak(self):
        labels, feats = chargeback_dataset()
        pit = PointInTimeJoiner(feats).join(labels)
        fraud_share = sum(l.label for l in labels) / len(labels)
        acc = best_threshold_accuracy(pit)
        # no better than always predicting the majority class
        self.assertAlmostEqual(acc, 1 - fraud_share, places=9)
        self.assertTrue(all(v in (0.0, 1.0) for _, v in pit))

    def test_picks_latest_value_before_event(self):
        j = PointInTimeJoiner([FeatureRow("a", 10, 1.0), FeatureRow("a", 20, 2.0),
                               FeatureRow("a", 30, 3.0)])
        self.assertEqual(j.lookup("a", 25), 2.0)
        self.assertEqual(j.lookup("a", 20), 2.0)   # inclusive by default
        self.assertEqual(j.lookup("a", 99), 3.0)

    def test_strict_excludes_same_timestamp(self):
        j = PointInTimeJoiner([FeatureRow("a", 10, 1.0), FeatureRow("a", 20, 2.0)], strict=True)
        self.assertEqual(j.lookup("a", 20), 1.0)

    def test_before_first_value_and_unknown_entity_use_default(self):
        j = PointInTimeJoiner([FeatureRow("a", 10, 1.0)], default=-1.0)
        self.assertEqual(j.lookup("a", 5), -1.0)
        self.assertEqual(j.lookup("zzz", 50), -1.0)

    def test_ttl_drops_stale_values(self):
        j = PointInTimeJoiner([FeatureRow("a", 0, 1.0)], ttl_s=7 * DAY, default=None)
        self.assertEqual(j.lookup("a", 7 * DAY), 1.0)
        self.assertIsNone(j.lookup("a", 7 * DAY + 1))

    def test_ingestion_delay_respected(self):
        # Value for day 10 only reached the store on day 12: not usable on day 11.
        j = PointInTimeJoiner([FeatureRow("a", 0, 1.0),
                               FeatureRow("a", 10 * DAY, 2.0, ingested_ts=12 * DAY)])
        self.assertEqual(j.lookup("a", 11 * DAY), 1.0)
        self.assertEqual(j.lookup("a", 12 * DAY), 2.0)

    def test_late_backfill_of_older_value_does_not_override_newer(self):
        j = PointInTimeJoiner([FeatureRow("a", 20, 2.0, ingested_ts=21),
                               FeatureRow("a", 10, 1.0, ingested_ts=50)])  # backfilled late
        self.assertEqual(j.lookup("a", 60), 2.0)
        self.assertEqual(j.lookup("a", 30), 2.0)

    def test_input_order_irrelevant(self):
        labels, feats = chargeback_dataset(n_users=100)
        shuffled = feats[:]
        random.Random(1).shuffle(shuffled)
        self.assertEqual(PointInTimeJoiner(feats).join(labels),
                         PointInTimeJoiner(shuffled).join(labels))


if __name__ == "__main__":
    unittest.main()
