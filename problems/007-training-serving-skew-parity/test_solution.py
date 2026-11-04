import random
import unittest

from solution import (FEATURES, Order, ParityChecker, build_training_rows, compute_features,
                      offline_features_legacy, online_features_legacy, serve_features)


def make_history(seed=7, customers=500):
    rng = random.Random(seed)
    history = {}
    for c in range(customers):
        n = rng.choice([0, 0, 1, 2, 3, 5, 8, 13])
        orders = []
        for _ in range(n):
            ts = rng.choice([rng.uniform(0, 100), float(rng.randint(60, 100))])  # some on day boundaries
            status = "refunded" if rng.random() < 0.1 else "paid"
            orders.append(Order(ts, round(rng.uniform(5, 300), 2), status))
        history[f"c{c}"] = orders
    return history


def score_times(history, seed=11):
    rng = random.Random(seed)
    return [(c, float(rng.choice([90, 95, 100]))) for c in history]


class ReproduceSkew(unittest.TestCase):
    def test_legacy_paths_disagree_on_real_traffic(self):
        history = make_history()
        pc = ParityChecker()
        for c, t in score_times(history):
            pc.compare((c, t), online_features_legacy(history[c], t), offline_features_legacy(history[c], t))
        report = pc.report()
        self.assertIn("avg_order_value_30d", pc.failing())
        self.assertIn("order_count_30d", pc.failing())
        self.assertGreater(report["avg_order_value_30d"], 0.3)

    def test_specific_divergences(self):
        refund = [Order(95, 100.0, "paid"), Order(96, 300.0, "refunded")]
        self.assertEqual(offline_features_legacy(refund, 100)["avg_order_value_30d"], 100.0)
        self.assertEqual(online_features_legacy(refund, 100)["avg_order_value_30d"], 200.0)
        boundary = [Order(100.0, 50.0, "paid")]        # order at the scoring instant
        self.assertEqual(offline_features_legacy(boundary, 100)["order_count_30d"], 0.0)
        self.assertEqual(online_features_legacy(boundary, 100)["order_count_30d"], 1.0)
        self.assertEqual(offline_features_legacy([], 100)["avg_order_value_30d"], 0.0)
        self.assertEqual(online_features_legacy([], 100)["avg_order_value_30d"], 42.0)


class SharedDefinition(unittest.TestCase):
    def test_online_and_offline_identical(self):
        history = make_history()
        labels = [(c, t, 0) for c, t in score_times(history)]
        rows = build_training_rows(history, labels)
        pc = ParityChecker()
        for row in rows:
            offline = {k: row[k] for k in FEATURES}
            pc.compare((row["customer"], row["as_of"]), serve_features(history, row["customer"], row["as_of"]), offline)
        self.assertEqual(pc.failing(max_rate=0.0), [])
        self.assertTrue(all(r == 0.0 for r in pc.report().values()))

    def test_semantics(self):
        orders = [Order(69.9, 10.0, "paid"),     # outside 30d window
                  Order(70.0, 20.0, "paid"),     # window start is inclusive
                  Order(99.0, 40.0, "refunded"),
                  Order(100.0, 999.0, "paid")]   # at as_of: not visible
        f = compute_features(orders, 100.0)
        self.assertEqual(f["order_count_30d"], 1.0)
        self.assertEqual(f["avg_order_value_30d"], 20.0)
        self.assertAlmostEqual(f["refund_rate_90d"], 1 / 3, places=6)

    def test_unknown_customer(self):
        self.assertEqual(serve_features({}, "ghost", 10.0),
                         {"order_count_30d": 0.0, "avg_order_value_30d": 0.0, "refund_rate_90d": 0.0})


class ParityCheckerEdges(unittest.TestCase):
    def test_tolerance_and_nan(self):
        pc = ParityChecker(rel_tol=1e-6)
        pc.compare("k", {"x": 1.0000001, "y": float("nan")}, {"x": 1.0, "y": float("nan")})
        self.assertEqual(pc.failing(0.0), [])

    def test_missing_feature_counts_as_mismatch(self):
        pc = ParityChecker()
        pc.compare("k", {"a": 1.0}, {"a": 1.0, "b": 2.0})
        self.assertEqual(pc.report(), {"a": 0.0, "b": 1.0})
        self.assertEqual(pc.examples["b"], [("k", None, 2.0)])

    def test_none_and_strings(self):
        pc = ParityChecker()
        pc.compare(1, {"s": "IN", "n": None}, {"s": "IN", "n": None})
        pc.compare(2, {"s": "IN", "n": None}, {"s": "US", "n": 0.0})
        self.assertEqual(pc.report(), {"n": 0.5, "s": 0.5})
        self.assertEqual(pc.failing(0.5), [])
        self.assertEqual(pc.failing(0.4), ["n", "s"])


if __name__ == "__main__":
    unittest.main()
