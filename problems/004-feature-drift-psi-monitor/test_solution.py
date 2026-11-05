import math
import random
import unittest

from solution import CategoricalBaseline, DriftMonitor, NumericBaseline, psi


def rows(n, seed, amount_scale=1.0, mu=50.0, country_w=(0.6, 0.3, 0.1), missing=0.0):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        amt = None if rng.random() < missing else rng.lognormvariate(math.log(mu), 0.8) * amount_scale
        out.append({"amount": amt, "age_days": rng.uniform(0, 3650),
                    "country": rng.choices(["IN", "US", "GB"], weights=country_w)[0]})
    return out


class DriftTests(unittest.TestCase):
    def setUp(self):
        self.mon = DriftMonitor()
        self.mon.fit(rows(20_000, seed=1), numeric=["amount", "age_days"], categorical=["country"])

    def report(self, window):
        return {r.feature: r for r in self.mon.check(window)}

    def test_same_distribution_is_ok(self):
        rep = self.report(rows(5_000, seed=2))
        self.assertTrue(all(r.level == "ok" for r in rep.values()), rep)
        self.assertLess(rep["amount"].psi, 0.02)

    def test_unit_change_rupees_to_paise_is_alert(self):
        # upstream service started sending amounts x100 (paise) — the incident
        rep = self.report(rows(5_000, seed=3, amount_scale=100))
        self.assertEqual(rep["amount"].level, "alert")
        self.assertGreater(rep["amount"].psi, 5)
        self.assertEqual(rep["age_days"].level, "ok")    # only the broken feature flagged

    def test_moderate_shift_warns(self):
        rep = self.report(rows(5_000, seed=4, mu=68.0))   # median amount +36%
        self.assertEqual(rep["amount"].level, "warn")

    def test_categorical_mix_shift(self):
        rep = self.report(rows(5_000, seed=5, country_w=(0.2, 0.3, 0.5)))
        self.assertEqual(rep["country"].level, "alert")

    def test_unseen_category_goes_to_other_bucket(self):
        b = CategoricalBaseline.fit(["a"] * 99 + ["b"])
        self.assertGreater(b.score(["zzz"] * 100), 0.2)

    def test_missing_rate_spike_alerts_even_if_values_look_fine(self):
        rep = self.report(rows(5_000, seed=6, missing=0.3))
        self.assertEqual(rep["amount"].level, "alert")
        self.assertAlmostEqual(rep["amount"].missing_delta, 0.3, delta=0.03)

    def test_psi_properties(self):
        self.assertEqual(psi([0.5, 0.5], [0.5, 0.5]), 0.0)
        self.assertGreater(psi([0.5, 0.5], [0.0, 1.0]), 0)        # empty bin is finite
        self.assertTrue(math.isfinite(psi([0.5, 0.5], [0.0, 1.0])))
        with self.assertRaises(ValueError):
            psi([1.0], [0.5, 0.5])

    def test_point_mass_feature_does_not_crash(self):
        b = NumericBaseline.fit([0.0] * 900 + [1.0] * 100)
        s, _ = b.score([0.0] * 900 + [1.0] * 100)
        self.assertAlmostEqual(s, 0.0)

    def test_zero_inflated_feature_has_no_empty_expected_bin(self):
        b = NumericBaseline.fit([0.0] * 900 + [1.0] * 100)
        self.assertEqual(b.expected, [0.9, 0.1])
        self.assertNotIn(0.0, NumericBaseline.fit([1.0, 2.0, 3.0]).expected)

    def test_few_values_below_training_minimum_do_not_warn(self):
        # 3% small negative refunds next to a feature whose training minimum is 0
        b = NumericBaseline.fit([0.0] * 900 + [1.0] * 100)
        s, _ = b.score([-0.5] * 30 + [0.0] * 870 + [1.0] * 100)
        self.assertLess(s, 0.1)

    def test_small_window_rejected(self):
        with self.assertRaises(ValueError):
            self.mon.check(rows(100, seed=7))


if __name__ == "__main__":
    unittest.main()
