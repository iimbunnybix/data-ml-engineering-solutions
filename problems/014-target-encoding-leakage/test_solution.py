import random
import unittest

from solution import TargetEncoder, auc, kfold_indices, out_of_fold_encode


class World:
    """Merchants with long-tail traffic; 5% of them are truly risky."""

    def __init__(self, seed=0, n_merchants=5_000):
        rng = random.Random(seed)
        self.rng = rng
        self.weights = [1 / (r + 1) ** 1.1 for r in range(n_merchants)]
        self.risk = [0.30 if rng.random() < 0.05 else 0.05 for _ in range(n_merchants)]
        self.ids = list(range(n_merchants))

    def sample(self, n):
        cats = self.rng.choices(self.ids, weights=self.weights, k=n)
        labels = [1 if self.rng.random() < self.risk[c] else 0 for c in cats]
        return cats, labels


def split(cats, labels, frac=0.8, seed=1):
    idx = list(range(len(cats)))
    random.Random(seed).shuffle(idx)
    cut = int(len(idx) * frac)
    pick = lambda ids: ([cats[i] for i in ids], [labels[i] for i in ids])
    return pick(idx[:cut]), pick(idx[cut:])


class ReproduceLeak(unittest.TestCase):
    def setUp(self):
        self.world = World(seed=0)
        self.cats, self.labels = self.world.sample(20_000)
        self.next_cats, self.next_labels = self.world.sample(20_000)   # next month

    def test_encoding_before_split_inflates_validation(self):
        # Original pipeline: encode on ALL rows (no smoothing), then split.
        enc = TargetEncoder(smoothing=0).fit(self.cats, self.labels)
        (_, _), (val_c, val_y) = split(self.cats, self.labels)
        val_auc = auc(enc.transform(val_c), val_y)    # val rows were in the fit

        # What production saw: encoder from training data, scored on new data.
        prod_auc = auc(enc.transform(self.next_cats), self.next_labels)
        self.assertGreater(val_auc, 0.75)
        self.assertLess(prod_auc, 0.60)
        self.assertGreater(val_auc - prod_auc, 0.15)
        self.assertEqual((round(val_auc, 2), round(prod_auc, 2)), (0.77, 0.55))

    def test_gap_holds_across_seeds(self):
        for seed in range(3):
            w = World(seed)
            c, l = w.sample(20_000)
            nc, nl = w.sample(20_000)
            _, (vc, vy) = split(c, l)
            enc = TargetEncoder(smoothing=0).fit(c, l)
            gap = auc(enc.transform(vc), vy) - auc(enc.transform(nc), nl)
            self.assertGreater(gap, 0.17, seed)

    def test_singleton_category_encodes_its_own_label(self):
        enc = TargetEncoder(smoothing=0).fit(["a", "b", "b"], [1, 0, 0])
        self.assertEqual(enc.transform(["a"]), [1.0])


class LeakFreeEncoding(unittest.TestCase):
    def setUp(self):
        self.world = World(seed=0)
        cats, labels = self.world.sample(20_000)
        (self.tr_c, self.tr_y), (self.va_c, self.va_y) = split(cats, labels)
        self.next_cats, self.next_labels = self.world.sample(20_000)

    def test_validation_predicts_production(self):
        enc = TargetEncoder(smoothing=20).fit(self.tr_c, self.tr_y)
        val_auc = auc(enc.transform(self.va_c), self.va_y)
        prod_auc = auc(enc.transform(self.next_cats), self.next_labels)
        self.assertLess(abs(val_auc - prod_auc), 0.03)
        self.assertEqual((round(val_auc, 2), round(prod_auc, 2)), (0.53, 0.55))

    def test_out_of_fold_training_feature_is_honest(self):
        in_sample = TargetEncoder(smoothing=0).fit(self.tr_c, self.tr_y).transform(self.tr_c)
        oof = out_of_fold_encode(self.tr_c, self.tr_y, k=5, smoothing=20, seed=3)
        enc = TargetEncoder(smoothing=20).fit(self.tr_c, self.tr_y)
        prod_auc = auc(enc.transform(self.next_cats), self.next_labels)
        self.assertGreater(auc(in_sample, self.tr_y) - prod_auc, 0.15)  # leaked
        self.assertLess(abs(auc(oof, self.tr_y) - prod_auc), 0.03)      # honest

    def test_smoothing_and_unseen(self):
        enc = TargetEncoder(smoothing=10).fit(["a"] * 2 + ["b"] * 8, [1, 1] + [0] * 8)
        self.assertAlmostEqual(enc.prior, 0.2)
        self.assertAlmostEqual(enc.transform(["a"])[0], (2 + 10 * 0.2) / 12)
        self.assertEqual(enc.transform(["zzz"]), [0.2])

    def test_out_of_fold_excludes_own_row(self):
        cats = ["x", "x", "y", "y"]
        oof = out_of_fold_encode(cats, [1, 0, 1, 0], k=4, smoothing=0, seed=0)
        self.assertEqual(sorted(oof), [0.0, 0.0, 1.0, 1.0])
        # each row got the OTHER row of its category
        self.assertEqual(oof[0], 0.0)
        self.assertEqual(oof[1], 1.0)

    def test_kfold_partitions(self):
        folds = kfold_indices(10, 3, seed=1)
        self.assertEqual(sorted(i for f in folds for i in f), list(range(10)))
        with self.assertRaises(ValueError):
            kfold_indices(3, 5, 0)

    def test_auc(self):
        self.assertEqual(auc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]), 0.75)
        self.assertEqual(auc([0.5, 0.5], [0, 1]), 0.5)
        with self.assertRaises(ValueError):
            auc([1, 2], [1, 1])

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            TargetEncoder().fit([], [])
        with self.assertRaises(ValueError):
            TargetEncoder(smoothing=-1)

    def test_transform_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            TargetEncoder().transform(["m1", "m2"])

    def test_out_of_fold_rejects_more_labels_than_categories(self):
        with self.assertRaises(ValueError):
            out_of_fold_encode(["a", "a", "b", "b"], [1, 0, 1, 0, 1, 1], k=2, smoothing=0)

    def test_out_of_fold_rejects_fewer_labels_than_categories(self):
        with self.assertRaises(ValueError):
            out_of_fold_encode(["a", "a", "b", "b"], [1, 0], k=2, smoothing=0)


if __name__ == "__main__":
    unittest.main()
