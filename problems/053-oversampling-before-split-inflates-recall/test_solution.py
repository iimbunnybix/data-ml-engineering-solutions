import random
import unittest

from solution import (
    FRAUD,
    LEGIT,
    NearestNeighbourModel,
    Transaction,
    assert_no_shared_rows,
    fixed_prepare,
    fraud_recall,
    naive_prepare,
    oversample_minority,
    shared_row_ids,
    simulate_transactions,
)

HISTORY_SIZE = 8_000
PRODUCTION_SIZE = 8_000
PRODUCTION_FIRST_ID = 1_000_000


def history() -> list[Transaction]:
    return simulate_transactions(HISTORY_SIZE, random.Random(53))


def production_traffic() -> list[Transaction]:
    """Fresh rows the model never saw, as after launch."""
    return simulate_transactions(PRODUCTION_SIZE, random.Random(530), PRODUCTION_FIRST_ID)


def count_fraud(rows: list[Transaction]) -> int:
    return sum(1 for row in rows if row.label == FRAUD)


class ReproduceProblem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train, cls.test = naive_prepare(history(), random.Random(1))
        cls.model = NearestNeighbourModel(cls.train)

    def test_every_test_fraud_row_also_appears_in_train(self):
        test_fraud_ids = {row.row_id for row in self.test if row.label == FRAUD}
        self.assertTrue(test_fraud_ids)
        self.assertEqual(shared_row_ids(self.train, self.test) & test_fraud_ids, test_fraud_ids)

    def test_offline_recall_is_perfect(self):
        self.assertEqual(fraud_recall(self.model, self.test), 1.0)

    def test_production_recall_is_far_below_offline_recall(self):
        self.assertLess(fraud_recall(self.model, production_traffic()), 0.5)

    def test_test_fold_has_an_unrealistic_fraud_share(self):
        self.assertGreater(count_fraud(self.test) / len(self.test), 0.4)


class FixSplitBeforeOversampling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train, cls.test = fixed_prepare(history(), random.Random(1))
        cls.model = NearestNeighbourModel(cls.train)

    def test_no_source_row_is_in_both_folds(self):
        self.assertEqual(shared_row_ids(self.train, self.test), set())

    def test_train_fold_is_balanced(self):
        self.assertEqual(count_fraud(self.train), len(self.train) - count_fraud(self.train))

    def test_test_fold_keeps_the_real_fraud_share(self):
        self.assertLess(count_fraud(self.test) / len(self.test), 0.06)

    def test_offline_recall_is_close_to_production_recall(self):
        offline = fraud_recall(self.model, self.test)
        production = fraud_recall(self.model, production_traffic())
        self.assertLess(abs(offline - production), 0.1)


class FixHelpers(unittest.TestCase):
    def test_guard_rejects_a_row_shared_by_both_folds(self):
        row = Transaction(7, (0.0, 0.0), FRAUD)
        with self.assertRaises(ValueError):
            assert_no_shared_rows([row], [row])

    def test_oversampling_without_fraud_returns_the_rows_unchanged(self):
        rows = [Transaction(1, (0.0, 0.0), LEGIT), Transaction(2, (1.0, 1.0), LEGIT)]
        self.assertEqual(oversample_minority(rows, random.Random(0)), rows)

    def test_oversampling_adds_copies_not_new_rows(self):
        rows = [Transaction(1, (0.0, 0.0), LEGIT), Transaction(2, (0.1, 0.1), LEGIT),
                Transaction(3, (0.2, 0.2), LEGIT), Transaction(4, (1.0, 1.0), FRAUD)]
        balanced = oversample_minority(rows, random.Random(0))
        self.assertEqual({row.row_id for row in balanced}, {1, 2, 3, 4})
        self.assertEqual(count_fraud(balanced), 3)

    def test_recall_without_fraud_rows_is_an_error(self):
        model = NearestNeighbourModel([Transaction(1, (0.0, 0.0), LEGIT)])
        with self.assertRaises(ValueError):
            fraud_recall(model, [Transaction(2, (0.0, 0.0), LEGIT)])


if __name__ == "__main__":
    unittest.main()
