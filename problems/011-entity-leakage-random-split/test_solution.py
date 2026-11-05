import unittest

from solution import (EntityLeakageError, NearestNeighbour, Row, accuracy,
                      assert_no_entity_leakage, group_kfold, group_split,
                      leakage_report, make_ticket_dataset, random_row_split)

ROWS = make_ticket_dataset(customers=200, tickets_per_customer=10, seed=0)


class ReproduceLeakage(unittest.TestCase):
    def test_row_split_puts_every_test_customer_in_train(self):
        train, test = random_row_split(ROWS, 0.2, seed=1)
        rep = leakage_report(train, test)
        self.assertEqual(rep.test_rows_with_seen_entity, 1.0)
        with self.assertRaises(EntityLeakageError):
            assert_no_entity_leakage(train, test)

    def test_row_split_reports_inflated_accuracy(self):
        train, test = random_row_split(ROWS, 0.2, seed=1)
        acc = accuracy(NearestNeighbour().fit(train), test)
        self.assertGreater(acc, 0.80)       # 84%: the model recognises customers


class GroupSplitFix(unittest.TestCase):
    def test_no_customer_on_both_sides(self):
        train, test = group_split(ROWS, 0.2)
        assert_no_entity_leakage(train, test)
        self.assertEqual(len(train) + len(test), len(ROWS))

    def test_honest_accuracy_is_near_chance(self):
        train, test = group_split(ROWS, 0.2)
        acc = accuracy(NearestNeighbour().fit(train), test)
        self.assertLess(acc, 0.60)          # 46%: nothing generalises to new customers

    def test_group_kfold_folds_are_disjoint_and_cover_everything(self):
        seen_validation = []
        accs = []
        for train, val in group_kfold(ROWS, 5):
            assert_no_entity_leakage(train, val)
            self.assertEqual(len(train) + len(val), len(ROWS))
            seen_validation.extend(val)
            accs.append(accuracy(NearestNeighbour().fit(train), val))
        self.assertEqual(sorted(map(id, seen_validation)), sorted(map(id, ROWS)))
        self.assertLess(sum(accs) / len(accs), 0.60)

    def test_assignment_is_stable_when_data_grows(self):
        _, test_before = group_split(ROWS[:1000], 0.2)
        _, test_after = group_split(ROWS, 0.2)       # second half of customers added
        before = {r.entity for r in test_before}
        after = {r.entity for r in test_after}
        self.assertTrue(before <= after)              # nobody switched sides

    def test_salt_changes_assignment(self):
        _, a = group_split(ROWS, 0.2, salt="v1")
        _, b = group_split(ROWS, 0.2, salt="v2")
        self.assertNotEqual({r.entity for r in a}, {r.entity for r in b})

    def test_fraction_is_roughly_respected(self):
        _, test = group_split(ROWS, 0.2)
        entities = {r.entity for r in test}
        self.assertTrue(25 <= len(entities) <= 55)    # ~40 of 200 customers

    def test_bad_arguments(self):
        with self.assertRaises(ValueError):
            group_split(ROWS, 0)
        with self.assertRaises(ValueError):
            list(group_kfold(ROWS, 1))

    def test_folds_of_the_training_side_are_never_empty_after_a_holdout(self):
        train, _ = group_split(ROWS, 0.2)
        for fold_train, validation in group_kfold(train, 5):
            self.assertGreater(len(validation), 0)
            self.assertGreater(len(fold_train), 0)

    def test_accuracy_on_empty_set_raises_clear_error(self):
        model = NearestNeighbour().fit(ROWS[:5])
        with self.assertRaises(ValueError):
            accuracy(model, [])

    def test_empty_test_set_report(self):
        rep = leakage_report([Row("a", (0.0,), 1)], [])
        self.assertEqual(rep.test_rows_with_seen_entity, 0.0)


if __name__ == "__main__":
    unittest.main()
