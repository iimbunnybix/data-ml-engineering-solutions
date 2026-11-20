import random
import unittest

from solution import (InsufficientPositivesError, Transaction, UndefinedMetricError,
                      choose_model, flags_above, make_transactions, naive_choose_model,
                      naive_random_split, naive_recall, never_flag, recall, stratified_split)

# New-market launch: 2,000 labelled transactions, 10 of them fraud (0.5%).
ROWS = make_transactions(rows=2000, frauds=10, seed=0)
MODELS = {"threshold-0.7": flags_above(0.7), "never-flag": never_flag}
VALIDATION_FRACTION = 0.2
SEEDS = range(200)
# With this seed the naive random split puts no fraud at all in validation.
ZERO_POSITIVE_SEED = 13


def fraud_count(rows):
    return sum(1 for row in rows if row.is_fraud)


class ReproduceProblem(unittest.TestCase):
    def test_some_random_splits_leave_validation_without_fraud(self):
        empty = 0
        for seed in SEEDS:
            _, validation = naive_random_split(ROWS, VALIDATION_FRACTION, seed)
            if fraud_count(validation) == 0:
                empty += 1
        self.assertEqual(empty, 23)

    def test_validation_fraud_count_swings_between_runs(self):
        counts = set()
        for seed in SEEDS:
            _, validation = naive_random_split(ROWS, VALIDATION_FRACTION, seed)
            counts.add(fraud_count(validation))
        self.assertEqual((min(counts), max(counts)), (0, 8))

    def test_zero_positives_is_reported_as_recall_zero(self):
        _, validation = naive_random_split(ROWS, VALIDATION_FRACTION, ZERO_POSITIVE_SEED)
        self.assertEqual(naive_recall(MODELS["threshold-0.7"], validation), 0.0)

    def test_model_that_never_flags_fraud_gets_promoted(self):
        _, validation = naive_random_split(ROWS, VALIDATION_FRACTION, ZERO_POSITIVE_SEED)
        self.assertEqual(naive_choose_model(MODELS, validation), "never-flag")


class FixStratifiedSplit(unittest.TestCase):
    def test_every_seed_puts_two_frauds_in_validation_and_eight_in_train(self):
        for seed in SEEDS:
            train, validation = stratified_split(ROWS, VALIDATION_FRACTION, seed)
            self.assertEqual((fraud_count(train), fraud_count(validation)), (8, 2))

    def test_split_keeps_every_row_exactly_once(self):
        train, validation = stratified_split(ROWS, VALIDATION_FRACTION, seed=1)
        ids = sorted(row.transaction_id for row in train + validation)
        self.assertEqual(ids, list(range(len(ROWS))))
        self.assertEqual(len(validation), 400)

    def test_same_seed_gives_same_split_whatever_the_input_order(self):
        shuffled = list(ROWS)
        random.Random(7).shuffle(shuffled)
        self.assertEqual(stratified_split(shuffled, VALIDATION_FRACTION, seed=3),
                         stratified_split(ROWS, VALIDATION_FRACTION, seed=3))

    def test_minimum_positives_is_given_even_when_the_fraction_rounds_lower(self):
        rows = make_transactions(rows=1000, frauds=6, seed=2)
        _, validation = stratified_split(rows, 0.1, seed=0, min_positives=2)
        self.assertEqual(fraud_count(validation), 2)

    def test_too_few_positives_refuses_to_split(self):
        rows = make_transactions(rows=1000, frauds=3, seed=2)
        with self.assertRaises(InsufficientPositivesError):
            stratified_split(rows, VALIDATION_FRACTION, seed=0)

    def test_no_positives_at_all_refuses_to_split(self):
        rows = make_transactions(rows=100, frauds=0, seed=2)
        with self.assertRaises(InsufficientPositivesError):
            stratified_split(rows, VALIDATION_FRACTION, seed=0)

    def test_useful_model_is_promoted_on_a_stratified_split(self):
        _, validation = stratified_split(ROWS, VALIDATION_FRACTION, ZERO_POSITIVE_SEED)
        self.assertEqual(choose_model(MODELS, validation), "threshold-0.7")


class FixUndefinedMetrics(unittest.TestCase):
    def test_recall_without_positives_is_undefined_not_zero(self):
        legit_only = [Transaction(1, 0.9, False), Transaction(2, 0.1, False)]
        result = recall(MODELS["threshold-0.7"], legit_only)
        self.assertFalse(result.is_defined)
        self.assertIsNone(result.value)
        self.assertIn("no positive rows", result.reason)

    def test_recall_with_positives_is_a_number(self):
        rows = [Transaction(1, 0.9, True), Transaction(2, 0.65, True), Transaction(3, 0.1, False)]
        self.assertEqual(recall(MODELS["threshold-0.7"], rows).value, 0.5)

    def test_model_selection_stops_when_validation_has_no_fraud(self):
        _, validation = naive_random_split(ROWS, VALIDATION_FRACTION, ZERO_POSITIVE_SEED)
        with self.assertRaises(UndefinedMetricError):
            choose_model(MODELS, validation)


if __name__ == "__main__":
    unittest.main()
