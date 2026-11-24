import random
import unittest

from solution import (SECONDS_PER_DAY, Transaction, count_borrowed_history, lag_features,
                      naive_lag_features)

CUSTOMERS = 1000
MAX_TRANSACTIONS_PER_CUSTOMER = 6
START = 1_700_000_000


def make_history(seed: int = 69) -> tuple[list[Transaction], set[str]]:
    """Seeded transactions for many customers, shuffled, plus the ids of each customer's first one."""
    rng = random.Random(seed)
    transactions, first_ids = [], set()
    for customer_number in range(CUSTOMERS):
        customer_id = f"cust-{customer_number:04d}"
        timestamp = START + rng.randrange(0, 90 * SECONDS_PER_DAY)
        for sequence in range(rng.randint(1, MAX_TRANSACTIONS_PER_CUSTOMER)):
            transaction_id = f"{customer_id}-t{sequence}"
            if sequence == 0:
                first_ids.add(transaction_id)
            transactions.append(Transaction(transaction_id, customer_id, timestamp,
                                            round(rng.uniform(5, 500), 2)))
            timestamp += rng.randrange(SECONDS_PER_DAY, 20 * SECONDS_PER_DAY)
    rng.shuffle(transactions)
    return transactions, first_ids


def by_id(features):
    return {feature.transaction_id: feature for feature in features}


def two_customers():
    return [
        Transaction("a1", "alice", START, 900.0),
        Transaction("a2", "alice", START + 30 * SECONDS_PER_DAY, 950.0),
        Transaction("b1", "bob", START + 2 * SECONDS_PER_DAY, 40.0),
        Transaction("b2", "bob", START + 5 * SECONDS_PER_DAY, 60.0),
    ]


class ReproduceProblem(unittest.TestCase):
    def test_new_customer_inherits_previous_customers_last_amount(self):
        features = by_id(naive_lag_features(two_customers()))
        self.assertEqual(features["b1"].previous_amount, 950.0)
        self.assertFalse(features["b1"].is_first_transaction)

    def test_days_since_previous_goes_negative_across_customers(self):
        features = by_id(naive_lag_features(two_customers()))
        self.assertEqual(features["b1"].days_since_previous, -28.0)

    def test_almost_every_first_transaction_borrows_history(self):
        transactions, first_ids = make_history()
        borrowed = count_borrowed_history(naive_lag_features(transactions), first_ids)
        self.assertEqual(borrowed, CUSTOMERS - 1)


class FixPartitionedLag(unittest.TestCase):
    def test_first_transaction_has_no_previous_value(self):
        features = by_id(lag_features(two_customers()))
        self.assertIsNone(features["b1"].previous_amount)
        self.assertIsNone(features["b1"].days_since_previous)
        self.assertTrue(features["b1"].is_first_transaction)

    def test_later_transaction_uses_same_customers_previous(self):
        features = by_id(lag_features(two_customers()))
        self.assertEqual(features["b2"].previous_amount, 40.0)
        self.assertEqual(features["b2"].days_since_previous, 3.0)

    def test_no_first_transaction_borrows_history(self):
        transactions, first_ids = make_history()
        self.assertEqual(count_borrowed_history(lag_features(transactions), first_ids), 0)

    def test_no_gap_is_negative(self):
        transactions, _ = make_history()
        for feature in lag_features(transactions):
            if feature.days_since_previous is not None:
                self.assertGreaterEqual(feature.days_since_previous, 0)

    def test_result_does_not_depend_on_input_order(self):
        transactions, _ = make_history()
        reversed_input = list(reversed(transactions))
        self.assertEqual(lag_features(transactions), lag_features(reversed_input))

    def test_equal_timestamps_are_ordered_by_transaction_id(self):
        same_time = [Transaction("x2", "carol", START, 20.0), Transaction("x1", "carol", START, 10.0)]
        features = by_id(lag_features(same_time))
        self.assertEqual(features["x2"].previous_amount, 10.0)
        self.assertEqual(features["x2"].days_since_previous, 0.0)

    def test_empty_input_gives_no_features(self):
        self.assertEqual(lag_features([]), [])

    def test_one_feature_row_per_transaction(self):
        transactions, first_ids = make_history()
        features = lag_features(transactions)
        self.assertEqual(len(features), len(transactions))
        self.assertEqual(sum(1 for feature in features if feature.is_first_transaction), len(first_ids))


if __name__ == "__main__":
    unittest.main()
