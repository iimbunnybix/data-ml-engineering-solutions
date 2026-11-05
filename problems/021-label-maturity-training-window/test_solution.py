import random
import unittest
from datetime import date, timedelta

from solution import (
    MAX_CHARGEBACK_DAYS,
    Transaction,
    fraud_rate,
    fraud_rate_given_new_device,
    label_maturity_days,
    mature_training_set,
    naive_training_set,
    settled_chargeback_delays,
    simulate_transactions,
)

START = date(2026, 1, 1)
SIMULATED_DAYS = 240
SNAPSHOT = START + timedelta(days=SIMULATED_DAYS - 1)
WINDOW_DAYS = 90
SETTLE_DAYS = 120


def simulated_payments() -> list[Transaction]:
    return simulate_transactions(START, SIMULATED_DAYS, per_day=500, rng=random.Random(21))


def with_true_labels(rows: list[tuple[Transaction, bool]]) -> list[tuple[Transaction, bool]]:
    """Relabel rows with the eventual truth (does a chargeback ever arrive)."""
    return [(txn, txn.chargeback_date is not None) for txn, _ in rows]


def payment(txn_date: date, chargeback_date: date | None = None) -> Transaction:
    return Transaction(0, txn_date, new_device=False, chargeback_date=chargeback_date)


class ReproduceProblem(unittest.TestCase):
    def test_recent_fraud_is_labelled_legitimate_in_the_naive_window(self):
        rows = naive_training_set(simulated_payments(), SNAPSHOT, WINDOW_DAYS)
        observed = fraud_rate(rows)
        actual = fraud_rate(with_true_labels(rows))
        # About a third of the window's fraud has no chargeback yet.
        self.assertLess(observed, 0.7 * actual)

    def test_naive_window_teaches_that_new_devices_are_safer_than_they_are(self):
        rows = naive_training_set(simulated_payments(), SNAPSHOT, WINDOW_DAYS)
        learned = fraud_rate_given_new_device(rows)
        actual = fraud_rate_given_new_device(with_true_labels(rows))
        self.assertLess(learned, 0.7 * actual)

    def test_yesterdays_fraud_is_never_labelled(self):
        txn = payment(SNAPSHOT, chargeback_date=SNAPSHOT + timedelta(days=20))
        self.assertFalse(txn.is_fraud_as_of(SNAPSHOT))


class FixLabelMaturity(unittest.TestCase):
    def setUp(self) -> None:
        self.payments = simulated_payments()
        delays = settled_chargeback_delays(self.payments, SNAPSHOT, SETTLE_DAYS)
        self.maturity = label_maturity_days(delays)

    def test_maturity_window_covers_the_slowest_chargebacks(self):
        self.assertEqual(self.maturity, MAX_CHARGEBACK_DAYS)

    def test_mature_window_labels_match_the_truth(self):
        rows = mature_training_set(self.payments, SNAPSHOT, WINDOW_DAYS, self.maturity)
        self.assertEqual(fraud_rate(rows), fraud_rate(with_true_labels(rows)))

    def test_mature_window_has_the_same_number_of_days(self):
        rows = mature_training_set(self.payments, SNAPSHOT, WINDOW_DAYS, self.maturity)
        distinct_days = {txn.txn_date for txn, _ in rows}
        self.assertEqual(len(distinct_days), WINDOW_DAYS)

    def test_mature_window_learns_the_true_new_device_risk(self):
        rows = mature_training_set(self.payments, SNAPSHOT, WINDOW_DAYS, self.maturity)
        self.assertEqual(fraud_rate_given_new_device(rows),
                         fraud_rate_given_new_device(with_true_labels(rows)))


class LabelMaturityDaysEdgeCases(unittest.TestCase):
    def test_coverage_uses_the_ceiling_rank(self):
        delays = list(range(1, 151))   # 99% of 150 delays needs the 149th value
        self.assertEqual(label_maturity_days(delays, coverage=0.99), 149)

    def test_full_coverage_returns_the_maximum_delay(self):
        self.assertEqual(label_maturity_days([5, 40, 12], coverage=1.0), 40)

    def test_empty_delays_are_rejected(self):
        with self.assertRaises(ValueError):
            label_maturity_days([])

    def test_invalid_coverage_is_rejected(self):
        with self.assertRaises(ValueError):
            label_maturity_days([5], coverage=0)


class MatureTrainingSetEdgeCases(unittest.TestCase):
    def test_transaction_exactly_maturity_days_old_is_included(self):
        txn = payment(SNAPSHOT - timedelta(days=60))
        rows = mature_training_set([txn], SNAPSHOT, window_days=1, maturity_days=60)
        self.assertEqual(len(rows), 1)

    def test_transaction_one_day_too_young_is_excluded(self):
        txn = payment(SNAPSHOT - timedelta(days=59))
        rows = mature_training_set([txn], SNAPSHOT, window_days=1, maturity_days=60)
        self.assertEqual(rows, [])

    def test_negative_maturity_is_rejected(self):
        with self.assertRaises(ValueError):
            mature_training_set([], SNAPSHOT, WINDOW_DAYS, maturity_days=-1)

    def test_chargeback_on_snapshot_day_counts_as_fraud(self):
        self.assertTrue(payment(START, chargeback_date=SNAPSHOT).is_fraud_as_of(SNAPSHOT))


if __name__ == "__main__":
    unittest.main()
