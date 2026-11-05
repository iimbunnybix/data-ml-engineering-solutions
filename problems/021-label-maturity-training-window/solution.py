"""Build fraud training sets only from transactions whose labels have matured.

Fraud labels come from chargebacks, which banks file days or weeks after the
payment. A nightly retrain that takes "every transaction up to today, label =
fraud if a chargeback exists yet" treats recent fraud as legitimate, because
its chargeback has not arrived. The model learns that risky patterns are safe
and recall drops. This module picks a maturity window from the observed
chargeback delays and keeps only transactions older than that window.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import date, timedelta

# Share of eventual chargebacks the maturity window must already include.
DEFAULT_LABEL_COVERAGE = 0.99
FRAUD_RATE = 0.02
NEW_DEVICE_RATE_LEGIT = 0.10
NEW_DEVICE_RATE_FRAUD = 0.60
MIN_CHARGEBACK_DAYS = 3
MAX_CHARGEBACK_DAYS = 60


@dataclass(frozen=True)
class Transaction:
    txn_id: int
    txn_date: date
    new_device: bool
    # None means "never charged back" (the transaction really was legitimate).
    chargeback_date: date | None

    def is_fraud_as_of(self, snapshot_date: date) -> bool:
        """Label as the warehouse sees it on snapshot_date."""
        if self.chargeback_date is None:
            return False
        return self.chargeback_date <= snapshot_date


def rows_in_window(transactions: list[Transaction], snapshot_date: date,
                   window_end: date, window_days: int) -> list[tuple[Transaction, bool]]:
    """Transactions in the `window_days` days ending at window_end (inclusive).

    Labels are always what the warehouse knows on snapshot_date.
    """
    window_start = window_end - timedelta(days=window_days - 1)
    rows = []
    for txn in transactions:
        if window_start <= txn.txn_date <= window_end:
            rows.append((txn, txn.is_fraud_as_of(snapshot_date)))
    return rows


def naive_training_set(transactions: list[Transaction], snapshot_date: date,
                       window_days: int) -> list[tuple[Transaction, bool]]:
    """Original job: the most recent window_days, labelled with what is known today."""
    return rows_in_window(transactions, snapshot_date, snapshot_date, window_days)


def label_maturity_days(observed_delays_days: list[int],
                        coverage: float = DEFAULT_LABEL_COVERAGE) -> int:
    """Smallest delay (in days) that includes `coverage` of past chargebacks.

    Uses delays from transactions old enough that all their chargebacks have
    arrived; otherwise the slow tail is missing and the window comes out short.
    """
    if not observed_delays_days:
        raise ValueError("need at least one observed chargeback delay")
    if not 0 < coverage <= 1:
        raise ValueError("coverage must be in (0, 1]")
    ordered = sorted(observed_delays_days)
    # ceil so that, e.g., 99% of 150 delays needs the 149th value, not the 148th.
    needed_count = math.ceil(coverage * len(ordered))
    return ordered[needed_count - 1]


def mature_training_set(transactions: list[Transaction], snapshot_date: date,
                        window_days: int, maturity_days: int) -> list[tuple[Transaction, bool]]:
    """Fixed job: the same-size window, shifted back so every row is mature.

    A transaction exactly maturity_days old has had the full window, so the
    window ends on that day (inclusive).
    """
    if maturity_days < 0:
        raise ValueError("maturity_days must not be negative")
    window_end = snapshot_date - timedelta(days=maturity_days)
    return rows_in_window(transactions, snapshot_date, window_end, window_days)


def fraud_rate(rows: list[tuple[Transaction, bool]]) -> float:
    """Share of rows labelled fraud."""
    if not rows:
        return 0.0
    fraud_count = 0
    for _txn, is_fraud in rows:
        if is_fraud:
            fraud_count += 1
    return fraud_count / len(rows)


def fraud_rate_given_new_device(rows: list[tuple[Transaction, bool]]) -> float:
    """P(fraud | new device): what a one-feature model would learn from rows."""
    new_device_rows = [row for row in rows if row[0].new_device]
    return fraud_rate(new_device_rows)


def simulate_transactions(start: date, days: int, per_day: int,
                          rng: random.Random) -> list[Transaction]:
    """Seeded payments where fraud is charged back 3-60 days after the payment."""
    transactions = []
    txn_id = 0
    for day_offset in range(days):
        txn_date = start + timedelta(days=day_offset)
        for _ in range(per_day):
            is_fraud = rng.random() < FRAUD_RATE
            device_rate = NEW_DEVICE_RATE_FRAUD if is_fraud else NEW_DEVICE_RATE_LEGIT
            chargeback_date = None
            if is_fraud:
                delay = rng.randint(MIN_CHARGEBACK_DAYS, MAX_CHARGEBACK_DAYS)
                chargeback_date = txn_date + timedelta(days=delay)
            transactions.append(Transaction(txn_id, txn_date, rng.random() < device_rate,
                                            chargeback_date))
            txn_id += 1
    return transactions


def settled_chargeback_delays(transactions: list[Transaction], snapshot_date: date,
                              settle_days: int) -> list[int]:
    """Delays of chargebacks for transactions older than settle_days (complete history)."""
    delays = []
    for txn in transactions:
        old_enough = txn.txn_date <= snapshot_date - timedelta(days=settle_days)
        if old_enough and txn.is_fraud_as_of(snapshot_date):
            delays.append((txn.chargeback_date - txn.txn_date).days)
    return delays
