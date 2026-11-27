"""Walk-forward backtest whose training labels peek into the test period.

The churn label is "did the customer cancel within the 30 days after the
snapshot date". A walk-forward split put snapshots before day T in training
and snapshots from day T on in test, but a training snapshot taken on day
T-10 has a label window that runs to day T+20: its label is a test-period
outcome. The model learned the test period's churn, so the backtest score was
better than anything production could reach. The fix purges (embargoes)
every training row whose label window ends after the test start, so training
uses only outcomes that were known on day T.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

SEED = 7
LABEL_HORIZON_DAYS = 30           # label = cancelled within 30 days after snapshot
HISTORY_DAYS = 180                # snapshot days 0..179
TEST_START_DAY = 150              # walk-forward cut: test snapshots are days 150..179
CUSTOMERS_PER_SEGMENT_PER_DAY = 40
SEGMENTS = ["web", "mobile"]
BASE_DAILY_CANCEL_RATE = 0.005    # about 14% cancel within 30 days
TRAINING_WINDOW_DAYS = 60         # the model learns from the 60 latest snapshot days
# A competitor's app launches on day 155: mobile customers start leaving.
REGIME_CHANGE_DAY = 155
MOBILE_DAILY_CANCEL_RATE_AFTER = 0.04


@dataclass(frozen=True)
class Snapshot:
    """One customer on one snapshot date, with its churn label."""
    day: int            # snapshot date, as a day number
    segment: str
    cancelled: int      # 1 if the customer cancelled within LABEL_HORIZON_DAYS


# A split function: all snapshots in, (train, test) out.
Split = Callable[[list[Snapshot]], tuple[list[Snapshot], list[Snapshot]]]


# ---------------------------------------------------------------- data
def daily_cancel_rate(segment: str, day: int) -> float:
    """Chance a customer cancels on a given day; mobile jumps after the launch."""
    if segment == "mobile" and day >= REGIME_CHANGE_DAY:
        return MOBILE_DAILY_CANCEL_RATE_AFTER
    return BASE_DAILY_CANCEL_RATE


def cancelled_within_horizon(segment: str, snapshot_day: int, rng: random.Random) -> int:
    """Simulate the 30 days after a snapshot; 1 if the customer cancels."""
    for day in range(snapshot_day + 1, snapshot_day + LABEL_HORIZON_DAYS + 1):
        if rng.random() < daily_cancel_rate(segment, day):
            return 1
    return 0


def build_snapshots(seed: int = SEED) -> list[Snapshot]:
    """Daily customer snapshots with fully matured (hindsight) labels."""
    rng = random.Random(seed)
    snapshots: list[Snapshot] = []
    for day in range(HISTORY_DAYS):
        for segment in SEGMENTS:
            for _ in range(CUSTOMERS_PER_SEGMENT_PER_DAY):
                label = cancelled_within_horizon(segment, day, rng)
                snapshots.append(Snapshot(day, segment, label))
    return snapshots


# ---------------------------------------------------------------- splits
def label_window_end(snapshot: Snapshot) -> int:
    """Last day whose outcome can change this snapshot's label."""
    return snapshot.day + LABEL_HORIZON_DAYS


def naive_walk_forward_split(snapshots: list[Snapshot],
                             test_start: int = TEST_START_DAY) -> tuple[list[Snapshot], list[Snapshot]]:
    """Original: split on the snapshot date only, ignoring the label window."""
    train = [row for row in snapshots if row.day < test_start]
    test = [row for row in snapshots if row.day >= test_start]
    return train, test


def purged_walk_forward_split(snapshots: list[Snapshot],
                              test_start: int = TEST_START_DAY) -> tuple[list[Snapshot], list[Snapshot]]:
    """Fix: keep a training row only if its label was known on the test start.

    A row's label is final once its whole window has passed. Rows whose
    window ends after test_start depend on test-period outcomes, so they are
    purged. This leaves a gap of LABEL_HORIZON_DAYS between train and test.
    """
    train: list[Snapshot] = []
    for row in snapshots:
        if row.day < test_start and label_window_end(row) <= test_start:
            train.append(row)
    test = [row for row in snapshots if row.day >= test_start]
    return train, test


def leaking_training_rows(train: list[Snapshot], test_start: int) -> list[Snapshot]:
    """Guard: training rows whose label window reaches past the test start."""
    leaking: list[Snapshot] = []
    for row in train:
        if label_window_end(row) > test_start:
            leaking.append(row)
    return leaking


# ---------------------------------------------------------------- model
def most_recent_rows(train: list[Snapshot], window_days: int = TRAINING_WINDOW_DAYS) -> list[Snapshot]:
    """Keep the latest window_days of training snapshots (recent data wins)."""
    if not train:
        return []
    latest_day = max(row.day for row in train)
    return [row for row in train if row.day > latest_day - window_days]


def fit_segment_rates(train: list[Snapshot]) -> dict[str, float]:
    """The model: each segment's churn rate over the latest training window."""
    train = most_recent_rows(train)
    totals: dict[str, int] = {}
    cancels: dict[str, int] = {}
    for row in train:
        totals[row.segment] = totals.get(row.segment, 0) + 1
        cancels[row.segment] = cancels.get(row.segment, 0) + row.cancelled
    rates: dict[str, float] = {}
    for segment, total in totals.items():
        rates[segment] = cancels[segment] / total
    return rates


def evaluate_split(snapshots: list[Snapshot], split: Split) -> tuple[dict[str, float], float]:
    """Fit on the split's training rows; return learned rates and test Brier score."""
    train, test = split(snapshots)
    rates = fit_segment_rates(train)
    return rates, brier_score(rates, test)


def brier_score(rates: dict[str, float], test: list[Snapshot]) -> float:
    """Mean squared error between predicted probability and the 0/1 label.

    Lower is better. It rewards probabilities that match what happened.
    """
    squared_error = 0.0
    for row in test:
        error = rates[row.segment] - row.cancelled
        squared_error += error * error
    return squared_error / len(test)
