"""Deduplicate a MERGE source batch per key before upserting it.

A billing partner's daily export lists every status change of the day, so one
subscription can appear several times in the same batch. The loader upserted
rows in file order and the last row processed won. File parts are loaded in
parallel, so "last" was effectively random and an older status often
overwrote the newest one. This module keeps that naive merge for comparison
and fixes it: pick exactly one row per key by the latest updated_at,
quarantine keys whose latest rows tie with different contents, and never let
a source row overwrite a newer target row.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

STATUSES = ("trial", "active", "past_due", "cancelled")
MINUTES_PER_DAY = 24 * 60
MAX_CHANGES_PER_DAY = 4


@dataclass(frozen=True)
class StatusChange:
    """One row of the export. updated_at is minutes since the start of the day."""
    subscription_id: int
    status: str
    updated_at: int


@dataclass
class DedupResult:
    """One winning row per key, plus keys that need a human decision."""
    winners: dict[int, StatusChange]
    conflicts: dict[int, list[StatusChange]]


def naive_merge(target: dict[int, StatusChange], batch: list[StatusChange]) -> None:
    """Original loader: upsert rows in the order they arrive; the last one wins."""
    for row in batch:
        target[row.subscription_id] = row


def latest_rows_per_key(batch: list[StatusChange]) -> dict[int, list[StatusChange]]:
    """For each key, every distinct row that carries the key's latest updated_at."""
    latest: dict[int, list[StatusChange]] = {}
    for row in batch:
        current = latest.get(row.subscription_id)
        if current is None or row.updated_at > current[0].updated_at:
            latest[row.subscription_id] = [row]
        elif row.updated_at == current[0].updated_at and row not in current:
            # Same timestamp, different content: keep both so it can be reported.
            current.append(row)
    return latest


def dedupe_batch(batch: list[StatusChange]) -> DedupResult:
    """Choose one row per key; the result does not depend on row order.

    Exact duplicates collapse to one row. Two different rows with the same
    latest timestamp cannot be ordered from the data, so the key is
    quarantined instead of guessed.
    """
    winners: dict[int, StatusChange] = {}
    conflicts: dict[int, list[StatusChange]] = {}
    for key, rows in latest_rows_per_key(batch).items():
        if len(rows) == 1:
            winners[key] = rows[0]
        else:
            conflicts[key] = sorted(rows, key=lambda row: row.status)
    return DedupResult(winners, conflicts)


def fixed_merge(target: dict[int, StatusChange], batch: list[StatusChange]) -> DedupResult:
    """Deduplicate the batch, then upsert only rows newer than the target's.

    The second check protects against an old file being reloaded after a
    newer one (a backfill or a retried job).
    """
    result = dedupe_batch(batch)
    for key, row in result.winners.items():
        existing = target.get(key)
        if existing is None or row.updated_at >= existing.updated_at:
            target[key] = row
    return result


def simulate_daily_export(subscription_count: int, rng: random.Random) -> list[StatusChange]:
    """Each subscription changes status 1-4 times; rows come back shuffled.

    Shuffling stands in for file parts loaded in parallel: the order rows
    reach the loader is not the order the changes happened.
    """
    batch = []
    for subscription_id in range(subscription_count):
        change_count = rng.randint(1, MAX_CHANGES_PER_DAY)
        minutes = sorted(rng.sample(range(MINUTES_PER_DAY), change_count))
        for minute in minutes:
            batch.append(StatusChange(subscription_id, rng.choice(STATUSES), minute))
    rng.shuffle(batch)
    return batch


def true_final_status(batch: list[StatusChange]) -> dict[int, StatusChange]:
    """Ground truth for the simulation: each key's row with the latest timestamp."""
    truth: dict[int, StatusChange] = {}
    for row in batch:
        current = truth.get(row.subscription_id)
        if current is None or row.updated_at > current.updated_at:
            truth[row.subscription_id] = row
    return truth


def count_wrong_keys(target: dict[int, StatusChange], truth: dict[int, StatusChange]) -> int:
    """How many keys in the target do not hold their true latest row."""
    wrong = 0
    for key, expected in truth.items():
        if target.get(key) != expected:
            wrong += 1
    return wrong
