"""Assign users to concurrent A/B experiments independently of each other.

Every experiment on the platform put a user in treatment when
hash(user_id) % 100 was below 50. Because the hash ignored the experiment,
the same half of the users was in treatment for every test at once, so the
effect of one experiment (a free-shipping banner) showed up as a fake win for
another one (a new button colour) that changed nothing. This module salts the
bucket hash with the experiment id, which makes assignments in different
experiments independent, and adds a chi-square check that flags experiments
whose assignments are correlated.
"""
from __future__ import annotations

import hashlib
import random
from collections.abc import Callable

BUCKET_COUNT = 100
TREATMENT_BUCKETS = 50            # buckets 0-49 get the treatment: a 50/50 split
# Chi-square critical value for 1 degree of freedom at p = 0.05.
CHI_SQUARE_CRITICAL_1_DOF = 3.841
BASE_CONVERSION_RATE = 0.10
FREE_SHIPPING_LIFT = 0.03         # true effect of the banner experiment
BUTTON_COLOUR_LIFT = 0.0          # the button colour truly changes nothing


def bucket_for(salt: str, user_id: int) -> int:
    """Stable bucket in [0, BUCKET_COUNT) from a cryptographic hash."""
    digest = hashlib.sha256(f"{salt}:{user_id}".encode()).hexdigest()
    # The first 8 hex digits (32 bits) are plenty for 100 buckets.
    return int(digest[:8], 16) % BUCKET_COUNT


def naive_in_treatment(experiment_id: str, user_id: int) -> bool:
    """Original assignment: the experiment id is not part of the hash."""
    return bucket_for("", user_id) < TREATMENT_BUCKETS


def in_treatment(experiment_id: str, user_id: int) -> bool:
    """Fixed assignment: each experiment hashes with its own salt."""
    return bucket_for(experiment_id, user_id) < TREATMENT_BUCKETS


def simulate_conversions(user_ids: list[int], banner_treated: set[int],
                         seed: int) -> set[int]:
    """Users who bought something; only the banner experiment has a real effect."""
    generator = random.Random(seed)
    converted = set()
    for user_id in user_ids:
        rate = BASE_CONVERSION_RATE + BUTTON_COLOUR_LIFT
        if user_id in banner_treated:
            rate += FREE_SHIPPING_LIFT
        if generator.random() < rate:
            converted.add(user_id)
    return converted


def conversion_rate(group: list[int], converted: set[int]) -> float:
    """Share of the group that converted (0.0 for an empty group)."""
    if not group:
        return 0.0
    hits = 0
    for user_id in group:
        if user_id in converted:
            hits += 1
    return hits / len(group)


def measured_lift(user_ids: list[int], treated: set[int], converted: set[int]) -> float:
    """Treatment conversion rate minus control conversion rate."""
    treatment_group = []
    control_group = []
    for user_id in user_ids:
        if user_id in treated:
            treatment_group.append(user_id)
        else:
            control_group.append(user_id)
    return conversion_rate(treatment_group, converted) - conversion_rate(control_group, converted)


def contingency_table(user_ids: list[int], treated_a: set[int],
                      treated_b: set[int]) -> list[list[int]]:
    """2x2 counts: rows = in A treatment (no, yes), columns = in B treatment (no, yes)."""
    table = [[0, 0], [0, 0]]
    for user_id in user_ids:
        row = 1 if user_id in treated_a else 0
        column = 1 if user_id in treated_b else 0
        table[row][column] += 1
    return table


def chi_square_statistic(table: list[list[int]]) -> float:
    """Pearson chi-square for a 2x2 table; 0 means the assignments look independent."""
    total = sum(sum(row) for row in table)
    row_totals = [sum(row) for row in table]
    column_totals = [table[0][column] + table[1][column] for column in range(2)]
    statistic = 0.0
    for row in range(2):
        for column in range(2):
            expected = row_totals[row] * column_totals[column] / total
            # A zero expected count means everyone is on one side of an
            # experiment; that is a broken split, reported as maximally dependent.
            if expected == 0:
                return float("inf")
            statistic += (table[row][column] - expected) ** 2 / expected
    return statistic


def assignments_are_independent(user_ids: list[int], treated_a: set[int],
                                treated_b: set[int]) -> bool:
    """Pre-launch check: False if two experiments' assignments are correlated."""
    if not user_ids:
        raise ValueError("need at least one user to check independence")
    table = contingency_table(user_ids, treated_a, treated_b)
    return chi_square_statistic(table) < CHI_SQUARE_CRITICAL_1_DOF


def treated_users(user_ids: list[int], experiment_id: str,
                  assign: Callable[[str, int], bool]) -> set[int]:
    """Set of users the given assignment function puts in treatment."""
    treated = set()
    for user_id in user_ids:
        if assign(experiment_id, user_id):
            treated.add(user_id)
    return treated
