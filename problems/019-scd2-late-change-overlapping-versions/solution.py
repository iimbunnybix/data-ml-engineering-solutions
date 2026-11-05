"""Apply late-arriving changes to a type-2 slowly changing dimension safely.

The customer dimension keeps one row per version of a customer, each valid
for [valid_from, valid_to). The original loader always closed the *current*
row at the change's effective time and appended a new current row. When a
change arrived late (effective before the current row started), that produced
overlapping versions, so point-in-time joins returned two rows and doubled
revenue. This module inserts a change into the version whose interval
contains its effective time, splitting that interval, so the history stays a
gap-free, non-overlapping timeline no matter what order changes arrive in.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

# Open-ended upper bound for the current version (like 9999-12-31 in SQL).
END_OF_TIME = 10**18


@dataclass(frozen=True)
class Version:
    customer_id: str
    tier: str
    valid_from: int      # effective time, inclusive (epoch seconds)
    valid_to: int        # exclusive; END_OF_TIME for the current version


@dataclass(frozen=True)
class Change:
    customer_id: str
    tier: str
    effective_from: int


def naive_apply(history: list[Version], change: Change) -> list[Version]:
    """Original loader: close the current version and append a new one.

    Correct only if changes arrive in effective-time order.
    """
    result = []
    for version in history:
        if version.customer_id == change.customer_id and version.valid_to == END_OF_TIME:
            version = replace(version, valid_to=change.effective_from)
        result.append(version)
    result.append(Version(change.customer_id, change.tier, change.effective_from, END_OF_TIME))
    return result


def _split_containing_version(versions: list[Version], change: Change) -> list[Version]:
    """Cut the version that contains the change's effective time in two.

    The part before the change keeps the old tier; the part after takes the
    new tier and keeps the old end, so later versions are untouched.
    """
    result = []
    for version in versions:
        contains = version.valid_from < change.effective_from < version.valid_to
        if not contains:
            result.append(version)
            continue
        result.append(replace(version, valid_to=change.effective_from))
        result.append(replace(version, tier=change.tier, valid_from=change.effective_from))
    return result


def _apply_to_one_customer(versions: list[Version], change: Change) -> list[Version]:
    """Insert one change into a single customer's sorted, non-overlapping versions."""
    if not versions:
        return [Version(change.customer_id, change.tier, change.effective_from, END_OF_TIME)]

    # Same effective time as an existing version: this is a correction of
    # that version, not a new one, so replacing the tier keeps reloads idempotent.
    for index, version in enumerate(versions):
        if version.valid_from == change.effective_from:
            versions[index] = replace(version, tier=change.tier)
            return versions

    first = versions[0]
    if change.effective_from < first.valid_from:
        # Change predates all known history: it is valid until the first version starts.
        earliest = Version(change.customer_id, change.tier, change.effective_from, first.valid_from)
        return [earliest, *versions]

    return _split_containing_version(versions, change)


def apply_change(history: list[Version], change: Change) -> list[Version]:
    """Return a new history with the change applied at its effective time."""
    others = [v for v in history if v.customer_id != change.customer_id]
    mine = sorted((v for v in history if v.customer_id == change.customer_id),
                  key=lambda version: version.valid_from)
    return others + _apply_to_one_customer(mine, change)


def versions_as_of(history: list[Version], customer_id: str, as_of: int) -> list[Version]:
    """All versions valid at `as_of`; a healthy dimension returns exactly one."""
    matches = []
    for version in history:
        if version.customer_id == customer_id and version.valid_from <= as_of < version.valid_to:
            matches.append(version)
    return matches


def find_overlaps(history: list[Version]) -> list[tuple[Version, Version]]:
    """Pairs of versions of the same customer whose intervals overlap or are inverted.

    Run this as a data-quality check after every load.
    """
    problems = []
    by_customer: dict[str, list[Version]] = {}
    for version in history:
        by_customer.setdefault(version.customer_id, []).append(version)
    for versions in by_customer.values():
        versions.sort(key=lambda version: (version.valid_from, version.valid_to))
        for earlier, later in zip(versions, versions[1:]):
            inverted = earlier.valid_to < earlier.valid_from
            if inverted or earlier.valid_to > later.valid_from:
                problems.append((earlier, later))
    return problems
