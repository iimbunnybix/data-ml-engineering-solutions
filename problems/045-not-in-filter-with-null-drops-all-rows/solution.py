"""Stop one NULL in a blocklist from emptying the daily "exclude blocked users" output.

The pipeline removed blocklisted users with ``WHERE user_id NOT IN (SELECT
user_id FROM blocklist)``. In SQL, ``x NOT IN (a, b, NULL)`` is never TRUE: it
is FALSE when x matches and UNKNOWN otherwise, and WHERE keeps only TRUE rows.
So a single NULL row in the blocklist silently removed every event. This module
models SQL's three-valued logic, keeps the original NOT IN filter for
comparison, and fixes it with NOT EXISTS (anti-join) semantics plus a guard that
reports NULL blocklist entries so the bad row is fixed at the source.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Truth(Enum):
    """SQL's three truth values. WHERE keeps a row only when the result is TRUE."""

    TRUE = "TRUE"
    FALSE = "FALSE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class Event:
    """One row of the daily events table. ``user_id`` may be NULL (None)."""

    event_id: int
    user_id: int | None


def sql_equals(left: int | None, right: int | None) -> Truth:
    """SQL ``left = right``: comparing anything with NULL is UNKNOWN, not FALSE."""
    if left is None or right is None:
        return Truth.UNKNOWN
    if left == right:
        return Truth.TRUE
    return Truth.FALSE


def sql_not_in(value: int | None, candidates: list[int | None]) -> Truth:
    """SQL ``value NOT IN (candidates)``, i.e. NOT (value = c1 OR value = c2 ...)."""
    saw_unknown = False
    for candidate in candidates:
        comparison = sql_equals(value, candidate)
        if comparison is Truth.TRUE:
            # A definite match makes NOT IN definitely FALSE.
            return Truth.FALSE
        if comparison is Truth.UNKNOWN:
            saw_unknown = True
    # No definite match: if any comparison was UNKNOWN, SQL cannot say the
    # value is absent, so the whole predicate is UNKNOWN (and the row is dropped).
    if saw_unknown:
        return Truth.UNKNOWN
    return Truth.TRUE


def naive_exclude_blocked(events: list[Event], blocklist: list[int | None]) -> list[Event]:
    """Original query: ``WHERE user_id NOT IN (SELECT user_id FROM blocklist)``."""
    kept = []
    for event in events:
        if sql_not_in(event.user_id, blocklist) is Truth.TRUE:
            kept.append(event)
    return kept


def build_blocked_set(blocklist: list[int | None]) -> set[int]:
    """Blocked ids for the anti-join; NULL entries can never match a row, so skip them."""
    blocked = set()
    for user_id in blocklist:
        if user_id is not None:
            blocked.add(user_id)
    return blocked


def exclude_blocked(events: list[Event], blocklist: list[int | None]) -> list[Event]:
    """Fixed query: ``WHERE NOT EXISTS (SELECT 1 FROM blocklist b WHERE b.user_id = e.user_id)``.

    NOT EXISTS asks "is there a matching blocklist row?", which is a plain yes
    or no, so a NULL in the blocklist matches nothing and removes nothing.
    Events with a NULL user_id are kept, because no blocklist row equals them.
    """
    blocked = build_blocked_set(blocklist)
    kept = []
    for event in events:
        if event.user_id is None or event.user_id not in blocked:
            kept.append(event)
    return kept


def count_null_entries(blocklist: list[int | None]) -> int:
    """How many blocklist rows have no user id (bad data that should be fixed upstream)."""
    null_count = 0
    for user_id in blocklist:
        if user_id is None:
            null_count += 1
    return null_count


def blocklist_warnings(blocklist: list[int | None]) -> list[str]:
    """Data-quality warnings to send to the pipeline's alert channel."""
    null_count = count_null_entries(blocklist)
    if null_count == 0:
        return []
    return [f"blocklist has {null_count} row(s) with NULL user_id; "
            "NOT IN filters would drop every event"]


def make_events(user_count: int, events_per_user: int) -> list[Event]:
    """Deterministic daily events: every user has the same number of events."""
    events = []
    event_id = 0
    for user_id in range(1, user_count + 1):
        for _ in range(events_per_user):
            events.append(Event(event_id, user_id))
            event_id += 1
    return events
