"""Deduplicate events by (source, event_id), not by event_id alone.

An order pipeline merges events from two regional checkout services and drops
duplicates (at-least-once delivery retries) by remembering every event_id it
has seen. Each region numbers its events independently starting at 1, so an
EU order and a US order routinely share an id and the second one to arrive was
silently thrown away as a "duplicate" of a different order.

Fix: an event's identity is the pair (source, event_id). The dedup key is a
tuple of both parts, so ids are only compared within the producer that issued
them, and an event with no source is rejected instead of guessed.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OrderEvent:
    source: str       # the producer that issued event_id, e.g. "checkout-eu"
    event_id: str     # unique only within its source
    amount_cents: int


class MissingSourceError(ValueError):
    """Raised when an event cannot be identified because it has no source."""


# --------------------------------------------------------------- original bug
def naive_dedup(events: list[OrderEvent]) -> list[OrderEvent]:
    """Original: keep the first event for each event_id, whatever its source."""
    seen_ids: set[str] = set()
    kept = []
    for event in events:
        # BUG: event_id is only unique per producer, so two different orders
        # from two regions collide here and the later one is dropped.
        if event.event_id in seen_ids:
            continue
        seen_ids.add(event.event_id)
        kept.append(event)
    return kept


# ----------------------------------------------------------------------- fix
def dedup_key(event: OrderEvent) -> tuple[str, str]:
    """The identity of an event: which producer issued it, plus that producer's id."""
    if not event.source:
        # Without a source we cannot tell a retry from another region's order.
        raise MissingSourceError(f"event {event.event_id!r} has no source")
    # A tuple, not f"{source}-{event_id}": a joined string makes
    # ("eu", "1-2") and ("eu-1", "2") the same key.
    return (event.source, event.event_id)


def dedup(events: list[OrderEvent]) -> list[OrderEvent]:
    """Keep the first event for each (source, event_id); drop real retries only."""
    seen_keys: set[tuple[str, str]] = set()
    kept = []
    for event in events:
        key = dedup_key(event)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        kept.append(event)
    return kept


# ------------------------------------------------------------------- helpers
def total_revenue_cents(events: list[OrderEvent]) -> int:
    """Revenue the finance dashboard reports from the deduplicated table."""
    total = 0
    for event in events:
        total += event.amount_cents
    return total
