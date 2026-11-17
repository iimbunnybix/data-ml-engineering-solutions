"""Fingerprint JSON records by content, not by how they happened to be serialised.

An ingestion job deduplicated payment events by hashing the raw JSON text.
The same event re-sent by a retrying producer (a different SDK version, or a
replay through another service) had its keys in another order, different
whitespace, 250.0 instead of 250, or a fresh ingested_at timestamp, so its
hash differed and it was counted twice. This module builds a canonical form
first: volatile metadata removed, keys sorted, compact separators, and
whole-number floats written as integers. Equal events then hash equally.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Callable

# Fields the transport adds on every delivery; they are not part of the event.
VOLATILE_FIELDS = frozenset({"ingested_at", "delivery_attempt", "producer_host"})


def naive_fingerprint(raw_json: str) -> str:
    """Original code: hash the bytes exactly as received."""
    return hashlib.sha256(raw_json.encode("utf-8")).hexdigest()


def normalise_value(value: Any) -> Any:
    """Return value with numbers in one canonical form, recursing into containers."""
    # bool is an int subclass; leave True/False alone so they never become 1/0.
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("NaN/Infinity is not valid in an event")
        # 250.0 and 250 are the same amount; one SDK writes floats, one ints.
        if value.is_integer():
            return int(value)
        return value
    if isinstance(value, dict):
        return {key: normalise_value(item) for key, item in value.items()}
    if isinstance(value, list):
        # List order is meaningful (e.g. line items), so it is kept as is.
        return [normalise_value(item) for item in value]
    return value


def canonical_json(record: dict[str, Any]) -> str:
    """One fixed text form per logical event."""
    content = {}
    for key, value in record.items():
        if key not in VOLATILE_FIELDS:
            content[key] = normalise_value(value)
    # sort_keys fixes key order at every nesting level; compact separators fix
    # whitespace; ensure_ascii=False so "é" and "é" produce the same text.
    return json.dumps(content, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def fingerprint(raw_json: str) -> str:
    """Hash of the canonical form of one JSON object."""
    record = json.loads(raw_json)
    if not isinstance(record, dict):
        raise ValueError("each event must be a JSON object")
    return hashlib.sha256(canonical_json(record).encode("utf-8")).hexdigest()


def deduplicate(raw_events: list[str],
                fingerprint_function: Callable[[str], str] = fingerprint) -> list[str]:
    """Keep the first copy of every distinct event, in arrival order."""
    seen: set[str] = set()
    kept: list[str] = []
    for raw_event in raw_events:
        key = fingerprint_function(raw_event)
        if key in seen:
            continue
        seen.add(key)
        kept.append(raw_event)
    return kept


def total_amount(raw_events: list[str]) -> float:
    """Sum the amount field, as the daily revenue report does."""
    total = 0.0
    for raw_event in raw_events:
        total += json.loads(raw_event)["amount"]
    return total
