"""Flatten nested JSON events without silently overwriting columns.

The clickstream loader flattened each event into warehouse columns by
joining nested keys with "_": {"user": {"id": ...}} became "user_id". The
legacy top-level field "user_id" (our internal account id) and the new SDK's
nested "user.id" (a hashed external id) therefore mapped to the same column.
Whichever came last in the JSON won, so many events carried a hashed id in
"user_id" and no longer joined to the accounts table.

Fix: remember which path produced each column and refuse to write a column
twice. Colliding events are quarantined with both paths named, instead of
one value silently replacing the other.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SEPARATOR = "_"


class FlattenCollision(ValueError):
    """Two different JSON paths produced the same column name."""

    def __init__(self, column: str, first_path: str, second_path: str) -> None:
        super().__init__(f"column {column!r} produced by both {first_path!r} and {second_path!r}")
        self.column = column
        self.paths = (first_path, second_path)


def original_flatten(record: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Original loader: join keys with "_"; a later key silently overwrites."""
    flat: dict[str, Any] = {}
    for key, value in record.items():
        column = prefix + key
        if isinstance(value, dict):
            flat.update(original_flatten(value, column + SEPARATOR))
        else:
            flat[column] = value
    return flat


def flatten(record: dict[str, Any]) -> dict[str, Any]:
    """Flatten ``record``; raise FlattenCollision if two paths share a column."""
    flat: dict[str, Any] = {}
    source_paths: dict[str, str] = {}
    _flatten_into(record, [], flat, source_paths)
    return flat


def _flatten_into(node: dict[str, Any], path: list[str], flat: dict[str, Any],
                  source_paths: dict[str, str]) -> None:
    """Walk ``node`` depth-first, writing leaves into ``flat``."""
    for key, value in node.items():
        child_path = path + [key]
        if isinstance(value, dict):
            _flatten_into(value, child_path, flat, source_paths)
            continue
        column = SEPARATOR.join(child_path)
        # WHY: the dotted path shows where each value came from, which is
        # exactly what an engineer needs to fix the collision.
        dotted_path = ".".join(child_path)
        if column in source_paths:
            raise FlattenCollision(column, source_paths[column], dotted_path)
        source_paths[column] = dotted_path
        flat[column] = value


@dataclass
class LoadResult:
    rows: list[dict[str, Any]] = field(default_factory=list)
    quarantined: list[tuple[dict[str, Any], str]] = field(default_factory=list)


def load_events(events: list[dict[str, Any]]) -> LoadResult:
    """Flatten every event; colliding events go to quarantine with the reason."""
    result = LoadResult()
    for event in events:
        try:
            result.rows.append(flatten(event))
        except FlattenCollision as collision:
            result.quarantined.append((event, str(collision)))
    return result


def joinable_share(rows: list[dict[str, Any]], known_account_ids: set[int]) -> float:
    """Fraction of rows whose user_id matches an account (what the dashboard needs)."""
    if not rows:
        return 0.0
    matched = 0
    for row in rows:
        if row.get("user_id") in known_account_ids:
            matched += 1
    return matched / len(rows)
