"""Incremental table sync that also removes rows deleted at the source.

An ELT job copied the `customers` table into the warehouse by selecting rows
with `updated_at > watermark` and upserting them. A hard `DELETE` at the source
leaves no row behind to select, so deleted customers (closed accounts, GDPR
erasures) stayed in the warehouse forever and were still counted as active.
This module keeps that upsert-only sync and adds a key reconciliation pass:
compare the source's current primary keys with the warehouse's and delete the
missing ones, with a safety limit so a broken or empty extract cannot wipe the
table.
"""
from __future__ import annotations

from dataclasses import dataclass

# If more than this share of warehouse rows would be deleted in one run, the
# key extract is more likely broken (empty result, wrong filter) than real.
MAX_DELETE_FRACTION = 0.05


@dataclass(frozen=True)
class CustomerRow:
    customer_id: int
    status: str
    updated_at: int          # seconds since epoch; set by the source on every change


class UnsafeDeleteError(Exception):
    """Raised when reconciliation would delete a suspiciously large share of rows."""


def changed_rows(source: dict[int, CustomerRow], watermark: int) -> list[CustomerRow]:
    """What `SELECT * FROM customers WHERE updated_at > :watermark` returns."""
    rows = []
    for row in source.values():
        if row.updated_at > watermark:
            rows.append(row)
    return rows


def naive_incremental_sync(source: dict[int, CustomerRow],
                           warehouse: dict[int, CustomerRow], watermark: int) -> int:
    """Original job: upsert changed rows, return the new watermark.

    A deleted row is simply absent from `source`, so nothing here ever
    removes it from `warehouse`.
    """
    new_watermark = watermark
    for row in changed_rows(source, watermark):
        warehouse[row.customer_id] = row
        new_watermark = max(new_watermark, row.updated_at)
    return new_watermark


def keys_to_delete(source_keys: set[int], warehouse: dict[int, CustomerRow]) -> list[int]:
    """Warehouse keys that no longer exist at the source (an anti-join on the key)."""
    missing = []
    for customer_id in warehouse:
        if customer_id not in source_keys:
            missing.append(customer_id)
    return sorted(missing)


def check_delete_is_safe(delete_count: int, warehouse_size: int) -> None:
    """Refuse to delete more than MAX_DELETE_FRACTION of the table in one run.

    An extract that silently returned zero keys would otherwise look like
    "every customer was deleted" and empty the warehouse table.
    """
    if warehouse_size == 0:
        return
    if delete_count / warehouse_size > MAX_DELETE_FRACTION:
        raise UnsafeDeleteError(
            f"refusing to delete {delete_count} of {warehouse_size} rows; "
            "check the source key extract")


def reconcile_deletes(source_keys: set[int], warehouse: dict[int, CustomerRow]) -> list[int]:
    """Delete warehouse rows whose key is gone from the source; return the deleted keys."""
    missing = keys_to_delete(source_keys, warehouse)
    check_delete_is_safe(len(missing), len(warehouse))
    for customer_id in missing:
        del warehouse[customer_id]
    return missing


def incremental_sync(source: dict[int, CustomerRow],
                     warehouse: dict[int, CustomerRow], watermark: int) -> int:
    """Fixed job: upsert changed rows, then reconcile deletes by primary key.

    The key list is cheap to extract (`SELECT customer_id FROM customers`, an
    index-only scan) compared with re-copying whole rows.
    """
    new_watermark = naive_incremental_sync(source, warehouse, watermark)
    reconcile_deletes(set(source.keys()), warehouse)
    return new_watermark


def active_count(warehouse: dict[int, CustomerRow]) -> int:
    """The dashboard metric that the stale rows inflated."""
    count = 0
    for row in warehouse.values():
        if row.status == "active":
            count += 1
    return count
