"""Revenue-by-region join that silently lost and mis-attributed orders.

The daily revenue report joins order facts to the customer dimension on
``customer_id``. Order keys arrive as ``None``, ``""``, ``" C1 "`` or ``"c2"``,
so the raw-key inner join drops them, and a placeholder dimension row whose
key is ``None`` "matches" every order with a missing key. This module keeps
that original join for comparison and adds a fixed join that normalises keys,
never lets a missing key match anything, routes unjoinable rows to explicit
buckets and refuses to publish unless input and output totals reconcile.
"""
from __future__ import annotations

from dataclasses import dataclass

# Buckets for rows that cannot be attributed to a real region. They are
# reported, not dropped, so the report total always equals the input total.
MISSING_KEY_BUCKET = "MISSING_KEY"
UNMATCHED_KEY_BUCKET = "UNMATCHED_KEY"


class ReconciliationError(Exception):
    """Raised when the joined report does not add up to the input facts."""


@dataclass(frozen=True)
class RegionReport:
    """Revenue per region in integer cents, plus how many rows went where."""

    revenue_cents: dict[str, int]
    rows_per_bucket: dict[str, int]


def naive_revenue_by_region(orders: list[dict], customers: list[dict]) -> dict[str, int]:
    """Original job: inner join on the raw key, exactly as the SQL did.

    Rows whose key is not found are silently skipped (inner join), and a
    ``None`` key in the dimension matches every ``None`` order key.
    """
    region_by_customer = {}
    for customer in customers:
        region_by_customer[customer["customer_id"]] = customer["region"]

    revenue: dict[str, int] = {}
    for order in orders:
        key = order["customer_id"]
        if key not in region_by_customer:
            continue
        region = region_by_customer[key]
        revenue[region] = revenue.get(region, 0) + order["amount_cents"]
    return revenue


def normalize_key(raw_key: object) -> str | None:
    """Return a canonical customer key, or None when the key is missing.

    Customer IDs are case-insensitive codes, so surrounding whitespace and
    letter case are not meaningful. An empty or blank string is as missing
    as NULL and must be treated the same way.
    """
    if raw_key is None:
        return None
    text = str(raw_key).strip()
    if text == "":
        return None
    return text.upper()


def build_region_lookup(customers: list[dict]) -> dict[str, str]:
    """Map normalised customer key -> region, skipping rows with no key.

    A dimension row without a key can never be a valid join target; letting
    it in is what made the "Test accounts" row absorb every keyless order.
    Duplicate keys after normalisation are a data bug, so fail loudly.
    """
    lookup: dict[str, str] = {}
    for customer in customers:
        key = normalize_key(customer["customer_id"])
        if key is None:
            continue
        if key in lookup and lookup[key] != customer["region"]:
            raise ValueError(f"customer key {key!r} maps to two regions")
        lookup[key] = customer["region"]
    return lookup


def assign_bucket(order: dict, lookup: dict[str, str]) -> str:
    """Pick the region (or explicit problem bucket) one order belongs to."""
    key = normalize_key(order["customer_id"])
    if key is None:
        return MISSING_KEY_BUCKET
    if key not in lookup:
        return UNMATCHED_KEY_BUCKET
    return lookup[key]


def reconcile(orders: list[dict], report: RegionReport) -> None:
    """Fail unless every input row and every cent appears in the report."""
    input_rows = len(orders)
    input_cents = 0
    for order in orders:
        input_cents += order["amount_cents"]

    output_rows = sum(report.rows_per_bucket.values())
    output_cents = sum(report.revenue_cents.values())
    if input_rows != output_rows or input_cents != output_cents:
        raise ReconciliationError(
            f"input {input_rows} rows / {input_cents} cents, "
            f"report {output_rows} rows / {output_cents} cents")


def revenue_by_region(orders: list[dict], customers: list[dict]) -> RegionReport:
    """Fixed job: normalised left join with explicit buckets, then reconcile."""
    lookup = build_region_lookup(customers)
    revenue: dict[str, int] = {}
    rows: dict[str, int] = {}
    for order in orders:
        bucket = assign_bucket(order, lookup)
        revenue[bucket] = revenue.get(bucket, 0) + order["amount_cents"]
        rows[bucket] = rows.get(bucket, 0) + 1

    report = RegionReport(revenue_cents=revenue, rows_per_bucket=rows)
    # Checked on every run so a future code change that drops rows is caught
    # before the report is published, not by finance a month later.
    reconcile(orders, report)
    return report
