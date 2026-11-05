"""Parse a vendor CSV export without losing rows that contain commas or newlines.

The nightly product import read the vendor's CSV one line at a time and split
each line on ",". Product descriptions are quoted and often contain commas and
line breaks, so those records were split into pieces with the wrong number of
columns, and the loader quietly skipped "malformed" rows. A few percent of the
catalogue disappeared every night and nobody noticed.

The fix parses with the ``csv`` module (which understands quoting and
embedded newlines), rejects any record with the wrong column count instead of
skipping it, and reconciles the parsed row count with the count the vendor
sends in its manifest, so a silent loss becomes a failed run.
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass

EXPECTED_COLUMNS = ("sku", "name", "description", "price_cents")


class MalformedRecord(ValueError):
    """A record does not have the expected number of columns."""


class RowCountMismatch(ValueError):
    """Parsed row count differs from what the vendor says it sent."""


@dataclass(frozen=True)
class Product:
    sku: str
    name: str
    description: str
    price_cents: int


def naive_parse(text: str) -> list[Product]:
    """The original loader: split on newlines and commas, skip odd rows."""
    products = []
    lines = text.splitlines()
    for line in lines[1:]:                      # skip the header
        fields = line.split(",")
        if len(fields) != len(EXPECTED_COLUMNS):
            continue                            # "malformed", silently dropped
        products.append(Product(fields[0], fields[1], fields[2], int(fields[3])))
    return products


def check_header(header: list[str]) -> None:
    """Fail fast if the vendor renamed, removed or reordered a column."""
    if tuple(header) != EXPECTED_COLUMNS:
        raise MalformedRecord(f"unexpected header {header!r}, want {list(EXPECTED_COLUMNS)}")


def record_to_product(record: list[str], record_number: int) -> Product:
    """Convert one parsed CSV record, refusing records of the wrong width."""
    if len(record) != len(EXPECTED_COLUMNS):
        raise MalformedRecord(
            f"record {record_number} has {len(record)} columns, "
            f"want {len(EXPECTED_COLUMNS)}: {record!r}"
        )
    sku, name, description, price = record
    return Product(sku, name, description, int(price))


def parse_products(text: str) -> list[Product]:
    """Parse the export with real CSV rules (quotes, embedded commas/newlines)."""
    # newline="" is what the csv module expects: it handles line endings
    # itself, which is what lets a quoted field contain a line break.
    reader = csv.reader(io.StringIO(text, newline=""))
    header = next(reader, None)
    if header is None:
        raise MalformedRecord("file is empty, expected a header row")
    check_header(header)
    products = []
    for record_number, record in enumerate(reader, start=1):
        products.append(record_to_product(record, record_number))
    return products


def load_products(text: str, manifest_row_count: int) -> list[Product]:
    """Parse and reconcile against the vendor's manifest before loading."""
    products = parse_products(text)
    if len(products) != manifest_row_count:
        raise RowCountMismatch(
            f"parsed {len(products)} products but the manifest says {manifest_row_count}"
        )
    return products
