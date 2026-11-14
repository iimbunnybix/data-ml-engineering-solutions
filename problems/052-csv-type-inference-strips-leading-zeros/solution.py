"""Load CSV identifier columns as text so leading zeros survive.

A nightly job loaded partner order files with automatic type inference: any
value made only of digits became an integer. ZIP code "02134" became 2134 and
SKU "007" became 7, the same as SKU "7". Orders from ZIP codes starting with 0
then failed the join to the region table, and two different products were
merged in revenue reports. This module keeps that inferring loader and adds a
schema-driven loader that parses only declared numeric columns, keeps every
other column as the exact text, and validates fixed-width identifiers.
"""
from __future__ import annotations

import csv
import io
from typing import Callable

UNKNOWN_REGION = "unknown"
ZIP_CODE_LENGTH = 5

# The fixed loader's contract: only these columns are numbers. Everything not
# listed stays a string, because "looks like a number" is not "is a number".
ORDER_SCHEMA: dict[str, Callable[[str], object]] = {
    "order_id": str,
    "zip_code": str,
    "sku": str,
    "amount": float,
}


def infer_value(raw: str) -> object:
    """Original type inference: digits become int, decimals become float."""
    if raw.isdigit():
        return int(raw)
    try:
        return float(raw)
    except ValueError:
        return raw


def naive_load_orders(csv_text: str) -> list[dict[str, object]]:
    """Original loader: infer every column's type from its value."""
    rows = []
    for record in csv.DictReader(io.StringIO(csv_text)):
        row = {}
        for column, raw in record.items():
            row[column] = infer_value(raw)
        rows.append(row)
    return rows


def check_columns(header: list[str], schema: dict[str, Callable[[str], object]]) -> None:
    """Fail loudly if the file's columns differ from the schema, instead of guessing."""
    if sorted(header) != sorted(schema):
        raise ValueError(f"columns {header} do not match schema {sorted(schema)}")


def validate_zip_code(zip_code: str) -> None:
    """A US ZIP is exactly five digits; a four-digit one means zeros were already lost upstream."""
    if len(zip_code) != ZIP_CODE_LENGTH or not zip_code.isdigit():
        raise ValueError(f"invalid ZIP code {zip_code!r}")


def load_orders(csv_text: str,
                schema: dict[str, Callable[[str], object]] = ORDER_SCHEMA) -> list[dict[str, object]]:
    """Fixed loader: convert each column with the schema's parser, validate ZIP codes."""
    reader = csv.DictReader(io.StringIO(csv_text))
    check_columns(list(reader.fieldnames or []), schema)
    rows = []
    for record in reader:
        row = {}
        for column, raw in record.items():
            row[column] = schema[column](raw)
        validate_zip_code(str(row["zip_code"]))
        rows.append(row)
    return rows


def region_for_each_order(orders: list[dict[str, object]],
                          region_by_zip: dict[str, str]) -> list[str]:
    """Join orders to the region table, which is keyed by the 5-character ZIP text."""
    regions = []
    for order in orders:
        # The naive loader hands us an int; str(2134) is "2134", not "02134".
        zip_key = str(order["zip_code"])
        regions.append(region_by_zip.get(zip_key, UNKNOWN_REGION))
    return regions


def revenue_by_sku(orders: list[dict[str, object]]) -> dict[str, float]:
    """Sum order amounts per SKU, keyed by the SKU as text."""
    totals: dict[str, float] = {}
    for order in orders:
        sku = str(order["sku"])
        totals[sku] = totals.get(sku, 0.0) + float(order["amount"])
    return totals
