"""Group sales by unit price without float keys splitting one price into several rows.

The "units sold per price point" report computed each line's unit price as
``float(line_total) / quantity`` and grouped on that float. 2.10 / 3 is
0.7000000000000001 while a single item at 0.70 is 0.7, so one price point
became two groups that both print as "$0.70", and each row undercounted.
This module keeps that original report and fixes it by parsing money as
Decimal and rounding the unit price to whole cents before using it as a key.
"""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal

# Group keys are whole cents; this is the quantum Decimal rounds to.
CENT = Decimal("0.01")
CENTS_PER_DOLLAR = 100


@dataclass(frozen=True)
class SaleLine:
    """One order line as it arrives from the CSV export: money is still text."""

    line_total: str
    quantity: int


def naive_units_by_price(lines: list[SaleLine]) -> Counter:
    """Original report: unit price as a float, used directly as the group key."""
    units: Counter = Counter()
    for line in lines:
        unit_price = float(line.line_total) / line.quantity
        units[unit_price] += line.quantity
    return units


def unit_price_cents(line: SaleLine) -> Decimal:
    """Exact unit price rounded to the cent, so equal prices give equal keys.

    Decimal("2.10") is exactly 2.10 (a float is not), and quantizing to CENT
    turns 0.70 and 0.7000... into the same key. Half-even (banker's) rounding
    matches how most finance systems round and avoids an upward bias.
    """
    if line.quantity <= 0:
        raise ValueError(f"quantity must be positive, got {line.quantity}")
    exact_total = Decimal(line.line_total)
    unit_price = exact_total / line.quantity
    return unit_price.quantize(CENT, rounding=ROUND_HALF_EVEN)


def units_by_price(lines: list[SaleLine]) -> Counter:
    """Fixed report: one group per real price point."""
    units: Counter = Counter()
    for line in lines:
        units[unit_price_cents(line)] += line.quantity
    return units


def format_price(price: float | Decimal) -> str:
    """How the dashboard prints a price key."""
    return f"${price:.2f}"


def duplicate_labels(report: Counter) -> list[str]:
    """Labels that appear on more than one row: a sign the keys are not exact."""
    label_counts: Counter = Counter()
    for price in report:
        label_counts[format_price(price)] += 1
    duplicates = []
    for label, count in sorted(label_counts.items()):
        if count > 1:
            duplicates.append(label)
    return duplicates


def make_sales(rng: random.Random, line_count: int, price_points_cents: list[int],
               max_quantity: int) -> list[SaleLine]:
    """Deterministic sales lines: a catalog price times a random quantity."""
    lines = []
    for _ in range(line_count):
        price_cents = rng.choice(price_points_cents)
        quantity = rng.randint(1, max_quantity)
        total_cents = price_cents * quantity
        # Format from integer cents so the CSV text is exactly what a POS would export.
        dollars, cents = divmod(total_cents, CENTS_PER_DOLLAR)
        lines.append(SaleLine(f"{dollars}.{cents:02d}", quantity))
    return lines
