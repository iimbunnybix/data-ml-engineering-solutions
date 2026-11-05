"""Daily ledger totals that are the same on every rerun and in every tool.

The batch job summed a day's wire amounts as floats: each partition added its
rows into a running float, and the partition results were then added in the
order the tasks finished. Float addition rounds after every step, so the
result depends on the order of the additions. Reruns (different shuffle and
task order) and the dashboard (one pass in event order) each got slightly
different totals, and the reconciliation check between them flapped.

The fix parses each amount once into integer minor units (cents). Integer
addition is exact, so the total is the same in any order, and the
reconciliation can demand exact equality. ``fsum_total`` shows the
alternative for data that is already float.
"""
from __future__ import annotations

import math
from decimal import Decimal, InvalidOperation

MINOR_UNITS_PER_MAJOR = 100            # cents per dollar
CENT = Decimal("0.01")


# --------------------------------------------------------------- original bug
def naive_float_total(amounts: list[float]) -> float:
    """Running float sum, like ``SUM(amount)`` over a DOUBLE column.

    WHY an explicit loop: since Python 3.12 the built-in ``sum()`` uses
    compensated summation for floats, which would hide the problem; Spark and
    SQL engines add DOUBLE values one at a time like this.
    """
    total = 0.0
    for amount in amounts:
        total += amount
    return total


def naive_partitioned_total(partitions: list[list[float]], finish_order: list[int]) -> float:
    """Original batch job: sum each partition, then add the partials as tasks finish."""
    total = 0.0
    for index in finish_order:
        total += naive_float_total(partitions[index])
    return total


def naive_reconciles(batch_total: float, dashboard_total: float) -> bool:
    """Original check: the two float totals must be equal to the cent."""
    return round(batch_total, 2) == round(dashboard_total, 2)


# ------------------------------------------------------------------------ fix
def to_minor_units(amount: str) -> int:
    """Parse a decimal amount string such as ``"1234.50"`` into integer cents.

    ``Decimal`` reads the text exactly (no binary rounding). Amounts with
    more than two decimals are rejected rather than silently rounded, because
    a sub-cent value means the upstream format is not what we think it is.
    """
    try:
        value = Decimal(amount.strip())
    except InvalidOperation:
        raise ValueError(f"not a decimal amount: {amount!r}") from None
    if not value.is_finite():
        raise ValueError(f"amount must be finite: {amount!r}")
    if value != value.quantize(CENT):
        raise ValueError(f"more than two decimals: {amount!r}")
    return int(value * MINOR_UNITS_PER_MAJOR)


def exact_partition_total(amounts: list[str]) -> int:
    """Total of one partition in cents. Integer addition never rounds."""
    total = 0
    for amount in amounts:
        total += to_minor_units(amount)
    return total


def exact_partitioned_total(partitions: list[list[str]], finish_order: list[int]) -> int:
    """Batch job with exact partials: any finish order gives the same cents."""
    total = 0
    for index in finish_order:
        total += exact_partition_total(partitions[index])
    return total


def reconciles(batch_cents: int, dashboard_cents: int) -> bool:
    """Exact totals can be compared exactly; any difference is a real data gap."""
    return batch_cents == dashboard_cents


def format_minor_units(cents: int) -> str:
    """Cents as a dollar string, e.g. 123450 -> '1234.50' (no float on the way)."""
    sign = "-" if cents < 0 else ""
    whole, rest = divmod(abs(cents), MINOR_UNITS_PER_MAJOR)
    return f"{sign}{whole}.{rest:02d}"


def fsum_total(amounts: list[float]) -> float:
    """Correctly rounded float sum: order-independent, but only within one call.

    ``math.fsum`` tracks the lost low-order bits, so any order of the same
    values gives the same result. Adding per-partition fsum results with
    ``+`` would round again; merge them with another ``fsum``.
    """
    return math.fsum(amounts)
