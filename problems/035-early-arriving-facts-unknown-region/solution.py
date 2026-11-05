"""Orders that arrive before their customer record must not stay "Unknown" forever.

New customers place their first orders minutes after signing up, but the CRM
sync that loads ``dim_customer`` runs later. The original fact loader looked
the customer up at load time and, finding nothing, stored the fixed
"Unknown" key. When the customer row arrived an hour later, nothing went back
to fix those orders, so revenue from new customers piled up under "Unknown".

The fix uses the data-warehouse "inferred member" pattern: when a fact
references a customer the dimension has not seen, the loader creates a
placeholder dimension row with a real surrogate key and stores that key on
the fact. When the CRM row arrives, it fills the placeholder in place, so
every earlier fact picks up the correct region with no fact rewrite.
"""
from __future__ import annotations

from dataclasses import dataclass

UNKNOWN_KEY = -1              # the single shared "Unknown" row of the original design
UNKNOWN_REGION = "Unknown"


@dataclass(frozen=True)
class Order:
    order_id: int
    customer_id: str
    amount_cents: int


@dataclass(frozen=True)
class CustomerRecord:
    customer_id: str
    region: str


@dataclass
class DimensionRow:
    surrogate_key: int
    customer_id: str
    region: str | None        # None while the row is an inferred placeholder
    inferred: bool


@dataclass(frozen=True)
class FactRow:
    order_id: int
    customer_key: int
    amount_cents: int


class CustomerDimension:
    """``dim_customer``: one row per customer, addressed by a surrogate key."""

    def __init__(self) -> None:
        self.rows_by_key: dict[int, DimensionRow] = {}
        self.key_by_customer: dict[str, int] = {}
        self.next_key = 1

    def lookup(self, customer_id: str) -> int | None:
        """Surrogate key of a customer, or None if the dimension has no row yet."""
        return self.key_by_customer.get(customer_id)

    def _insert(self, customer_id: str, region: str | None, inferred: bool) -> int:
        """Add a new row and return its freshly allocated surrogate key."""
        key = self.next_key
        self.next_key += 1
        self.rows_by_key[key] = DimensionRow(key, customer_id, region, inferred)
        self.key_by_customer[customer_id] = key
        return key

    def key_for_fact(self, customer_id: str) -> int:
        """Key to store on a fact; creates an inferred placeholder if needed.

        WHY a real key instead of UNKNOWN_KEY: the placeholder can be filled in
        later, while the shared Unknown row can never tell whose order it is.
        """
        existing = self.lookup(customer_id)
        if existing is not None:
            return existing
        return self._insert(customer_id, region=None, inferred=True)

    def upsert(self, record: CustomerRecord) -> int:
        """Load a CRM row: fill an inferred placeholder, update, or insert."""
        existing = self.lookup(record.customer_id)
        if existing is None:
            return self._insert(record.customer_id, record.region, inferred=False)
        row = self.rows_by_key[existing]
        # Filling the placeholder in place keeps the key, so earlier facts
        # that point at it now show the real region.
        row.region = record.region
        row.inferred = False
        return existing

    def region_of(self, key: int) -> str:
        """Region for reporting; placeholders and the Unknown key report as Unknown."""
        row = self.rows_by_key.get(key)
        if row is None or row.region is None:
            return UNKNOWN_REGION
        return row.region

    def inferred_customers(self) -> list[str]:
        """Customers still waiting for a CRM row: monitor this list, do not ignore it."""
        return sorted(row.customer_id for row in self.rows_by_key.values() if row.inferred)


def naive_load_order(order: Order, dimension: CustomerDimension) -> FactRow:
    """Original loader: a customer missing at load time is pinned to Unknown forever."""
    key = dimension.lookup(order.customer_id)
    if key is None:
        key = UNKNOWN_KEY
    return FactRow(order.order_id, key, order.amount_cents)


def load_order(order: Order, dimension: CustomerDimension) -> FactRow:
    """Fixed loader: an early-arriving fact gets an inferred member's key."""
    key = dimension.key_for_fact(order.customer_id)
    return FactRow(order.order_id, key, order.amount_cents)


def revenue_by_region(facts: list[FactRow], dimension: CustomerDimension) -> dict[str, int]:
    """Join facts to the dimension at query time and sum cents per region."""
    totals: dict[str, int] = {}
    for fact in facts:
        region = dimension.region_of(fact.customer_key)
        totals[region] = totals.get(region, 0) + fact.amount_cents
    return totals
