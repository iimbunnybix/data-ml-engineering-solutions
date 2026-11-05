"""Sum order revenue that was charged in several currencies.

The revenue pipeline added up ``amount`` across all orders as if every amount
were in US dollars. Orders are actually charged in INR, USD, EUR and more, so
one 83,000-rupee order (about 1,000 dollars) counted as 83,000 dollars. A
first "fix" converted everything with today's FX rate, which silently
rewrites last month's revenue every time a rate moves.

The fix: every amount carries its currency, each order is converted with the
rate that was in effect on its transaction date (the latest published rate on
or before that date), missing/unknown/stale rates raise an error instead of
being guessed, and all money math uses ``Decimal``.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from datetime import date
from decimal import ROUND_HALF_EVEN, Decimal

REPORTING_CURRENCY = "USD"
CENTS = Decimal("0.01")
# A rate older than this is treated as missing: a gap in the FX feed should
# stop the pipeline, not quietly reuse a two-month-old rate.
MAX_RATE_AGE_DAYS = 7


class FxRateError(Exception):
    """Base class: the pipeline cannot convert an amount reliably."""


class UnknownCurrencyError(FxRateError):
    """No rates at all exist for this currency (typo, new market, bad data)."""


class MissingRateError(FxRateError):
    """No rate was published on or before the date, or it is too old."""


@dataclass(frozen=True)
class Order:
    order_id: str
    order_date: date
    amount: Decimal    # in ``currency``, never a float
    currency: str      # ISO 4217 code, e.g. "INR"

    def __post_init__(self) -> None:
        # Floats cannot represent most cent values exactly; reject them early.
        if not isinstance(self.amount, Decimal):
            raise TypeError(f"{self.order_id}: amount must be Decimal, got {type(self.amount).__name__}")


# --- the original bugs ------------------------------------------------------
def naive_total(orders: list[Order]) -> Decimal:
    """Original pipeline: adds amounts in different currencies together."""
    total = Decimal("0")
    for order in orders:
        total += order.amount
    return total


def naive_total_with_todays_rate(orders: list[Order], todays_rates: dict[str, Decimal]) -> Decimal:
    """First attempted fix: converts every order with today's rate.

    Old orders get re-valued whenever a rate moves, so last month's revenue
    changes every day. Unknown currencies are silently skipped.
    """
    total = Decimal("0")
    for order in orders:
        if order.currency == REPORTING_CURRENCY:
            total += order.amount
        elif order.currency in todays_rates:
            total += order.amount * todays_rates[order.currency]
    return total.quantize(CENTS, rounding=ROUND_HALF_EVEN)


# --- the fix ----------------------------------------------------------------
class FxRateTable:
    """Historical rates: units of REPORTING_CURRENCY per 1 unit of a currency."""

    def __init__(self, rates: dict[str, list[tuple[date, Decimal]]]) -> None:
        self._dates: dict[str, list[date]] = {}
        self._rates: dict[str, list[Decimal]] = {}
        for currency, published in rates.items():
            # Sort by effective date so we can binary-search on lookup.
            ordered = sorted(published)
            self._dates[currency] = [effective for effective, _ in ordered]
            self._rates[currency] = [rate for _, rate in ordered]

    def rate_on(self, currency: str, on_date: date) -> Decimal:
        """Rate effective on ``on_date``: the latest one published on or before it."""
        if currency == REPORTING_CURRENCY:
            return Decimal("1")
        if currency not in self._dates:
            raise UnknownCurrencyError(f"no FX rates for currency {currency!r}")
        dates = self._dates[currency]
        # bisect_right counts the rates published on or before on_date.
        position = bisect_right(dates, on_date)
        if position == 0:
            raise MissingRateError(f"no {currency} rate on or before {on_date}")
        effective_date = dates[position - 1]
        age_days = (on_date - effective_date).days
        if age_days > MAX_RATE_AGE_DAYS:
            raise MissingRateError(
                f"latest {currency} rate before {on_date} is from {effective_date} ({age_days} days old)")
        return self._rates[currency][position - 1]


def convert(order: Order, table: FxRateTable) -> Decimal:
    """Order amount in REPORTING_CURRENCY at the rate of its transaction date (unrounded)."""
    return order.amount * table.rate_on(order.currency, order.order_date)


def total_revenue(orders: list[Order], table: FxRateTable) -> Decimal:
    """Revenue in REPORTING_CURRENCY, rounded to cents once at the end.

    Rounding each order first would add up to half a cent of error per order.
    Any conversion error stops the whole total: a partial number on a
    dashboard looks exactly like a real one.
    """
    total = Decimal("0")
    for order in orders:
        total += convert(order, table)
    return total.quantize(CENTS, rounding=ROUND_HALF_EVEN)


def revenue_by_currency(orders: list[Order]) -> dict[str, Decimal]:
    """Totals per original currency, for reconciling with each payment provider."""
    totals: dict[str, Decimal] = {}
    for order in orders:
        totals[order.currency] = totals.get(order.currency, Decimal("0")) + order.amount
    return totals
