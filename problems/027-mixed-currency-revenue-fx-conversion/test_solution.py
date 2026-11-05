import unittest
from datetime import date
from decimal import Decimal

from solution import (FxRateTable, MissingRateError, Order, UnknownCurrencyError, convert,
                      naive_total, naive_total_with_todays_rate, revenue_by_currency, total_revenue)

# Illustrative weekly rates: US dollars per 1 unit of each currency.
RATES = {
    "INR": [(date(2026, 10, 1), Decimal("0.0120")), (date(2026, 10, 8), Decimal("0.0119")),
            (date(2026, 10, 15), Decimal("0.0118")), (date(2026, 11, 5), Decimal("0.0115"))],
    "EUR": [(date(2026, 10, 1), Decimal("1.05")), (date(2026, 10, 8), Decimal("1.07")),
            (date(2026, 10, 15), Decimal("1.08")), (date(2026, 11, 5), Decimal("1.10"))],
}
TODAYS_RATES = {"INR": Decimal("0.0115"), "EUR": Decimal("1.10")}


def october_orders() -> list[Order]:
    """Four October orders in three currencies, like the real dashboard input."""
    return [
        Order("o1", date(2026, 10, 2), Decimal("83000.00"), "INR"),
        Order("o2", date(2026, 10, 3), Decimal("1000.00"), "USD"),
        Order("o3", date(2026, 10, 10), Decimal("1000.00"), "EUR"),
        Order("o4", date(2026, 10, 20), Decimal("41500.50"), "INR"),
    ]


def rate_table() -> FxRateTable:
    return FxRateTable(RATES)


class ReproduceProblem(unittest.TestCase):
    def test_naive_total_adds_rupees_to_dollars(self):
        self.assertEqual(naive_total(october_orders()), Decimal("126500.50"))

    def test_todays_rate_gives_a_different_total_than_transaction_date_rates(self):
        todays = naive_total_with_todays_rate(october_orders(), TODAYS_RATES)
        self.assertEqual(todays, Decimal("3531.76"))
        self.assertNotEqual(todays, total_revenue(october_orders(), rate_table()))

    def test_todays_rate_silently_drops_unknown_currency(self):
        orders = october_orders() + [Order("o5", date(2026, 10, 21), Decimal("500"), "GBP")]
        self.assertEqual(naive_total_with_todays_rate(orders, TODAYS_RATES), Decimal("3531.76"))


class FixConvertAtTransactionDateRate(unittest.TestCase):
    def test_total_uses_the_rate_effective_on_each_order_date(self):
        # 83000*0.0120 + 1000 + 1000*1.07 + 41500.50*0.0118 = 3555.7059
        self.assertEqual(total_revenue(october_orders(), rate_table()), Decimal("3555.71"))

    def test_rate_published_on_the_order_date_is_used(self):
        self.assertEqual(rate_table().rate_on("EUR", date(2026, 10, 8)), Decimal("1.07"))

    def test_latest_rate_before_the_order_date_is_used(self):
        self.assertEqual(rate_table().rate_on("EUR", date(2026, 10, 7)), Decimal("1.05"))

    def test_rates_given_out_of_order_are_sorted(self):
        table = FxRateTable({"EUR": list(reversed(RATES["EUR"]))})
        self.assertEqual(table.rate_on("EUR", date(2026, 10, 9)), Decimal("1.07"))

    def test_reporting_currency_is_not_converted(self):
        order = Order("o2", date(2026, 1, 1), Decimal("19.99"), "USD")
        self.assertEqual(convert(order, rate_table()), Decimal("19.99"))

    def test_total_does_not_change_when_todays_rate_moves(self):
        later_rates = {currency: list(rates) for currency, rates in RATES.items()}
        later_rates["EUR"].append((date(2026, 11, 20), Decimal("1.25")))
        self.assertEqual(total_revenue(october_orders(), FxRateTable(later_rates)), Decimal("3555.71"))

    def test_rounding_happens_once_on_the_total(self):
        # Each order is 0.005 USD; rounding per order (half-even) would give 0.00.
        orders = [Order(f"p{index}", date(2026, 10, 2), Decimal("0.5"), "INR") for index in range(3)]
        table = FxRateTable({"INR": [(date(2026, 10, 1), Decimal("0.01"))]})
        self.assertEqual(total_revenue(orders, table), Decimal("0.02"))

    def test_per_currency_totals_keep_original_amounts(self):
        self.assertEqual(revenue_by_currency(october_orders()),
                         {"INR": Decimal("124500.50"), "USD": Decimal("1000.00"), "EUR": Decimal("1000.00")})

    def test_empty_order_list_totals_zero(self):
        self.assertEqual(total_revenue([], rate_table()), Decimal("0.00"))


class FixFailLoudly(unittest.TestCase):
    def test_unknown_currency_raises(self):
        orders = october_orders() + [Order("o5", date(2026, 10, 21), Decimal("500"), "GBP")]
        with self.assertRaises(UnknownCurrencyError):
            total_revenue(orders, rate_table())

    def test_order_before_the_first_rate_raises(self):
        with self.assertRaises(MissingRateError):
            rate_table().rate_on("INR", date(2026, 9, 30))

    def test_stale_rate_raises(self):
        # The INR feed has a gap from Oct 15 to Nov 5; Oct 30 is 15 days after the last rate.
        with self.assertRaises(MissingRateError):
            rate_table().rate_on("INR", date(2026, 10, 30))

    def test_rate_exactly_seven_days_old_is_accepted(self):
        self.assertEqual(rate_table().rate_on("INR", date(2026, 10, 22)), Decimal("0.0118"))

    def test_float_amount_is_rejected(self):
        with self.assertRaises(TypeError):
            Order("o6", date(2026, 10, 2), 0.1, "USD")


if __name__ == "__main__":
    unittest.main()
