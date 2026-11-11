import random
import unittest
from decimal import Decimal

from solution import (SaleLine, duplicate_labels, make_sales, naive_units_by_price,
                      unit_price_cents, units_by_price)

SEED = 7
LINE_COUNT = 10_000
# 99 catalog prices: $0.10, $0.20, ... $9.90.
PRICE_POINTS_CENTS = list(range(10, 1000, 10))
MAX_QUANTITY = 6


def daily_sales() -> list[SaleLine]:
    """The same 10,000 order lines on every run."""
    return make_sales(random.Random(SEED), LINE_COUNT, PRICE_POINTS_CENTS, MAX_QUANTITY)


def seventy_cent_lines() -> list[SaleLine]:
    """One item bought alone and three bought together, all at $0.70."""
    return [SaleLine("0.70", 1), SaleLine("2.10", 3)]


class ReproduceProblem(unittest.TestCase):
    def test_same_price_lands_in_two_groups(self):
        report = naive_units_by_price(seventy_cent_lines())
        self.assertEqual(len(report), 2)

    def test_both_groups_print_the_same_label(self):
        report = naive_units_by_price(seventy_cent_lines())
        self.assertEqual(duplicate_labels(report), ["$0.70"])

    def test_daily_report_has_more_rows_than_price_points(self):
        report = naive_units_by_price(daily_sales())
        self.assertGreater(len(report), len(PRICE_POINTS_CENTS))


class FixWithDecimalCentKeys(unittest.TestCase):
    def test_same_price_lands_in_one_group(self):
        report = units_by_price(seventy_cent_lines())
        self.assertEqual(report, {Decimal("0.70"): 4})

    def test_daily_report_has_one_row_per_price_point(self):
        report = units_by_price(daily_sales())
        self.assertEqual(len(report), len(PRICE_POINTS_CENTS))
        self.assertEqual(duplicate_labels(report), [])

    def test_no_units_are_lost_or_added(self):
        sales = daily_sales()
        expected_units = sum(line.quantity for line in sales)
        self.assertEqual(sum(units_by_price(sales).values()), expected_units)

    def test_uneven_split_rounds_to_the_cent(self):
        self.assertEqual(unit_price_cents(SaleLine("1.00", 3)), Decimal("0.33"))

    def test_half_cent_rounds_half_even(self):
        # 0.25 / 2 = 0.125 -> 0.12 (to even), not 0.13.
        self.assertEqual(unit_price_cents(SaleLine("0.25", 2)), Decimal("0.12"))

    def test_zero_quantity_is_rejected(self):
        with self.assertRaises(ValueError):
            unit_price_cents(SaleLine("1.00", 0))


if __name__ == "__main__":
    unittest.main()
