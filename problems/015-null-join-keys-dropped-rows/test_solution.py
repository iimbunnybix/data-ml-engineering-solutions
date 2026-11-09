import unittest

from solution import (MISSING_KEY_BUCKET, UNMATCHED_KEY_BUCKET, RegionReport,
                      ReconciliationError, naive_revenue_by_region,
                      normalize_key, reconcile, revenue_by_region)


def make_customers() -> list[dict]:
    """Dimension table, including the placeholder row with a NULL key."""
    return [
        {"customer_id": "C1", "region": "North"},
        {"customer_id": "C2", "region": "South"},
        {"customer_id": None, "region": "Test accounts"},
    ]


def make_orders() -> list[dict]:
    """The day's facts as they actually arrived from the checkout service."""
    return [
        {"customer_id": "C1", "amount_cents": 10_000},
        {"customer_id": "C1 ", "amount_cents": 5_000},
        {"customer_id": "c2", "amount_cents": 7_000},
        {"customer_id": None, "amount_cents": 3_000},
        {"customer_id": "", "amount_cents": 2_000},
        {"customer_id": "C9", "amount_cents": 4_000},
    ]


def total_cents(orders: list[dict]) -> int:
    return sum(order["amount_cents"] for order in orders)


class ReproduceProblem(unittest.TestCase):
    def test_naive_join_reports_less_than_half_of_revenue(self):
        report = naive_revenue_by_region(make_orders(), make_customers())
        self.assertEqual(sum(report.values()), 13_000)
        self.assertEqual(total_cents(make_orders()), 31_000)

    def test_naive_join_drops_whitespace_and_case_variants(self):
        report = naive_revenue_by_region(make_orders(), make_customers())
        self.assertEqual(report["North"], 10_000)
        self.assertNotIn("South", report)

    def test_naive_join_attributes_keyless_orders_to_placeholder_row(self):
        report = naive_revenue_by_region(make_orders(), make_customers())
        self.assertEqual(report["Test accounts"], 3_000)


class FixNormalizeKey(unittest.TestCase):
    def test_whitespace_and_case_are_ignored(self):
        self.assertEqual(normalize_key("  c1 "), "C1")

    def test_none_empty_and_blank_are_all_missing(self):
        for raw in (None, "", "   "):
            self.assertIsNone(normalize_key(raw))

    def test_float_nan_is_missing_not_the_text_nan(self):
        self.assertIsNone(normalize_key(float("nan")))


class FixRevenueByRegion(unittest.TestCase):
    def setUp(self):
        self.report = revenue_by_region(make_orders(), make_customers())

    def test_normalised_keys_join_to_their_region(self):
        self.assertEqual(self.report.revenue_cents["North"], 15_000)
        self.assertEqual(self.report.revenue_cents["South"], 7_000)

    def test_missing_keys_never_match_the_placeholder_row(self):
        self.assertNotIn("Test accounts", self.report.revenue_cents)
        self.assertEqual(self.report.revenue_cents[MISSING_KEY_BUCKET], 5_000)
        self.assertEqual(self.report.rows_per_bucket[MISSING_KEY_BUCKET], 2)

    def test_unknown_customer_goes_to_unmatched_bucket(self):
        self.assertEqual(self.report.revenue_cents[UNMATCHED_KEY_BUCKET], 4_000)

    def test_report_total_equals_input_total(self):
        self.assertEqual(sum(self.report.revenue_cents.values()), 31_000)

    def test_empty_input_gives_empty_report(self):
        report = revenue_by_region([], make_customers())
        self.assertEqual(report.revenue_cents, {})

    def test_conflicting_duplicate_dimension_keys_fail_loudly(self):
        customers = make_customers() + [{"customer_id": "c1 ", "region": "East"}]
        with self.assertRaises(ValueError):
            revenue_by_region(make_orders(), customers)


class FixNanKeys(unittest.TestCase):
    """pandas and many file readers represent a missing key as float NaN."""

    def test_nan_order_key_goes_to_missing_bucket(self):
        orders = [{"customer_id": float("nan"), "amount_cents": 3_000}]
        report = revenue_by_region(orders, make_customers())
        self.assertEqual(report.revenue_cents, {MISSING_KEY_BUCKET: 3_000})

    def test_nan_keyed_dimension_row_never_matches_nan_orders(self):
        customers = make_customers() + [
            {"customer_id": float("nan"), "region": "Test accounts"}]
        orders = [{"customer_id": float("nan"), "amount_cents": 3_000},
                  {"customer_id": "C1", "amount_cents": 1}]
        report = revenue_by_region(orders, customers)
        self.assertEqual(report.revenue_cents,
                         {MISSING_KEY_BUCKET: 3_000, "North": 1})


class FixReconciliation(unittest.TestCase):
    def test_report_missing_a_row_is_rejected(self):
        short_report = RegionReport(revenue_cents={"North": 10_000},
                                    rows_per_bucket={"North": 1})
        with self.assertRaises(ReconciliationError):
            reconcile(make_orders(), short_report)


if __name__ == "__main__":
    unittest.main()
