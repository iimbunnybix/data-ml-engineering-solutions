import random
import unittest

from solution import (
    UNKNOWN_REGION,
    load_orders,
    naive_load_orders,
    region_for_each_order,
    revenue_by_sku,
)

REGION_BY_ZIP = {
    "02134": "Northeast", "07030": "Northeast", "06103": "Northeast",
    "10001": "Northeast", "60601": "Midwest", "75201": "South",
    "94105": "West", "98101": "West", "30301": "South", "80202": "West",
}
SKUS = ["007", "7", "1200", "0420"]
ORDER_COUNT = 1000
SEED = 52
HEADER = "order_id,zip_code,sku,amount"


def order_file() -> str:
    """Seeded partner file: ZIPs and SKUs picked uniformly, whole-dollar amounts."""
    rng = random.Random(SEED)
    zip_codes = sorted(REGION_BY_ZIP)
    lines = [HEADER]
    for order_number in range(ORDER_COUNT):
        zip_code = rng.choice(zip_codes)
        sku = rng.choice(SKUS)
        amount = rng.randint(5, 200)
        lines.append(f"A{order_number:05d},{zip_code},{sku},{amount}.00")
    return "\n".join(lines) + "\n"


def unknown_region_count(orders: list[dict[str, object]]) -> int:
    return region_for_each_order(orders, REGION_BY_ZIP).count(UNKNOWN_REGION)


class ReproduceProblem(unittest.TestCase):
    def test_leading_zero_is_stripped_from_zip_code(self):
        orders = naive_load_orders(HEADER + "\nA1,02134,7,10.00\n")
        self.assertEqual(orders[0]["zip_code"], 2134)

    def test_orders_from_zero_prefixed_zips_lose_their_region(self):
        self.assertEqual(unknown_region_count(naive_load_orders(order_file())), 290)

    def test_two_different_skus_are_merged(self):
        totals = revenue_by_sku(naive_load_orders(order_file()))
        self.assertNotIn("007", totals)
        self.assertEqual(sorted(totals), ["1200", "420", "7"])

    def test_merged_sku_revenue_is_roughly_doubled(self):
        naive_totals = revenue_by_sku(naive_load_orders(order_file()))
        true_totals = revenue_by_sku(load_orders(order_file()))
        self.assertEqual(naive_totals["7"], 53071.0)
        self.assertEqual(true_totals["7"], 26743.0)


class FixSchemaLoader(unittest.TestCase):
    def test_zip_code_keeps_its_leading_zero(self):
        orders = load_orders(HEADER + "\nA1,02134,7,10.00\n")
        self.assertEqual(orders[0]["zip_code"], "02134")

    def test_every_order_finds_its_region(self):
        self.assertEqual(unknown_region_count(load_orders(order_file())), 0)

    def test_distinct_skus_stay_distinct(self):
        totals = revenue_by_sku(load_orders(order_file()))
        self.assertEqual(sorted(totals), ["007", "0420", "1200", "7"])

    def test_total_revenue_is_unchanged(self):
        naive_total = sum(revenue_by_sku(naive_load_orders(order_file())).values())
        fixed_total = sum(revenue_by_sku(load_orders(order_file())).values())
        self.assertEqual(naive_total, fixed_total)

    def test_amount_is_still_parsed_as_a_number(self):
        orders = load_orders(HEADER + "\nA1,02134,7,10.50\n")
        self.assertEqual(orders[0]["amount"], 10.5)


class FixValidation(unittest.TestCase):
    def test_zip_code_already_stripped_upstream_is_rejected(self):
        with self.assertRaises(ValueError):
            load_orders(HEADER + "\nA1,2134,7,10.00\n")

    def test_non_digit_zip_code_is_rejected(self):
        with self.assertRaises(ValueError):
            load_orders(HEADER + "\nA1,0213X,7,10.00\n")

    def test_unexpected_column_is_rejected(self):
        with self.assertRaises(ValueError):
            load_orders("order_id,zip_code,sku,amount,coupon\nA1,02134,7,10.00,X\n")

    def test_missing_column_is_rejected(self):
        with self.assertRaises(ValueError):
            load_orders("order_id,zip_code,amount\nA1,02134,10.00\n")

    def test_empty_file_with_header_loads_no_rows(self):
        self.assertEqual(load_orders(HEADER + "\n"), [])


if __name__ == "__main__":
    unittest.main()
