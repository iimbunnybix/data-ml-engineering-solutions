import unittest

from solution import (SIZE_ORDER, UnknownCategory, encode, encode_column, fit_ordinal_encoding,
                      flag_large_orders, naive_fit_encoding, size_return_correlation)

# Orders per size and how many of them were returned: bigger sizes come back more.
ORDERS_AND_RETURNS = {"S": (400, 40), "M": (500, 60), "L": (400, 80), "XL": (200, 60), "XXL": (100, 35)}


def build_orders():
    """Expand the summary table into one (size, returned) row per order."""
    sizes, returned = [], []
    for size, (order_count, return_count) in ORDERS_AND_RETURNS.items():
        sizes.extend([size] * order_count)
        returned.extend([1] * return_count + [0] * (order_count - return_count))
    return sizes, returned


class ReproduceProblem(unittest.TestCase):
    def test_alphabetical_codes_put_s_above_l(self):
        codes = naive_fit_encoding(["S", "M", "L", "XL"])
        self.assertEqual(codes, {"L": 0, "M": 1, "S": 2, "XL": 3})
        self.assertGreater(codes["S"], codes["L"])

    def test_large_size_rule_flags_every_order(self):
        sizes = ["S", "M", "L", "XL"]
        codes = naive_fit_encoding(sizes)
        self.assertEqual(flag_large_orders(sizes, codes, "L"), [True, True, True, True])

    def test_alphabetical_codes_weaken_the_size_signal(self):
        sizes, returned = build_orders()
        naive = size_return_correlation(sizes, returned, naive_fit_encoding(sizes))
        fixed = size_return_correlation(sizes, returned, fit_ordinal_encoding(SIZE_ORDER))
        self.assertLess(naive, fixed / 2)

    def test_codes_change_when_a_size_is_missing_from_the_batch(self):
        self.assertNotEqual(naive_fit_encoding(["S", "M", "L"])["S"], naive_fit_encoding(["S", "L"])["S"])


class FixExplicitOrder(unittest.TestCase):
    def setUp(self):
        self.codes = fit_ordinal_encoding(SIZE_ORDER)

    def test_codes_follow_the_declared_order(self):
        self.assertLess(self.codes["S"], self.codes["M"])
        self.assertLess(self.codes["M"], self.codes["L"])
        self.assertLess(self.codes["L"], self.codes["XL"])

    def test_large_size_rule_flags_only_l_and_above(self):
        flags = flag_large_orders(["S", "M", "L", "XL"], self.codes, "L")
        self.assertEqual(flags, [False, False, True, True])

    def test_codes_do_not_depend_on_which_sizes_are_in_the_batch(self):
        self.assertEqual(encode("S", self.codes), encode("S", fit_ordinal_encoding(SIZE_ORDER)))

    def test_spelling_variants_are_normalised(self):
        self.assertEqual(encode(" xl ", self.codes), self.codes["XL"])

    def test_unknown_size_is_rejected(self):
        with self.assertRaises(UnknownCategory):
            encode("XXXL", self.codes)

    def test_column_error_lists_every_unknown_size(self):
        with self.assertRaises(UnknownCategory) as caught:
            encode_column(["S", "3XL", "one size", "M"], self.codes)
        self.assertIn("3XL", str(caught.exception))
        self.assertIn("one size", str(caught.exception))

    def test_duplicate_in_the_declared_order_is_a_config_error(self):
        with self.assertRaises(ValueError):
            fit_ordinal_encoding(["S", "M", "M", "L"])

    def test_size_signal_is_strongly_positive(self):
        sizes, returned = build_orders()
        self.assertGreater(size_return_correlation(sizes, returned, self.codes), 0.15)


if __name__ == "__main__":
    unittest.main()
