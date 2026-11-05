import functools
import math
import random
import unittest

from solution import (exact_partition_total, exact_partitioned_total, format_minor_units, fsum_total,
                      naive_float_total, naive_partitioned_total, naive_reconciles, reconciles,
                      to_minor_units)

WIRES_PER_DAY = 400_000
PARTITIONS = 64
SMALLEST_WIRE_CENTS = 10 ** 3           # $10
LARGEST_WIRE_EXPONENT = 9.7             # 10**9.7 cents, about $50M
DATA_SEED = 7
RERUN_SEEDS = range(5)


def make_wire_day(seed: int) -> list[str]:
    """One day of wire amounts as the source file has them ("1234.50"), in event order.

    Amounts are log-uniform from $10 to about $50M: many small wires, a few huge ones.
    """
    rng = random.Random(seed)
    wires = []
    for _ in range(WIRES_PER_DAY):
        cents = int(10 ** rng.uniform(3, LARGEST_WIRE_EXPONENT))
        wires.append(format_minor_units(cents))
    return wires


def partition(wires: list[str], seed: int) -> list[list[str]]:
    """Spread wires over partitions the way a hash shuffle would."""
    rng = random.Random(seed)
    partitions = [[] for _ in range(PARTITIONS)]
    for wire in wires:
        partitions[rng.randrange(PARTITIONS)].append(wire)
    return partitions


def rerun_layout(partitions: list[list[str]], seed: int) -> tuple[list[list[str]], list[int]]:
    """A rerun: rows arrive in a different order inside each partition, tasks finish in another order."""
    rng = random.Random(seed)
    shuffled = []
    for rows in partitions:
        rows = list(rows)
        rng.shuffle(rows)
        shuffled.append(rows)
    finish_order = list(range(PARTITIONS))
    rng.shuffle(finish_order)
    return shuffled, finish_order


def as_floats(partitions: list[list[str]]) -> list[list[float]]:
    return [[float(wire) for wire in rows] for rows in partitions]


@functools.cache
def wire_day_fixture() -> dict:
    """The day, its partitions and five reruns, built once for every test class."""
    wires = make_wire_day(DATA_SEED)
    partitions = partition(wires, DATA_SEED)
    reruns = [rerun_layout(partitions, seed) for seed in RERUN_SEEDS]
    return {
        "wires": wires,
        "true_cents": sum(to_minor_units(wire) for wire in wires),
        "exact_reruns": [exact_partitioned_total(rows, order) for rows, order in reruns],
        # Float copies are built once: parsing is the slow part, not the sums.
        "float_reruns": [(as_floats(rows), order) for rows, order in reruns],
        "dashboard_float": naive_float_total([float(wire) for wire in wires]),
    }


class WireDay(unittest.TestCase):
    """Gives each test the shared, read-only wire day."""

    @classmethod
    def setUpClass(cls):
        for name, value in wire_day_fixture().items():
            setattr(cls, name, value)


class ReproduceProblem(WireDay):
    def test_adding_the_same_three_amounts_in_another_order_changes_the_float(self):
        self.assertEqual((0.1 + 0.2) + 0.3, 0.6000000000000001)
        self.assertEqual((0.3 + 0.2) + 0.1, 0.6)

    def test_reruns_of_the_same_data_get_different_float_totals(self):
        totals = set()
        for values, finish_order in self.float_reruns:
            totals.add(naive_partitioned_total(values, finish_order))
        self.assertEqual(len(totals), 3)                        # 5 reruns, 3 different answers

    def test_dashboard_single_pass_is_two_cents_off(self):
        dashboard = self.dashboard_float
        self.assertEqual(format_minor_units(self.true_cents), "1301782510419.50")
        self.assertEqual(f"{dashboard:.2f}", "1301782510419.52")

    def test_reconciliation_fails_although_no_wire_is_missing(self):
        values, finish_order = self.float_reruns[0]
        batch = naive_partitioned_total(values, finish_order)
        self.assertFalse(naive_reconciles(batch, self.dashboard_float))


class FixIntegerMinorUnits(WireDay):
    def test_every_rerun_gets_the_exact_total(self):
        self.assertEqual(self.exact_reruns, [self.true_cents] * len(RERUN_SEEDS))

    def test_batch_and_dashboard_reconcile_exactly(self):
        dashboard = exact_partition_total(self.wires)          # one pass in event order
        self.assertTrue(reconciles(self.exact_reruns[0], dashboard))

    def test_one_missing_cent_is_a_real_failure(self):
        self.assertFalse(reconciles(self.exact_reruns[0], self.true_cents - 1))

    def test_fsum_is_order_independent_for_float_data(self):
        totals = set()
        for values, _ in self.float_reruns:
            partials = [fsum_total(partition_values) for partition_values in values]
            totals.add(fsum_total(partials))
        self.assertEqual(len(totals), 1)


class ParsingAndFormatting(unittest.TestCase):
    def test_amounts_parse_to_exact_cents(self):
        self.assertEqual(to_minor_units("0.10"), 10)
        self.assertEqual(to_minor_units(" 1234.5 "), 123450)
        self.assertEqual(to_minor_units("-12.34"), -1234)
        self.assertEqual(to_minor_units("7"), 700)

    def test_sub_cent_amount_is_rejected(self):
        with self.assertRaises(ValueError):
            to_minor_units("12.345")

    def test_non_numbers_and_nan_are_rejected(self):
        for text in ("", "abc", "nan", "inf"):
            with self.assertRaises(ValueError):
                to_minor_units(text)

    def test_cents_format_without_float(self):
        self.assertEqual(format_minor_units(5), "0.05")
        self.assertEqual(format_minor_units(-123450), "-1234.50")
        self.assertEqual(format_minor_units(0), "0.00")

    def test_empty_day_totals_zero(self):
        self.assertEqual(exact_partitioned_total([[], []], [1, 0]), 0)
        self.assertEqual(naive_float_total([]), 0.0)
        self.assertTrue(math.isclose(fsum_total([]), 0.0))


if __name__ == "__main__":
    unittest.main()
