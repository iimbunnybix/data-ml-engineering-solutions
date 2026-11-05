import random
import unittest
from collections import Counter

from solution import naive_head_sample, reservoir_sample, share_by_key

REGIONS = ["APAC", "EMEA", "AMER"]
ROWS_PER_REGION = 4_000
SAMPLE_SIZE = 1_000
ALLOWED_SHARE_ERROR = 0.05


def export_sorted_by_region():
    """A daily export written region by region, like the real file."""
    for region in REGIONS:
        for index in range(ROWS_PER_REGION):
            yield {"region": region, "txn_id": f"{region}-{index}"}


class ReproduceProblem(unittest.TestCase):
    def test_head_sample_contains_only_the_first_region(self):
        sample = naive_head_sample(export_sorted_by_region(), SAMPLE_SIZE)
        self.assertEqual(share_by_key(sample, "region"), {"APAC": 1.0})


class FixReservoirSampling(unittest.TestCase):
    def test_sample_has_the_requested_size(self):
        sample = reservoir_sample(export_sorted_by_region(), SAMPLE_SIZE, random.Random(1))
        self.assertEqual(len(sample), SAMPLE_SIZE)

    def test_every_region_is_represented_in_proportion(self):
        sample = reservoir_sample(export_sorted_by_region(), SAMPLE_SIZE, random.Random(1))
        shares = share_by_key(sample, "region")
        for region in REGIONS:
            self.assertAlmostEqual(shares[region], 1 / 3, delta=ALLOWED_SHARE_ERROR)

    def test_sample_has_no_duplicate_rows(self):
        sample = reservoir_sample(export_sorted_by_region(), SAMPLE_SIZE, random.Random(2))
        self.assertEqual(len({row["txn_id"] for row in sample}), SAMPLE_SIZE)

    def test_each_position_is_chosen_with_equal_probability(self):
        stream_length, sample_size, runs = 10, 3, 20_000
        rng = random.Random(3)
        chosen = Counter()
        for _ in range(runs):
            chosen.update(reservoir_sample(range(stream_length), sample_size, rng))
        expected = runs * sample_size / stream_length
        for position in range(stream_length):
            self.assertAlmostEqual(chosen[position] / expected, 1.0, delta=0.05)

    def test_same_seed_gives_the_same_sample(self):
        first = reservoir_sample(range(10_000), 50, random.Random(7))
        second = reservoir_sample(range(10_000), 50, random.Random(7))
        self.assertEqual(first, second)

    def test_stream_shorter_than_sample_returns_everything(self):
        self.assertEqual(reservoir_sample(range(5), 10, random.Random(0)), [0, 1, 2, 3, 4])

    def test_zero_sample_size_returns_empty_list(self):
        self.assertEqual(reservoir_sample(range(5), 0, random.Random(0)), [])

    def test_negative_sample_size_is_rejected(self):
        with self.assertRaises(ValueError):
            reservoir_sample(range(5), -1, random.Random(0))


if __name__ == "__main__":
    unittest.main()
