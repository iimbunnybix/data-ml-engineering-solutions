import random
import unittest
from collections import Counter

from solution import (ShuffleStats, find_hot_keys, plain_join, salted_join,
                      stable_hash)

PARTITIONS = 32


def skewed_data(seed=42, rows=50_000, hot_share=0.4, merchants=2_000):
    """Payments where one merchant (m0) has `hot_share` of all rows."""
    rng = random.Random(seed)
    big = []
    for i in range(rows):
        m = "m0" if rng.random() < hot_share else f"m{rng.randint(1, merchants - 1)}"
        big.append((m, i))
    small = [(f"m{i}", f"tier{i % 3}") for i in range(merchants)]
    return big, small


class ReproduceSkew(unittest.TestCase):
    def test_hot_key_makes_one_task_a_straggler(self):
        big, small = skewed_data()
        _, stats = plain_join(big, small, PARTITIONS)
        self.assertGreater(stats.skew_ratio, 10)    # slowest task >10x the median
        hot_task_rows = Counter(k for k, _ in big)["m0"]
        self.assertGreaterEqual(stats.max, hot_task_rows)


class SaltedJoin(unittest.TestCase):
    def setUp(self):
        self.big, self.small = skewed_data()

    def test_same_result_as_plain_join(self):
        plain, _ = plain_join(self.big, self.small, PARTITIONS)
        salted, _ = salted_join(self.big, self.small, PARTITIONS)
        self.assertEqual(Counter(plain), Counter(salted))
        self.assertEqual(len(salted), len(self.big))

    def test_balances_tasks(self):
        _, before = plain_join(self.big, self.small, PARTITIONS)
        _, after = salted_join(self.big, self.small, PARTITIONS)
        self.assertLess(after.skew_ratio, 2.0)
        self.assertLess(after.max * 5, before.max)

    def test_detects_only_the_hot_key(self):
        self.assertEqual(find_hot_keys((k for k, _ in self.big), PARTITIONS), {"m0"})

    def test_small_side_replication_is_bounded(self):
        # Only hot keys are replicated: small side grows by (salts-1) per hot key.
        _, stats = salted_join(self.big, self.small, PARTITIONS, salts=8)
        self.assertEqual(sum(stats.rows_per_task), len(self.big) + len(self.small) + 7)

    def test_no_hot_keys_is_identical_to_plain(self):
        rng = random.Random(1)
        big = [(f"k{rng.randint(0, 999)}", i) for i in range(10_000)]
        small = [(f"k{i}", i) for i in range(1000)]
        self.assertEqual(find_hot_keys((k for k, _ in big), PARTITIONS), set())
        p, ps = plain_join(big, small, PARTITIONS)
        s, ss = salted_join(big, small, PARTITIONS)
        self.assertEqual(Counter(p), Counter(s))
        self.assertEqual(ps.rows_per_task, ss.rows_per_task)

    def test_unmatched_keys_and_duplicates_on_small_side(self):
        big = [("hot", i) for i in range(100)] + [("orphan", 1)]
        small = [("hot", "a"), ("hot", "b")]
        p, _ = plain_join(big, small, 4)
        s, _ = salted_join(big, small, 4, salts=4, hot_keys={"hot"})
        self.assertEqual(Counter(p), Counter(s))
        self.assertEqual(len(s), 200)

    def test_deterministic_across_runs(self):
        a, _ = salted_join(self.big, self.small, PARTITIONS)
        b, _ = salted_join(self.big, self.small, PARTITIONS)
        self.assertEqual(a, b)
        self.assertEqual(stable_hash("m0"), stable_hash("m0"))

    def test_invalid_salts(self):
        with self.assertRaises(ValueError):
            salted_join(self.big, self.small, PARTITIONS, salts=0)

    def test_stats_helpers(self):
        st = ShuffleStats([1, 2, 3, 10])
        self.assertEqual((st.max, st.median, st.skew_ratio), (10, 2.5, 4.0))


if __name__ == "__main__":
    unittest.main()
