import random
import unittest

from solution import (MissingSourceError, OrderEvent, dedup, dedup_key, naive_dedup,
                      total_revenue_cents)

EU_ORDER_COUNT = 10_000
US_ORDER_COUNT = 6_000
RETRY_SHARE = 0.03  # share of events the broker delivers a second time
SEED = 71


def region_orders(source: str, count: int, rng: random.Random) -> list[OrderEvent]:
    """One region's real orders; its event ids start at 1 like a per-region sequence."""
    orders = []
    for number in range(1, count + 1):
        orders.append(OrderEvent(source, str(number), rng.randint(500, 20_000)))
    return orders


def delivered_stream(real_orders: list[OrderEvent], rng: random.Random) -> list[OrderEvent]:
    """What the pipeline receives: every order, some twice, regions interleaved."""
    stream = list(real_orders)
    for order in real_orders:
        if rng.random() < RETRY_SHARE:
            stream.append(order)
    rng.shuffle(stream)
    return stream


def make_workload() -> tuple[list[OrderEvent], list[OrderEvent]]:
    """Return (real orders, delivered stream with retries)."""
    rng = random.Random(SEED)
    real = region_orders("checkout-eu", EU_ORDER_COUNT, rng) + region_orders("checkout-us", US_ORDER_COUNT, rng)
    return real, delivered_stream(real, rng)


class ReproduceProblem(unittest.TestCase):
    def setUp(self):
        self.real, self.stream = make_workload()

    def test_naive_dedup_drops_thousands_of_real_orders(self):
        kept = naive_dedup(self.stream)
        self.assertEqual(len(kept), EU_ORDER_COUNT)
        self.assertEqual(len(self.real) - len(kept), US_ORDER_COUNT)

    def test_naive_revenue_is_far_below_the_real_revenue(self):
        reported = total_revenue_cents(naive_dedup(self.stream))
        real = total_revenue_cents(self.real)
        self.assertLess(reported / real, 0.7)

    def test_bug_is_invisible_with_one_region(self):
        eu_only = [event for event in self.stream if event.source == "checkout-eu"]
        self.assertEqual(len(naive_dedup(eu_only)), EU_ORDER_COUNT)


class FixDedupBySourceAndId(unittest.TestCase):
    def setUp(self):
        self.real, self.stream = make_workload()

    def test_every_real_order_is_kept_exactly_once(self):
        self.assertEqual(sorted(dedup(self.stream), key=dedup_key), sorted(self.real, key=dedup_key))

    def test_revenue_matches_the_real_orders(self):
        self.assertEqual(total_revenue_cents(dedup(self.stream)), total_revenue_cents(self.real))

    def test_retry_from_the_same_source_is_dropped(self):
        order = OrderEvent("checkout-eu", "7", 1_000)
        self.assertEqual(dedup([order, order]), [order])

    def test_same_id_from_different_sources_is_kept(self):
        eu = OrderEvent("checkout-eu", "7", 1_000)
        us = OrderEvent("checkout-us", "7", 2_000)
        self.assertEqual(dedup([eu, us]), [eu, us])

    def test_key_parts_cannot_run_into_each_other(self):
        first = OrderEvent("eu", "1-2", 100)
        second = OrderEvent("eu-1", "2", 200)
        self.assertEqual(dedup([first, second]), [first, second])

    def test_event_without_source_is_rejected(self):
        with self.assertRaises(MissingSourceError):
            dedup([OrderEvent("", "7", 1_000)])

    def test_empty_stream_gives_empty_result(self):
        self.assertEqual(dedup([]), [])


if __name__ == "__main__":
    unittest.main()
