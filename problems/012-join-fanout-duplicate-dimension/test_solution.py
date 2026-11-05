import random
import unittest
from decimal import Decimal

from solution import (CardinalityError, ReconciliationError, assert_measure_preserved,
                      dedupe_latest, duplicate_keys, naive_left_join, revenue_by, total,
                      validated_join)


def make_data(seed=3, customers=1_000, orders=20_000, reingested=150):
    rng = random.Random(seed)
    dim = [{"customer_id": c, "region": rng.choice(["EU", "US", "APAC"]),
            "updated_at": "2026-10-01"} for c in range(customers)]
    # The CRM re-sync appended a second row for some customers instead of updating.
    for c in rng.sample(range(customers), reingested):
        dim.append({**dim[c], "updated_at": "2026-10-15"})
    facts = [{"order_id": i, "customer_id": rng.randrange(customers),
              "amount": f"{rng.randint(500, 50_000) / 100:.2f}"} for i in range(orders)]
    return facts, dim


class ReproduceFanOut(unittest.TestCase):
    def test_duplicate_dimension_rows_inflate_revenue_silently(self):
        facts, dim = make_data()
        joined = naive_left_join(facts, dim, "customer_id")
        inflation = total(joined, "amount") / total(facts, "amount") - 1
        self.assertGreater(len(joined), len(facts))
        self.assertGreater(inflation, Decimal("0.13"))   # ~15% more revenue than was sold
        self.assertLess(inflation, Decimal("0.17"))

    def test_every_order_of_a_duplicated_customer_is_counted_twice(self):
        facts = [{"order_id": 1, "customer_id": 7, "amount": "10.00"}]
        dim = [{"customer_id": 7, "region": "EU"}, {"customer_id": 7, "region": "EU"}]
        self.assertEqual(len(naive_left_join(facts, dim, "customer_id")), 2)


class ValidatedJoinFix(unittest.TestCase):
    def test_many_to_one_join_rejects_duplicate_dimension_keys(self):
        facts, dim = make_data()
        with self.assertRaises(CardinalityError) as ctx:
            validated_join(facts, dim, "customer_id", "many_to_one")
        self.assertEqual(ctx.exception.side, "right")
        self.assertEqual(len(ctx.exception.duplicates), 150)
        self.assertIn("150 duplicate", str(ctx.exception))

    def test_dedupe_then_join_preserves_revenue(self):
        facts, dim = make_data()
        clean = dedupe_latest(dim, "customer_id", "updated_at")
        self.assertEqual(len(clean), 1_000)
        joined = validated_join(facts, clean, "customer_id", "many_to_one")
        assert_measure_preserved(facts, joined, "amount")
        self.assertEqual(sum(revenue_by(joined, "region").values()), total(facts, "amount"))

    def test_dedupe_keeps_latest_row(self):
        rows = [{"id": 1, "v": "old", "ts": 1}, {"id": 1, "v": "new", "ts": 2},
                {"id": 2, "v": "only", "ts": 1}]
        out = {r["id"]: r["v"] for r in dedupe_latest(rows, "id", "ts")}
        self.assertEqual(out, {1: "new", 2: "only"})

    def test_dedupe_refuses_ambiguous_ties(self):
        rows = [{"id": 1, "v": "a", "ts": 1}, {"id": 1, "v": "b", "ts": 1}]
        with self.assertRaises(CardinalityError):
            dedupe_latest(rows, "id", "ts")
        identical = [{"id": 1, "v": "a", "ts": 1}, {"id": 1, "v": "a", "ts": 1}]
        self.assertEqual(len(dedupe_latest(identical, "id", "ts")), 1)

    def test_tie_resolved_by_a_later_row(self):
        rows = [{"id": 1, "v": "a", "ts": 1}, {"id": 1, "v": "b", "ts": 1},
                {"id": 1, "v": "c", "ts": 2}]
        self.assertEqual(dedupe_latest(rows, "id", "ts")[0]["v"], "c")

    def test_reconciliation_catches_fanout_without_declared_cardinality(self):
        facts, dim = make_data()
        joined = validated_join(facts, dim, "customer_id", "many_to_many")
        with self.assertRaises(ReconciliationError):
            assert_measure_preserved(facts, joined, "amount")

    def test_unmatched_and_null_keys_are_kept_once(self):
        facts = [{"order_id": 1, "customer_id": 99, "amount": "5.00"},
                 {"order_id": 2, "customer_id": None, "amount": "7.00"}]
        dim = [{"customer_id": 1, "region": "EU"}, {"customer_id": None, "region": "X"},
               {"customer_id": None, "region": "Y"}]
        self.assertEqual(duplicate_keys(dim, "customer_id"), {})
        joined = validated_join(facts, dim, "customer_id")
        self.assertEqual(len(joined), 2)

    def test_dedupe_passes_null_key_rows_through_unchanged(self):
        dim = [{"customer_id": 1, "region": "EU", "updated_at": "2026-10-01"},
               {"customer_id": None, "region": "X", "updated_at": "2026-10-01"},
               {"customer_id": None, "region": "Y", "updated_at": "2026-10-01"}]
        deduped = dedupe_latest(dim, "customer_id", "updated_at")
        null_regions = sorted(r["region"] for r in deduped if r["customer_id"] is None)
        self.assertEqual(null_regions, ["X", "Y"])
        self.assertEqual(len(deduped), 3)

    def test_row_with_timestamp_beats_row_without_one(self):
        rows = [{"customer_id": 1, "region": "EU", "updated_at": "2026-10-01"},
                {"customer_id": 1, "region": "US", "updated_at": None}]
        self.assertEqual(dedupe_latest(rows, "customer_id", "updated_at")[0]["region"], "EU")
        self.assertEqual(dedupe_latest(rows[::-1], "customer_id", "updated_at")[0]["region"], "EU")

    def test_two_different_rows_without_timestamp_are_an_ambiguous_tie(self):
        rows = [{"customer_id": 1, "region": "EU", "updated_at": None},
                {"customer_id": 1, "region": "US", "updated_at": None}]
        with self.assertRaises(CardinalityError):
            dedupe_latest(rows, "customer_id", "updated_at")

    def test_one_to_one_checks_left_side_too(self):
        left = [{"k": 1}, {"k": 1}]
        with self.assertRaises(CardinalityError) as ctx:
            validated_join(left, [{"k": 1}], "k", "one_to_one")
        self.assertEqual(ctx.exception.side, "left")

    def test_unknown_cardinality(self):
        with self.assertRaises(ValueError):
            validated_join([], [], "k", "m:1")


if __name__ == "__main__":
    unittest.main()
