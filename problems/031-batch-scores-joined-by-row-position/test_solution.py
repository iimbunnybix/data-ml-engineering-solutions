import random
import unittest

from solution import (CHUNK_SIZE, Customer, JoinMismatchError, ScoredCustomer, churn_score,
                      join_scores_by_key, naive_score_batch, score_batch, top_risk_ids,
                      wrong_score_share)

CUSTOMER_COUNT = 20_000
OUTREACH_LIST_SIZE = 500
DATA_SEED = 31
POOL_SEED = 4


def make_customers(count: int, seed: int) -> list[Customer]:
    """Customers sorted by id, as the scoring job reads them from the table."""
    rng = random.Random(seed)
    customers = []
    for i in range(count):
        days_inactive = int(rng.expovariate(1 / 15))
        tickets = rng.choice([0, 0, 0, 1, 1, 2, 3, 5])
        customers.append(Customer(f"cust-{i:06d}", days_inactive, tickets))
    return customers


def true_scores(customers: list[Customer]) -> dict[str, float]:
    """Each customer's own score, computed one by one with no parallel step."""
    return {c.customer_id: churn_score(c) for c in customers}


def outreach_overlap(scores: dict[str, float], customers: list[Customer]) -> float:
    """Share of the real top-risk customers that make it onto the call list."""
    real = top_risk_ids(true_scores(customers), OUTREACH_LIST_SIZE)
    called = top_risk_ids(scores, OUTREACH_LIST_SIZE)
    return len(real & called) / OUTREACH_LIST_SIZE


class ReproduceProblem(unittest.TestCase):
    def setUp(self):
        self.customers = make_customers(CUSTOMER_COUNT, DATA_SEED)

    def test_positional_zip_gives_most_customers_someone_elses_score(self):
        scores = naive_score_batch(self.customers, random.Random(POOL_SEED))
        self.assertGreater(wrong_score_share(scores, self.customers), 0.9)

    def test_call_list_misses_most_of_the_real_high_risk_customers(self):
        scores = naive_score_batch(self.customers, random.Random(POOL_SEED))
        self.assertLess(outreach_overlap(scores, self.customers), 0.2)

    def test_bug_is_invisible_in_summary_statistics(self):
        scores = naive_score_batch(self.customers, random.Random(POOL_SEED))
        # Same customers, same multiset of scores: row counts and averages all look normal.
        self.assertEqual(set(scores), {c.customer_id for c in self.customers})
        self.assertEqual(sorted(scores.values()), sorted(true_scores(self.customers).values()))

    def test_single_chunk_hides_the_bug_in_small_test_runs(self):
        small = self.customers[:CHUNK_SIZE]
        scores = naive_score_batch(small, random.Random(POOL_SEED))
        self.assertEqual(wrong_score_share(scores, small), 0.0)


class FixJoinByKey(unittest.TestCase):
    def setUp(self):
        self.customers = make_customers(CUSTOMER_COUNT, DATA_SEED)

    def test_every_customer_gets_their_own_score(self):
        scores = score_batch(self.customers, random.Random(POOL_SEED))
        self.assertEqual(scores, true_scores(self.customers))

    def test_result_is_the_same_for_any_completion_order(self):
        for pool_seed in range(5):
            scores = score_batch(self.customers, random.Random(pool_seed))
            self.assertEqual(wrong_score_share(scores, self.customers), 0.0)

    def test_call_list_matches_the_real_high_risk_customers(self):
        scores = score_batch(self.customers, random.Random(POOL_SEED))
        self.assertEqual(outreach_overlap(scores, self.customers), 1.0)

    def test_lost_chunk_is_rejected(self):
        ids = ["a", "b", "c"]
        results = [[ScoredCustomer("a", 0.1), ScoredCustomer("b", 0.2)]]
        with self.assertRaisesRegex(JoinMismatchError, "1 customers were not scored"):
            join_scores_by_key(ids, results)

    def test_retried_chunk_scored_twice_is_rejected(self):
        chunk = [ScoredCustomer("a", 0.1)]
        with self.assertRaisesRegex(JoinMismatchError, "scored twice"):
            join_scores_by_key(["a"], [chunk, chunk])

    def test_score_for_unknown_customer_is_rejected(self):
        with self.assertRaisesRegex(JoinMismatchError, "unknown customer"):
            join_scores_by_key(["a"], [[ScoredCustomer("a", 0.1), ScoredCustomer("z", 0.9)]])

    def test_duplicate_input_ids_are_rejected(self):
        with self.assertRaisesRegex(JoinMismatchError, "duplicate"):
            join_scores_by_key(["a", "a"], [[ScoredCustomer("a", 0.1)]])

    def test_empty_batch_scores_nothing(self):
        self.assertEqual(score_batch([], random.Random(POOL_SEED)), {})


if __name__ == "__main__":
    unittest.main()
