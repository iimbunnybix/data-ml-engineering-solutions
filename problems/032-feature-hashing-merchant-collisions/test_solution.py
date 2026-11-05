import random
import unittest

from solution import (HASH_BUCKETS, MIN_TRANSACTIONS_FOR_OWN_INDEX, NaiveHashedEncoder, Transaction,
                      VocabularyWithHashedTail, count_by_merchant, declined_share,
                      expected_colliding_keys, heavy_merchant_collisions, is_declined,
                      stable_bucket, train_index_fraud_rates)

HONEST_MERCHANTS = 5_000
FRAUD_RING_MERCHANTS = 50
FRAUD_RING_VOLUME = 400
HONEST_FRAUD_RATE = 0.002
RING_FRAUD_RATE = 0.3
MAX_MERCHANT_VOLUME = 20_000
DATA_SEED, TRAIN_SEED, TEST_SEED = 1, 2, 3


def make_merchants(seed: int) -> list[tuple[str, int, float]]:
    """(merchant_id, monthly transactions, fraud rate): heavy-tailed honest volume plus fraud rings."""
    rng = random.Random(seed)
    merchants = []
    for i in range(HONEST_MERCHANTS):
        volume = min(int(rng.paretovariate(1.1) * 10), MAX_MERCHANT_VOLUME)
        merchants.append((f"m-{i:05d}", volume, HONEST_FRAUD_RATE))
    for i in range(FRAUD_RING_MERCHANTS):
        merchants.append((f"r-{i:05d}", FRAUD_RING_VOLUME, RING_FRAUD_RATE))
    return merchants


def make_transactions(merchants: list[tuple[str, int, float]], seed: int) -> list[Transaction]:
    """One month of labelled transactions drawn from each merchant's fraud rate."""
    rng = random.Random(seed)
    transactions = []
    for merchant_id, volume, fraud_rate in merchants:
        for _ in range(volume):
            transactions.append(Transaction(merchant_id, rng.random() < fraud_rate))
    return transactions


def is_honest(merchant_id: str) -> bool:
    return merchant_id.startswith("m-")


class MonthOfTraffic(unittest.TestCase):
    """Builds the training and evaluation months once; they are read-only."""

    @classmethod
    def setUpClass(cls):
        merchants = make_merchants(DATA_SEED)
        cls.training = make_transactions(merchants, TRAIN_SEED)
        cls.evaluation = make_transactions(merchants, TEST_SEED)
        cls.honest_traffic = [tx for tx in cls.evaluation if is_honest(tx.merchant_id)]
        cls.ring_traffic = [tx for tx in cls.evaluation if not is_honest(tx.merchant_id)]
        cls.counts = count_by_merchant(cls.training)

    def declined_honest_merchants(self, encoder, rates) -> set[str]:
        declined = set()
        for tx in self.honest_traffic:
            if is_declined(tx.merchant_id, encoder, rates):
                declined.add(tx.merchant_id)
        return declined


class ReproduceProblem(MonthOfTraffic):
    def setUp(self):
        self.encoder = NaiveHashedEncoder()
        self.rates = train_index_fraud_rates(self.training, self.encoder)

    def test_honest_merchants_sharing_a_bucket_with_a_ring_are_declined(self):
        self.assertEqual(len(self.declined_honest_merchants(self.encoder, self.rates)), 220)

    def test_three_percent_of_honest_transactions_are_declined(self):
        self.assertAlmostEqual(declined_share(self.honest_traffic, self.encoder, self.rates),
                               0.0311, places=4)

    def test_fraud_ring_still_looks_mostly_caught(self):
        self.assertAlmostEqual(declined_share(self.ring_traffic, self.encoder, self.rates), 0.96, places=2)

    def test_a_declined_high_volume_merchant_shares_its_bucket_with_a_ring(self):
        self.assertEqual(self.encoder.index("m-04732"), self.encoder.index("r-00014"))
        self.assertTrue(is_declined("m-04732", self.encoder, self.rates))

    def test_high_volume_merchants_collide_with_each_other(self):
        self.assertEqual(len(heavy_merchant_collisions(self.counts, self.encoder)), 25)

    def test_almost_every_merchant_shares_a_bucket_at_this_size(self):
        self.assertGreater(expected_colliding_keys(5_050, HASH_BUCKETS) / 5_050, 0.99)


class FixVocabularyWithHashedTail(MonthOfTraffic):
    def setUp(self):
        self.encoder = VocabularyWithHashedTail(self.training)
        self.rates = train_index_fraud_rates(self.training, self.encoder)

    def test_no_honest_merchant_is_declined(self):
        self.assertEqual(self.declined_honest_merchants(self.encoder, self.rates), set())

    def test_every_fraud_ring_transaction_is_still_declined(self):
        self.assertEqual(declined_share(self.ring_traffic, self.encoder, self.rates), 1.0)

    def test_high_volume_merchants_never_share_an_index(self):
        self.assertEqual(heavy_merchant_collisions(self.counts, self.encoder), {})

    def test_vocabulary_holds_only_merchants_with_enough_volume(self):
        for merchant in self.encoder.vocabulary:
            self.assertGreaterEqual(self.counts[merchant], MIN_TRANSACTIONS_FOR_OWN_INDEX)

    def test_unseen_merchant_goes_to_a_tail_bucket_after_the_vocabulary(self):
        index = self.encoder.index("m-new-onboarded")
        self.assertGreaterEqual(index, len(self.encoder.vocabulary))
        self.assertLess(index, len(self.encoder.vocabulary) + HASH_BUCKETS)


class HashingHelpers(unittest.TestCase):
    def test_stable_bucket_is_deterministic_and_in_range(self):
        self.assertEqual(stable_bucket("m-00001", 1024), stable_bucket("m-00001", 1024))
        self.assertTrue(0 <= stable_bucket("m-00001", 7) < 7)

    def test_expected_collisions_for_tiny_inputs(self):
        self.assertEqual(expected_colliding_keys(0, 10), 0.0)
        self.assertEqual(expected_colliding_keys(1, 10), 0.0)
        self.assertAlmostEqual(expected_colliding_keys(2, 10), 0.2)

    def test_small_merchant_with_one_fraud_is_not_declined_after_smoothing(self):
        training = [Transaction("big", False) for _ in range(10_000)]
        training += [Transaction("tiny", True)]
        encoder = VocabularyWithHashedTail(training)
        rates = train_index_fraud_rates(training, encoder)
        self.assertFalse(is_declined("tiny", encoder, rates))


if __name__ == "__main__":
    unittest.main()
