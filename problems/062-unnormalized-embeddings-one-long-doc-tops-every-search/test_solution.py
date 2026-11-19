import unittest

from solution import CosineIndex, ZeroVectorError, l2_normalize, naive_search, top_result_counts

# Tiny 3-dimensional "embeddings": axes are roughly refunds, delivery, account.
ARTICLES = {
    "refund-policy": [0.9, 0.1, 0.0],
    "track-delivery": [0.1, 0.8, 0.1],
    "reset-password": [0.0, 0.1, 0.7],
    # A long legal article touches every topic a little; its vector is long.
    "terms-of-service": [2.0, 2.0, 2.0],
}
QUERIES = {
    "how do I get my money back": [1.0, 0.0, 0.0],
    "where is my parcel": [0.0, 1.0, 0.1],
    "forgot my password": [0.0, 0.0, 1.0],
}


def build_index():
    index = CosineIndex()
    for doc_id, vector in ARTICLES.items():
        index.add(doc_id, vector)
    return index


class ReproduceProblem(unittest.TestCase):
    def test_terms_of_service_is_first_for_every_query(self):
        results = [naive_search(ARTICLES, query, k=1) for query in QUERIES.values()]
        self.assertEqual(top_result_counts(results), {"terms-of-service": 3})

    def test_relevant_article_loses_on_raw_dot_product(self):
        hits = naive_search(ARTICLES, QUERIES["how do I get my money back"], k=2)
        self.assertEqual([hit.doc_id for hit in hits], ["terms-of-service", "refund-policy"])


class FixCosineRanking(unittest.TestCase):
    def test_each_query_finds_its_own_article(self):
        index = build_index()
        firsts = [index.search(query, k=1)[0].doc_id for query in QUERIES.values()]
        self.assertEqual(firsts, ["refund-policy", "track-delivery", "reset-password"])

    def test_scaling_a_document_vector_does_not_change_its_score(self):
        index = CosineIndex()
        index.add("short", [0.6, 0.8])
        index.add("long", [6.0, 8.0])
        hits = index.search([0.6, 0.8], k=2)
        self.assertAlmostEqual(hits[0].score, hits[1].score)

    def test_scores_are_between_minus_one_and_one(self):
        for hit in build_index().search([5.0, -3.0, 2.0], k=4):
            self.assertLessEqual(abs(hit.score), 1.0 + 1e-9)

    def test_equal_scores_are_ordered_by_doc_id(self):
        index = CosineIndex()
        index.add("b", [1.0, 0.0])
        index.add("a", [2.0, 0.0])
        self.assertEqual([hit.doc_id for hit in index.search([1.0, 0.0], k=2)], ["a", "b"])


class FixRejectsBadVectors(unittest.TestCase):
    def test_zero_document_vector_is_rejected(self):
        with self.assertRaises(ZeroVectorError):
            CosineIndex().add("empty-article", [0.0, 0.0, 0.0])

    def test_zero_query_vector_is_rejected(self):
        with self.assertRaises(ZeroVectorError):
            build_index().search([0.0, 0.0, 0.0], k=1)

    def test_dimension_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            build_index().search([1.0, 0.0], k=1)

    def test_normalised_vector_has_length_one(self):
        unit = l2_normalize([3.0, 4.0])
        self.assertAlmostEqual(unit[0], 0.6)
        self.assertAlmostEqual(unit[1], 0.8)


if __name__ == "__main__":
    unittest.main()
