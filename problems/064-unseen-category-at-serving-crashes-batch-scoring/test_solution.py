import random
import unittest

from solution import (UNKNOWN_INDEX, CategoryEncoder, FraudRateModel, UnknownReport,
                      naive_encode, naive_encode_with_default, naive_fit_index, score_batch,
                      unknown_rate_alerts)

COLUMNS = ["payment_method", "city"]
# (payment_method, rows, frauds): crypto and gift_card are too rare for their own index.
PAYMENT_METHODS = [("card", 600, 12), ("apple_pay", 300, 3), ("paypal", 100, 4),
                   ("crypto", 3, 2), ("gift_card", 2, 1)]
CITIES = ["mumbai", "delhi", "pune"]


def make_training_rows(seed):
    rng = random.Random(seed)
    rows, labels = [], []
    for method, count, frauds in PAYMENT_METHODS:
        for position in range(count):
            rows.append({"payment_method": method, "city": rng.choice(CITIES)})
            labels.append(position < frauds)
    return rows, labels


def make_serving_batch(seed):
    """Tonight's batch: 1,000 payments, 150 using 'bnpl' (buy now, pay later) launched today."""
    rng = random.Random(seed)
    batch = [{"payment_method": "bnpl", "city": rng.choice(CITIES)} for _ in range(150)]
    for _ in range(850):
        batch.append({"payment_method": rng.choice(["card", "apple_pay", "paypal"]),
                      "city": rng.choice(CITIES)})
    rng.shuffle(batch)
    return batch


TRAIN_ROWS, TRAIN_LABELS = make_training_rows(seed=0)
BATCH = make_serving_batch(seed=1)


def fitted():
    encoder = CategoryEncoder.fit(TRAIN_ROWS, COLUMNS, min_count=20)
    model = FraudRateModel.fit(encoder.encode(TRAIN_ROWS), TRAIN_LABELS)
    return encoder, model


class ReproduceProblem(unittest.TestCase):
    def test_new_payment_method_raises_key_error_and_fails_the_job(self):
        index = naive_fit_index(TRAIN_ROWS, "payment_method")
        with self.assertRaises(KeyError):
            for row in BATCH:
                naive_encode(index, row["payment_method"])

    def test_default_index_zero_is_a_real_category(self):
        index = naive_fit_index(TRAIN_ROWS, "payment_method")
        self.assertEqual(naive_encode_with_default(index, "bnpl"), index["apple_pay"])

    def test_new_payment_method_is_silently_scored_as_apple_pay(self):
        index = naive_fit_index(TRAIN_ROWS, "payment_method")
        encoded = [{"payment_method": naive_encode_with_default(index, row["payment_method"])}
                   for row in TRAIN_ROWS]
        model = FraudRateModel.fit(encoded, TRAIN_LABELS)
        bnpl = model.score({"payment_method": naive_encode_with_default(index, "bnpl")})
        apple_pay = model.score({"payment_method": index["apple_pay"]})
        self.assertEqual(bnpl, apple_pay)
        self.assertAlmostEqual(bnpl, 0.01)


class FixUnknownBucket(unittest.TestCase):
    def test_index_zero_is_reserved_for_unknown(self):
        encoder, _ = fitted()
        self.assertNotIn(UNKNOWN_INDEX, encoder.indexes["payment_method"].values())
        self.assertEqual(encoder.indexes["payment_method"],
                         {"apple_pay": 1, "card": 2, "paypal": 3})

    def test_unseen_value_maps_to_unknown(self):
        encoder, _ = fitted()
        self.assertEqual(encoder.encode_value("payment_method", "bnpl"), UNKNOWN_INDEX)

    def test_rare_training_values_map_to_unknown(self):
        encoder, _ = fitted()
        self.assertEqual(encoder.encode_value("payment_method", "crypto"), UNKNOWN_INDEX)
        self.assertEqual(encoder.encode_value("payment_method", "gift_card"), UNKNOWN_INDEX)

    def test_missing_value_maps_to_unknown(self):
        encoder, _ = fitted()
        self.assertEqual(encoder.encode([{"payment_method": None}])[0],
                         {"payment_method": UNKNOWN_INDEX, "city": UNKNOWN_INDEX})

    def test_unknown_bucket_has_a_weight_learned_from_rare_values(self):
        _, model = fitted()
        self.assertAlmostEqual(model.rates["payment_method"][UNKNOWN_INDEX], 0.6)

    def test_whole_batch_is_scored_without_crashing(self):
        encoder, model = fitted()
        scores, _, _ = score_batch(encoder, model, BATCH)
        self.assertEqual(len(scores), len(BATCH))

    def test_new_payment_method_is_no_longer_scored_as_apple_pay(self):
        encoder, model = fitted()
        bnpl = model.score(encoder.encode([{"payment_method": "bnpl", "city": "pune"}])[0])
        apple_pay = model.score(encoder.encode([{"payment_method": "apple_pay", "city": "pune"}])[0])
        self.assertGreater(bnpl, apple_pay)


class FixUnknownMonitoring(unittest.TestCase):
    def test_unknowns_are_counted_per_column(self):
        encoder, model = fitted()
        _, report, _ = score_batch(encoder, model, BATCH)
        self.assertEqual(report.rows, 1000)
        self.assertEqual(report.unknown_counts["payment_method"], 150)
        self.assertEqual(report.unknown_counts["city"], 0)
        self.assertEqual(report.unseen_values["payment_method"]["bnpl"], 150)

    def test_new_payment_method_above_threshold_raises_an_alert(self):
        encoder, model = fitted()
        _, _, alerts = score_batch(encoder, model, BATCH)
        self.assertEqual(len(alerts), 1)
        self.assertIn("payment_method: 15.0% unknown", alerts[0])
        self.assertIn("bnpl", alerts[0])

    def test_unknown_rate_at_or_below_threshold_does_not_alert(self):
        report = UnknownReport(rows=100)
        report.unknown_counts["city"] = 5
        report.unseen_values["city"] = {"jaipur": 5}
        self.assertEqual(unknown_rate_alerts(report, threshold=0.05), [])

    def test_empty_batch_has_zero_unknown_rate_and_no_alerts(self):
        encoder, model = fitted()
        scores, report, alerts = score_batch(encoder, model, [])
        self.assertEqual((scores, report.unknown_rate("payment_method"), alerts), ([], 0.0, []))

    def test_report_accumulates_across_chunks_of_one_job(self):
        encoder, _ = fitted()
        report = UnknownReport()
        encoder.encode(BATCH[:500], report)
        encoder.encode(BATCH[500:], report)
        self.assertEqual((report.rows, report.unknown_counts["payment_method"]), (1000, 150))


if __name__ == "__main__":
    unittest.main()
