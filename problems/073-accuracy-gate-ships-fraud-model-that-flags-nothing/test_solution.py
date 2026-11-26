import unittest

from solution import (EVAL_SIZE, FRAUD, FRAUD_COUNT, LEGIT, accuracy, collapsed_model, confusion,
                      make_labels, naive_release_gate, precision, recall, release_gate,
                      useful_model)


class ReproduceProblem(unittest.TestCase):
    def setUp(self):
        self.labels = make_labels()

    def test_model_that_flags_nothing_is_99_2_percent_accurate(self):
        counts = confusion(self.labels, collapsed_model(self.labels))
        self.assertAlmostEqual(accuracy(counts), 0.992)

    def test_accuracy_gate_ships_the_model_that_flags_nothing(self):
        self.assertTrue(naive_release_gate(self.labels, collapsed_model(self.labels)))

    def test_accuracy_barely_separates_a_useless_and_a_useful_model(self):
        useless = accuracy(confusion(self.labels, collapsed_model(self.labels)))
        useful = accuracy(confusion(self.labels, useful_model(self.labels)))
        self.assertAlmostEqual(useful - useless, 0.002)

    def test_accuracy_prefers_a_model_that_misses_half_the_fraud(self):
        cautious = useful_model(self.labels, caught=40, false_alarms=0)
        self.assertGreater(accuracy(confusion(self.labels, cautious)),
                           accuracy(confusion(self.labels, useful_model(self.labels))))


class FixGateOnRecallAndPrecision(unittest.TestCase):
    def setUp(self):
        self.labels = make_labels()

    def test_eval_set_has_the_expected_fraud_rate(self):
        self.assertEqual(sum(self.labels), FRAUD_COUNT)
        self.assertEqual(len(self.labels), EVAL_SIZE)

    def test_model_that_flags_nothing_is_rejected_for_recall(self):
        shipped, reason = release_gate(self.labels, collapsed_model(self.labels))
        self.assertFalse(shipped)
        self.assertIn("recall 0.00", reason)

    def test_useful_model_is_shipped(self):
        shipped, reason = release_gate(self.labels, useful_model(self.labels))
        self.assertTrue(shipped)
        self.assertEqual(reason, "recall 0.75, precision 0.60")

    def test_model_missing_half_the_fraud_is_rejected(self):
        cautious = useful_model(self.labels, caught=40, false_alarms=0)
        self.assertFalse(release_gate(self.labels, cautious)[0])

    def test_model_drowning_analysts_in_false_alarms_is_rejected(self):
        noisy = useful_model(self.labels, caught=70, false_alarms=300)
        shipped, reason = release_gate(self.labels, noisy)
        self.assertFalse(shipped)
        self.assertIn("precision 0.19", reason)

    def test_precision_is_zero_when_nothing_is_flagged(self):
        self.assertEqual(precision(confusion([FRAUD, LEGIT], [LEGIT, LEGIT])), 0.0)

    def test_eval_set_without_fraud_cannot_be_judged(self):
        with self.assertRaises(ValueError):
            recall(confusion([LEGIT, LEGIT], [LEGIT, LEGIT]))

    def test_mismatched_lengths_are_rejected(self):
        with self.assertRaises(ValueError):
            confusion([FRAUD], [FRAUD, LEGIT])


if __name__ == "__main__":
    unittest.main()
