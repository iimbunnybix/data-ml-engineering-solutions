import unittest

from solution import (NEGATIVE_SAMPLING_RATE, TRUE_CTR_BY_SLOT, Impression,
                      bid_per_impression, correct_for_sampling, corrected_model,
                      downsample_negatives, naive_train, simulate_impressions,
                      weighted_train)

VALUE_PER_CLICK = 20.0  # rupees the advertiser is willing to pay per click


def sampled_training_set() -> tuple[list[Impression], list[Impression]]:
    """300,000 real impressions and the cheap training set built from them."""
    impressions = simulate_impressions(count_per_slot=100_000, seed=7)
    sampled = downsample_negatives(impressions, NEGATIVE_SAMPLING_RATE, seed=8)
    return impressions, sampled


class ReproduceProblem(unittest.TestCase):
    def test_sampled_model_overestimates_every_slot_several_times(self):
        _, sampled = sampled_training_set()
        model = naive_train(sampled)
        for slot, true_ctr in TRUE_CTR_BY_SLOT.items():
            self.assertGreater(model[slot], 6 * true_ctr, slot)

    def test_top_slot_is_predicted_at_about_34_percent_instead_of_5(self):
        _, sampled = sampled_training_set()
        self.assertAlmostEqual(naive_train(sampled)["top"], 0.3405, places=4)

    def test_bidder_pays_far_more_than_an_impression_is_worth(self):
        _, sampled = sampled_training_set()
        naive_bid = bid_per_impression(naive_train(sampled)["side"], VALUE_PER_CLICK)
        fair_bid = bid_per_impression(TRUE_CTR_BY_SLOT["side"], VALUE_PER_CLICK)
        self.assertGreater(naive_bid / fair_bid, 8)


class FixUndoesTheSampling(unittest.TestCase):
    def test_weighted_training_recovers_the_true_rates(self):
        _, sampled = sampled_training_set()
        model = weighted_train(sampled, NEGATIVE_SAMPLING_RATE)
        for slot, true_ctr in TRUE_CTR_BY_SLOT.items():
            self.assertAlmostEqual(model[slot], true_ctr, delta=0.15 * true_ctr)

    def test_closed_form_correction_matches_weighted_training(self):
        _, sampled = sampled_training_set()
        corrected = corrected_model(naive_train(sampled), NEGATIVE_SAMPLING_RATE)
        weighted = weighted_train(sampled, NEGATIVE_SAMPLING_RATE)
        for slot in TRUE_CTR_BY_SLOT:
            self.assertAlmostEqual(corrected[slot], weighted[slot], places=9)

    def test_corrected_model_is_close_to_training_on_all_data(self):
        impressions, sampled = sampled_training_set()
        corrected = corrected_model(naive_train(sampled), NEGATIVE_SAMPLING_RATE)
        full = naive_train(impressions)
        for slot in TRUE_CTR_BY_SLOT:
            self.assertAlmostEqual(corrected[slot], full[slot], delta=0.1 * full[slot])

    def test_training_set_is_much_smaller_than_the_traffic(self):
        impressions, sampled = sampled_training_set()
        self.assertLess(len(sampled), 0.15 * len(impressions))


class CorrectionFormulaEdgeCases(unittest.TestCase):
    def test_rate_of_one_changes_nothing(self):
        self.assertAlmostEqual(correct_for_sampling(0.3, 1.0), 0.3)

    def test_zero_and_one_stay_at_the_bounds(self):
        self.assertEqual(correct_for_sampling(0.0, 0.1), 0.0)
        self.assertEqual(correct_for_sampling(1.0, 0.1), 1.0)

    def test_half_at_ten_percent_sampling_is_about_nine_percent(self):
        self.assertAlmostEqual(correct_for_sampling(0.5, 0.1), 1 / 11)

    def test_invalid_rate_is_rejected(self):
        for rate in (0.0, -0.1, 1.5):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                correct_for_sampling(0.2, rate)

    def test_invalid_probability_is_rejected(self):
        with self.assertRaises(ValueError):
            correct_for_sampling(1.2, 0.1)


if __name__ == "__main__":
    unittest.main()
