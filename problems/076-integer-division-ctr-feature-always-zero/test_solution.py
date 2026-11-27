import unittest

from solution import (AdStats, build_ads, ctr_feature, feature_collapse_problems, global_ctr,
                      mean_true_ctr_of_top_ads, new_ads_in_top, original_ctr, raw_float_ctr,
                      smoothed_ctr)


def ad(impressions: int, clicks: int) -> AdStats:
    """A hand-made ad; true_ctr is irrelevant for feature tests."""
    return AdStats(ad_id=0, impressions=impressions, clicks=clicks, true_ctr=0.0)


def average_true_ctr(ads: list[AdStats]) -> float:
    """What a random ranking would score on average."""
    return sum(item.true_ctr for item in ads) / len(ads)


class ReproduceProblem(unittest.TestCase):
    def setUp(self):
        self.ads = build_ads()
        self.feature = [original_ctr(item) for item in self.ads]

    def test_integer_division_gives_zero_when_clicks_are_fewer_than_impressions(self):
        self.assertEqual(original_ctr(ad(impressions=1000, clicks=40)), 0)

    def test_only_one_click_in_one_impression_ads_get_ctr_one(self):
        ones = [item for item in self.ads if original_ctr(item) == 1]
        self.assertEqual([(item.impressions, item.clicks) for item in ones], [(1, 1)] * 3)

    def test_feature_is_zero_for_almost_every_ad(self):
        zeros = sum(1 for value in self.feature if value == 0)
        self.assertEqual(zeros, 997)

    def test_ranking_by_the_feature_is_no_better_than_random(self):
        top = mean_true_ctr_of_top_ads(self.ads, self.feature)
        self.assertAlmostEqual(top, 0.0442, places=4)
        self.assertAlmostEqual(average_true_ctr(self.ads), 0.0431, places=4)


class FixSmoothedCtr(unittest.TestCase):
    def setUp(self):
        self.ads = build_ads()
        self.feature = ctr_feature(self.ads)

    def test_large_ad_keeps_almost_its_own_ctr(self):
        value = smoothed_ctr(ad(impressions=100_000, clicks=4000), prior_ctr=0.01)
        self.assertAlmostEqual(value, 0.04, places=3)

    def test_one_click_in_one_impression_stays_near_the_prior(self):
        value = smoothed_ctr(ad(impressions=1, clicks=1), prior_ctr=0.04)
        self.assertAlmostEqual(value, 0.0495, places=4)

    def test_zero_impressions_gives_the_prior(self):
        self.assertEqual(smoothed_ctr(ad(impressions=0, clicks=0), prior_ctr=0.04), 0.04)

    def test_zero_impressions_without_smoothing_does_not_crash(self):
        value = smoothed_ctr(ad(impressions=0, clicks=0), prior_ctr=0.04, prior_strength=0.0)
        self.assertEqual(value, 0.04)

    def test_without_smoothing_it_is_plain_float_division(self):
        value = smoothed_ctr(ad(impressions=1000, clicks=40), prior_ctr=0.5, prior_strength=0.0)
        self.assertEqual(value, 0.04)

    def test_global_ctr_of_no_impressions_is_zero(self):
        self.assertEqual(global_ctr([ad(impressions=0, clicks=0)]), 0.0)

    def test_ranking_by_smoothed_ctr_finds_high_ctr_ads(self):
        self.assertAlmostEqual(mean_true_ctr_of_top_ads(self.ads, self.feature), 0.0756, places=4)

    def test_smoothing_keeps_lucky_new_ads_out_of_the_top(self):
        raw = [raw_float_ctr(item) for item in self.ads]
        self.assertEqual(new_ads_in_top(self.ads, raw), 12)
        self.assertEqual(new_ads_in_top(self.ads, self.feature), 0)

    def test_smoothed_beats_raw_float_ctr(self):
        raw = [raw_float_ctr(item) for item in self.ads]
        self.assertAlmostEqual(mean_true_ctr_of_top_ads(self.ads, raw), 0.0690, places=4)
        self.assertGreater(mean_true_ctr_of_top_ads(self.ads, self.feature),
                           mean_true_ctr_of_top_ads(self.ads, raw))


class FixCollapseCheck(unittest.TestCase):
    def test_flags_the_truncated_feature(self):
        feature = [original_ctr(item) for item in build_ads()]
        self.assertEqual(len(feature_collapse_problems(feature)), 2)

    def test_passes_the_fixed_feature(self):
        self.assertEqual(feature_collapse_problems(ctr_feature(build_ads())), [])

    def test_flags_an_all_zero_column(self):
        problems = feature_collapse_problems([0.0] * 100)
        self.assertIn("only 1 distinct values (minimum 10)", problems)

    def test_flags_one_value_dominating_despite_many_distinct_values(self):
        values = [0.0] * 990 + [index / 100 for index in range(1, 11)]
        self.assertEqual(feature_collapse_problems(values), ["value 0.0 on 99.0% of rows (maximum 95%)"])

    def test_flags_an_empty_column(self):
        self.assertEqual(feature_collapse_problems([]), ["feature column is empty"])

    def test_data_is_the_same_on_every_run(self):
        self.assertEqual(build_ads(), build_ads())


if __name__ == "__main__":
    unittest.main()
