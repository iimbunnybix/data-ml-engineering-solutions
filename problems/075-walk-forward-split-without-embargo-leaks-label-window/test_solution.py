import unittest

from solution import (LABEL_HORIZON_DAYS, TEST_START_DAY, Snapshot, brier_score,
                      build_snapshots, evaluate_split, fit_segment_rates,
                      label_window_end, leaking_training_rows, naive_walk_forward_split,
                      purged_walk_forward_split)


def actual_test_rates(snapshots: list[Snapshot]) -> dict[str, float]:
    """What really happened in the test period, per segment."""
    _, test = naive_walk_forward_split(snapshots)
    return fit_segment_rates(test)


class ReproduceProblem(unittest.TestCase):
    def setUp(self):
        self.snapshots = build_snapshots()
        self.train, self.test = naive_walk_forward_split(self.snapshots)

    def test_training_rows_near_the_cut_have_labels_from_the_test_period(self):
        leaking = leaking_training_rows(self.train, TEST_START_DAY)
        self.assertEqual(len(leaking), 2320)           # 29 snapshot days x 80 customers

    def test_leaky_model_partly_learns_the_mobile_churn_spike_it_could_not_know(self):
        rates, _ = evaluate_split(self.snapshots, naive_walk_forward_split)
        self.assertAlmostEqual(rates["mobile"], 0.264, places=3)

    def test_leaky_backtest_reports_a_flattering_brier_score(self):
        _, score = evaluate_split(self.snapshots, naive_walk_forward_split)
        self.assertAlmostEqual(score, 0.262, places=3)


class FixPurgedSplit(unittest.TestCase):
    def setUp(self):
        self.snapshots = build_snapshots()
        self.train, self.test = purged_walk_forward_split(self.snapshots)

    def test_no_training_label_depends_on_the_test_period(self):
        self.assertEqual(leaking_training_rows(self.train, TEST_START_DAY), [])

    def test_purged_model_only_knows_the_pre_launch_churn_rate(self):
        rates, _ = evaluate_split(self.snapshots, purged_walk_forward_split)
        self.assertAlmostEqual(rates["mobile"], 0.135, places=3)

    def test_honest_brier_score_is_worse_than_the_leaky_one(self):
        _, honest = evaluate_split(self.snapshots, purged_walk_forward_split)
        _, leaky = evaluate_split(self.snapshots, naive_walk_forward_split)
        self.assertAlmostEqual(honest, 0.327, places=3)
        self.assertGreater(honest, leaky)

    def test_mobile_churn_really_jumped_in_the_test_period(self):
        self.assertAlmostEqual(actual_test_rates(self.snapshots)["mobile"], 0.708, places=3)

    def test_test_set_is_the_same_as_in_the_naive_split(self):
        _, naive_test = naive_walk_forward_split(self.snapshots)
        self.assertEqual(self.test, naive_test)

    def test_gap_between_train_and_test_equals_the_label_horizon(self):
        last_train_day = max(row.day for row in self.train)
        self.assertEqual(TEST_START_DAY - last_train_day, LABEL_HORIZON_DAYS)


class FixPurgeBoundaries(unittest.TestCase):
    def test_row_whose_window_ends_on_the_test_start_is_kept(self):
        row = Snapshot(TEST_START_DAY - LABEL_HORIZON_DAYS, "web", 0)
        train, _ = purged_walk_forward_split([row])
        self.assertEqual(train, [row])

    def test_row_one_day_later_is_purged(self):
        row = Snapshot(TEST_START_DAY - LABEL_HORIZON_DAYS + 1, "web", 0)
        train, _ = purged_walk_forward_split([row])
        self.assertEqual(train, [])

    def test_snapshot_on_the_test_start_goes_to_test(self):
        row = Snapshot(TEST_START_DAY, "mobile", 1)
        train, test = purged_walk_forward_split([row])
        self.assertEqual((train, test), ([], [row]))

    def test_label_window_end_is_snapshot_day_plus_horizon(self):
        self.assertEqual(label_window_end(Snapshot(10, "web", 0)), 10 + LABEL_HORIZON_DAYS)

    def test_empty_training_set_gives_an_empty_model(self):
        self.assertEqual(fit_segment_rates([]), {})


class FixBrierScore(unittest.TestCase):
    def test_perfect_predictions_score_zero(self):
        rows = [Snapshot(0, "web", 1), Snapshot(0, "mobile", 0)]
        self.assertEqual(brier_score({"web": 1.0, "mobile": 0.0}, rows), 0.0)

    def test_coin_flip_predictions_score_a_quarter(self):
        rows = [Snapshot(0, "web", 1), Snapshot(0, "web", 0)]
        self.assertEqual(brier_score({"web": 0.5}, rows), 0.25)

    def test_data_is_the_same_on_every_run(self):
        self.assertEqual(build_snapshots(), build_snapshots())


if __name__ == "__main__":
    unittest.main()
