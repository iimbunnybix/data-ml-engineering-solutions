import random
import unittest

from solution import (Example, LinearModel, LockedTestSet, TestSetReusedError, accuracy, best_candidate,
                      make_examples, naive_select_and_report, random_search_candidates, select_and_report)

CANDIDATES = 100
VALIDATION_SIZE = 300
TEST_SIZE = 300
PRODUCTION_SIZE = 5000
SWEEPS = 40


def run_sweep(seed: int) -> tuple[list[LinearModel], list[Example], list[Example], list[Example]]:
    """One seeded hyperparameter sweep: candidates plus validation, test and production data."""
    rng = random.Random(seed)
    candidates = random_search_candidates(CANDIDATES, rng)
    validation = make_examples(VALIDATION_SIZE, rng)
    test = make_examples(TEST_SIZE, rng)
    production = make_examples(PRODUCTION_SIZE, rng)
    return candidates, validation, test, production


def optimism_per_sweep(use_fix: bool) -> list[float]:
    """Reported accuracy minus production accuracy of the chosen model, for every sweep."""
    gaps = []
    for seed in range(SWEEPS):
        candidates, validation, test, production = run_sweep(seed)
        if use_fix:
            report = select_and_report(candidates, validation, LockedTestSet(test))
        else:
            report = naive_select_and_report(candidates, test)
        gaps.append(report.reported_accuracy - accuracy(candidates[report.chosen_index], production))
    return gaps


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def count_positive(values: list[float]) -> int:
    return sum(1 for value in values if value > 0)


ALWAYS_ONE = LinearModel((1.0,))
EXAMPLES = [Example((1.0,), 1), Example((2.0,), 1), Example((3.0,), 0), Example((4.0,), 0)]


class ReproduceProblem(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.gaps = optimism_per_sweep(use_fix=False)

    def test_choosing_on_the_test_set_overstates_production_accuracy_on_average(self):
        self.assertGreater(mean(self.gaps), 0.02)

    def test_choosing_on_the_test_set_overstates_accuracy_in_most_sweeps(self):
        self.assertGreaterEqual(count_positive(self.gaps), 30)

    def test_naive_report_is_the_score_used_for_selection(self):
        candidates, _, test, _ = run_sweep(seed=0)
        report = naive_select_and_report(candidates, test)
        self.assertEqual(report.reported_accuracy, report.selection_accuracy)


class FixLockedTestSet(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.gaps = optimism_per_sweep(use_fix=True)

    def test_reported_accuracy_is_close_to_production_on_average(self):
        self.assertLess(abs(mean(self.gaps)), 0.01)

    def test_reported_accuracy_overstates_only_about_half_the_time(self):
        self.assertEqual(count_positive(self.gaps), SWEEPS // 2)

    def test_second_evaluation_is_refused(self):
        locked = LockedTestSet(EXAMPLES)
        locked.evaluate_once(ALWAYS_ONE)
        with self.assertRaises(TestSetReusedError):
            locked.evaluate_once(ALWAYS_ONE)

    def test_selection_uses_validation_not_test(self):
        good_on_validation = LinearModel((1.0,))
        good_on_test = LinearModel((-1.0,))
        validation = [Example((1.0,), 1)]
        test = [Example((1.0,), 0)]
        report = select_and_report([good_on_test, good_on_validation], validation, LockedTestSet(test))
        self.assertEqual(report.chosen_index, 1)
        self.assertEqual(report.reported_accuracy, 0.0)


class Helpers(unittest.TestCase):
    def test_accuracy_counts_correct_predictions(self):
        self.assertEqual(accuracy(ALWAYS_ONE, EXAMPLES), 0.5)

    def test_accuracy_of_empty_dataset_is_an_error(self):
        with self.assertRaises(ValueError):
            accuracy(ALWAYS_ONE, [])

    def test_best_candidate_keeps_the_first_on_ties(self):
        self.assertEqual(best_candidate([ALWAYS_ONE, LinearModel((2.0,))], EXAMPLES), (0, 0.5))

    def test_best_candidate_needs_candidates(self):
        with self.assertRaises(ValueError):
            best_candidate([], EXAMPLES)


if __name__ == "__main__":
    unittest.main()
