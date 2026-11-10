import random
import unittest

from solution import (DECLINE_THRESHOLD, EXPLORATION_RATE, Transaction, apply_policy,
                      generate_traffic, labelled_only, naive_retrain, run_retrain_cycles,
                      weighted_fraud_rate, weighted_retrain)

RISKY = "new_device_foreign_ip"
FRAUD_RATES = {"domestic_card": 0.01, "foreign_card": 0.03, RISKY: 0.30}
SEGMENTS = list(FRAUD_RATES)
COUNT_PER_SEGMENT = 10_000
CYCLES = 6
SEED = 7


def simulate(use_fix: bool) -> list[dict]:
    """Six deterministic retrain cycles with or without the fix."""
    return run_retrain_cycles(FRAUD_RATES, COUNT_PER_SEGMENT, CYCLES, SEED, use_fix)


def risky_is_declined(cycle: dict) -> bool:
    """True when that cycle's model declines the risky segment."""
    return cycle["model"][RISKY] > DECLINE_THRESHOLD


def total_approved_fraud(history: list[dict]) -> int:
    """Fraud let through across all cycles."""
    return sum(cycle["approved_fraud"] for cycle in history)


def one_period(model: dict[str, float], exploration_rate: float) -> list[Transaction]:
    """One period of traffic decided by ``model``."""
    rng = random.Random(SEED)
    traffic = generate_traffic(rng, FRAUD_RATES, COUNT_PER_SEGMENT)
    return apply_policy(traffic, model, rng, exploration_rate)


BLOCKING_MODEL = {"domestic_card": 0.01, "foreign_card": 0.03, RISKY: 0.30}


class ReproduceProblem(unittest.TestCase):
    def test_declined_segment_has_no_labels(self):
        labelled = labelled_only(one_period(BLOCKING_MODEL, exploration_rate=0.0))
        risky_labels = [item for item in labelled if item.segment == RISKY]
        self.assertEqual(risky_labels, [])

    def test_naive_retrain_rates_the_blocked_segment_as_low_risk(self):
        labelled = labelled_only(one_period(BLOCKING_MODEL, exploration_rate=0.0))
        model = naive_retrain(labelled, SEGMENTS)
        self.assertLess(model[RISKY], DECLINE_THRESHOLD)

    def test_naive_model_flips_between_blocking_and_approving(self):
        decisions = [risky_is_declined(cycle) for cycle in simulate(use_fix=False)]
        self.assertEqual(decisions, [True, False, True, False, True, False])

    def test_naive_loop_lets_through_about_3400_frauds_every_other_cycle(self):
        approving_cycles = [cycle for cycle in simulate(use_fix=False)
                            if not risky_is_declined(cycle)]
        for cycle in approving_cycles:
            self.assertGreater(cycle["approved_fraud"], 3_000)


class FixExplorationAndWeighting(unittest.TestCase):
    def test_risky_segment_stays_declined_on_every_retrain(self):
        decisions = [risky_is_declined(cycle) for cycle in simulate(use_fix=True)]
        self.assertEqual(decisions, [True] * CYCLES)

    def test_fix_lets_through_far_less_fraud_overall(self):
        naive_total = total_approved_fraud(simulate(use_fix=False))
        fixed_total = total_approved_fraud(simulate(use_fix=True))
        self.assertLess(fixed_total, naive_total / 3)

    def test_exploration_gives_the_risky_segment_labels(self):
        labelled = labelled_only(one_period(BLOCKING_MODEL, EXPLORATION_RATE))
        risky_labels = [item for item in labelled if item.segment == RISKY]
        self.assertGreater(len(risky_labels), 400)
        self.assertLess(len(risky_labels), 600)

    def test_weighted_overall_rate_matches_all_traffic(self):
        decided = one_period(BLOCKING_MODEL, EXPLORATION_RATE)
        true_rate = sum(1 for item in decided if item.is_fraud) / len(decided)
        weighted = weighted_fraud_rate(labelled_only(decided))
        self.assertAlmostEqual(weighted, true_rate, delta=0.02)

    def test_unweighted_overall_rate_underestimates_fraud(self):
        decided = one_period(BLOCKING_MODEL, EXPLORATION_RATE)
        labelled = labelled_only(decided)
        unweighted = sum(1 for item in labelled if item.is_fraud) / len(labelled)
        true_rate = sum(1 for item in decided if item.is_fraud) / len(decided)
        self.assertLess(unweighted, true_rate / 2)

    def test_segment_without_labels_keeps_previous_estimate(self):
        labelled = [Transaction("domestic_card", False, True, 1.0)]
        model = weighted_retrain(labelled, SEGMENTS, BLOCKING_MODEL)
        self.assertEqual(model[RISKY], BLOCKING_MODEL[RISKY])

    def test_weighted_rate_of_no_rows_is_none(self):
        self.assertIsNone(weighted_fraud_rate([]))

    def test_low_propensity_fraud_counts_for_more(self):
        rows = [Transaction(RISKY, True, True, 0.5), Transaction(RISKY, False, True, 1.0)]
        self.assertAlmostEqual(weighted_fraud_rate(rows), 2 / 3)


if __name__ == "__main__":
    unittest.main()
