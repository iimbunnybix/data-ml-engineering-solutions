import unittest

from solution import (
    ArmCounts,
    Phase,
    allocation_changed,
    naive_pooled_difference,
    stratified_difference,
)


def ramped_experiment() -> list[Phase]:
    """Week 1: 10% treatment, normal traffic. Week 2: 50% treatment, holiday sale.

    Inside each week the new checkout converts worse than control
    (9.5% vs 10%, then 19% vs 20%).
    """
    week_1 = Phase("week 1 (10% ramp)",
                   control=ArmCounts(users=90_000, conversions=9_000),
                   treatment=ArmCounts(users=10_000, conversions=950))
    week_2 = Phase("week 2 (50% ramp, sale)",
                   control=ArmCounts(users=50_000, conversions=10_000),
                   treatment=ArmCounts(users=50_000, conversions=9_500))
    return [week_1, week_2]


def constant_split_experiment() -> list[Phase]:
    """Same baselines, but a 50/50 split in both weeks."""
    week_1 = Phase("week 1", ArmCounts(50_000, 5_000), ArmCounts(50_000, 4_750))
    week_2 = Phase("week 2", ArmCounts(50_000, 10_000), ArmCounts(50_000, 9_500))
    return [week_1, week_2]


class ReproduceProblem(unittest.TestCase):
    def test_pooled_analysis_reports_a_large_win(self):
        difference = naive_pooled_difference(ramped_experiment())
        self.assertAlmostEqual(difference, 10_450 / 60_000 - 19_000 / 140_000)
        self.assertGreater(difference, 0.038)

    def test_treatment_loses_inside_every_week(self):
        for phase in ramped_experiment():
            self.assertLess(phase.treatment.rate(), phase.control.rate())


class FixStratifiesByRampPhase(unittest.TestCase):
    def test_ramp_is_detected_as_an_allocation_change(self):
        self.assertTrue(allocation_changed(ramped_experiment()))

    def test_constant_split_is_not_flagged(self):
        self.assertFalse(allocation_changed(constant_split_experiment()))

    def test_stratified_difference_is_negative(self):
        result = stratified_difference(ramped_experiment())
        # Both weeks have 100,000 users, so the weights are equal:
        # (-0.005 + -0.010) / 2.
        self.assertAlmostEqual(result.difference, -0.0075)

    def test_interval_excludes_zero_so_the_loss_is_significant(self):
        result = stratified_difference(ramped_experiment())
        self.assertLess(result.interval_high, 0.0)
        self.assertLess(result.interval_low, result.difference)

    def test_with_constant_split_pooled_and_stratified_agree(self):
        phases = constant_split_experiment()
        self.assertAlmostEqual(naive_pooled_difference(phases),
                               stratified_difference(phases).difference)

    def test_phase_with_an_empty_arm_is_excluded_and_reported(self):
        paused = Phase("paused day", ArmCounts(3_000, 300), ArmCounts(0, 0))
        result = stratified_difference(ramped_experiment() + [paused])
        self.assertEqual(result.excluded_phases, ["paused day"])
        self.assertAlmostEqual(result.difference, -0.0075)

    def test_no_usable_phase_raises(self):
        only_control = Phase("control only", ArmCounts(1_000, 100), ArmCounts(0, 0))
        with self.assertRaises(ValueError):
            stratified_difference([only_control])

    def test_single_phase_matches_simple_difference(self):
        phase = ramped_experiment()[0]
        result = stratified_difference([phase])
        self.assertAlmostEqual(result.difference, 0.095 - 0.100)


if __name__ == "__main__":
    unittest.main()
