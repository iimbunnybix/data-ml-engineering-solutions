import unittest

from solution import (
    END_OF_TIME,
    Change,
    Version,
    apply_change,
    find_overlaps,
    naive_apply,
    versions_as_of,
)

DAY = 86_400


def load(changes: list[Change], applier) -> list[Version]:
    """Apply changes one by one, in arrival order, with the given loader."""
    history: list[Version] = []
    for change in changes:
        history = applier(history, change)
    return history


def late_upgrade_arrival_order() -> list[Change]:
    """Basic on day 0, Gold on day 20, then a Silver change for day 10 arrives late."""
    return [
        Change("c1", "basic", 0),
        Change("c1", "gold", 20 * DAY),
        Change("c1", "silver", 10 * DAY),
    ]


def tiers(history: list[Version]) -> list[tuple[str, int, int]]:
    ordered = sorted(history, key=lambda version: version.valid_from)
    return [(v.tier, v.valid_from // DAY, v.valid_to) for v in ordered]


class ReproduceProblem(unittest.TestCase):
    def test_late_change_creates_two_versions_for_one_day(self):
        history = load(late_upgrade_arrival_order(), naive_apply)
        self.assertEqual(len(versions_as_of(history, "c1", 5 * DAY)), 1)
        self.assertEqual(len(versions_as_of(history, "c1", 12 * DAY)), 2)

    def test_late_change_overwrites_newer_current_tier(self):
        history = load(late_upgrade_arrival_order(), naive_apply)
        [current] = versions_as_of(history, "c1", 30 * DAY)
        self.assertEqual(current.tier, "silver")

    def test_overlap_check_flags_naive_history(self):
        history = load(late_upgrade_arrival_order(), naive_apply)
        self.assertGreater(len(find_overlaps(history)), 0)


class FixLateChanges(unittest.TestCase):
    def test_late_change_splits_the_interval_it_falls_in(self):
        history = load(late_upgrade_arrival_order(), apply_change)
        self.assertEqual(tiers(history), [
            ("basic", 0, 10 * DAY),
            ("silver", 10, 20 * DAY),
            ("gold", 20, END_OF_TIME),
        ])

    def test_every_point_in_time_has_exactly_one_version(self):
        history = load(late_upgrade_arrival_order(), apply_change)
        for day in range(0, 40):
            self.assertEqual(len(versions_as_of(history, "c1", day * DAY)), 1, day)

    def test_result_does_not_depend_on_arrival_order(self):
        in_order = sorted(late_upgrade_arrival_order(), key=lambda c: c.effective_from)
        self.assertEqual(tiers(load(in_order, apply_change)),
                         tiers(load(late_upgrade_arrival_order(), apply_change)))

    def test_change_before_all_history_ends_where_history_starts(self):
        history = load([Change("c1", "gold", 20 * DAY), Change("c1", "basic", 0)], apply_change)
        self.assertEqual(tiers(history), [("basic", 0, 20 * DAY), ("gold", 20, END_OF_TIME)])

    def test_replayed_change_does_not_add_a_version(self):
        changes = late_upgrade_arrival_order()
        history = load(changes + changes, apply_change)
        self.assertEqual(len(history), 3)
        self.assertEqual(find_overlaps(history), [])

    def test_same_effective_time_corrects_the_tier(self):
        history = load([Change("c1", "basic", 0), Change("c1", "gold", 0)], apply_change)
        self.assertEqual(tiers(history), [("gold", 0, END_OF_TIME)])

    def test_other_customers_are_untouched(self):
        changes = [Change("c2", "basic", 0), *late_upgrade_arrival_order()]
        history = load(changes, apply_change)
        [other] = versions_as_of(history, "c2", 15 * DAY)
        self.assertEqual((other.tier, other.valid_to), ("basic", END_OF_TIME))

    def test_overlap_check_flags_a_single_inverted_version(self):
        inverted = Version("c1", "gold", 20 * DAY, 10 * DAY)
        self.assertEqual(find_overlaps([inverted]), [(inverted, inverted)])

    def test_overlap_check_flags_an_inverted_last_version(self):
        first = Version("c1", "basic", 0, 10 * DAY)
        inverted_last = Version("c1", "gold", 20 * DAY, 15 * DAY)
        self.assertEqual(find_overlaps([first, inverted_last]), [(inverted_last, inverted_last)])

    def test_unknown_customer_has_no_versions(self):
        history = load(late_upgrade_arrival_order(), apply_change)
        self.assertEqual(versions_as_of(history, "nobody", 5 * DAY), [])


if __name__ == "__main__":
    unittest.main()
