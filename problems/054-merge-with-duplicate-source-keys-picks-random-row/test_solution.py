import random
import unittest

from solution import (
    StatusChange,
    count_wrong_keys,
    dedupe_batch,
    fixed_merge,
    naive_merge,
    simulate_daily_export,
    true_final_status,
)

SUBSCRIPTIONS = 2_000


def daily_export() -> list[StatusChange]:
    return simulate_daily_export(SUBSCRIPTIONS, random.Random(54))


def reordered(batch: list[StatusChange], seed: int) -> list[StatusChange]:
    """Same rows, different arrival order (another parallel load)."""
    copy = list(batch)
    random.Random(seed).shuffle(copy)
    return copy


def load_naively(batch: list[StatusChange]) -> dict[int, StatusChange]:
    target: dict[int, StatusChange] = {}
    naive_merge(target, batch)
    return target


def load_fixed(batch: list[StatusChange]) -> dict[int, StatusChange]:
    target: dict[int, StatusChange] = {}
    fixed_merge(target, batch)
    return target


class ReproduceProblem(unittest.TestCase):
    def test_naive_merge_leaves_hundreds_of_keys_with_an_older_row(self):
        batch = daily_export()
        wrong = count_wrong_keys(load_naively(batch), true_final_status(batch))
        self.assertGreater(wrong, 900)

    def test_naive_result_changes_when_the_same_rows_arrive_in_another_order(self):
        batch = daily_export()
        first = load_naively(batch)
        second = load_naively(reordered(batch, seed=7))
        self.assertNotEqual(first, second)


class FixDedupeBeforeMerge(unittest.TestCase):
    def test_fixed_merge_holds_the_latest_row_for_every_key(self):
        batch = daily_export()
        self.assertEqual(count_wrong_keys(load_fixed(batch), true_final_status(batch)), 0)

    def test_fixed_result_does_not_depend_on_row_order(self):
        batch = daily_export()
        self.assertEqual(load_fixed(batch), load_fixed(reordered(batch, seed=7)))

    def test_exact_duplicate_rows_collapse_to_one_winner(self):
        row = StatusChange(1, "active", 600)
        result = dedupe_batch([row, row])
        self.assertEqual(result.winners, {1: row})
        self.assertEqual(result.conflicts, {})

    def test_different_rows_with_the_same_latest_time_are_quarantined(self):
        active = StatusChange(1, "active", 600)
        cancelled = StatusChange(1, "cancelled", 600)
        result = dedupe_batch([cancelled, StatusChange(1, "trial", 10), active])
        self.assertEqual(result.winners, {})
        self.assertEqual(result.conflicts, {1: [active, cancelled]})

    def test_a_tie_on_an_older_row_is_not_a_conflict(self):
        latest = StatusChange(1, "cancelled", 900)
        batch = [StatusChange(1, "trial", 10), StatusChange(1, "active", 10), latest]
        self.assertEqual(dedupe_batch(batch).winners, {1: latest})

    def test_quarantined_key_keeps_its_current_target_row(self):
        current = StatusChange(1, "trial", 5)
        target = {1: current}
        fixed_merge(target, [StatusChange(1, "active", 600), StatusChange(1, "cancelled", 600)])
        self.assertEqual(target, {1: current})

    def test_reloading_an_older_batch_does_not_overwrite_a_newer_target_row(self):
        newer = StatusChange(1, "cancelled", 900)
        target = {1: newer}
        fixed_merge(target, [StatusChange(1, "active", 300)])
        self.assertEqual(target, {1: newer})

    def test_empty_batch_changes_nothing(self):
        target = {1: StatusChange(1, "active", 5)}
        fixed_merge(target, [])
        self.assertEqual(target, {1: StatusChange(1, "active", 5)})


if __name__ == "__main__":
    unittest.main()
