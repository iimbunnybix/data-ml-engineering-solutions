import unittest
from datetime import date, timedelta

from solution import (PartitionConflict, naive_new_partitions, parse_partition_date,
                      plan_incremental_load)

SEPTEMBER_30 = date(2026, 9, 30)


def padded_name(day: date) -> str:
    """Partition name written by the old exporter: dt=2026-09-30."""
    return f"dt={day:%Y-%m-%d}"


def unpadded_name(day: date) -> str:
    """Partition name written by the new exporter: dt=2026-10-9."""
    return f"dt={day.year}-{day.month}-{day.day}"


def october_unpadded() -> list[str]:
    """All 31 October partitions, as the new exporter names them."""
    names = []
    for offset in range(1, 32):
        names.append(unpadded_name(SEPTEMBER_30 + timedelta(days=offset)))
    return names


def run_daily_naive(names: list[str], last_loaded: str) -> list[str]:
    """Simulate one naive run per day as each partition lands."""
    loaded = []
    for index in range(len(names)):
        landed_so_far = names[: index + 1]
        for name in naive_new_partitions(landed_so_far, last_loaded):
            loaded.append(name)
            last_loaded = name
    return loaded


class ReproduceProblem(unittest.TestCase):
    def test_text_order_puts_day_10_before_day_9(self):
        self.assertLess("dt=2026-10-10", "dt=2026-10-9")

    def test_daily_naive_runs_skip_october_10_to_31(self):
        loaded = run_daily_naive(october_unpadded(), padded_name(SEPTEMBER_30))
        self.assertEqual(len(loaded), 9)
        self.assertEqual(loaded[-1], "dt=2026-10-9")


class FixParsing(unittest.TestCase):
    def test_padded_and_unpadded_names_parse_to_the_same_date(self):
        self.assertEqual(parse_partition_date("dt=2026-10-09"), date(2026, 10, 9))
        self.assertEqual(parse_partition_date("dt=2026-10-9"), date(2026, 10, 9))

    def test_impossible_date_is_not_parsed(self):
        self.assertIsNone(parse_partition_date("dt=2026-02-30"))

    def test_unrelated_folder_is_not_parsed(self):
        self.assertIsNone(parse_partition_date("_temporary"))


class FixIncrementalPlan(unittest.TestCase):
    def test_daily_runs_load_all_31_october_days(self):
        names = october_unpadded()
        watermark = SEPTEMBER_30
        loaded = []
        for index in range(len(names)):
            plan = plan_incremental_load(names[: index + 1], watermark)
            loaded.extend(plan.to_load)
            watermark = plan.new_watermark
        self.assertEqual(len(loaded), 31)
        self.assertEqual(watermark, date(2026, 10, 31))

    def test_partitions_are_loaded_in_date_order(self):
        plan = plan_incremental_load(["dt=2026-10-10", "dt=2026-10-9"], date(2026, 10, 8))
        self.assertEqual(plan.to_load, ["dt=2026-10-9", "dt=2026-10-10"])

    def test_already_loaded_days_are_not_reloaded(self):
        plan = plan_incremental_load(october_unpadded(), date(2026, 10, 31))
        self.assertEqual(plan.to_load, [])
        self.assertEqual(plan.new_watermark, date(2026, 10, 31))

    def test_two_spellings_of_one_day_are_refused(self):
        with self.assertRaises(PartitionConflict):
            plan_incremental_load(["dt=2026-10-09", "dt=2026-10-9"], SEPTEMBER_30)

    def test_unparseable_names_are_reported_not_loaded(self):
        plan = plan_incremental_load(["dt=2026-10-1", "dt=20261002"], SEPTEMBER_30)
        self.assertEqual(plan.to_load, ["dt=2026-10-1"])
        self.assertEqual(plan.unparseable, ["dt=20261002"])

    def test_gap_is_reported_and_watermark_stops_before_it(self):
        names = ["dt=2026-10-1", "dt=2026-10-3"]
        plan = plan_incremental_load(names, SEPTEMBER_30)
        self.assertEqual(plan.missing_days, [date(2026, 10, 2)])
        self.assertEqual(plan.to_load, ["dt=2026-10-1"])
        self.assertEqual(plan.new_watermark, date(2026, 10, 1))

    def test_late_day_is_loaded_once_it_lands(self):
        names = ["dt=2026-10-1", "dt=2026-10-3", "dt=2026-10-2"]
        plan = plan_incremental_load(names, date(2026, 10, 1))
        self.assertEqual(plan.to_load, ["dt=2026-10-2", "dt=2026-10-3"])
        self.assertEqual(plan.missing_days, [])

    def test_month_and_year_boundaries_are_ordered_by_date(self):
        names = ["dt=2027-1-1", "dt=2026-12-31"]
        plan = plan_incremental_load(names, date(2026, 12, 30))
        self.assertEqual(plan.to_load, ["dt=2026-12-31", "dt=2027-1-1"])


if __name__ == "__main__":
    unittest.main()
