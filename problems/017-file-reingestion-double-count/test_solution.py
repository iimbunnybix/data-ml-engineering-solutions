import unittest

from solution import (
    ALREADY_LOADED,
    CHANGED_AFTER_LOAD,
    DUPLICATE_CONTENT,
    LOADED,
    LandingZone,
    Warehouse,
    exactly_once_ingest,
    naive_ingest,
    run_until_success,
)


def sales_file(*amounts: int, prefix: str = "o") -> bytes:
    """Build a small sales CSV with one order per amount."""
    lines = ["order_id,amount"]
    for index, amount in enumerate(amounts):
        lines.append(f"{prefix}{index},{amount}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def three_day_zone(crash_on: str | None = None) -> LandingZone:
    """Landing folder with three daily files totalling 600."""
    files = {
        "sales/2026-11-01.csv": sales_file(100, 50, prefix="a"),
        "sales/2026-11-02.csv": sales_file(200, prefix="b"),
        "sales/2026-11-03.csv": sales_file(150, 100, prefix="c"),
    }
    return LandingZone(files=files, crash_on_archive=crash_on)


EXPECTED_TOTAL = 600


class ReproduceProblem(unittest.TestCase):
    def test_crash_before_archive_double_counts_on_retry(self):
        zone = three_day_zone(crash_on="sales/2026-11-02.csv")
        warehouse = Warehouse()
        run_until_success(naive_ingest, zone, warehouse)
        self.assertEqual(warehouse.total_amount(), EXPECTED_TOTAL + 200)

    def test_rerunning_with_file_put_back_double_counts(self):
        warehouse = Warehouse()
        naive_ingest(three_day_zone(), warehouse)
        naive_ingest(three_day_zone(), warehouse)
        self.assertEqual(warehouse.total_amount(), 2 * EXPECTED_TOTAL)


class FixExactlyOnceIngest(unittest.TestCase):
    def test_crash_before_archive_does_not_double_count(self):
        zone = three_day_zone(crash_on="sales/2026-11-02.csv")
        warehouse = Warehouse()
        outcomes = run_until_success(exactly_once_ingest, zone, warehouse)
        self.assertEqual(warehouse.total_amount(), EXPECTED_TOTAL)
        self.assertEqual(outcomes["sales/2026-11-02.csv"], ALREADY_LOADED)
        self.assertEqual(outcomes["sales/2026-11-03.csv"], LOADED)

    def test_every_file_is_archived_after_recovery(self):
        zone = three_day_zone(crash_on="sales/2026-11-01.csv")
        run_until_success(exactly_once_ingest, zone, Warehouse())
        self.assertEqual(zone.files, {})
        self.assertEqual(len(zone.archived), 3)

    def test_replaying_the_same_files_loads_nothing_new(self):
        warehouse = Warehouse()
        exactly_once_ingest(three_day_zone(), warehouse)
        outcomes = exactly_once_ingest(three_day_zone(), warehouse)
        self.assertEqual(warehouse.total_amount(), EXPECTED_TOTAL)
        self.assertEqual(set(outcomes.values()), {ALREADY_LOADED})

    def test_same_content_under_new_name_is_skipped(self):
        warehouse = Warehouse()
        exactly_once_ingest(three_day_zone(), warehouse)
        reupload = LandingZone(files={"sales/2026-11-02-resend.csv": sales_file(200, prefix="b")})
        outcomes = exactly_once_ingest(reupload, warehouse)
        self.assertEqual(outcomes["sales/2026-11-02-resend.csv"], DUPLICATE_CONTENT)
        self.assertEqual(warehouse.total_amount(), EXPECTED_TOTAL)

    def test_corrected_file_with_same_name_is_held_for_review(self):
        warehouse = Warehouse()
        exactly_once_ingest(three_day_zone(), warehouse)
        corrected = LandingZone(files={"sales/2026-11-02.csv": sales_file(250, prefix="b")})
        outcomes = exactly_once_ingest(corrected, warehouse)
        self.assertEqual(outcomes["sales/2026-11-02.csv"], CHANGED_AFTER_LOAD)
        self.assertEqual(warehouse.total_amount(), EXPECTED_TOTAL)
        self.assertIn("sales/2026-11-02.csv", corrected.files)

    def test_manifest_records_row_count_per_file(self):
        warehouse = Warehouse()
        exactly_once_ingest(three_day_zone(), warehouse)
        counts = sorted(warehouse.manifest.values())
        self.assertEqual(counts, [1, 2, 2])

    def test_empty_landing_folder_is_a_no_op(self):
        warehouse = Warehouse()
        outcomes = exactly_once_ingest(LandingZone(files={}), warehouse)
        self.assertEqual(outcomes, {})
        self.assertEqual(warehouse.total_amount(), 0)


if __name__ == "__main__":
    unittest.main()
