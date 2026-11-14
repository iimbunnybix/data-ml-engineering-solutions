import unittest

from solution import (
    CustomerRow,
    UnsafeDeleteError,
    active_count,
    incremental_sync,
    keys_to_delete,
    naive_incremental_sync,
    reconcile_deletes,
)

CUSTOMER_COUNT = 1000
DELETED_COUNT = 30           # GDPR erasures and closed accounts in one day
FIRST_SYNC_TIME = 1000
DELETE_TIME = 2000


def make_source() -> dict[int, CustomerRow]:
    """1,000 active customers, all last changed before the first sync."""
    source = {}
    for customer_id in range(1, CUSTOMER_COUNT + 1):
        source[customer_id] = CustomerRow(customer_id, "active", FIRST_SYNC_TIME)
    return source


def delete_some(source: dict[int, CustomerRow]) -> None:
    """Hard-delete the first DELETED_COUNT customers and update one other."""
    for customer_id in range(1, DELETED_COUNT + 1):
        del source[customer_id]
    source[500] = CustomerRow(500, "suspended", DELETE_TIME)


def synced_copy(sync) -> tuple[dict[int, CustomerRow], dict[int, CustomerRow]]:
    """Run one initial sync and one sync after the deletes; return (source, warehouse)."""
    source = make_source()
    warehouse: dict[int, CustomerRow] = {}
    watermark = sync(source, warehouse, 0)
    delete_some(source)
    sync(source, warehouse, watermark)
    return source, warehouse


class ReproduceProblem(unittest.TestCase):
    def test_deleted_customers_stay_in_the_warehouse(self):
        source, warehouse = synced_copy(naive_incremental_sync)
        self.assertEqual(len(warehouse) - len(source), DELETED_COUNT)

    def test_active_count_is_inflated_by_deleted_customers(self):
        source, warehouse = synced_copy(naive_incremental_sync)
        self.assertEqual(active_count(source), 969)
        self.assertEqual(active_count(warehouse), 999)

    def test_rerunning_the_naive_sync_never_heals_it(self):
        source, warehouse = synced_copy(naive_incremental_sync)
        naive_incremental_sync(source, warehouse, 0)
        self.assertIn(1, warehouse)


class FixKeyReconciliation(unittest.TestCase):
    def test_warehouse_matches_source_after_deletes(self):
        source, warehouse = synced_copy(incremental_sync)
        self.assertEqual(warehouse, source)

    def test_updates_are_still_applied(self):
        _, warehouse = synced_copy(incremental_sync)
        self.assertEqual(warehouse[500].status, "suspended")

    def test_active_count_matches_source(self):
        source, warehouse = synced_copy(incremental_sync)
        self.assertEqual(active_count(warehouse), active_count(source))

    def test_keys_to_delete_lists_only_missing_keys(self):
        warehouse = {key: CustomerRow(key, "active", 1) for key in (1, 2, 3)}
        self.assertEqual(keys_to_delete({2}, warehouse), [1, 3])

    def test_nothing_is_deleted_when_nothing_is_missing(self):
        source = make_source()
        warehouse = dict(source)
        self.assertEqual(reconcile_deletes(set(source), warehouse), [])
        self.assertEqual(len(warehouse), CUSTOMER_COUNT)


class FixSafetyLimit(unittest.TestCase):
    def test_empty_key_extract_does_not_wipe_the_table(self):
        warehouse = make_source()
        with self.assertRaises(UnsafeDeleteError):
            reconcile_deletes(set(), warehouse)
        self.assertEqual(len(warehouse), CUSTOMER_COUNT)

    def test_delete_just_above_the_limit_is_refused(self):
        warehouse = make_source()
        source_keys = set(range(52, CUSTOMER_COUNT + 1))       # 51 of 1,000 missing
        with self.assertRaises(UnsafeDeleteError):
            reconcile_deletes(source_keys, warehouse)

    def test_delete_at_the_limit_is_allowed(self):
        warehouse = make_source()
        source_keys = set(range(51, CUSTOMER_COUNT + 1))       # 50 of 1,000 missing
        self.assertEqual(len(reconcile_deletes(source_keys, warehouse)), 50)

    def test_empty_warehouse_is_fine(self):
        self.assertEqual(reconcile_deletes(set(), {}), [])


if __name__ == "__main__":
    unittest.main()
