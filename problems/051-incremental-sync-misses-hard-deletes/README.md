# 051 · Deleted customers kept showing up as active in the warehouse, and re-running the sync never removed them

## Problem
A subscription business copies its `customers` table from PostgreSQL into the
warehouse every 15 minutes with an incremental job: `SELECT * FROM customers
WHERE updated_at > :watermark`, upsert the rows, advance the watermark. Account
closures and GDPR erasure requests are handled by the application with a hard
`DELETE`.

The "active customers" KPI in the warehouse slowly drifted above the number in
the billing system. Marketing emailed people who had asked to be erased, which
turned a reporting bug into a compliance incident. In the test scenario (1,000
active customers, 30 hard-deleted in one day), the warehouse still reports
**999 active** while the source has **969**, and re-running the job changes
nothing.

## Root cause
An incremental extract can only see rows that exist. `updated_at > :watermark`
finds inserts and updates, but a deleted row has no `updated_at` any more: it
is simply absent from the result, and absence looks exactly like "unchanged".
Upserts never remove anything, so every hard delete since the first load is
still in the warehouse. A full reload would fix it, which is why the problem
seemed to "come back" between occasional full refreshes.

## Solution
- After the usual upsert, `incremental_sync` runs `reconcile_deletes`: extract
  only the source's primary keys (`SELECT customer_id FROM customers`, an
  index-only scan, far cheaper than re-copying rows) and delete every warehouse
  key that is not in that set (an anti-join).
- `check_delete_is_safe` refuses to delete more than 5% of the table in one run
  (`MAX_DELETE_FRACTION`). An extract that failed and returned zero keys would
  otherwise look like "everyone was deleted" and empty the table; the test
  `test_empty_key_extract_does_not_wipe_the_table` covers that.
- The tests show the naive job leaving 30 extra rows and the fixed job making
  the warehouse equal to the source, with the update to customer 500 still
  applied.

## Trade-offs
- The key extract is O(table size) per run. For very large tables, compare
  per-range hashes or counts first and only list keys in ranges that differ.
- The safety limit can block a real mass deletion (a big cleanup); that needs
  an operator override, which is the point.
- Better long-term options are soft deletes (`deleted_at` set and synced like
  any update) or log-based CDC, which emits delete events directly. Both need
  changes in the source system; reconciliation works with what exists today.
- Rows deleted and re-inserted with the same key between runs look like an
  update, which is the correct result for the warehouse.

## Run
```bash
python -m unittest -v test_solution.py
```
