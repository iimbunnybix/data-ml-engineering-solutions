# 054 · Cancelled subscriptions showed as active in the warehouse, and re-running the load "fixed" different customers each time

## Problem
A SaaS company loads a daily export from its billing partner into the
warehouse table `subscription_status` with an upsert (MERGE on
`subscription_id`). Finance uses the table for revenue and churn, and
customer success uses it to decide who to call.

Customers who had cancelled were still listed as active, and some who had
upgraded showed as `trial`. When an engineer re-ran the load to "refresh"
it, a different set of customers came out wrong. In the seeded test day
(2,000 subscriptions, 5,006 rows, 1,496 subscriptions appearing more than
once) the naive load leaves **959 keys holding an older row**, and **704**
of them show the wrong status. Re-running the same rows in another order
leaves **905 keys** on a different row than the first run.

## Root cause
The export is a change log, not a snapshot: a subscription that went
`trial -> active -> cancelled` during the day appears three times. The
loader assumed one row per key and upserted rows in arrival order, so the
**last row processed won**. The export arrives as several file parts that
are loaded in parallel, so arrival order has nothing to do with the order
the changes happened, and the final value is effectively random.

SQL engines differ here: PostgreSQL and SQL Server reject a MERGE that
would update the same target row twice, other warehouses make the check
configurable, and hand-written upsert loops like this one never complain. Either way, the
source must have one row per key before it reaches the MERGE.

## Solution
- `dedupe_batch` keeps one row per key: the one with the latest
  `updated_at` (the SQL version is `ROW_NUMBER() OVER (PARTITION BY
  subscription_id ORDER BY updated_at DESC) = 1`). The result does not
  depend on row order, which the tests prove by loading the same rows in
  two orders.
- Exact duplicate rows collapse to one. Two **different** rows with the
  same latest timestamp cannot be ordered from the data, so the key goes to
  `conflicts` and its target row is left unchanged, instead of guessing.
- `fixed_merge` only overwrites a target row with a source row that is at
  least as new, so reloading an old file (a retry or backfill) cannot roll
  a status back.
- With the fix, all 2,000 keys hold their true latest row.

## Trade-offs
- `updated_at` in the export has minute resolution, so real ties happen.
  Quarantining them is safe but needs a human or a second ordering column.
  Ask the producer for a strictly increasing change sequence number; with
  one, ties disappear (see problem 008 for sequence-guarded CDC applies).
- A tie with the target row (`updated_at` equal) lets the batch win, so a
  corrected row re-sent with the same timestamp is applied. A source that
  re-sends stale rows with fresh timestamps cannot be caught this way.
- Dedupe needs the whole batch (or one partition of keys) in memory or a
  shuffle by key; for very large batches do it in the warehouse with the
  window function above.
- If intermediate changes matter (for example, time spent `past_due`), the
  load should keep the full history (an SCD2 or event table) rather than
  only the latest row.

## Run
```bash
python -m unittest -v test_solution.py
```
