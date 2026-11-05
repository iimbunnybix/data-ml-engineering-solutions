# 013 · The warehouse quietly disagreed with production: a few orders per day were missing or stuck at an old status, forever

## Problem
An incremental ELT job copies the `orders` table from PostgreSQL into the
warehouse every 5 minutes: `SELECT * FROM orders WHERE updated_at >
:watermark ORDER BY updated_at LIMIT 500`, upsert the rows, then save the
largest `updated_at` it saw as the new watermark. It was fast and cheap and
passed every test.

Months later, finance found orders in the payment processor's report that did
not exist in the warehouse, and customer support found orders shown as
`pending` in dashboards that had shipped weeks earlier. Re-running the job did
nothing. Each day only a handful of rows were wrong, which is why no row-count
alert noticed, but they accumulated and nobody trusted the order tables.

## Root cause
1. **Timestamps are assigned before rows become visible.** `updated_at`
   defaults to `now()`, which in PostgreSQL is the start time of the
   transaction. A transaction that starts at 10:00:00 and commits at 10:01:30
   (a bulk status update, a slow checkout, a lock wait) writes rows stamped
   10:00:00 that no reader can see until 10:01:30. If the extract runs at
   10:01:00, it moves the watermark past 10:00:00 using other rows, and the
   late rows are never selected again.
2. **Ties at page boundaries.** With second-precision timestamps, many rows
   share the same `updated_at`. `updated_at > :ts LIMIT 500` drops any rows
   with the page's last timestamp that did not fit in the page.

In the simulation (one day of 20,000 writes over 5,000 orders, 2% of
transactions taking 10–90 s, extract every 5 minutes), the naive job left
**12 of 4,907 rows wrong** (2 missing, 10 stuck at an older version). Over 5
seeded days: 73 of 24,513 rows (0.30%).

## Solution
`solution.py`:

- **Settled upper bound.** `SafeExtractor` only reads rows with
  `updated_at <= now - settle_seconds`. If no transaction lasts longer than
  `settle_seconds`, every row with such a timestamp has already committed,
  so the cursor never passes a row that is still invisible. Enforce the bound
  in the database with `statement_timeout` / `idle_in_transaction_session_timeout`
  below the settle window, so the assumption is guaranteed.
- **Keyset cursor on `(updated_at, id)`.** Pages are read with
  `WHERE (updated_at, id) > (:ts, :id) ORDER BY updated_at, id`, so ties are
  never skipped regardless of page size.
- **Idempotent upsert by version.** The target keeps the highest version, so
  re-running or overlapping extracts never corrupts data.
- **Reconciliation.** `reconcile` compares source and target for the settled
  range and reports missing and stale ids grouped by hour. In production,
  compare per-hour counts and checksums and drill down only where they differ.
  It is the safety net for any transaction that broke the settle assumption.

Tests: with `settle_seconds=120`, 0 wrong rows across 5 seeded days; page-
boundary ties are kept; a full re-extract is a no-op; and when transactions
run up to 15 minutes (longer than the settle window), reconciliation reports
the affected rows by hour.

## Trade-offs
- The settle window adds latency: data in the warehouse is at least
  `settle_seconds` old. Pick it from the real timeout you enforce, not from
  typical transaction duration.
- Reading the source repeatedly costs OLTP load. Index `(updated_at, id)` and
  run against a replica (then add the replica lag to the settle window).
- Deletes are invisible to timestamp-based extraction. Use soft deletes or
  log-based change data capture (Debezium, logical replication), which reads
  changes in commit order and avoids this whole class of bug (see 008 for
  ordering CDC events).
- A transaction-commit-time column (e.g. via `pg_xact_commit_timestamp` with
  `track_commit_timestamp = on`) is another option, with its own overhead.
- An overlap window (`updated_at > watermark - 10 min`) is a common quick fix.
  It works the same way as the settle bound when combined with idempotent
  upserts, but re-reads overlapping rows on every run.

## Run
```bash
python -m unittest -v test_solution.py
```
