# 005 · Re-running a failed backfill inflated September revenue by 60% on the finance dashboard

## Problem
A bug fix to the merchant revenue logic required a 30-day backfill of the
`daily_revenue` table that feeds finance dashboards and merchant payouts. The
job died on day 18 when an executor was lost. The on-call engineer re-ran the
same command for the full month. The next morning, September revenue for days
1-17 was exactly double. In the reproduction test (seeded data, ~30 payments a
day across 5 merchants) the month total ends up **1.60×** the true value.
Payout reconciliation was frozen for two days while the table was rebuilt by hand.

## Root cause
The job wrote with `INSERT` (append) and kept no record of which partitions
were finished. That means:

- any re-run of a day adds a second copy of its rows;
- a crash halfway through a day leaves a partial partition;
- there is no safe way to "just re-run" anything, which is the first thing
  anyone does during an incident.

## Solution
Make each partition write **idempotent and atomic**, and make the job
**restartable**:

- **Partition overwrite.** `write_partition` runs `DELETE WHERE day = ?` and
  `INSERT` in one transaction. Writing a day once or ten times gives the same
  result, and a crash mid-write rolls back to the previous state (tested).
- **Deterministic output.** Each partition is a pure function of that day's
  source rows, so overlapping or repeated ranges (Sep 1-20, then Sep 10-30
  with `force`) still sum to the exact source total.
- **Run manifest with input fingerprints.** After each partition commits, the
  job stores a SHA-256 of its source rows in `backfill_manifest` (in the same
  transaction). A re-run skips days with an unchanged fingerprint: after the
  day-18 crash, the re-run skips 17 days, writes 13, and the total is exact.
- **Changed inputs are picked up.** A late payment for Sep 5 changes that
  day's fingerprint, so only Sep 5 is recomputed. Deleted source rows disappear
  from the output too, which append-only jobs can never do.

## Trade-offs
- Fingerprinting reads every source row again. On large tables, use cheaper
  change signals (partition row count + max `updated_at`, file listing etags,
  or the table format's snapshot id in Iceberg/Delta).
- `DELETE + INSERT` in one transaction is easy in a database. In a data lake,
  use the engine's equivalent: Spark `insertInto` with
  `partitionOverwriteMode=dynamic`, Iceberg/Delta `REPLACE WHERE`, or
  write-to-staging-then-swap the partition path.
- Readers can see a mix of old and new partitions *during* a backfill.
  If that matters (e.g. month-end close), write to a shadow table and swap it
  in once all partitions are done.
- Idempotence relies on the partition key being the unit of work. Jobs that
  update rows across many partitions need `MERGE` on a natural key instead.

## Run
```bash
python -m unittest -v test_solution.py
```
