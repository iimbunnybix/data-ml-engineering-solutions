# 041 · After a 30-day backfill, the "weekly active users" chart was a flat line at 2,346 for all of October

## Problem
A product-analytics pipeline writes one partition per day to a
`wau_daily` table: the number of distinct users active on that date or the
six days before it. The metric was added in November, so an engineer
backfilled October by triggering the job for each date from 2026-10-01 to
2026-10-30 on the morning of 2026-11-09. Every one of the 30 partitions came
out with the same value. October was a month of steady growth from a
referral campaign, and the growth team spent a day trying to explain why the
campaign "had no effect" before someone noticed the line was perfectly flat.

The daily scheduled runs had looked correct for weeks, and a retried run
was also wrong in a quieter way: re-running the 2026-11-05 partition on
2026-11-08 changed its value from 2,252 to 2,321.

On the test fixture (37,260 seeded events, September to November), the
original job writes 2,346 to every October partition. The correct values
rise from 1,299 on 2026-10-01 to 2,077 on 2026-10-30.

## Root cause
The job was asked to build a partition for a **logical date**, but computed
its 7-day window from the **wall clock**: `today() - 1 day`. It then labelled
the result with the logical date. On a scheduled run the two agree (the job
runs the morning after the day it builds), which hid the bug. A backfill or a
late retry runs on a different day, so every partition got the window ending
"yesterday" relative to when the job ran, not the day it was building.

The output was therefore a function of *when the job ran*, not just of its
inputs, so runs were not reproducible and backfills were silently wrong.

## Solution
- `build_partition` takes the partition date as its only notion of time and
  derives the window from it (`window_for(partition_date)` is the date and the
  six days before). Its result depends only on the input events and the
  partition date, so a backfill, a retry next week and the original
  scheduled run all produce identical rows. The test checks every backfilled
  October row against what the on-schedule run produced.
- The clock is kept for one job only: refusing to build a partition whose
  day has not ended (`partition_date >= today` raises
  `IncompleteDateError`), because publishing a partial day under-counts.
- `naive_build_partition` keeps the original behaviour for comparison.

Orchestrators pass this logical date explicitly (Airflow's `ds` /
`data_interval_end`, Dagster partition keys); the fix is to use it everywhere
instead of `now()` or `CURRENT_DATE` inside the query.

## Trade-offs
- The "day is complete" guard uses the calendar only. Late-arriving events
  can still land after a partition is built; pair this with a lateness
  allowance or a scheduled re-run of the last few partitions.
- Every hidden "now" must be found: SQL `CURRENT_DATE`, default arguments,
  `datetime.now()` in helper libraries, and TTL filters. A lint rule or test
  that runs the job under a fake clock set to a different day finds them.
- Computing each partition from scratch over a 7-day window re-reads data a
  backfill already read; incremental approaches are faster but harder to make
  reproducible.
- Data that is mutable "as of now" (for example a dimension table without
  history) still makes old partitions differ from what was originally
  published; that needs snapshots or SCD2 tables, not just a logical date.

## Run
```bash
python -m unittest -v test_solution.py
```
