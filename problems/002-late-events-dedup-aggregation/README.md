# 002 · Daily revenue dashboard over-reports by 3% and changes after it is published

## Problem
A streaming job sums order events into daily revenue. Finance noticed two
issues:

1. Revenue is ~3% too high — the mobile SDK retries on network errors, so
   the same `order_id` event arrives two or three times.
2. Yesterday's number keeps changing — events from phones that were offline
   arrive hours late, and some arrive days late after the report was sent.

## Root cause
- The job assumed **exactly-once delivery**; real pipelines are
  *at-least-once*, so duplicates are normal.
- It aggregated by **arrival time** in some places and **event time** in
  others, and had no rule for when a day is "closed".

## Solution
An event-time aggregator with:

- **Deduplication** by `event_id`, remembered for a retention period
  (default 7 days) that covers how late a retry can arrive, then forgotten so
  memory stays bounded. Late events are de-duplicated too.
- **Watermark** = max event time seen − allowed lateness (e.g. 6 hours).
  A day's window is **finalised** once the watermark passes its end.
- Events for an already finalised day are not silently dropped or added:
  they go to a **late-events ledger** that finance can reconcile as an
  adjustment the next day.
- **Future-time guard**: with a processing-time `clock`, events stamped more
  than 5 minutes ahead of it (a phone with a wrong clock) go to a
  `future_quarantine` list instead of advancing the watermark. Otherwise one
  such event would finalise today's window early and evict every remembered
  event id.

This mirrors how Flink / Spark Structured Streaming / Beam handle it
(`withWatermark` + `dropDuplicates`), in ~60 lines of plain Python.

## Trade-offs
- Larger allowed lateness = more accurate first numbers but later publishing.
- Dedup state must be bounded; a duplicate arriving after the retention
  period would be double counted, so retention is set from observed retry delays.
- Exact numbers for finance are still reconciled from the source-of-truth DB.

## Run
```bash
python -m unittest -v test_solution.py
```
