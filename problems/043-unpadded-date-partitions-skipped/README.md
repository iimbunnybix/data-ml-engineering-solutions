# 043 · The orders warehouse had no data for October 10-31, and the loader reported "nothing new" every night

## Problem
A nightly loader copies daily `orders` partitions from a data-lake folder
(`orders/dt=YYYY-MM-DD/`) into the warehouse. It remembers the name of the
last partition it loaded and, on each run, loads every partition whose name
sorts after it. On October 1 the upstream team replaced its exporter, and the
new one wrote dates without zero padding: `dt=2026-10-1`, `dt=2026-10-9`,
`dt=2026-10-10`. The first nine days of October loaded normally. From
October 10 the loader logged "0 new partitions" every night, all its checks
stayed green, and the finance team's month-end close found 22 days of orders
missing. November 1-9 then loaded again, which made the gap look like an
upstream outage rather than a loader bug.

In the tests, running the original logic once per day over October loads
9 of 31 partitions and stops at `dt=2026-10-9`. Extended to November 10, it
loads 18 partitions: October 1-9 and November 1-9.

## Root cause
The loader compared **partition names as text**. Text comparison goes
character by character, so `"dt=2026-10-10" < "dt=2026-10-9"` because `'1'`
comes before `'9'`. With zero padding, text order and date order happen to be
the same, which is why the bug stayed hidden until the naming changed. Once
the watermark was `dt=2026-10-9`, every later day of October (and November 10
onward) sorted "before" it and looked already loaded. Nothing checked that the
set of days loaded had no holes.

## Solution
`plan_incremental_load` works with dates, not names:

1. `parse_partition_date` turns each folder name into a `date` with a strict
   pattern, accepting padded and unpadded spellings. Names that do not parse
   (`_temporary`, `dt=20261002`, `dt=2026-02-30`) are reported in
   `unparseable` instead of being silently compared.
2. Two folders for the same date (`dt=2026-10-09` and `dt=2026-10-9`) raise
   `PartitionConflict`, because loading both would double count the day.
3. The watermark is a `date`. New partitions are those after it, loaded in
   date order.
4. Days between the watermark and the newest partition that have no folder
   are reported in `missing_days`, and the watermark stops just before the
   first gap, so a day that lands late is still loaded on the next run.

With daily runs over October, the fixed loader loads all 31 partitions and
ends with the watermark at 2026-10-31.

## Trade-offs
- Stopping at a gap holds back later days until the missing one arrives (or
  someone marks it as intentionally empty). That delays data, but it never
  skips any. A loader that overwrites whole partitions (problem 005) could
  instead load later days and keep re-checking the gap.
- The pattern only accepts `dt=Y-M-D`. Hourly or differently named layouts need
  their own parser; guessing formats is what caused this incident.
- Listing the folders and comparing them with the set of loaded days is
  cheap for daily partitions; for millions of small partitions, keep a
  manifest of loaded partitions instead of a single watermark.
- Asking the producer to keep zero-padded ISO dates is still worthwhile, but
  the loader should not depend on it.

## Run
```bash
python -m unittest -v test_solution.py
```
