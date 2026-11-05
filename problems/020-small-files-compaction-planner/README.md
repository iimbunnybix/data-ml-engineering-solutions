# 020 · Dashboard queries on the clickstream table got slower every day although data volume was flat

## Problem
A streaming job writes clickstream events to a Parquet table on object
storage, partitioned by day. It flushes every minute, and each flush writes
one 1-3 MB file per partition: **1,440 files per day partition** (~2.8 GB in
the test simulation). Dashboard queries over the last week spent most of
their time listing and opening files rather than reading data, and the query
engine's planning time grew with the file count.

The first fix was a nightly job that rewrote each partition into **one file**.
That made things worse in a different way: the day partition became a single
~2.8 GB file (over 20× the 128 MB target, so it could not be read in parallel),
and every run rewrote the already-compacted data again. For a partition with
ten healthy 128 MB files plus one new hour of small files, it rewrote
**~1,403 MB to fix ~123 MB** of small files.

## Root cause
Query engines pay a fixed cost per file (object-store list and GET calls,
reading Parquet footers, scheduling a task). Many tiny files multiply that
cost without adding data. The naive compaction ignored two things: files
that are already big enough should not be rewritten, and outputs should be
close to a target size, not as large as possible.

## Solution
`plan_compaction` builds a rewrite plan per partition:

1. Only files smaller than **75% of the 128 MB target** are candidates, so
   compacted output is never rewritten again (no write amplification).
2. Partitions with fewer than 5 small files are skipped: the rewrite would
   cost more than it saves.
3. Candidates are **bin-packed first-fit decreasing** into groups that do not
   exceed the target. Largest-first placement leaves the smallest files to fill gaps.
4. Groups of one file are dropped, since copying a file alone changes nothing.
5. Files from different partitions are never mixed in one output.

In the simulation, one day partition goes from **1,440 files to 23**, with
no output above 128 MB. The partition with ten healthy files and an hour of
new small files goes from 70 files to 11, rewriting only the ~123 MB of new data.

## Trade-offs
- Compaction is extra I/O and compute. Running it often keeps file counts low
  but costs more; running it once a day leaves today's partition slow.
- Rewriting files must be atomic for readers (a table format like Delta Lake,
  Iceberg or Hudi commits the swap); on a plain directory of files, readers
  may see duplicates or missing data mid-rewrite.
- A better root fix is fewer, larger writes upstream (longer trigger interval,
  or repartitioning before the sink), traded against data freshness.
- First-fit decreasing is not optimal bin packing, but it is close in practice
  and fast. Sorting rows by a common filter column while rewriting would also
  improve data skipping; this planner only decides which files to merge.

## Run
```bash
python -m unittest -v test_solution.py
```
