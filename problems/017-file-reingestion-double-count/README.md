# 017 · Sales table showed a day's revenue twice after the nightly load job was restarted

## Problem
A nightly job picks up the CSV files a payment partner drops into a landing
bucket (one per day, tens of thousands of rows each), appends them to the
`sales` table and then moves each file to an `archive/` prefix. One night the
job's pod was killed during the move step. The scheduler retried the job, the
file was still in the landing folder, and its rows were appended again.
Finance's daily revenue report for that date came out double, and the same
thing happened whenever an engineer "re-dropped" a file to re-trigger a load.

## Root cause
The job used **"the file is still in the landing folder" as its record of
"not loaded yet"**. Loading rows and archiving the file are two separate
steps with no transaction across them, so a crash between them leaves the
rows in the table and the file in the folder. Nothing in the warehouse said
which files had already been loaded, so a retry could not tell.

## Solution
`exactly_once_ingest` keeps a **manifest table** keyed by
`(path, SHA-256 checksum)` with the number of rows loaded:

- For each pending file, the checksum is computed and looked up in the manifest.
- New files are written with `commit_file`, which inserts the rows **and** the
  manifest entry in one transaction — both are visible, or neither is.
- Archiving becomes clean-up only. If the job crashes before or during it,
  the retry finds the file in the manifest (`already_loaded`) and just archives it.
- The same bytes under a new name (a partner re-send) are skipped as
  `duplicate_content`.
- The same name with different bytes (a corrected file) is **not** appended;
  it is left in the landing folder as `changed_after_load` for a person to
  decide whether to replace that day's rows.

In the tests, a crash while archiving the 2026-11-02 file makes the naive job
load 800 instead of 600; the fixed job loads 600 after the retry.

## Trade-offs
- The guarantee depends on the rows and the manifest living in the same
  transactional store. If they are in different systems you need an
  idempotent sink instead (e.g. `MERGE` on `order_id`, or overwrite by
  partition as in problem 005).
- Checksumming reads every file once more; for very large files, object-store
  ETags or a hash recorded at upload time are cheaper.
- Corrected files need a human or a replace-partition flow; this code
  deliberately refuses to guess.
- Row-level dedup on `order_id` would also catch duplicates *inside* a file,
  which a file manifest cannot.

## Run
```bash
python -m unittest -v test_solution.py
```
