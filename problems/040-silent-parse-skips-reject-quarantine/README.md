# 040 · Yesterday's payments table was missing almost a third of transactions, and the ingestion job said "success"

## Problem
A payments company loads app payment events (JSON lines, about 10,000 per
hourly batch in the test, millions a day in production) into the warehouse
that feeds settlement reports and the fraud model. A mobile app release
changed the `amount` field from a JSON number (`1299.0`) to a formatted string
(`"1,299.00"`). About 30% of users updated on the first day.

The ingestion job kept finishing green. The next morning, finance found that
settlement totals were about 30% below what the payment gateway reported, and
the fraud model had been scoring a population with no users of the new app
version. Nobody was paged; the gap was found by a person comparing two
numbers.

In the simulation (10,000 events, 30% from the new app version, 0.2% of lines
truncated as background noise, seed 11), the original job loaded **6,898**
rows and quietly dropped **3,102** (31.0%).

## Root cause
The parsing loop was:

```python
for line in lines:
    try:
        events.append(parse_event(line))
    except Exception:
        continue
```

The `try/except continue` was added years earlier for the occasional
truncated line from the message queue, which is normal and harmless at that
rate. But it treated "1 broken line" and "a third of the batch is unreadable"
the same way: skipped, not counted, not kept. The job measured success as "the
process did not crash", not "the rows that came in are the rows that went out".

## Solution
`solution.py`:

- **Strict per-row parsing with reason codes.** `parse_event` raises
  `RowRejected` with a short, countable reason: `invalid_json`,
  `missing_<field>`, `amount_not_a_number`, `amount_not_finite` (JSON `NaN`),
  `negative_amount`, `bad_created_at`, `not_an_object`.
- **Quarantine, not skip.** `parse_batch` puts every row in exactly one
  place: `accepted`, or `rejected` as a `RejectedRow` with line number, raw
  text and reason. Rows in = accepted + rejected, always. On the incident
  batch the quarantine says `amount_not_a_number: 3,087, invalid_json: 15`,
  which points straight at the format change.
- **Reject-rate gate.** `ingest_batch` raises `BatchRejectedError` (with the
  quarantine attached) when more than 1% of rows are rejected:
  `reject rate 31.0% exceeds 1.0%`. The normal background noise (15 rows,
  0.15%) passes, and those rows stay in quarantine for replay.
- **Minimum-row guard.** Below 200 rows a rate is noise (1 bad row in 10 is
  10%), so small batches fail only above 5 rejects. A batch where every row
  is rejected always fails, whatever its size.

## Trade-offs
- Failing the batch delays good rows too. That is the point: publishing 69%
  of the data as if it were 100% is worse than publishing late. If partial
  data must flow, publish it marked as partial, never silently.
- The 1% threshold must be set from the observed baseline (here 0.15%). Too
  tight pages on normal noise; too loose hides a format change that hits a
  small share of users.
- The quarantine holds raw payloads, which may contain personal data. Give it
  the same access controls and retention as the main table.
- Quarantined rows need a replay path: after the parser is changed on purpose
  to accept the new format, re-run them so no transaction is lost.
- This complements, not replaces, a schema contract (001), a manifest row
  count (025) and write-audit-publish (009): those catch whole-file problems;
  this catches per-row losses inside a file that looks fine.

## Run
```bash
python -m unittest -v test_solution.py
```
