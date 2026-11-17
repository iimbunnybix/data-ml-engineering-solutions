# 057 · Daily payments revenue was overstated after a producer retry storm, although the dedup step reported "0 duplicates"

## Problem
A payments data pipeline ingests payment events from several producer
services into a raw table, deduplicates them, and feeds the finance revenue
dashboard. Deduplication hashed each event's raw JSON text with SHA-256 and
dropped any event whose hash had been seen before.

During a broker incident, producers retried thousands of payments. Some
retries came from an older SDK (different key order and spacing), some were
replayed through a gateway that re-serialised amounts as `250.0`, and every
redelivery got a fresh `ingested_at` timestamp. The dedup job reported
almost no duplicates, and that day's revenue on the dashboard was inflated
until an analyst noticed the totals disagreed with the payment provider's
settlement report. In the tests, one payment of ₹250 plus four retried
copies produces five distinct raw hashes and ₹1,250 of revenue.

## Root cause
Hashing raw text treats *serialisation* differences as *content*
differences. JSON does not define key order, whitespace, or whether `250`
and `250.0` are distinct, and transport metadata such as `ingested_at`
changes on every delivery. So the same logical event can have many byte
representations, and a byte-level hash sees each one as new.

## Solution
`fingerprint` hashes a **canonical form** of the event instead of its bytes:

- Remove **volatile transport fields** (`ingested_at`, `delivery_attempt`,
  `producer_host`); they describe the delivery, not the event.
- **Sort keys** at every level and use **compact separators**, so key order
  and whitespace no longer matter.
- Write **whole-number floats as integers** (`250.0` → `250`), keep
  `true`/`false` distinct from `1`/`0`, and reject `NaN`/`Infinity`.
- Use `ensure_ascii=False` so `"José"` and `"José"` give the same text.
- **List order is kept**, because the order of line items is real content.

With this, all five copies collapse to one event and revenue is ₹250.

## Trade-offs
- The best fix is upstream: producers send a stable `event_id` (an
  idempotency key) and dedup uses only that. Canonical hashing is the fallback
  when you cannot trust or do not have one.
- The volatile-field list has to be maintained. A field that should have been
  excluded makes duplicates look distinct; one that should not have been
  excluded makes distinct events look equal.
- Treating `250` and `250.0` as equal is right for amounts but not for every
  field; floats such as `0.1` are kept as Python prints them, which is stable
  across Python versions but may differ from another language's output. For
  cross-language hashing use a standard like RFC 8785 (JSON Canonicalization
  Scheme).
- The `seen` set grows with the data. In production it is bounded by a time
  window (dedup within N days) and kept in a keyed store.

## Run
```bash
python -m unittest -v test_solution.py
```
