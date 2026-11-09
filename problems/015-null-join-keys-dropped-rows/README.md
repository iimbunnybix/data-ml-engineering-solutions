# 015 · The daily revenue report showed 42% of the day's sales, and keyless orders were booked to "Test accounts"

## Problem
A nightly job builds revenue by region by joining the `orders` fact table
(about 2 million rows a day) to the `customers` dimension on `customer_id`.
After the checkout service started sending guest checkouts with an empty
`customer_id`, and a partner integration started sending IDs with trailing
spaces and lowercase letters, the regional dashboard dropped sharply while
payment-provider totals stayed flat. The job never failed. Finance spent two
days chasing a "sales drop" that was never real, and a placeholder dimension
row with a NULL key (used for internal test accounts) was credited with every
order that had no customer at all.

The test fixture reproduces it: of 31,000 cents of orders, the original join
reports 13,000 (42%), and 3,000 cents of keyless orders land in "Test accounts".

## Root cause
- An **inner join** on the raw key silently discards every fact row whose key
  does not match exactly, so `"C1 "` and `"c2"` vanish.
- `""`, `"   "`, NULL and NaN are different values to the join, but all mean
  "no customer", and none of them is handled on purpose.
- Some join engines (pandas `merge`, Python dict lookups) treat NULL as equal
  to NULL, so a NULL-key dimension row matches every NULL-key fact row.
- Nothing compared the report's row count and money total with the input.

## Solution
- `normalize_key` canonicalises keys (strip, upper-case) and maps empty or
  blank strings and float `NaN` (how pandas writes a missing value) to
  `None`, so there is exactly one "missing" value. Without the NaN check,
  `str(nan)` becomes the key `"NAN"` and a NaN-keyed placeholder row would
  again absorb every NaN-keyed order.
- Dimension rows without a key are excluded from the lookup, so a missing
  key can never match anything. Conflicting duplicate keys raise.
- The join is a **left join with explicit buckets**: `MISSING_KEY` for orders
  without a key and `UNMATCHED_KEY` for keys not in the dimension. These show
  up in the report instead of disappearing.
- `reconcile` checks that rows and cents in equal rows and cents out, and the
  job raises `ReconciliationError` before publishing if they differ.

On the fixture the fixed report is North 15,000, South 7,000, MISSING_KEY
5,000 (2 rows), UNMATCHED_KEY 4,000, total 31,000.

## Trade-offs
- Case-folding is only correct if the source system treats IDs as
  case-insensitive; confirm that with the owning team before normalising.
- Buckets keep totals right but do not fix attribution; a growing
  `UNMATCHED_KEY` bucket should alert so late-arriving dimension rows or
  upstream bugs get fixed.
- Reconciling totals catches dropped rows but not rows moved to the wrong
  region; pair it with a row-count check per key for high-value reports.
- In SQL the same fix is `LEFT JOIN ... ON TRIM(UPPER(o.id)) = TRIM(UPPER(c.id))`
  with `COALESCE(region, 'UNMATCHED_KEY')`; normalising once at ingestion is
  cheaper than in every join.

## Run
```bash
python -m unittest -v test_solution.py
```
