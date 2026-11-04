# 012 · Revenue by region jumped 15% overnight with no change in orders, and the sales team started celebrating

## Problem
A nightly job builds `fct_orders_enriched` by joining the orders table to
`dim_customers` on `customer_id` to attach region, segment and account
manager. Every revenue dashboard and the commission report read from it. One
morning, revenue for the previous month was about 15% higher than the day
before, even though the orders table itself had not changed and the payment
processor's totals matched the old number. Regional managers had already sent
congratulation emails, and commission statements generated that morning
overpaid several reps before finance caught it.

## Root cause
**The dimension table stopped being unique on its join key, and the join
silently multiplied rows.** A CRM re-sync was changed from upsert to append
and wrote a second row for every customer it touched. A left join returns one
output row per *matching pair*, so each order of a duplicated customer
appeared twice and every `SUM(amount)` downstream counted it twice. No step
failed: the join is valid SQL and the output schema was unchanged.

The reproduction (1,000 customers, 20,000 orders, 150 customers duplicated by
the re-sync) turns 20,000 orders into **22,990 joined rows** and inflates
total revenue by **15.3%**.

## Solution
`solution.py`:

1. **Declare join cardinality and enforce it.** `validated_join(...,
   cardinality="many_to_one")` checks that the right-hand keys are unique
   before joining and raises `CardinalityError` naming how many keys are
   duplicated and showing samples (150 in the reproduction). This is the same
   contract as pandas' `merge(validate="m:1")` or a dbt `unique` test on the
   dimension's key, run *before* the join, so bad data stops the job instead of
   reaching dashboards.
2. **Resolve duplicates explicitly when they are expected.**
   `dedupe_latest` keeps the most recently updated row per key, and refuses to
   guess when two different rows tie on the timestamp.
3. **Reconcile a measure across the join.** `assert_measure_preserved`
   checks that row count and `SUM(amount)` are the same before and after an
   enriching left join. It catches fan-out whatever the cause, including a join
   someone declared `many_to_many` by mistake.

NULL keys follow SQL semantics (they never match), so they are neither
treated as duplicates nor multiplied; a test caught this in the first version
of the join.

## Trade-offs
- Uniqueness checks cost a `GROUP BY key HAVING COUNT(*) > 1` over the
  dimension. That is cheap next to the fact table, but on huge dimensions run
  it once when the dimension is built, not in every consumer.
- `dedupe_latest` is a business choice: for slowly changing dimensions the
  right answer may be the row valid *at order time* (SCD type 2 with
  `valid_from`/`valid_to`), not the latest.
- Failing the job delays dashboards. That is usually better than publishing
  wrong revenue, but pair it with an alert and an owner for the upstream table.
- Fix the source too: a primary key or unique constraint on the dimension
  (or a `MERGE` instead of append in the sync) prevents the duplicates.

## Run
```bash
python -m unittest -v test_solution.py
```
