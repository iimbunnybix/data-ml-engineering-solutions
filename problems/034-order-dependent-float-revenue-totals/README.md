# 034 · The daily wire-total reconciliation alerted every morning over two cents, and each rerun of the job gave a slightly different total

## Problem
A bank's wholesale payments team loads about 400,000 wires a day (from $10 to
about $50M each, roughly $1.3 trillion in total) into the warehouse. A Spark
job computes the daily total over 64 partitions. A control check compares it
with the total on the treasury dashboard, which sums the same rows in event
order, and with the previous run of the job. The check failed almost every
morning. On the replayed day in `test_solution.py`:

- the exact total is **$1,301,782,510,419.50**;
- the dashboard (one pass in event order) shows **…419.52**, 2.3 cents off;
- five reruns of the batch job on identical input give **three different
  float totals** (…419.5005, …419.5007, …419.5010);
- the "equal to the cent" reconciliation fails on **all 5 reruns**, although
  no wire is missing.

On-call engineers and the finance-controls team spent hours a week proving
that nothing was lost. Then someone proposed widening the tolerance to $1. That
would also have hidden a genuinely missing small wire.

This is not about rounding one invoice wrong (see backend-api-solutions 012).
Here every single amount is right. Only the **order of the additions** changes
between runs and between tools.

## Root cause
**The amounts were summed as binary floats, and float addition is not
associative.** Each `+` rounds the result to the nearest representable
double. Near $1.3 trillion, adjacent doubles are 2⁻¹² ≈ $0.000244 apart, so
every addition of a small wire to the running total is rounded on that grid,
and the rounding errors depend on what the running total was at each step.
`(0.1 + 0.2) + 0.3` is `0.6000000000000001`, while `(0.3 + 0.2) + 0.1` is `0.6`.
At scale:

- **Reruns** differ because the shuffle delivers rows to each task in a
  different order and the tasks finish, and are merged, in a different order.
  Partition sums are smaller (about 1/64 of the total), so each step loses
  less, and the reruns stay within 0.1 cent of the exact total. But they are still
  not equal.
- **The dashboard** adds all 400,000 rows into one running total near a
  trillion dollars, so it loses more at each step: 2.3 cents here.
- The amounts are not exact to begin with: `float("0.10")` is not 0.10.

## Solution
**Parse each amount once into integer minor units (cents) and sum integers**
(`solution.py`):

1. `to_minor_units` reads the decimal string with `Decimal` (exact) and
   converts it to an `int` number of cents. Amounts with sub-cent digits,
   `nan`, `inf` or non-numbers are rejected instead of silently rounded.
2. `exact_partition_total` and `exact_partitioned_total` add integers.
   Integer addition is exact and associative, so every partition layout,
   row order and merge order gives the same number: all five reruns and the
   dashboard return exactly 130,178,251,041,950 cents.
3. `reconciles` can now demand **exact equality**. A difference of even one
   cent is a real gap (tested), so the check stops flapping and becomes
   stricter at the same time.
4. `format_minor_units` turns cents back into a dollar string without going
   through a float.

For data that only exists as floats, `math.fsum` (`fsum_total`) returns the
correctly rounded sum of its inputs, so any order of the same values gives
the same result. Partition partials must then be merged with another `fsum`,
not `+`. On this day the fsum total is the same for all reruns, and equal to the
exact $1,301,782,510,419.50.

## Trade-offs
- **Minor units depend on the currency**: 2 decimals for USD, 0 for JPY, 3
  for BHD. Store the exponent per currency, and never add amounts of
  different currencies (see problem 027).
- In a warehouse, the equivalent is a `DECIMAL(p, 2)` column or `BIGINT`
  cents instead of `DOUBLE`/`FLOAT`. A `BIGINT` holds about 9.2 × 10¹⁸
  cents, so overflow is not a concern for one day. Python's `int` never
  overflows.
- `fsum` is order-independent, not exact: it sums the binary approximations
  of the amounts and rounds once. It cannot be computed incrementally across
  machines with plain `+`, and two tools that both use doubles but a
  different summation method will still disagree. Integer cents avoid all of
  this.
- Compensated summation (Kahan/Neumaier, which Python's built-in `sum()` uses
  for floats since 3.12) makes the error much smaller but does not guarantee
  the same result in every order. That is also why `naive_float_total` uses
  an explicit loop, to behave like a plain SQL/Spark `SUM` over doubles.
- A tolerance on the reconciliation (`abs(a - b) < x`) is the common
  workaround. Any `x` large enough to stop the flapping on big days also
  hides real small gaps, and it has to grow with volume.
- Converting at ingest costs a parse per row and a schema change for
  existing `DOUBLE` columns. Historical totals computed in float will differ
  by fractions of a cent from the recomputed exact ones, which needs a
  one-time sign-off.

## Run
```bash
python -m unittest -v test_solution.py
```
