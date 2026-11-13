# 019 · Revenue by customer tier was higher than total revenue after a CRM backfill

## Problem
A subscription business keeps a type-2 slowly changing dimension
`dim_customer` in the warehouse: one row per version of a customer, each with
`valid_from` / `valid_to`, so finance can report revenue by the tier a
customer had *on the day of the sale*. The loader reads tier changes from the
CRM's change feed. After the CRM team replayed a few weeks of changes that a
broken webhook had missed, the "revenue by tier" dashboard summed to more
than total revenue, and some customers who had upgraded to Gold were shown
as Silver again in the current view.

## Root cause
The loader assumed changes arrive in effective-time order. For every change
it closed the customer's *current* row at the change's effective time and
appended a new current row. A late change breaks both steps. In the test
scenario a customer goes Basic (day 0) → Gold (day 20), then a Silver change
effective on day 10 arrives late. The naive loader produces:

| tier | valid_from | valid_to |
|---|---|---|
| basic | day 0 | day 20 |
| silver | day 10 | end of time |
| gold | day 20 | day 10 (inverted) |

On day 12 the point-in-time join finds two rows (basic and silver), so the
sale is counted twice. From day 20 on only Silver is found, so the newer Gold
tier is lost.

## Solution
`apply_change` treats the history of each customer as a timeline and inserts
the change where it belongs instead of always at the end:

- If the effective time falls **inside** an existing version, that version is
  split: the first part keeps the old tier and ends at the change, the second
  part takes the new tier and keeps the old end. Later versions are untouched.
- If the effective time **equals** a version's start, it is a correction, so
  the tier is replaced. Replaying the same change feed twice is a no-op.
- If it is **before all history**, a new first version ends where the old
  first version starts.

For changes with **distinct** effective times the result is the same whatever
order they arrive in, and every point in time maps to exactly one version.
Two different changes with the *same* effective time are treated as a
correction, so the one that arrives last wins; if a replay can reorder such
ties, give each change a source sequence number and keep the highest. `find_overlaps` is a cheap data-quality
check to run after every load; it flags overlapping or inverted intervals.

## Trade-offs
- A split rewrites one existing row, so the table must allow updates (Delta,
  Iceberg, or a `MERGE` in the warehouse), not append-only inserts.
- Facts already joined to the old version are not re-attributed automatically;
  downstream aggregates for the affected range must be recomputed.
- A late change only affects the gap until the next known change. If the
  source actually means "from now on" rather than "as of that date", a late
  change should not override later versions, and this model matches that.
- Changes must carry a reliable effective time from the source system; using
  load time instead hides the problem but makes history wrong.
- The in-memory version sorts each customer's history per change. In a
  warehouse, the same logic is a `MERGE` on the version whose interval
  contains `effective_from`.

## Run
```bash
python -m unittest -v test_solution.py
```
