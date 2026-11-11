# 046 · The "units sold per price point" report listed $0.70 twice, and neither row matched the stock count

## Problem
A grocery chain's merchandising team uses a daily report of units sold per
unit price to judge price changes. The job reads order lines from the POS
export (line total as text plus quantity), computes each line's unit price and
groups by it. Analysts noticed the report had several rows with the same
label, e.g. two "$0.70" rows with 216 and 159 units, while stock movements said
375 units sold at that price. Price-elasticity models trained on the report saw
each real price point as several smaller ones.

In the simulation (10,000 lines, 99 catalog prices from $0.10 to $9.90,
quantities 1-6, seed 7) the report had **140 rows instead of 99**, and 41
price labels appeared more than once.

## Root cause
The group key was a binary float: `float(line_total) / quantity`. Most
decimal amounts, such as 0.70 and 2.10, have no exact binary representation,
so dividing does not give back the same float. `2.10 / 3` is
`0.7000000000000001` while `0.70 / 1` is `0.7`. They are different keys, so
GROUP BY puts them in different groups, and the dashboard's two-decimal
formatting hides the difference.

## Solution
- `naive_units_by_price` keeps the original float-keyed report.
- `unit_price_cents` parses the text as `Decimal` (exact for decimal money),
  divides, and quantizes to whole cents with half-even rounding. Equal prices
  now produce equal keys regardless of quantity.
- `units_by_price` groups on that key: 99 rows for 99 price points, $0.70
  shows 375 units, and the total number of units is unchanged.
- `duplicate_labels` is a cheap check worth keeping in the pipeline: two
  rows with the same printed label mean the keys are not what they look like.

## Trade-offs
- Rounding to the cent merges prices that differ by less than a cent (a 3-for-$1.00
  deal becomes $0.33). That is usually right for a price report but must be a
  conscious choice; keep the exact total and quantity if you need to recover it.
- Storing money as integer cents (or a fixed-point `DECIMAL(12,2)` column)
  from the start avoids the problem everywhere, not just in this report. In SQL
  engines, `ROUND(x, 2)` on a DOUBLE still returns a DOUBLE, so cast to
  DECIMAL before grouping.
- Decimal arithmetic is slower than float. For a daily report that does not
  matter; for very large jobs, integer cents give the same exactness at
  native speed.

## Run
```bash
python -m unittest -v test_solution.py
```
