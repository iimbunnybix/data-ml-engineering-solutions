# 049 · The first week of January showed 14% more orders than any other week, and the last week of December came up short

## Problem
An e-commerce analytics team publishes a weekly orders dashboard built from a
daily orders table. Each day is labelled with a week key and summed. In
January, the finance team noticed the week "2025-W01" had **800** orders in
a test where every day has exactly 100 orders (8 days instead of 7, so
14% above a normal 700-order week), while "2026-W01" had only **400**.
Year-over-year comparisons for the holiday weeks were wrong, and the bad
keys also showed up as a 54th week ("2024-W01" with only 2 days) on the
chart for a period that contains exactly 53 complete ISO weeks.

## Root cause
The job built the key with `strftime("%Y-W%V")`. `%V` is the **ISO** week
number, but `%Y` is the **calendar** year. ISO weeks start on Monday and
week 1 is the week containing the year's first Thursday, so a few days at
the edges of a year belong to the neighbouring ISO year:

| Day | `%Y-W%V` (naive) | ISO year-week |
|---|---|---|
| 30 Dec 2024 | 2024-W01 | 2025-W01 |
| 29-31 Dec 2025 | 2025-W01 | 2026-W01 |
| 1 Jan 2027 | 2027-W53 | 2026-W53 |

So 29-31 Dec 2025 were added to the first week of January **2025**, a year
earlier, and the real first week of 2026 lost them. A key such as "2027-W53"
names a week that does not exist.

## Solution
- `iso_week_key` takes both parts from `date.isocalendar()` (equivalent to
  `strftime("%G-W%V")`, where `%G` is the ISO year). Over 30 Dec 2024 – 4 Jan
  2026 it produces 53 weeks of exactly 7 days each.
- `overfull_weeks` is a cheap data-quality guard: any week label with more
  than 7 distinct days means two real weeks share a label. It flags
  "2025-W01" for the naive keys.
- `week_start` turns a key back into its Monday, which is a safer join key
  for charts and other tables than a string label.

## Trade-offs
- ISO weeks fit retail and finance reporting, but a business may define
  weeks starting on Sunday (US retail) or use a 4-4-5 fiscal calendar; then a
  calendar dimension table is the right tool, not `strftime`.
- Calendar-year totals cannot be computed by summing ISO weeks: an ISO year
  has 52 or 53 weeks and its edges cross 31 December.
- Fixing the key changes historical numbers. Backfill the affected weeks and
  announce the change, or year-over-year charts will compare old and new
  definitions.

## Run
```bash
python -m unittest -v test_solution.py
```
