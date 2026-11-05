# 036 · The risk model kept flagging customers who shop once a month, because their "7-day spend" included purchases from months ago

## Problem
A payments company scores every card transaction with a risk model. One of
its strongest features is `spend_7d`: how much the customer spent in the
last 7 days, computed nightly by a batch job from a `daily_spend` table
(one row per customer per day *with spend*). After a new model version gave
`spend_7d` more weight, customers who shop rarely started getting step-up
challenges and declines: a customer buying groceries once a month suddenly
"spent" as much per week as a heavy daily user. Support tickets about
declined cards doubled, and the fraud team could not explain why low-activity
customers looked like spending sprees.

## Root cause
**The window was defined in rows, not in days.** The job used
`SUM(spend) OVER (PARTITION BY user ORDER BY day ROWS BETWEEN 6 PRECEDING AND
CURRENT ROW)`. The table has no rows for days with no spend, so "the current
row and the 6 before it" is 7 days only for customers who spend every day.
For an occasional customer, the same 7 rows can reach back half a year.

The reproduction (100 daily and 100 occasional customers over 180 days)
shows daily customers are correct, but **456 of 572 feature rows (80%)** for
occasional customers are wrong: on average they are **3.7x** the true 7-day
spend, and the worst is **12.9x**.

## Solution
`solution.py` computes the feature over a **calendar window**: a row counts
only if its date is within the 6 days before the current date or on it. This
is SQL's `RANGE BETWEEN INTERVAL '6' DAY PRECEDING AND CURRENT ROW`.

- `rolling_spend_for_user` walks each customer's rows in date order with a
  deque and a running total: rows enter on the right and leave on the left
  once they are older than the window, so the job stays O(rows).
- `check_one_row_per_day` rejects two rows for the same customer and day,
  which would mean an upstream duplicate rather than real spend.
- `spend_as_of` is a deliberately simple reference definition (the same one
  the online service uses). The tests check that the batch job equals it on
  every row, so training and serving agree by construction.

## Trade-offs
- A dense alternative is to fill missing days with zero rows and keep the
  `ROWS` window. It works, but multiplies the table size for sparse users and
  breaks again the moment someone filters out the zero rows.
- The batch job only emits a value for days that have a row. A score on a
  day with no spend yet needs `spend_as_of` (or a dense calendar), otherwise
  the feature looks up a stale value.
- "Last 7 days" here means 7 calendar dates in one time zone. A
  rolling 168-hour window is more precise for real-time scoring but needs
  timestamps, not daily rows.

## Run
```bash
python -m unittest -v test_solution.py
```
