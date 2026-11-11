# 045 · The daily "active customers" table came out empty after someone added one blocklist row without a user id

## Problem
A marketing pipeline builds a daily table of customer events for campaign
targeting. One step removes users on a compliance blocklist (opted-out or
fraud-flagged accounts):

```sql
SELECT * FROM events
WHERE user_id NOT IN (SELECT user_id FROM blocklist)
```

It ran fine for months. Then a support agent added a blocklist row through an
admin form that left `user_id` empty (NULL). The next morning the output table
had **0 rows**. The job did not fail, row-count checks downstream only looked
for "table exists", and that day's campaigns went to nobody.

In the simulation (1,000 users × 10 events = 10,000 events, 20 blocked users),
the filter keeps 9,800 events. Add one NULL row to the blocklist and it keeps 0.

## Root cause
SQL uses three-valued logic: comparing anything with NULL gives UNKNOWN, not
FALSE. `x NOT IN (a, b, NULL)` means `NOT (x = a OR x = b OR x = NULL)`. For a
user that is not blocked this is `NOT (FALSE OR FALSE OR UNKNOWN)` =
`NOT UNKNOWN` = UNKNOWN. `WHERE` keeps only rows that are TRUE, so every row is
dropped. The query was correct only as long as the subquery never returned NULL.

## Solution
- `sql_not_in` models the SQL semantics exactly, and `naive_exclude_blocked`
  keeps the original filter so the failure can be reproduced in a test.
- `exclude_blocked` uses **NOT EXISTS** (anti-join) semantics:
  `WHERE NOT EXISTS (SELECT 1 FROM blocklist b WHERE b.user_id = e.user_id)`.
  EXISTS is always TRUE or FALSE, so a NULL blocklist row matches nothing and
  removes nothing, while every real blocked id is still removed.
- `blocklist_warnings` reports NULL blocklist rows. They are bad data even when
  the filter no longer breaks: the agent meant to block *someone*.

With the fix, the same 10,000 events and the NULL-containing blocklist give
9,800 rows again.

## Trade-offs
- NOT EXISTS keeps events whose own `user_id` is NULL; NOT IN with a non-empty
  blocklist drops them. Decide which you want explicitly (and test it) rather
  than inheriting it from the operator.
- An alternative is `NOT IN (SELECT user_id FROM blocklist WHERE user_id IS
  NOT NULL)`. It works, but every future copy of the query must remember the
  extra clause; NOT EXISTS is safe by construction. A `NOT NULL` constraint on
  `blocklist.user_id` stops the bad row at the source and is worth adding too.
- The real lesson is that "the job succeeded" is not a data check. A volume
  check comparing today's row count with recent days (problem 016) would have
  caught 0 rows before the campaigns ran.

## Run
```bash
python -m unittest -v test_solution.py
```
