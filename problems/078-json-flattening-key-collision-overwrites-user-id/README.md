# 078 · After the new mobile SDK shipped, half of the app's page views showed up as "unknown user" in every dashboard

## Problem
A consumer app sends clickstream events as JSON. A loader flattens each
event into warehouse columns and the analytics team joins `user_id` to the
accounts table for retention, funnels and attribution.

Two weeks after a new mobile SDK version rolled out, the share of events
attributed to a known account fell steadily as users upgraded. Retention and
funnel dashboards showed a wave of "unknown user" traffic, a growth
experiment was read on wrong numbers, and nothing in the loader logs
reported a problem: every event loaded "successfully".

## Root cause
The loader built column names by joining nested keys with `_`, so
`{"user": {"id": ...}}` became the column `user_id`. The new SDK kept the
legacy top-level `user_id` (the internal account id) **and** added a nested
`user` object whose `id` is a hashed external id. Both paths produced the
column `user_id`, and the loader simply assigned one after the other, so the
later key in the JSON won.

In the reproduction, an event with `user_id: 101` and `user.id:
"ext-799819"` loads with `user_id = "ext-799819"`. With two of four events
from the new SDK, only **50%** of rows join to an account. Swapping the key
order in the JSON changes which value survives, so the result even depended
on the SDK's serializer.

## Solution
`flatten` records which JSON path produced each column. If a second path
produces a column that already exists, it raises `FlattenCollision`, naming
the column and both paths (`user_id` and `user.id`). `load_events`
quarantines those events with that reason instead of loading a wrong value,
so every row that is loaded still joins (100% in the test), and the
quarantine count shows the problem the day the SDK ships. The check works
whatever the key order and at any depth (`a.b_c` versus `a.b.c`).

The longer-term fix is a mapping that names columns explicitly (for example
`user.id` -> `external_user_id`) once the collision is reported.

## Trade-offs
- Quarantining drops events from the main table until the mapping is fixed.
  If that is not acceptable, load them with only the colliding columns set to
  NULL, and alert on the count.
- A separator that cannot appear in keys (or escaping it) avoids the
  collision, but long or escaped column names are awkward for analysts, and
  the collision still has to be resolved somewhere.
- Many tools do the same silent overwrite (for example simple `json_normalize`-
  style flatteners). A schema registry that validates new fields when an SDK
  release is built catches this before any data arrives.
- Lists are kept as values here; exploding them into rows is a separate
  decision with its own fan-out risks.

## Run
```bash
python -m unittest -v test_solution.py
```
