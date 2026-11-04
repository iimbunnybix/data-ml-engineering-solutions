# 006 · Nightly merchant-enrichment job stuck at 199/200 tasks for hours after onboarding one large marketplace

## Problem
A nightly Spark job joins the day's payments with the merchants table
(tier, region, risk segment) before writing features for the risk model.
It normally finishes in about 25 minutes. After a large marketplace merchant
was onboarded, the stage would sit at 199/200 tasks finished for hours, one
executor ran out of memory and retried, and the features were not ready for
the morning model refresh, so the model scored a full day on stale inputs.

## Root cause
A shuffle join hash-partitions both sides on the join key, so **every row with
the same key goes to the same task**. The new merchant produced about 40% of
all payments, so one task received about 40% of the stage's data while the
other tasks were nearly idle. Adding executors does not help: a single key
cannot be split across tasks.

In the reproduction (50,000 payments, 2,000 merchants, one with a 40% share,
32 partitions) the slowest task gets **20,696 rows vs a median of 987**, a
21× skew. The stage takes as long as that one task.

## Solution
**Salt the hot keys** (`salted_join`):

1. **Detect** keys holding more than 2× a task's fair share
   (`find_hot_keys`). In production, run this on a sample or on yesterday's
   key counts. In the test it finds exactly the one hot merchant.
2. **Big side:** each row of a hot key gets a salt `0..S-1` (round-robin, so it
   is deterministic), turning one key into S sub-keys that hash to different tasks.
3. **Small side:** the hot key's merchant row is replicated S times, once per
   salt, so every sub-key still finds its match. Only hot keys are replicated,
   so the small side grows by `S-1` rows per hot key.
4. Non-hot keys keep their plain key and shuffle exactly as before (tested:
   with no hot keys, task loads are identical to the plain join).

The output is identical to the plain join (compared as a multiset). With the
default `S = 4 × partitions`, the slowest task drops from 20,696 to 2,507 rows
(**8.3× less**), and the slowest/median ratio from 21× to about 1.5×.

## Trade-offs
- Salted sub-keys are still hash-partitioned, so a few tasks get more than one
  sub-key. More salts smooth this out (in the test: 8 salts give 6.2×, 32 give
  2.0×, 128 give 1.5×) at the cost of more small-side replication.
- Needs a hot-key list. Hard-coding it goes stale; compute it per run from a
  sample, and re-check when the data changes.
- If the small side is small enough, a **broadcast join** avoids the shuffle
  entirely and is the simpler fix. Salting is for when both sides are large.
- Spark 3 Adaptive Query Execution (`spark.sql.adaptive.skewJoin.enabled`)
  splits skewed partitions automatically for sort-merge joins. Try it first;
  manual salting is still needed for aggregations by a hot key, other
  engines, and cases AQE's thresholds miss.
- Null or "unknown" keys are a common source of skew. Filter them out before
  the join if they can never match.

## Run
```bash
python -m unittest -v test_solution.py
```
