# 003 · Fraud model scored near-perfect offline but caught almost nothing after launch

## Problem
A card-fraud team trained a model on a year of labeled transactions. One
feature, `chargebacks_count` per user, looked like the star: on the
offline validation set the model was almost perfect. After deployment the
fraud-catch rate was barely better than the old rules engine, and the
feature's importance in production was near zero.

## Root cause
**Feature leakage through a "latest value" join.** The training set was
built by joining each labeled transaction to the user's *current* row in
the feature table. But `chargebacks_count` is incremented days or weeks
*after* a fraudulent transaction, precisely because it was fraud. So every
fraud label was joined to a value that only exists because of the label.
At serving time the model sees the value *as of the transaction*, which
does not yet contain that chargeback.

The test dataset reproduces it: 400 users, 38 fraud labels (seeded), and a
chargeback that lands 5-30 days after each fraud. With the latest-value
join, a single threshold on `chargebacks_count` classifies **100%** of rows
correctly, a classic leakage smell.

## Solution
`PointInTimeJoiner` performs an **as-of join**: for each label at time `t`
it returns the newest feature value whose *availability time* is `<= t`.

- Per entity, rows are sorted by availability and looked up with
  `bisect` (O(log n) per label).
- **Availability ≠ event time**: an optional `ingested_ts` models pipeline
  delay, so a value describing day 10 that only landed on day 12 is not used
  for a prediction on day 11.
- **TTL**: values older than `ttl_s` return the default instead of a stale
  number the online store would already have expired.
- **Late backfills**: if an older value is ingested after a newer one, the
  newest event-time value available still wins.
- `strict=True` excludes same-timestamp values when the serving path can't
  see writes from the same second.

With point-in-time features the same threshold test drops to **90.5%**, which
is exactly the majority-class baseline (362 of 400 non-fraud): the feature
carries no future information, which is the honest offline number.

`best_threshold_accuracy` is included as a cheap guard: run it per feature
in the training pipeline and flag any single feature that is "too good".

## Trade-offs
- Point-in-time correctness requires keeping **feature history**, not just
  the latest value; storage grows with update frequency.
- Getting `ingested_ts` right matters as much as the join. If you only log
  event time, you'll still leak values that arrived late.
- At warehouse scale this is the `ASOF JOIN` in DuckDB/Snowflake/ClickHouse
  or Feast/Tecton's `get_historical_features`; this in-memory version is for
  understanding and for tests.
- A lower offline score after the fix is expected and correct. It is the
  number that will match production.

## Run
```bash
python -m unittest -v test_solution.py
```
