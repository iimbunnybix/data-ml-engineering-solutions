# 029 · Delivery ETAs were 20 minutes too optimistic all evening while every dashboard stayed green

## Problem
A food-delivery app predicts an ETA for every checkout. The ETA model reads
two live features per delivery zone from an online feature store: pending
orders and active couriers over the last 5 minutes. A streaming job
recomputes them every minute. One evening the job's consumer hung at 17:00
after a broker restart. It did not crash, so no job alert fired, and the
feature store kept serving the last values it had, from the quiet afternoon.
During the 18:00 to 20:00 dinner peak customers were promised about 25
minutes and waited about 45. Late-order complaints and refund credits spiked,
and the cause was found the next morning, when someone noticed that the
feature's "last updated" time had not changed in 16 hours.

## Root cause
**The online read path had no freshness check.** The store keeps the latest
value per (zone, feature) and `naive_read_features` returns it however old
it is, while a feature that was never written becomes `0.0`. The model has no
idea that "20 pending orders" describes 16:59, not 18:30. Monitoring watched
the job's process and the store's latency, both healthy; nothing watched the
age of the data.

In the replay (`test_solution.py`: one request per minute from 16:00 to
21:00, job stalls at 17:00, pending orders jump from 20 to 60 at 18:00 with
20 couriers), the naive path predicts 25 minutes for every peak request
against a true 45: a mean error of **20.0 minutes**, with no signal at all.

## Solution
1. **Store the write time with every value** (`FeatureValue.written_at`).
2. **Per-feature `FreshnessPolicy`**: a maximum age (here 5 minutes for
   features refreshed every minute) and a fallback value (here the zone's
   historical evening average: 45 pending orders, 20 couriers).
3. **`read_fresh_features` checks age at read time** against an injected
   clock. A stale or missing value is replaced by its fallback, so the model
   gets a reasonable input instead of a confidently wrong one.
4. **Staleness metric and alert.** `StalenessMetrics` counts reads and stale
   reads per feature. `alerting_features()` returns features whose stale rate
   is above 5%, which pages the pipeline owner.

In the same replay, stale reads start at 17:05 (once the last write is more
than 5 minutes old): 235 of 300 reads per feature are stale (78%), both
features alert, and the mean peak ETA error drops to **7.5 minutes**
(37.5 predicted vs 45 true).

## Trade-offs
- Fallbacks are a guess. Between 17:05 and 18:00 the fallback *overestimates*
  the ETA (37.5 vs 25 minutes). A wrong-but-bounded default is better than a
  silently stale value, but the real fix is the alert getting the job
  restarted.
- Max age must match the refresh interval plus normal lag. Too tight and
  every small delay floods the fallback; too loose and stale data is served
  for longer. Per-feature policies matter: a daily batch feature can be a day
  old, a streaming one only minutes.
- The check uses the store's write time. If the pipeline writes fresh
  timestamps on recomputed-but-stale inputs (e.g. it re-emits the last
  window), you also need the event time of the newest input in the value.
- Some systems should fail closed instead (fraud scoring may route to manual
  review when key features are stale) rather than use a fallback.

## Run
```bash
python -m unittest -v test_solution.py
```
