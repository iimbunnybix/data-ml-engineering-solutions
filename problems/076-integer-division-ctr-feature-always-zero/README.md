# 076 · The new CTR feature got zero importance in the ad-ranking model, and the top ad slot was no better than random

## Problem
An ads team added "historical click-through rate" to its ranking model. The
feature was a column in the daily feature table, computed in the warehouse
as `COALESCE(clicks / NULLIF(impressions, 0), 0)`. The retrained model gave
it zero importance, offline metrics did not move, and the team concluded
"CTR doesn't help". Months later someone looked at the column itself.

In the seeded simulation (1,000 ads, 1.9 million impressions, overall CTR
4.39%, 207 new ads with 0-3 impressions), ranking by the feature puts ads
whose real CTR averages **4.42%** in the top 50 slots: no better than the
**4.31%** of a random pick. Ranking by real CTR would have reached 7.83%.

## Root cause
`clicks` and `impressions` are both INTEGER columns, and INTEGER / INTEGER in
most SQL engines (Postgres, SQL Server, SQLite, Redshift) is **integer
division**: it truncates, like Python's `clicks // impressions`. 40 clicks in
1,000 impressions becomes 0. In the simulation **997 of 1,000** ads get CTR 0
and the other 3 get CTR 1: new ads with one impression and one click. A
feature that is constant on 99.7% of rows has nothing to teach a model, so
the model ignored it, and the 3 "perfect" ads sat in the top slots on noise.
Nothing errored, and the column had no nulls, so every existing check passed.

## Solution
- **Float division with a zero guard.** `raw_float_ctr` divides as floats
  (in SQL: `clicks::float / NULLIF(impressions, 0)` or `1.0 * clicks / ...`).
  This alone lifts the top-50 real CTR to **6.90%**, but 12 of the top 50
  are new ads with a lucky click (1 of 2 impressions is "50% CTR").
- **Smoothing.** `smoothed_ctr` adds 100 "virtual impressions" at the global
  CTR, which is the mean of a Beta prior updated with the ad's clicks. Ads
  with thousands of impressions keep their own CTR; a 1-in-1 ad gets about
  5.3% instead of 100%; zero impressions gives the global CTR. Top-50 real CTR
  becomes **7.56%** with **0** new ads in the top 50.
- **Collapse check.** `feature_collapse_problems` fails a feature column with
  fewer than 10 distinct values or one value on more than 95% of rows. It
  reports both problems for the truncated column (2 distinct values, `0` on
  99.7%) and nothing for the fixed one (799 distinct values). Run it in the
  feature pipeline before training, next to the null checks.
- `original_ctr` is kept for comparison.

| Feature | Distinct values | Real CTR of top 50 ads | New ads in top 50 |
|---|---|---|---|
| `clicks // impressions` | 2 | 4.42% | 12 |
| float division | 784 | 6.90% | 12 |
| smoothed (100 virtual impressions) | 799 | 7.56% | 0 |

(The 12 new ads for the truncated feature are the 3 "CTR 1" ads plus ties
broken by input order; with no information, any order is arbitrary.)

## Trade-offs
- **The prior strength is a tuning knob.** 100 virtual impressions suits a
  4% CTR; too large and genuinely great new ads wait longer to rank, too
  small and lucky ads leak back in. Fit it to the data (empirical Bayes,
  method of moments on the Beta) or tune it on ranking metrics.
- **One global prior.** A per-campaign or per-placement prior is fairer
  (search ads click more than display ads) but needs enough data per group.
- **Shrinkage hides new-ad quality on purpose.** Exploration (Thompson
  sampling, an exploration slot) is the right way to give new ads a chance,
  not an unsmoothed CTR.
- **The collapse check has thresholds.** Genuine low-cardinality features
  (a boolean flag, a 5-level tier) need their own limits; configure the
  check per feature, or compare distinct counts with yesterday's run instead
  of a fixed minimum.
- Casting to float is not the only fix: some engines offer `DIVIDE` or
  `SAFE_DIVIDE` (BigQuery's `/` is already float). Write the cast anyway so
  the SQL stays correct if it is moved to another engine.

## Run
```bash
python -m unittest -v test_solution.py
```
