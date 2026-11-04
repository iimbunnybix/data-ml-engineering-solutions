# 007 · Fraud model's live scores disagreed with offline scores for the exact same transactions

## Problem
A card-fraud model scores every checkout in real time. Its offline evaluation
looked good and stable, but after launch the risk team replayed a day of
production traffic through the offline pipeline and found that the *same*
transactions got noticeably different scores online. New customers in
particular were scored as if they had an average spending history. Analysts
could not tell whether the model, the data or the service was wrong, and the
model's decline threshold had been tuned on numbers that production never saw.

## Root cause
**Training/serving skew from two implementations of the same features.** The
training set was built in SQL by the data team; the scoring API re-implemented
the features in application code. The two disagreed on three undocumented
details:

| Detail | Offline (training) | Online (serving) |
|---|---|---|
| Refunded orders in `avg_order_value_30d` / `order_count_30d` | excluded | included |
| Window end | `ts < as_of` | `ts <= as_of` (counts the order being scored) |
| No history | `0.0` | global mean `42.0` |

On the seeded test traffic (500 customers), the two paths disagree on
**53.8%** of `avg_order_value_30d` values, **21.2%** of `order_count_30d`
values and **2.4%** of `refund_rate_90d` values. The model learned one
definition and was served another. Nothing in the input data had changed, so
drift monitoring could not catch it.

## Solution
1. **One feature definition.** `compute_features(orders, as_of)` is the single
   source of truth: paid orders only, strictly-before-`as_of` visibility (which
   also keeps training point-in-time correct), `0.0` for no history. The
   offline builder (`build_training_rows`) and the online path
   (`serve_features`) both call it, so the parity test shows 0% mismatch on
   every feature.
2. **Parity checking as a release gate.** The serving path logs the feature
   vector it used. `ParityChecker` recomputes features offline for a sample of
   logged (customer, timestamp) pairs and reports per-feature mismatch rates
   with a float tolerance. NaN equals NaN, and a missing feature counts as a
   mismatch. Any feature above `max_rate` blocks the model release and shows
   up to 5 example keys for debugging.

## Trade-offs
- Sharing code is easy when both paths are in Python. When offline is
  Spark/SQL and online is Java or Go, use a feature store or a feature DSL that
  compiles to both. The parity check is still needed, because "same definition"
  can still drift through different data sources or freshness.
- Parity checks need logged serving features and a replayable offline source
  for the same timestamp. Log at a sampled rate to control storage.
- Online features are often fresher than offline snapshots (streaming vs
  daily batch). Compare against data as of the serving time, or some mismatch
  will be pure lag, not a bug; tune `max_rate` with that in mind.
- Rounding to 6 decimals in the shared function removes float-summation noise
  but means features are not bit-exact with any older pipeline.

## Run
```bash
python -m unittest -v test_solution.py
```
