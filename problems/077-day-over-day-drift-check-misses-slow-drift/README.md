# 077 · Fraud model's precision fell for two months while the drift dashboard stayed green every day

## Problem
A payments company scores every card order with a fraud model trained on
order value, basket size and customer history. A nightly job computes the
Population Stability Index (PSI) of each input feature and pages the ML team
if any PSI goes above 0.2.

Over two months fraud analysts saw precision slowly fall: more good
customers' orders were held for manual review, and the review queue doubled.
The drift dashboard was green every single night. When the team finally
compared current data with the training set by hand, average order value had
moved from about €50 to about €90 (prices rose and a premium product line
launched) and the model was scoring most orders in a range it had barely
seen.

## Root cause
The job compared **today with yesterday**. A feature that moves 1% a day
changes very little from one day to the next, so every daily comparison
passed, and the comparison point drifted along with the data. The question
the team needed answered was "does today look like what the model was
trained on?", but the check asked "does today look like yesterday?".

In the reproduction, the mean of a normal(50, 15) feature grows 1% per day for
60 days. The largest day-over-day PSI is **0.0034**, far below 0.2, so the
original monitor raises **no alerts**, while PSI between day 59 (mean 89.9)
and day 0 is **5.2**.

## Solution
`DriftMonitor` keeps a fixed reference: the snapshot of the feature from the
model's training data.
- Bin edges are the training data's deciles, computed **once** and frozen,
  so the bins cannot slide with the drift.
- Each day is compared with that reference. The PSI crosses 0.2 on **day 13**
  (mean 56.9, PSI 0.202; day 12 is 0.17) and stays above it every day after.
- Empty bins get a tiny floor share so values far outside the training range
  produce a large finite PSI instead of a division error.
- Day-over-day PSI is still useful for sudden jumps (a broken upstream join),
  but only as an additional check.

## Trade-offs
- The reference must be updated when the model is retrained, otherwise
  every retrain makes the monitor compare against data the model no longer
  uses. Store the reference with the model version.
- Seasonal features (for example weekend versus weekday) drift against a
  fixed reference on purpose. Compare with the same weekday, or use a
  reference that covers a full season.
- PSI thresholds of 0.1 and 0.2 are a rule of thumb, not a statistical test;
  PSI also depends on the number of bins. Tune thresholds against past
  incidents and look at the business metric (precision, review rate) too.
- A trend test over a rolling window (for example the slope of the daily
  mean) catches slow drift earlier than a threshold on PSI, at the cost of
  more false alarms.

## Run
```bash
python -m unittest -v test_solution.py
```
