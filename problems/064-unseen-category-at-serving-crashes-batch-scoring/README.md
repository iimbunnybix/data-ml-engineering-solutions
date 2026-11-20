# 064 · The nightly fraud-scoring job crashed with KeyError: 'bnpl' the day "buy now, pay later" launched, and the hotfix scored it as Apple Pay

## Problem
A payments company scores every transaction of the day in a nightly batch
job; the scores feed the next morning's manual review queue and the payouts
that are held back. The model uses categorical features such as
`payment_method` and `city`, each turned into an integer index with a
dictionary built from the training data.

The day the product team launched a "buy now, pay later" option, the job
failed after a few minutes with `KeyError: 'bnpl'`. No transaction was
scored, the review queue was empty the next morning, and payouts went out
unchecked. The on-call engineer changed `index[value]` to
`index.get(value, 0)` and re-ran the job. It succeeded, but every BNPL
payment was now scored as if it were Apple Pay, the lowest-risk method,
and nobody noticed until the BNPL fraud losses came in weeks later.

## Root cause
**The encoder had no place for values it had not seen in training.** Every
training value got an index from 0 upwards, so:

- a new value (a new payment method, a newly launched city, a typo from an
  upstream change) was not in the dictionary and the lookup raised, failing
  the whole batch for one value;
- the default `0` in the hotfix was not "unknown": it was the first real
  category in sorted order, `apple_pay`. In the tests the model's fraud rate
  for Apple Pay is 1%, so all 150 BNPL payments in a 1,000-row batch got
  exactly Apple Pay's score, with no error and no metric showing it.

Problem 024 is about one-hot columns built in a different *order* at serving
time; here the order is stable and the failure is a value with no slot.

## Solution
`CategoryEncoder` in `solution.py`:

1. **Index 0 is reserved for `__unknown__`.** Training values get indices
   from 1, so no real category can be mistaken for "unknown". Missing values
   (`None`) also go to the unknown bucket.
2. **Rare training values are folded into the unknown bucket.** Values seen
   fewer than `min_count` times (20 in the tests: `crypto` with 3 rows and
   `gift_card` with 2) get no index of their own. This means the model is
   trained on rows in the unknown bucket and learns a weight for "rare or
   new" instead of an arbitrary one. In the tests that bucket's fraud rate is
   60% (3 of 5 rows), so BNPL payments are scored as unfamiliar rather than as
   Apple Pay (1%).
3. **Unknowns never fail the job; they are counted and alerted on.**
   `encode` fills an `UnknownReport` with unknown counts per column and the
   most common unseen values, and works across chunks of one job.
   `unknown_rate_alerts` returns a message for each column whose unknown rate
   is above the threshold (5% by default). The launch batch produces one alert:
   `payment_method: 15.0% unknown (threshold 5%), top values [('bnpl', 150)]`.

## Trade-offs
- The unknown bucket's weight is learned from few rows (5 in the tests), so
  it is noisy, and new values are not necessarily like old rare ones: a big
  launch can be safer or riskier than the long tail. Treat the alert as a
  prompt to retrain with the new value, not as a permanent answer.
- `min_count` trades detail for stability: too low and rare values get
  weights learned from a handful of rows, too high and real signal is
  merged away. Pick it per column from the counts, and consider
  smoothing (shrinking rare categories towards the global rate) instead.
- Alerting on a fixed rate misses slow growth in unknowns and fires on
  small batches by chance; compare against a baseline rate or require a
  minimum row count where that matters.
- The encoder must be saved and deployed with the model as one artifact;
  refitting it on serving data would change the indices under the model.

## Run
```bash
python -m unittest -v test_solution.py
```
