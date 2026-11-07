# 038 · A batch of high-risk applicants scored as "average", and the same customer's risk changed from 39% to 13% between files

## Problem
A lender scores credit-line applications in nightly batches with a small
logistic default-risk model. The model was trained on z-scored features
(`utilization`, `income_k`). One night, marketing sent a targeted batch of
applicants who were all near their credit limit. Their average predicted
default risk came out at 12.9%, the same as a normal night, so the batch was
auto-approved at standard limits. Later, an analyst re-scored one applicant
together with a different file and got a different number. A separate
same-day "single application" job crashed with `ZeroDivisionError`.

## Root cause
**The scoring job recomputed the scaling statistics from the batch it was
scoring.** `(value - mean) / std` was computed with the mean and std of the
current file, not of the training data. That centres every batch on zero, so a
batch of risky applicants looks exactly like an average batch, and each row's
score depends on who else happens to be in the file. A one-row batch has a std
of 0 and the division fails.

In the simulation (model weights fixed; 999 applicants per batch), with batch
statistics:

- the same applicant scores **39.2%** in a general batch and **12.9%** in the
  high-risk batch;
- the mean score is **12.7%** for the general batch and **12.9%** for the
  high-risk batch (utilization 0.7–1.0 instead of 0.0–1.0).

## Solution
`solution.py`:

- **Fit once, on training data.** `FrozenScaler.fit` measures each feature's
  mean and std on the training set; these numbers are part of the model
  artifact, just like the weights.
- **Apply the same numbers to every row.** `score_batch` scales each row
  independently, so a row's score is identical whether it is scored alone or
  in any batch (39.0% for the example applicant), and the high-risk batch now
  averages **28.0%** against **12.8%** for the general batch.
- **Safe edge cases.** A feature that was constant in training is scaled by 1
  instead of 0; a row missing a feature raises `KeyError` instead of being
  scored with a silent default; a scaler without statistics for a model
  feature is rejected before scoring.

## Trade-offs
- Frozen statistics go stale when the population really drifts. That is the
  point: drift should show up as different scores and be caught by a drift
  monitor (see 004), not be normalised away. Retrain to update them.
- The scaler and the model must be versioned and deployed together; loading
  new weights with old statistics is the same class of bug. Store them in one
  artifact.
- The same mistake hides in other per-batch transforms: min-max scaling,
  per-batch quantile bins, filling missing values with the batch median, and
  batch-norm layers left in training mode. All of them must use training-time
  statistics at inference.

## Run
```bash
python -m unittest -v test_solution.py
```
