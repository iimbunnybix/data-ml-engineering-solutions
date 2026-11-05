# 030 · Churn model's launch-review accuracy kept coming in about 3 points lower in production

## Problem
A subscription business retrains its churn model each quarter. Every time,
the team runs a hyperparameter search of about 100 configurations, scores
each one on a held-out test set, ships the best and puts that test accuracy
in the launch review. The retention team sizes its outreach budget from that
number. Quarter after quarter, the accuracy measured on the following
month's real outcomes came in lower than the review promised, the outreach
budget was off, and nobody could find a data bug, because there wasn't one.

## Root cause
**The test set was used to choose the model, so it stopped being a test
set.** Each candidate's test accuracy is its true accuracy plus sampling
noise. Taking the maximum of 100 such numbers picks the candidate whose
noise happened to be most favourable on *those* rows. The reported number is
the winner's score on the data that crowned it, which is biased upward
(the "winner's curse", or adaptive overfitting to the holdout).

The simulation (`test_solution.py`) repeats a sweep 40 times with different
seeds: 100 random linear candidates, 300 validation rows, 300 test rows and
5,000 "production" rows per sweep, where churn depends weakly on one feature.
Choosing and reporting on the test set (`naive_select_and_report`):

| | Choose on test, report test (naive) | Choose on validation, report locked test (fix) |
|---|---|---|
| Mean (reported − production accuracy) | **+2.7 points** | +0.5 points |
| Sweeps where the report overstated production | **33 / 40** | 20 / 40 |
| Largest overstatement | 8.6 points | 5.3 points |

The fixed column still has noise (300 test rows give roughly ±3 points), but
it is no longer biased in one direction.

## Solution
1. **Split three ways**: train (fit candidates), validation (compare
   candidates), test (report the one chosen model).
2. **`select_and_report`** picks the best candidate on validation only, then
   asks the test set for exactly one score.
3. **`LockedTestSet.evaluate_once`** raises `TestSetReusedError` on a second
   call, so a sweep script or notebook cannot quietly turn the test set back
   into a selection set. Another look needs fresh data.

## Trade-offs
- Holding out a third split costs training data. With small datasets use
  nested cross-validation (inner loop selects, outer loop estimates) instead.
- The guard only protects one Python object. Teams also peek by re-running
  the evaluation in a new process; real protection is process: keep the test
  set in a separate location, log every access, and refresh it regularly.
- The validation score of the winner is still optimistic. Only use it to
  rank candidates, never as the reported number.
- A single unbiased test score is still noisy. Report a confidence interval
  with it, or the next quarter's "drop" may just be sampling noise.

## Run
```bash
python -m unittest -v test_solution.py
```
