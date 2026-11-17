# 058 · After switching to downsampled training data, the ad bidder started paying about 7-9x too much for every impression

## Problem
An ads platform predicts the click-through rate (CTR) of each impression and
bids `predicted CTR × value per click`. Training on every impression had
become slow and expensive, so the team kept **all clicks** but only **10% of
non-click impressions** — a standard trick, since non-clicks are over 95% of
the data.

The new model's offline AUC was unchanged, so it shipped. Within a day
campaign budgets were exhausted by mid-morning while clicks stayed flat: the
bidder believed impressions were far more likely to be clicked than they
were. In the simulation in the tests (300,000 impressions, true CTRs of 5%,
2% and 0.5% for the top, side and footer slots) the model predicted 34.0%,
16.7% and 4.7%: 6.8x, 8.4x and 9.4x too high.

## Root cause
Downsampling negatives changes the class balance the model learns from. With
a sampling rate `r`, every click is kept but each non-click survives with
probability `r`, so the **odds** of a click in the training data are `1/r`
times the real odds. A model fitted on that data is correctly calibrated
**for the sampled data**, not for production traffic. AUC only measures
ranking, which sampling does not change, so it could not reveal the problem.

## Solution
Keep the cheap sampling, but undo its effect on probabilities. Two
equivalent ways are implemented:

- **Weighted training** (`weighted_train`): each kept negative gets weight
  `1 / r`, so it stands for the negatives that were dropped.
- **Closed-form correction** (`correct_for_sampling`) applied after
  training: `p = q / (q + (1 - q) / r)`, where `q` is the model's
  prediction. This divides the odds by `1/r` again.

In the tests both give 4.91%, 1.97% and 0.49%, within 10% of a model trained
on all 300,000 impressions, using a training set of 36,826 rows.

## Trade-offs
- The correction must use the exact sampling rate used for training. Store
  it with the model version; a pipeline that changes the rate without
  retraining reintroduces the bug in a new size.
- The closed form is exact for models that output well-calibrated
  probabilities on the sampled data (logistic regression, frequency tables).
  For tree ensembles or neural nets, check calibration on an unsampled
  holdout and recalibrate (Platt scaling or isotonic regression) if needed.
- Downsampling still loses information about rare negatives; weights reduce
  bias but increase variance. Sampling rates are usually chosen so that
  negatives still outnumber positives several times.
- Monitor calibration (mean predicted CTR vs observed CTR) in production, not
  only AUC, so this class of bug is caught within hours.

## Run
```bash
python -m unittest -v test_solution.py
```
