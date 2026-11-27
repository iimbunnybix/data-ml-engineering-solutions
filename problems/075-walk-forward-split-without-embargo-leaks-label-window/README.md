# 075 · The churn model's backtest error was 20% lower than anything it achieved in production after a competitor launch

## Problem
A subscription business scores every customer daily with "probability of
cancelling in the next 30 days", and the retention team sizes its offers from
those probabilities. Models are evaluated with a walk-forward backtest: train
on snapshots before day T, test on snapshots from day T on. The model is a
per-segment churn rate over the latest 60 days of training snapshots (web or
mobile sign-up); in the seeded simulation there are 80 customers per day for
180 days, and a competitor's app launches on day 155, which pushes 30-day
mobile churn from about 14% to **70.8%**.

The backtest at T = day 150 reported a Brier score of **0.262**. A model
trained on day 150 can only know what was known on day 150, and that model
scores **0.327** on the same test rows: 25% more error. The backtest had
promised the team the model would "pick up the mobile spike"; in production it
predicted 13.5% mobile churn while 70.8% of mobile customers left, and the
offers were sent too late.

## Root cause
**The split cut on the snapshot date, but the label lives 30 days later.**
A snapshot from day 140 is labelled with whether the customer cancelled on
days 141-170, so its label is mostly a test-period outcome. In the naive
split, **2,320** training rows (snapshot days 121-149) have label windows
that reach past day 150. Those rows already contain the competitor launch,
so the leaky model learned a mobile churn rate of **26.4%** instead of the
**13.5%** that was knowable on day 150. On the test rows that looks better,
and nothing like it is available in production.

This is not label maturity (problem 021), where production retraining
labels *immature* rows as negative. Here every label in the backtest is
complete and correct; the problem is *when* it became known. Nor is it a
leaky feature (problem 050): the features are fine, the overlap is between
the training labels and the test period.

## Solution
- `label_window_end` gives the last day that can change a row's label:
  snapshot day + 30.
- `purged_walk_forward_split` keeps a training row only if its label window
  ends on or before the test start, i.e. it **purges** (embargoes) snapshot
  days 121-149, leaving a 30-day gap before the cut. The test set is
  unchanged. Training then holds exactly the rows whose labels were final
  on day 150, which is what production would have had.
- `leaking_training_rows` is a guard for any split: it returns training rows
  whose window reaches past the test start. It finds 2,320 rows in the naive
  split and none after purging; fail the backtest job if it is non-empty.
- `naive_walk_forward_split` is kept for comparison.

| Split | Training rows | Learned mobile churn | Test Brier score |
|---|---|---|---|
| Naive: snapshots before day 150 | 12,000 | 26.4% | 0.262 |
| Purged: label window ends by day 150 | 9,680 | 13.5% | 0.327 |

The purged number is worse, and it is the honest one: it is what the model
can deliver when the world changes after training.

## Trade-offs
- **Less, older training data.** Purging drops a horizon's worth of the
  freshest rows (here 29 of 150 snapshot days, 2,320 rows). With a long
  horizon this hurts, which is exactly the delay production faces too; the
  backtest should not hide it.
- **The gap must match the label definition.** If the label horizon changes
  (30 to 60 days), the purge must change with it, so it is computed from
  `LABEL_HORIZON_DAYS`, not hard-coded.
- **Feature lookbacks need an embargo too.** If features use windows that
  reach into the test period (for example after a test fold in k-fold
  time-series CV, where training data also sits *after* the test fold), add
  an embargo after the test fold as in López de Prado's purged k-fold.
- An alternative is to backtest by **as-of date**: rebuild the labels exactly
  as the warehouse showed them on day T. That is more faithful (it also
  catches late-arriving cancellations) but requires a versioned label store.
- One regime change in one simulated run is a worst case on purpose; with a
  stable world the two scores are close, which is why this leak usually goes
  unnoticed until something shifts.

## Run
```bash
python -m unittest -v test_solution.py
```
