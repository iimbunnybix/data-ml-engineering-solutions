# 063 · A fraud model that flags nothing passed validation and was promoted, because the validation set contained no fraud

## Problem
A payments company launched in a new market and trained a market-specific
fraud model on the first labelled data: 2,000 transactions, of which 10
(0.5%) were confirmed fraud. The training pipeline shuffled the rows, kept
80% for training and 20% (400 rows) for validation, and promoted whichever
candidate had the highest validation recall, breaking ties by the lower
false-positive rate.

Engineers noticed that re-running the pipeline with a different seed gave
completely different recall numbers. Then one run promoted a candidate that
in effect never flagged anything. For two weeks every fraudulent payment in
the new market was approved, and the chargebacks arrived a month later.

## Root cause
**The split ignored the label.** With 10 positives and a random 20% cut, the
number of frauds that land in validation is a lottery: the chance that none
of them do is 0.8^10 ≈ 11%. In the tests, over 200 seeds the validation set
had 0 frauds 23 times and exactly 1 fraud 57 times, and the count ranged from
0 to 8. Recall on 0, 1 or 2 positives is either undefined or one of 0%, 50%
or 100%.

**The metric hid the undefined case.** The recall code divided by
`max(positives, 1)`, so a validation set with no fraud gave every model
recall 0 instead of an error. All candidates tied, and the tie-break on
false-positive rate picked the model that never flags anything: on seed 13
the useful model (threshold 0.7) flags 13% of legitimate payments, the
"never flag" model 0%, and it won.

## Solution
`solution.py`:

1. **`stratified_split`** splits frauds and non-frauds separately with a
   seeded `random.Random`, so every split has the same fraud rate as the
   data. On all 200 seeds it puts exactly 2 frauds in validation and 8 in
   training. Rows are sorted by id before shuffling, so the split depends
   only on the seed, not on the order the rows arrived in.
2. **Minimum positives per split.** Validation gets at least
   `MIN_POSITIVES_PER_SPLIT` (2) frauds even when the fraction rounds lower,
   and if training would then get fewer than that minimum the function raises
   `InsufficientPositivesError` instead of producing a split that cannot be
   evaluated (3 frauds is refused, as is 0).
3. **Undefined metrics are reported, not invented.** `recall` returns a
   `MetricResult` with `value=None` and a reason when there are no positives,
   and `choose_model` raises `UndefinedMetricError` rather than ranking models
   on it. Returning NaN is no better than 0: `nan > x` is always false, so a
   NaN silently loses (or wins) any comparison it is part of.

## Trade-offs
- Two positives per split is a floor, not a good evaluation: recall can still
  only be 0%, 50% or 100%. With this few events, prefer stratified k-fold
  cross-validation (every fraud is validated once), report confidence
  intervals, and look at precision-recall curves on all folds together.
- Stratifying by label does not prevent other leakage: if one customer or
  card has several frauds, group the split by that entity as well (problem
  011), and for fraud a time-based split is usually the honest one.
- Refusing to split stops the pipeline. That is the point, but someone must
  own the decision of what to do instead (wait for more labels, reuse a model
  from a similar market).
- Libraries do this for you (`train_test_split(stratify=y)`,
  `StratifiedKFold` in scikit-learn, and recent versions warn about undefined
  recall); the minimum-positives check and the hard stop on undefined metrics
  usually still need to be added in the pipeline.

## Run
```bash
python -m unittest -v test_solution.py
```
