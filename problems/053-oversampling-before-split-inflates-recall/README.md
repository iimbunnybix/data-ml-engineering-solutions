# 053 · Fraud model passed review with 100% offline recall, then caught only 38% of fraud in production

## Problem
A payments team trains a card-fraud model on 8,000 labelled transactions,
of which 245 (about 3%) are fraud. To stop the model ignoring the rare
class, the training notebook balances the data with random oversampling,
duplicating fraud rows until there are as many fraud as legitimate rows,
and then splits the balanced data 75/25 into train and test.

The validation report showed **100% fraud recall** on the test fold, and the
model was approved. On 8,000 fresh transactions after launch it caught
**38.5%** of fraud (95 of 247). Chargebacks did not fall, and the risk team
lost a quarter re-examining a model that had never been as good as the
report said.

## Root cause
Oversampling ran **before** the split. Each of the 245 fraud rows was copied
about 32 times on average, and the shuffle scattered the copies across both folds.
In the seeded run every one of the 245 source fraud rows had copies in
both train and test. A model that memorises training rows (here a
1-nearest-neighbour model; deep trees and boosted ensembles behave
similarly on rows they have seen) simply looks up the copy, so test recall
measures memory, not generalisation.

A second distortion: the test fold was 50% fraud (1,938 of 3,878 rows),
not the real 3%, so precision and alert-volume estimates from it were also
meaningless.

## Solution
- `fixed_prepare` splits the **original** rows first, then calls
  `oversample_minority` on the training fold only. The test fold keeps the
  real class mix (70 fraud of 2,000 rows).
- `assert_no_shared_rows` fails the job if any source `row_id` appears in
  both folds, catching this mistake if someone reorders the steps again.
- With the fix, offline recall is **37.1%** (26 of 70) and recall on fresh
  production traffic is **33.6%** (83 of 247): the offline number now
  predicts production. The model is not better, but the decision to ship
  it is now made on an honest number.

## Trade-offs
- The same rule applies to any resampling or synthetic-sample method
  (SMOTE, undersampling) and to cross-validation: resample inside each
  training fold, never before `KFold`. In scikit-learn, put the sampler in
  an `imblearn` pipeline so it runs per fold.
- Oversampling is often unnecessary; class weights or adjusting the decision
  threshold achieve the same balancing without creating duplicate rows.
- The guard checks `row_id`, so it cannot spot near-duplicates (the same
  card retrying a payment). Those need a group split by card or customer
  (see problem 011).
- A small honest test fold (70 fraud rows) gives a noisy recall estimate;
  report a confidence interval or use repeated cross-validation.

## Run
```bash
python -m unittest -v test_solution.py
```
