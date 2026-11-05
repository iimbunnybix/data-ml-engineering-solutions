# 014 · The new chargeback model's merchant feature scored 0.77 AUC in validation, and 0.55 in production

## Problem
A payments team added `merchant_id` to its chargeback-risk model. There are
thousands of merchants with a long-tail distribution: a few big merchants
produce most transactions, and most merchants appear only a handful of times.
One-hot encoding was too wide, so a data scientist used **target encoding**:
replace each merchant with its historical chargeback rate.

Offline, the feature alone reached a validation AUC of 0.77, and the model
was approved. In production the same feature was close to useless (0.55), the
risk team's review queue filled with false positives, and real chargebacks
from new merchants went through. Nobody could reproduce the offline number.

## Root cause
**Target leakage through the encoding.** The per-merchant averages were
computed over the whole dataset *before* the train/validation split, so every
validation row's own label was inside its feature value. For a merchant seen
once, the encoding *is* that row's label (0.0 or 1.0). The model learned "the
feature equals the answer", which can never be true for new transactions.

In the simulation (20,000 transactions over 5,000 Zipf-distributed merchants,
5% of merchants truly risky), the leaky pipeline's validation AUC was **0.77**
while the AUC on next month's transactions was **0.55**. Across 3 seeds the
gap was always above 0.17 (validation 0.77–0.79, production 0.55–0.62).

## Solution
`solution.py`:

1. **Fit the encoder on training rows only.** `TargetEncoder.fit` runs on
   the training split; validation, test and serving rows are only
   `transform`ed. Unseen merchants get the global prior.
2. **Smoothing.** Encodings are `(sum_y + m * prior) / (count + m)` with a
   pseudo-count `m` (default 20), so a merchant with 1 transaction stays close
   to the prior instead of jumping to 0 or 1.
3. **Out-of-fold encoding for the training set.** Even with a proper split,
   encoding the *training* rows with an encoder fitted on those same rows
   leaks labels into training, and the model overweights the feature.
   `out_of_fold_encode` splits training data into k folds and encodes each
   fold with an encoder fitted on the other k−1, so no row sees its own label.
4. **`auc`** is a small, tie-aware ROC AUC so the tests need no packages.

Results from the tests (seed 0): honest validation AUC 0.53 vs production
0.55, within 0.03, so offline evaluation now predicts what production sees.
The training feature's in-sample AUC is 0.80 (leaked); out-of-fold it is
within 0.03 of production.

## Trade-offs
- The honest result is less exciting: in this data `merchant_id` carries
  only a weak signal. That is the point; the alternative was shipping a
  model on a number that did not exist.
- Smoothing strength `m` is a hyperparameter. Too low lets noise from rare
  merchants through; too high erases real differences between big merchants.
  Tune it with out-of-fold validation, not on the test set.
- Out-of-fold encoding costs k encoder fits and makes training features
  slightly noisier than serving features (which use all training data).
  This is the accepted trade-off (it is what scikit-learn's
  `TargetEncoder.fit_transform` does with internal cross fitting).
- For time-ordered data, use time-based folds or an expanding window: encode
  each transaction only with labels known before it (see 003 for
  point-in-time joins). Chargeback labels arrive weeks late, so only use
  labels that were available at prediction time.
- Alternatives: count/frequency encoding (no label used), hashing, or
  learned embeddings.

## Run
```bash
python -m unittest -v test_solution.py
```
