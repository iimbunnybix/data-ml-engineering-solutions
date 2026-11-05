# 024 · Fraud scores for iOS users jumped to the Android level whenever a serving batch had no Android traffic

## Problem
A fraud model scores card payments in micro-batches. One of its inputs is
the `device` category, one-hot encoded into `android`, `ios` and `web`
columns. Offline evaluation looked fine, but in production the score for the
same payment changed depending on which other payments happened to be in the
batch. Late at night, when batches often had no Android traffic, iOS
payments were scored with the Android weight and web payments with the iOS
weight. When a new `tv` app launched, its traffic shifted the columns again.
No error was raised; the model just returned wrong numbers.

## Root cause
Serving built the one-hot columns from the categories present **in the
current batch** (`sorted(set(batch))`), while the model's weights are tied
to the **training-time** column positions. In the tests (weights android
0.5, ios 0.1, web 0.9):

| Batch | Columns built at serving | Scores returned | Correct scores |
|---|---|---|---|
| `[ios, web]` | `[ios, web]` | `[0.5, 0.1]` | `[0.1, 0.9]` |
| `[web, tv]` | `[tv, web]` | `[0.1, 0.5]` | `[0.9, 0.7]` (tv = unknown) |

The scoring loop used `zip(weights, row)`, which silently stops at the
shorter input, so a wrong number of columns never raised an error either.

## Solution
- **Save the vocabulary with the model.** `Vocabulary.fit` records the
  sorted training categories; serving encodes every batch with it, so the
  width and order never depend on the batch.
- **Explicit unknown bucket.** Values not seen in training map to a final
  `device=__unknown__` column that has its own learned weight, instead of
  creating a new column. `unknown_share` reports how much traffic falls into
  it, so a new app or a renamed value is visible.
- **Column-name check before scoring.** The model stores its column names;
  `score` compares count, names and order with the caller's columns and
  raises `ColumnMismatch` on any difference, for example a vocabulary from a
  different training run.

## Trade-offs
- The unknown bucket only has a sensible weight if training data contained
  some rare or held-out categories mapped to it; otherwise its weight is
  arbitrary. Collapsing rare training categories into it is a common fix.
- Retraining with new categories changes the column list, so the model and
  its vocabulary must be deployed together as one artifact (or the feature
  pipeline must be part of the model, as in scikit-learn `Pipeline` or a
  TF/ONNX graph).
- Feature hashing avoids a vocabulary entirely with a fixed width, at the
  cost of collisions and harder debugging.
- Checking names on every row costs a little CPU; checking once per batch
  is enough when all rows come from the same encoder.

## Run
```bash
python -m unittest -v test_solution.py
```
