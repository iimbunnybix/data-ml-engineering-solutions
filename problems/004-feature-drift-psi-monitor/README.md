# 004 · Payment-risk model started declining good customers after an upstream release, and no alert fired

## Problem
A risk model scores every payment in real time. After an unrelated
release of the checkout service, the decline rate on legitimate payments
rose sharply over the next day. Infrastructure dashboards were green:
latency, error rate and throughput of the model server were all normal.
The model was "working"; it was just being fed different data.

## Root cause
The checkout service changed the `amount` field from rupees to paise
(×100). Nothing validated the *distribution* of model inputs, only that the
request schema was valid (a number is still a number). The model saw
amounts far outside anything in its training data and pushed them to the
high-risk end. Similar silent breaks: a category renamed upstream, a
default value changing from `null` to `0`, a join that starts returning
missing values.

## Solution
A per-feature **drift monitor** based on the Population Stability Index:

`PSI = Σ (actual − expected) · ln(actual / expected)` over bins, with the
standard thresholds **< 0.1 ok, 0.1-0.2 warn, > 0.2 alert**.

- **Numeric features**: bin edges are training-set deciles, so each bin
  holds ~10% of training data. Edges are de-duplicated for features with
  point masses (e.g. many zeros), and an edge equal to the training
  minimum is dropped, so no bin has an expected share of zero.
- **Categorical features**: categories under 1% share, and any category
  never seen in training, fall into an `__other__` bucket, so a renamed
  value shows up as drift instead of disappearing.
- **Missing rate** is tracked separately: a jump of more than 5 points
  alerts even if the remaining values look normal (PSI alone ignores nulls).
- Empty bins are floored at `1e-4` so PSI stays finite.
- Windows smaller than 500 rows are rejected because PSI is noisy on tiny
  samples.

Measured on the seeded test data (20,000 training rows, 5,000-row windows):

| Scenario | Feature | PSI | Level |
|---|---|---|---|
| Same distribution | amount | 0.003 | ok |
| Amount sent in paise (×100) | amount | 8.28 | alert |
| Median amount +36% | amount | 0.15 | warn |
| Country mix 60/30/10 → 20/30/50 | country | 1.08 | alert |
| 30% of amounts null | amount (missing +0.295) | 0.004 | alert |

In the unit-change scenario only `amount` alerts; unrelated features stay
`ok`, so the alert points at the broken input.

## Trade-offs
- PSI detects *input* drift, not whether the model is wrong. Pair it with
  delayed label-based monitoring (precision/recall once outcomes arrive).
- Real traffic drifts naturally (seasonality, promotions); thresholds should
  be tuned per feature and baselines refreshed after each retrain.
- Decile binning hides changes *inside* a bin and tail changes beyond the
  last edge are lumped together; KS or Wasserstein distance are
  alternatives with different sensitivity.
- The cheapest defence against unit changes is still a contract: validate
  ranges at the producer and version the schema. The monitor catches what
  the contract missed.

## Run
```bash
python -m unittest -v test_solution.py
```
