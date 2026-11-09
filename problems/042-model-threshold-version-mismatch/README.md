# 042 · The fraud review queue went from 2% to 20% of transactions the morning a retrained model went live

## Problem
A card-payments fraud service scores every transaction and sends those
scoring above a threshold to a manual review team sized for about 2% of
traffic. The model file is deployed by the ML platform's pipeline; the
threshold lives in a separate config store owned by the risk team. When the
retrained model v7 was rolled out, the threshold stayed at the value tuned
for v6. Within an hour the review queue was ten times its normal size,
legitimate payments were held for hours, and customers abandoned checkouts.
During the rollback the risk team pushed v7's threshold to the config store
first, so for a while v6 ran with v7's threshold and almost nothing was
reviewed, letting fraud through.

On the seeded fixture (20,000 transactions per sample):

| Model | Threshold | Review rate |
|---|---|---|
| v6 | v6's (0.539) | 2.01% |
| v7 | v6's (0.539) | 19.83% |
| v6 | v7's (0.930) | 0.02% |
| v7 | v7's (0.930), deployed as a bundle | 2.06% |

## Root cause
A threshold is only meaningful for the exact model whose score distribution
it was tuned on. Retraining (new features, scaling, class weights or
regularisation) moves that distribution, even when the model ranks
transactions just as well. The system treated the model and its threshold as
two independent settings, deployed by different pipelines at different
times, with nothing recording which model a threshold belonged to and
nothing checking the resulting review rate before taking traffic.

## Solution
- **One bundle per model version.** `build_bundle` tunes the threshold for a
  specific model on a tuning sample and packages the model, threshold,
  expected review rate and a SHA-256 **fingerprint of the model's weights**.
- **Verify before serving.** `verify_bundle` recomputes the fingerprint (a
  threshold copied from another model fails), then scores a fixed reference
  sample and fails if the review rate differs from the expected 2% by more
  than 0.5 points. A threshold tuned only on high-amount transactions gives
  0.38% on the reference sample and is rejected.
- **Atomic swap.** `FraudService.deploy` verifies first and then replaces the
  whole bundle in one assignment; a failed check leaves the old bundle
  serving. Each request reads the bundle once, so the model and threshold
  always come from the same version, and rollback means redeploying the
  previous bundle.
- `NaiveFraudService` keeps the original, independently updated design for
  comparison.

## Trade-offs
- The fingerprint proves the threshold and model belong together, not that
  the model is good; offline evaluation and canary analysis are still needed.
- The reference-sample check only guards the review *rate*. If live traffic
  shifts, the rate moves anyway; monitor the live rate and alert on it too.
- Tying the threshold to the model means a business change (say, the review
  team doubles) needs a new bundle version rather than a config edit. Some
  teams instead store a target *rate* in config and let the service derive
  the threshold from the bundle's score quantiles.
- Calibrating scores (Platt scaling, isotonic regression) makes thresholds
  more stable across retrains but does not remove the need to ship them
  together.

## Run
```bash
python -m unittest -v test_solution.py
```
