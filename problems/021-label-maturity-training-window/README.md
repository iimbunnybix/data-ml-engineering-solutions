# 021 · Fraud model recall dropped after every nightly retrain, while offline metrics looked fine

## Problem
A card-payments team retrains its fraud model every night on the last 90
days of transactions, labelling a payment as fraud if a chargeback exists
in the warehouse. Each new model scored well on its own validation split,
yet in production it caught less fraud than the model it replaced, and
fraud losses crept up month after month. Analysts noticed that the most
recent weeks of training data looked almost fraud-free.

## Root cause
The fraud label is a chargeback, and banks file chargebacks days or weeks
after the payment (3 to 60 days in this simulation). On the snapshot date,
recent fraudulent payments have no chargeback yet, so the training job
labels them **legitimate**. The model is taught that exactly the newest
fraud patterns are safe. Validation uses the same immature labels, so it
cannot see the problem.

Measured on the seeded simulation (500 payments/day for 240 days, 2% fraud,
chargebacks 3-60 days later, 90-day training window):

| Training window | Labelled fraud rate | True fraud rate | Learned P(fraud \| new device) | True P(fraud \| new device) |
|---|---|---|---|---|
| Naive: last 90 days | 1.19% | 1.78% | 6.80% | 10.52% |
| Fixed: 90 days ending 60 days ago | 1.98% | 1.98% | 10.97% | 10.97% |

In the naive window a third of the fraud is labelled legitimate, and a
one-feature model under-estimates the risk of a new device by about 35%.

## Solution
- `label_maturity_days` measures the chargeback delay distribution on
  transactions old enough to be fully settled (here, older than 120 days)
  and returns the smallest delay that covers 99% of chargebacks. On the
  simulation that is 60 days.
- `mature_training_set` keeps the training window the same size but ends it
  `maturity_days` before the snapshot, so every row has had time to receive
  its label. The boundary is inclusive: a payment exactly 60 days old is in.
- `naive_training_set` is kept for comparison.

## Trade-offs
- The model is trained on data at least 60 days old, so it reacts more
  slowly to brand-new fraud patterns. Teams often add faster, weaker labels
  (customer fraud reports, manual review outcomes) to cover recent weeks.
- A 99% coverage target still mislabels the slowest 1% of chargebacks;
  raising it lengthens the delay.
- The delay distribution itself changes (new card networks, dispute rule
  changes); recompute it regularly.
- Alternatives: weight recent rows by the probability their label has
  matured, or model "time to chargeback" with survival analysis.
- Validation must use the same maturity rule, or it will keep reporting the
  biased labels as truth.

## Run
```bash
python -m unittest -v test_solution.py
```
