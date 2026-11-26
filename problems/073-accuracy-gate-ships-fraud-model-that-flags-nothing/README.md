# 073 · A retrained fraud model passed the 99% accuracy gate and then flagged no fraud at all for a week

## Problem
A payments company retrains its card-fraud model weekly. The pipeline
evaluates each candidate on a held-out set of 10,000 recent transactions and
promotes it automatically if accuracy is at least 99%. About 0.8% of
transactions are fraud.

One week a label-join bug left almost no fraud examples in the training data,
and the retrained model learned to answer "legit" for everything. It scored
99.2% accuracy, passed the gate and replaced the previous model. For a week
the fraud review queue was empty. The team noticed only when chargebacks
arrived, by which time the losses were already booked.

## Root cause
**Accuracy is dominated by the majority class.** With 80 fraud cases in
10,000, the constant "legit" answer is right 9,920 times: **99.2%**
accuracy, above the gate, while catching **0** fraud. The working model it
replaced (60 of 80 fraud caught, 40 false alarms) scored **99.4%**, only
0.2 points higher. A cautious model that catches only 40 of 80 with no false
alarms scores **99.6%**, the best of the three, although it misses half the
fraud. Accuracy cannot tell these apart because the rare class is the one
that matters.

This is not a leakage problem (problems 011, 014, 053) or test-set reuse
(problem 030): the metric was computed correctly on clean data. It was
simply the wrong metric for the decision.

## Solution
`release_gate` measures the positive (fraud) class directly:

- **Recall** (fraud caught / all fraud) must be at least 0.60.
- **Precision** (real fraud / all flags) must be at least 0.30, so the review
  team is not flooded with false alarms.
- A model that flags nothing gets precision 0.0, not "undefined", so it is
  rejected rather than crashing the gate or slipping through.
- An evaluation set with no fraud raises an error: there is nothing to judge
  recall on, and silently passing would repeat the incident.

Results on the same evaluation set:

| Model | Accuracy | Recall | Precision | Accuracy gate | New gate |
|---|---|---|---|---|---|
| Flags nothing | 99.2% | 0.00 | 0.00 | ship | reject (recall) |
| Working model (60 caught, 40 false alarms) | 99.4% | 0.75 | 0.60 | ship | ship |
| Cautious (40 caught, 0 false alarms) | 99.6% | 0.50 | 1.00 | ship | reject (recall) |
| Noisy (70 caught, 300 false alarms) | 96.9% | 0.88 | 0.19 | reject | reject (precision) |

The gate returns a reason ("recall 0.00 below 0.60") so the release log
says why a candidate was blocked.

## Trade-offs
- **Thresholds are business decisions.** 0.60 recall and 0.30 precision
  stand for "losses we accept" and "analyst hours we have". They should come
  from the fraud and operations teams and be revisited.
- **One threshold, one operating point.** A real model outputs scores, and
  recall/precision depend on the score cut-off. Gate on recall at the
  precision the review team can handle, or on PR-AUC, and compare with the
  current production model rather than only fixed numbers.
- **Small counts are noisy.** 80 fraud cases means one case is 1.25 points
  of recall. Use a larger or longer window for evaluation, or put a
  confidence interval on recall before blocking a release on a small
  difference.
- Balanced accuracy or MCC are single-number alternatives that also resist
  class imbalance, but they are harder to explain to the people who own the
  thresholds.

## Run
```bash
python -m unittest -v test_solution.py
```
