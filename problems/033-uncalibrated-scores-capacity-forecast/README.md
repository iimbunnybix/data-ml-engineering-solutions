# 033 · The retention team staffed for 13,400 churners and 6,000 left; most customers flagged "more likely than not to leave" stayed

## Problem
A subscription business with about 50,000 customers up for renewal each month
plans its retention call centre from the churn model. Two rules were built on
the model's scores:

- **Capacity forecast:** "expected churners = sum of the scores", used to book
  agent hours and the discount budget.
- **Targeting:** "offer the save discount to everyone with a score above 0.5",
  because those customers are "more likely than not to leave".

In the replay in `test_solution.py`, the summed scores forecast **13,395
churners** for a month in which **6,016** actually churned (2.2×). The 0.5 rule
selected **11,551** customers, of whom only **3,400 (29%)** churned. So the
team paid for twice the agent hours it needed and gave discounts to about
8,000 customers who were staying anyway. The model's ranking was fine and its
AUC had passed review, which is why nobody questioned the numbers.

## Root cause
**The model's scores were treated as probabilities, but they were never
calibrated.** A score that ranks customers well (high AUC) only says that one
customer is riskier than another. It does not say that, of customers scored
0.6, 60% will leave. Boosted trees, heavily regularised models and models
trained on resampled or reweighted data all routinely produce scores whose
scale is off. Here the model is overconfident: its log-odds are 2.5× the true
log-odds plus a shift, so it reports an average score of 0.27 for a month with
a 12.0% churn rate. Summing scores and cutting at 0.5 both depend on the scale,
not just the order, so both were wrong. On the raw scores the expected
calibration error (ECE, 10 bins) is **0.156** and the Brier score is
**0.154**.

## Solution
**Fit a calibration map on a held-out month with known outcomes, and plan
only with calibrated probabilities** (`IsotonicCalibrator` in `solution.py`):

1. Take a calibration set the model was *not* trained on (here, last month's
   50,000 scored customers and whether they churned).
2. Fit **isotonic regression** with pool-adjacent-violators: sort by score,
   start with one block per distinct score, and merge neighbouring blocks
   whenever a higher-score block has a lower churn rate than the one before.
   The result is a non-decreasing step function from raw score to observed
   churn rate (69 steps here). It never reorders customers, and it assumes
   nothing about the shape of the miscalibration.
3. Map the new month's raw scores through it, then sum and threshold.
4. Measure with **ECE** (average gap between predicted and observed rate per
   score bin, weighted by bin size) and **Brier score** (mean squared error of
   the probability).

On the next month:

| | raw scores | calibrated |
|---|---|---|
| forecast churners (actual 6,016) | 13,395 | 6,002 |
| customers above 0.5 | 11,551 | 838 |
| of those, churned | 3,400 (29%) | 502 (60%) |
| ECE | 0.156 | 0.006 |
| Brier score | 0.154 | 0.092 |

With calibrated probabilities "above 0.5" means what the targeting rule
intended, and the forecast is within 0.3% of the real count (6,002 vs 6,016).

## Trade-offs
- **Needs fresh labelled data that the model did not train on.** Calibrating
  on training data learns the model's overfitting, not the real rates. The map
  must be refitted when the model is retrained or when churn behaviour drifts.
- **Isotonic regression is a step function** and can overfit with a small
  calibration set (a few hundred positives): steps become noisy, and the
  lowest or highest block can be exactly 0 or 1. Alternatives: **Platt
  scaling** (a logistic fit on the log-odds, two parameters, smooth, works
  with little data but assumes the miscalibration has that shape), or
  **histogram binning** (simplest, but loses resolution within a bin).
- Calibration is about averages. A forecast from summed probabilities is
  still a random quantity (here the standard deviation is about
  √(Σ p(1−p)) ≈ 68 customers) and says nothing about which customers churn.
- Calibration fixes the scale, not the ranking: it cannot make a weak model
  separate churners better, and ties in a step reduce resolution slightly.
- If only the base rate changed (for example, the model was trained on a
  balanced sample), a prior-shift correction is enough (see problem 018).
  Here the error depends on the score, so a single log-odds shift cannot fix it.
- ECE depends on the number of bins and hides errors in sparse bins; look at
  the reliability table as well as the single number.

## Run
```bash
python -m unittest -v test_solution.py
```
