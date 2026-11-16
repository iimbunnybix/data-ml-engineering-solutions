# 056 · Marketing cut ad spend because "customers stay 6 months", but half of them stay longer than 9

## Problem
A subscription meal-kit company sets its maximum cost per acquired customer
from lifetime value: monthly margin × average customer lifetime. The
analytics team computed the lifetime as the average tenure of customers who
had cancelled, which came out at **6.3 months**. Marketing used it to cap
bids and paused channels that looked unprofitable.

Two years later the oldest cohorts were clearly staying longer than that.
In the test data (24 monthly cohorts of 500 customers, 12,000 in total,
generated with a true 8% monthly churn), the real numbers are a **median
lifetime of 9 months** and a **mean of 12.5 months**, so the reported value
was about half the mean and the bid cap was far too low.

## Root cause
Lifetime data is **right-censored**: for 5,001 of the 12,000 customers we
only know "still active after N months", not when they will leave. Averaging
only the 6,999 cancelled customers drops exactly the long-lived ones, and
young cohorts (observed for a month or two) can only contribute short
lifetimes, so the average is biased short. The mirror mistake, averaging the
tenure of current customers only, measures how long ago people signed up,
not how long they stay (8.8 months here, also wrong).

## Solution
`kaplan_meier` builds a survival curve: for each month t it takes every
customer observed for at least t months as "at risk", counts how many of
them cancelled in month t, and multiplies `1 - cancelled / at_risk` into the
running survival probability. An active customer observed for 5 months
counts in the denominator for months 1-5 and is then simply no longer at
risk; nobody is dropped and nobody is assumed to cancel.

- On the test data the curve matches the true one (`0.92 ** t`) within
  0.002 at 3, 6 and 12 months, and `median_lifetime` returns **9**, the
  true median.
- If more than half of the customers are still active at the end of the
  data, `median_lifetime` returns `None` instead of a number: the data
  cannot answer yet (a young business with 6 months of history and 5%
  churn, in the tests).
- A small hand-checked example in the tests shows the arithmetic.

## Trade-offs
- Kaplan-Meier assumes censoring is unrelated to the chance of churning.
  If, say, a promotion cohort is both newest and more likely to churn,
  estimate per cohort or with a model that includes cohort.
- It gives a median reliably, but a **mean** only up to the longest
  observed tenure (the tail is unknown). LTV usually needs a parametric fit
  (for example geometric or Weibull) or a capped "restricted mean" lifetime.
- Monthly buckets are coarse; with daily data the same estimator works per
  day.
- Survival libraries (lifelines, R's `survival`) add confidence intervals
  and covariates; this standard-library version only shows the estimator.

## Run
```bash
python -m unittest -v test_solution.py
```
