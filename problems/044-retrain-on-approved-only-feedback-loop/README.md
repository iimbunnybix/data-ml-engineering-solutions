# 044 · Fraud on new devices from foreign IPs spiked every other week, right after each "routine" model retrain

## Problem
A card-payments fraud model scores every transaction and declines segments
whose estimated fraud rate is above 10%. It is retrained after every labelling
period on the chargebacks that came back. At launch it learned that
transactions from a new device on a foreign IP are about 30% fraud, and
declined them. After the next retrain it approved them again, the next
period's chargebacks spiked, the retrain after that declined them again, and so
on. The fraud team saw a sawtooth: a large loss on every second period, and
each retrain passed its offline evaluation because it was evaluated on the
same labelled (approved-only) data.

In the deterministic simulation (3 segments with true fraud rates of 1%, 3%
and 30%, 10,000 transactions per segment per period, seed 7), the original
retrain declines and approves the risky segment on alternating periods. On
each approving period it lets through about 3,400 frauds; over 6 periods it
lets through 11,455.

## Root cause
**The model only learns from the traffic it approved.** Chargeback labels
exist only for approved transactions, so as soon as the model declined the
risky segment, that segment disappeared from the next training set. The
retrain had no rows for it and fell back to the overall labelled fraud rate,
about 2%, which is low precisely *because* the risky traffic had been
declined. "No data" was read as "average risk". This is a selection-bias
feedback loop: the model's own decisions decide which labels it gets.

## Solution
1. **Exploration slice.** `apply_policy` approves a random 5%
   (`EXPLORATION_RATE`) of the transactions it would decline, so every
   segment keeps producing labels. In the simulation that is about 500
   labelled risky transactions per period.
2. **Log the propensity.** Each decision records the probability that the
   policy would approve a transaction like it: 1.0 for normal approvals, 0.05
   for explored ones.
3. **Inverse-propensity weights.** `weighted_fraud_rate` counts each labelled
   transaction as `1 / propensity` transactions, so the explored slice stands
   in for all the declined traffic. On one simulated period, the true fraud
   rate of all traffic is 11.3%; the weighted estimate from labelled rows is
   12.6%, while the plain labelled rate is 2.8%.
4. **No data keeps the old estimate.** `weighted_retrain` keeps the previous
   model's value for a segment with no labels instead of using the average.

With the fix, the risky segment stays declined on all 6 retrains and total
fraud let through over 6 periods is 3,231 instead of 11,455.

## Trade-offs
- Exploration has a real price: about 150 frauds per period are approved on
  purpose in the risky segment. The rate should be set from that cost versus
  the cost of a model that drifts blind; many teams use 1-5% and cap it by
  amount.
- Weights of 1 / 0.05 = 20 make estimates from a small slice noisy (12.6% vs
  11.3% above). Clipping weights, or a larger exploration slice on new or
  changing segments, trades bias for variance.
- Propensities must be logged at decision time. Reconstructing them later
  from model versions is error-prone.
- Alternatives: reject inference (estimating labels for declines from
  similar approved traffic), or keeping a small always-approve holdout of
  random customers for evaluation. Both still need some unbiased labels.
- Exploration on payments may need sign-off from risk and compliance, and
  very high-risk patterns (stolen-card lists) should be excluded from it.

## Run
```bash
python -m unittest -v test_solution.py
```
