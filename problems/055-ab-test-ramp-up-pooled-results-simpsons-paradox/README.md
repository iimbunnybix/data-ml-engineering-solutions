# 055 · The new checkout "won" its A/B test by 28%, then conversion dropped after it shipped to everyone

## Problem
An online retailer tested a redesigned checkout. To limit risk, the
experiment platform sent **10%** of users to the new checkout in week 1 and
ramped to **50%** in week 2. Week 2 happened to be a holiday sale, when
everyone converts about twice as often.

The experiment dashboard pooled both weeks per arm and reported:

| Arm | Users | Conversions | Rate |
|---|---|---|---|
| control | 140,000 | 19,000 | 13.57% |
| new checkout | 60,000 | 10,450 | 17.42% |

A **+3.85 point (+28% relative)** lift. The new checkout shipped to 100% of
traffic and conversion went down. Looking week by week, the new checkout
had lost **in every week**: 9.5% vs 10.0% in week 1 and 19.0% vs 20.0% in
week 2.

## Root cause
Pooling is only valid when every arm sees the same mix of time periods. With
the ramp, 83% of the treatment users (50,000 of 60,000) came from the sale
week, against 36% of control users (50,000 of 140,000). The pooled treatment
rate is mostly a "sale week" rate and the control rate is mostly a "normal
week" rate, so the comparison measured the sale, not the checkout. This is
Simpson's paradox: the sign of an aggregated difference flips relative to
every subgroup because the subgroups are mixed in different proportions.

## Solution
- `allocation_changed` flags an experiment whose treatment share differs
  between phases (here 10% vs 50%), which is when pooled numbers must not be
  shown.
- `stratified_difference` compares the arms **inside each phase**, where the
  split was constant, and combines the per-phase differences with weights
  equal to each phase's total traffic. Both arms get the same weights, so the
  phase mix can no longer leak into the result.
- It also returns a standard error and a 95% interval (normal approximation
  for a difference of proportions). On the test data the result is
  **-0.75 points** (interval about -1.14 to -0.36), so the new checkout
  is a significant loss, not a win.
- Phases where one arm had no users (a paused arm) are excluded and listed
  in `excluded_phases` rather than silently pooled.

With a constant 50/50 split the pooled and stratified numbers are identical,
which the tests check.

## Trade-offs
- Stratifying needs the phase (or day) of every exposure. Logging the
  allocation version alongside the assignment makes this possible.
- Weighting by traffic is one choice; inverse-variance weighting gives a
  tighter interval but answers a slightly different question. The
  Cochran-Mantel-Haenszel test is the standard alternative for stratified
  binary outcomes.
- The simplest safe option is to analyse only the period after the final
  ramp, at the cost of throwing away the early data.
- Stratifying by phase removes bias from time effects that change between
  phases, not from effects within a phase (for example novelty that fades
  over days). Long-running tests still need a look at the effect over time.

## Run
```bash
python -m unittest -v test_solution.py
```
