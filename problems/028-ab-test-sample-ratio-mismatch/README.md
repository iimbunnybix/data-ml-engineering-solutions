# 028 · A "9% conversion lift" from a new checkout page vanished after launch because the 50/50 test was really 52/48

## Problem
An online store ran a 50/50 A/B test of a redesigned checkout page on
100,000 users. The treatment page lived on a new URL, so treatment users
were sent through a redirect. After two weeks the experiment dashboard
showed treatment converting at 5.48% against 5.02% for control, a +9.2%
relative lift with p = 0.0013. The team shipped it. Over the next month
site-wide conversion did not move, and the product review asked why the
"significant winner" did nothing.

## Root cause
**Sample ratio mismatch (SRM) from biased tracking loss.** The redirect
sometimes navigated away before the experiment's exposure event was sent.
Users who bounced quickly during the redirect were never logged; users who
went on to buy were logged again on the checkout page. So the missing users
were almost all non-buyers, all from one variant.

In the seeded simulation (100,000 users, true conversion 5% in *both*
variants, 8% of non-buying treatment users lost):

| | Control users | Treatment users | Lift | p-value |
|---|---|---|---|---|
| What really happened | 50,045 | 49,955 | +0.8% | 0.78 |
| What the pipeline logged | 50,045 | 46,097 | +9.2% | 0.0013 |

3,858 non-buyers disappeared from treatment, which pushed its conversion
rate up. Nobody looked at the user counts: the split was 52.05% / 47.95%
instead of 50 / 50, and that alone shows the data is broken.

## Solution
Run an **SRM check before anyone can read the results** (`read_results`):

1. `chi_square_statistic` computes Pearson's chi-square goodness of fit of
   the observed user counts against the configured split:
   `sum((observed - expected)^2 / expected)`.
2. With two variants there is 1 degree of freedom. A chi-square with 1 dof is
   the square of a standard normal, so the p-value is
   `erfc(sqrt(chi_square / 2))` using only `math` (it gives 0.05 at 3.841 and
   0.001 at 10.828, matching standard tables).
3. If p < 0.001, `read_results` raises `SampleRatioMismatchError` with the
   observed shares, so the experiment cannot be read until the logging bug is
   found.

The biased experiment gives chi-square 162.1, p = 3.9e-37: blocked. The
healthy one gives chi-square 0.081, p = 0.78: readable (and not
significant). The check scales with sample size: 50.6% / 49.4% is flagged at
100,000 users (chi-square 14.4, p = 0.00015) but is normal noise at 10,000
users (chi-square 1.44, p = 0.23). On 2,000 fair simulated splits, p < 0.05
fires about 5% of the time, as it should.

## Trade-offs
- The 0.001 threshold is strict on purpose. SRM is checked on every
  experiment, often every day, so a loose threshold would block many healthy
  tests by chance. Real SRMs from bugs usually have tiny p-values.
- SRM detects that *something* is wrong, not what. Common causes: redirects,
  bot filtering that affects one variant more, crashes in new code, and
  assignment done after an eligibility filter. Check SRM per segment
  (browser, country, day) to find the cause.
- No SRM does not prove the data is unbiased: a bug that drops users equally
  from both variants, or that changes behaviour without changing counts, is
  invisible to this check.
- This version handles two variants (1 degree of freedom). For k variants,
  use k - 1 degrees of freedom and the regularized incomplete gamma function
  (for example `scipy.stats.chisquare`).
- Fixing an SRM means fixing the logging and re-running the experiment;
  re-weighting the logged users cannot recover the users who were never
  recorded.

## Run
```bash
python -m unittest -v test_solution.py
```
