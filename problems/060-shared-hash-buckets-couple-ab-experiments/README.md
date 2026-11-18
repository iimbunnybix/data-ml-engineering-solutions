# 060 · A button-colour test "won" with a +2.4 point conversion lift, but sales did not move at all after it shipped to everyone

## Problem
An e-commerce site runs many A/B tests at once on the same users. Two of
them ran in the same fortnight: a new **button colour** on product pages and
a **free-shipping banner** in the cart. The button test reported a lift of
about 2.4 percentage points in conversion, so the team shipped it to 100% of
traffic. Conversion did not change. A later audit showed the button colour
had no effect at all; its "win" was the banner's effect leaking in.

In the reproduction (40,000 users, 10% base conversion, banner truly worth
+3 points, button worth 0), the button's treatment group converted at 12.87%
and its control group at 10.48%: a measured lift of **+2.39 points** for a
change that does nothing.

## Root cause
Every experiment used the same assignment rule:
`treatment if hash(user_id) % 100 < 50`. The experiment id was not part of
the hash, so the same 19,916 users were in treatment for **every**
experiment. Concurrent experiments were not independent: the button
treatment group was exactly the banner treatment group, and any difference
between the groups was credited to whichever experiment was being analysed.
Sample-ratio checks still passed, because each test on its own had a clean
50/50 split.

## Solution
- **Salt the hash with the experiment id** (`in_treatment`):
  `sha256(f"{experiment_id}:{user_id}") % 100`. Each experiment gets its own
  pseudo-random split, still stable for a given user. In the test, 10,005 of
  the 20,149 button-treatment users (49.7%) see the banner, as independent
  50/50 splits should give.
- With independent splits the banner's effect lands equally in the button's
  treatment and control groups and cancels out: the button lift becomes
  **-0.04 points** (noise), while the banner is still measured at +3.61
  points.
- **Check independence before reading results**
  (`assignments_are_independent`): a Pearson chi-square test on the 2x2
  table of "in A treatment" x "in B treatment". The shared hash gives a
  statistic of 40,000 (fails); the salted hash gives 0.29, below the 3.841
  critical value for p = 0.05 (passes).

## Trade-offs
- Salting makes experiments independent, not non-interacting. If two changes
  interact (the banner works better with the new button), you need a
  factorial analysis or mutually exclusive layers for those experiments.
- Changing the salt of a running experiment reshuffles its users; salts must
  be fixed at launch and stored with the experiment config.
- Mutually exclusive "layers" (each user in at most one experiment per
  layer) avoid interaction completely but cut the traffic each test gets.
- With many experiment pairs, about 5% of independence checks fail by chance
  at p = 0.05; use a stricter threshold or correct for multiple comparisons
  when checking all pairs.

## Run
```bash
python -m unittest -v test_solution.py
```
