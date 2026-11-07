# 037 · Global checkout conversion "jumped" from 3.1% to 4.1% the week a tiny new market launched

## Problem
An e-commerce company's growth dashboard shows weekly checkout conversion per
country and globally. A nightly job writes one row per (country, day) with the
day's conversion *rate*, and the weekly tiles average those rows. The week a
new market launched (a handful of visits per day), global conversion went from
3.1% to 4.1%. The growth team presented it as a win, and a planned checkout
experiment was paused because "conversion is already up a full point". Nothing
had changed in the big markets. A week later the number fell back and nobody
could explain either move.

## Root cause
**An average of ratios is not the ratio of the totals.** Averaging daily rates
gives every (country, day) the same weight, whatever its traffic. The new
market had one day with 1 conversion out of 2 visits: a 50% "rate" that counted
as much as a day with 30,000 visits in India. A second, quieter version of the
same bug: a tracking outage day with 0 visits was stored as 0.0%, which halves
an average of two days instead of having no effect.

In the simulation (28 country-days, 414,616 visits, big markets converting at
3.1%), averaging the daily rates gives **4.11%**; summing conversions and visits
gives the true **3.10%**. Without the new market, both methods give 3.10%.

## Solution
`solution.py`:

- **Store the parts, not the rate.** `to_parts` keeps `conversions` and
  `visits` for each (country, day) in a `RatioParts`. These are additive, so
  any rollup (per country, global, per week) is just a sum.
- **Divide once, at the end.** `rollup` sums parts per level and `rates`
  divides only for display. Rolling up in two steps (days → countries →
  global) gives exactly the same totals as one step, which is not true for
  averaged rates.
- **No visits means no rate.** `RatioParts.rate` returns `None` when visits is
  0, so an outage day does not become a fake 0% and an empty group is shown as
  "no data" instead of 0%.
- **Validate inputs.** `DailyFunnel` rejects negative counts and more
  conversions than visits, which usually means a broken join.

## Trade-offs
- The weighted rate is dominated by big markets. That is correct for "global
  conversion", but a small market's own trend still needs its own tile.
  Show it next to its traffic (or a confidence interval) so a 1-in-19 week is
  not read as a reliable 5.3%.
- Some questions do want an unweighted mean, for example "how is the typical
  country doing?". Make that a separate, clearly named metric, not the rollup
  of the global one.
- Ratio metrics with non-additive denominators (distinct users, medians) cannot
  be summed this way; store sketches instead (see system-design-solutions 037
  for distinct counts).
- Keeping parts means two columns instead of one and a small change for every
  dashboard query that read the stored rate.

## Run
```bash
python -m unittest -v test_solution.py
```
