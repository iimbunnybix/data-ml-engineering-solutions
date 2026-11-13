# 050 · The demand forecast beat every baseline offline, then missed by noticeably more in production from day one

## Problem
A grocery chain forecasts tomorrow's orders per store to plan staffing and
fresh-food stock. The new model's headline feature was a "7-day rolling mean
of orders". In the offline backtest over a year of daily data (simulated in
the test), the simplest model "forecast = feature" had a mean absolute error
of **15.7 orders/day**. After launch, real errors were consistently higher,
and the team spent weeks looking for training/serving bugs in the feature
store. A correct backtest of the same feature gives **18.1 orders/day**:
the offline number was about 13% too optimistic, enough to have picked this
model over the existing one.

## Root cause
The rolling mean was copied from an analysis notebook that used a
**centered** window (`rolling(7, center=True)` in pandas): 3 days before,
the day itself, and 3 days after. Centering is fine for smoothing a chart,
but as a feature it averages in the very value being predicted plus three
days that are still in the future at prediction time. Every training row
leaked target information, so offline error was flattering. In production
those days do not exist yet, so the feature cannot be the same thing.

## Solution
- `trailing_features` builds the window from the 7 days **strictly before**
  the target day: exactly what the pipeline can compute the evening before
  the forecast. Offline and online values now match.
- Each `FeatureRow` records the day indexes it read, and `leaking_rows`
  flags any row whose inputs include the target day or later. It flags every
  centered row and none of the trailing rows; run it as a guard in the
  training pipeline.
- The tests compare offline error for both windows on the same seeded data,
  so the size of the leak is measured, not assumed.

## Trade-offs
- The trailing mean lags trend changes by about half a window. That lag is
  real and the model has to live with it; shorter windows or exponential
  weighting react faster but are noisier.
- Leakage can be subtler than a centered window: features computed from a
  table that is updated later (refunds, corrected orders) also leak. The
  `days_used` record only catches what it is told about; point-in-time joins
  (problem 003) cover the table case.
- A shifted window (`rolling(7).mean().shift(1)` in pandas) is the usual
  one-line fix; keeping the explicit day list makes the guard possible.

## Run
```bash
python -m unittest -v test_solution.py
```
