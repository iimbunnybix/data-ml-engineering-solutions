# 047 · The courier budget came in 15% short every week, although the cost model's accuracy looked fine

## Problem
A delivery company forecasts next week's courier spend by predicting the cost
of every scheduled delivery and summing the predictions. Costs are skewed (most
deliveries are a few dollars, a few long or delayed ones are expensive), so the
data science team trained a linear model on `log(cost)` and the serving code
returned `exp(prediction)`. Offline metrics on the log scale were good and
per-delivery predictions looked plausible, but finance saw the real spend beat
the forecast every single week and had to ask for extra budget each month.

In the simulation (5,000 training deliveries, 2,000 deliveries forecast for
next week, distance 1-20 km, log-scale noise with standard deviation 0.6,
seeds 47 and 4700) the real cost was **$23,506** and the naive forecast was
**$19,931, 15.2% too low**. 50.5% of deliveries cost more than their forecast:
the forecast was a median, not an average.

## Root cause
The model estimates the mean of `log(cost)`. `exp()` of the mean of a log is
the *median* of the cost, not the mean: by Jensen's inequality,
`exp(mean(log x)) <= mean(x)`, with equality only when there is no noise. For
right-skewed costs the mean is above the median, so every prediction is a bit
low. Each one looks fine on its own; when 2,000 of them are summed, the shortfall
adds up instead of cancelling. This is called retransformation bias. With normal
log-errors the gap is a factor of `exp(sigma² / 2)`, here `exp(0.18) = 1.197`.

## Solution
- `naive_predict_cost` keeps the original `exp(prediction)`.
- `smearing_factor` computes Duan's smearing estimator on the **training**
  residuals: the average of `exp(log(actual) - prediction)`. It needs no
  assumption about the error distribution and is one number stored next to
  the model. Here it is 1.198, matching the lognormal theory (1.197).
- `smeared_predict_cost` multiplies `exp(prediction)` by that factor. Next
  week's forecast becomes **$23,875, 1.6% above** the real $23,506, inside
  the 3% finance accepts.
- `total_forecast_error` is the metric to monitor: the error of the summed
  forecast, which is what the business uses, not the per-row log error.

## Trade-offs
- One global factor assumes the spread of residuals is the same everywhere. If
  long deliveries are noisier than short ones (heteroscedasticity), compute the
  factor per segment (e.g. distance band or city), or model the variance.
- The smeared prediction is the right number for *sums and averages*. If the
  product needs a "typical" cost, or a quote the customer will usually not
  exceed, the median (naive) or a quantile is the correct answer. Pick per use.
- The factor must be recomputed whenever the model is retrained and stored
  with the model version; a stale factor reintroduces bias.
- Alternatives: train directly on cost with a loss that targets the mean
  (Poisson/Gamma/Tweedie regression), or rescale predictions so their total
  matches a recent actual total (calibration). Both avoid the back-transform.
- Smearing on training residuals is slightly optimistic for overfitted models;
  residuals from a validation set are safer when the model is flexible.

## Run
```bash
python -m unittest -v test_solution.py
```
