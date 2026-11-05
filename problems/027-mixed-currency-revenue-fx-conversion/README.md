# 027 · The revenue dashboard showed 35x more money than the bank received after we launched in India

## Problem
An online store reports daily revenue in US dollars on a finance dashboard.
It started selling in India and Europe, charging customers in INR and EUR.
The month after launch, the dashboard total jumped far beyond anything the
payment providers had paid out, and the CFO's monthly report had to be pulled.
An engineer "fixed" it by converting every order with today's FX rate. The
total became plausible, but last month's revenue now changed every day as
rates moved, and it never matched the books, which value each sale at the
rate of its transaction date.

## Root cause
The `orders.amount` column held numbers in the customer's currency, but the
aggregation (`naive_total`) summed them as if they were all dollars. In the
test data, four October orders (83,000.00 INR, 1,000.00 USD, 1,000.00 EUR,
41,500.50 INR) sum to **126,500.50 "dollars"**; the correct value is
**3,555.71 USD**, so the naive total is about 35.6x too high. One rupee is
about a cent, so every INR order was inflated 83x to 85x (1 / 0.0120 to 1 / 0.0118).

The first fix (`naive_total_with_todays_rate`) used one rate per currency for
all orders: **3,531.76 USD** with the November 5 rates, 23.95 USD (0.67%)
below the correct value, and a different number every day. It also skipped
currencies it had no rate for, so a GBP order simply vanished from the total.

## Solution
1. **Amounts carry their currency.** `Order` has `amount: Decimal` and
   `currency`. A float amount raises `TypeError`, because floats cannot hold
   most cent values exactly.
2. **Convert at the rate of the transaction date.** `FxRateTable.rate_on`
   binary-searches each currency's sorted rate history for the latest rate
   published **on or before** the order date. October revenue then stays
   3,555.71 USD however the rates move later.
3. **Fail loudly.** An unknown currency raises `UnknownCurrencyError`. An
   order dated before the first rate, or whose latest rate is more than
   `MAX_RATE_AGE_DAYS` (7) days old, raises `MissingRateError`. A dashboard
   that shows no number gets fixed; one that shows a wrong number gets
   trusted.
4. **Round once.** Orders are converted unrounded and the total is rounded to
   cents (banker's rounding) at the end. Three 0.50 INR orders at 0.01 total
   0.02 USD; rounding each order first would give 0.00.
5. `revenue_by_currency` keeps per-currency totals in original units so each
   payment provider's settlement report can be reconciled directly.

## Trade-offs
- Which rate is "correct" is an accounting policy, not an engineering
  choice: daily close, monthly average or the payment provider's actual
  settlement rate. Agree it with finance first. Settlement rates match the
  bank exactly but arrive days later.
- Failing the whole total on one bad row blocks the dashboard. An alternative
  is to publish the total with an explicit "N orders unconverted" warning,
  but never to silently drop or guess.
- The 7-day staleness limit allows weekends and holidays without rates. Tune
  it to how often your FX feed publishes.
- The rate table is in memory here. In a warehouse, do the same thing as an
  as-of join (`rate_date = max(rate_date <= order_date)`) and store the
  rate used next to each converted row for auditing.
- Refunds and chargebacks should be converted at the rate of the original
  sale or of the refund date, depending on the accounting policy; this
  example does not model them.

## Run
```bash
python -m unittest -v test_solution.py
```
