# 069 · Fraud model rated first-time buyers as long-standing customers, and "days since last purchase" was sometimes negative

## Problem
A marketplace's fraud model scores every card transaction. Two of its
strongest features are **the amount of the customer's previous transaction**
and **days since that previous transaction**: a stolen card used for the
first time usually has no history, or a sudden jump from small to large
amounts.

Fraud losses on first purchases rose after the feature job was rewritten in
a "simpler" way, while the model's offline metrics looked fine. A data
scientist plotting `days_since_previous` found negative values, which cannot
happen for one person's purchase history.

## Root cause
`naive_lag_features` sorted the whole table by `(customer, time)` and took
**the row above** as "previous". That is a lag window without a partition:
at every customer boundary, the row above belongs to someone else.

```
cust      time   amount   previous_amount (naive)
alice     day 0    900    -
alice     day 30   950    900
bob       day 2     40    950   <- alice's purchase, days_since = -28
bob       day 5     60     40
```

Every customer's first transaction (except the very first customer in the
table) got a stranger's amount and a random, often negative, gap, and was
marked as a repeat purchase. The model learned that "previous amount"
was noisy for low-history rows and could not see "this is a first
purchase" at all.

In the seeded replay in `test_solution.py` (1,000 customers, 3,613
transactions, shuffled), **999 of 1,000** first transactions borrow another
customer's history and **749** of them get a negative `days_since_previous`.

## Solution
`lag_features` partitions the window by customer, the same thing as
`LAG(amount) OVER (PARTITION BY customer_id ORDER BY ts, id)` in SQL:

1. **Group by customer** and sort each group by time, with the transaction id
   as a tie breaker so equal timestamps give the same result on every run.
2. **Lag inside the group only** (`lag_features_for_customer`). The first
   transaction gets `None` for both features, never `0`, which would look
   like a real tiny amount.
3. **Add `is_first_transaction`**, so "no history" is an explicit input the
   model can learn from.

On the replay, no first transaction borrows history, no gap is negative, and
the output is the same for any input order.

## Trade-offs
- Grouping needs each customer's full history in one place. In Spark or a
  warehouse this is a shuffle by customer; very large customers (marketplace
  sellers, test accounts) can skew it.
- The online scoring path must compute the same features from the same
  history (a "last transaction per customer" lookup). If online and offline
  definitions differ, you get training/serving skew; test both against the
  same fixtures.
- `None` must be handled by the model: tree models usually accept missing
  values, linear models need an imputed value plus the flag.
- A first-purchase flag only knows about history inside the job's input. If
  the job reads 90 days, a customer who last bought 100 days ago looks new;
  pick the lookback on purpose and document it.

## Run
```bash
python -m unittest -v test_solution.py
```
