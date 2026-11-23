# 067 · The "large sizes need a fit check" rule sent every order, size S included, to manual review

## Problem
A fashion retailer's returns-risk pipeline turns each order into features
for a model and for a few hand-written rules. One rule sends orders of size
**L or larger** to a "check the fit" email flow, because large sizes are
returned far more often. On the training data, return rates were S 10%,
M 12%, L 20%, XL 30% and XXL 35%.

After the rule went live, every order got the fit-check email, including
size S, and the email's click-through collapsed. Separately, the data
scientists noticed that size barely mattered to the model, even though the
raw return rates clearly depend on it.

## Root cause
The pipeline encoded `size` with a generic "sorted unique values -> 0, 1,
2..." encoder (`naive_fit_encoding`). This is the default of many encoders,
including scikit-learn's `OrdinalEncoder(categories="auto")` and the
category codes of an unordered pandas `Categorical`. Alphabetical order is
not size order:

| Size | Alphabetical code | Real order |
|---|---|---|
| L | 0 | 2 |
| M | 1 | 1 |
| S | 2 | 0 |
| XL | 3 | 3 |

- The rule `size_code >= code("L")` became `size_code >= 0`, which is true
  for every order.
- The model saw the sequence L, M, S, XL, XXL, which is not monotonic in
  return rate. On the 1,600-order test table, the correlation between the
  size code and "was returned" is **0.09** with alphabetical codes and
  **0.20** with the real order, so a linear model throws away more than half
  of the signal.
- Codes also depended on which sizes were in the batch: without "M", "S"
  gets code 1 instead of 2.

## Solution
Treat the order of an ordinal feature as configuration, not something to
infer:

1. `SIZE_ORDER` lists the sizes from smallest to largest, and
   `fit_ordinal_encoding` turns that list into codes. A size listed twice is
   a configuration error.
2. `normalise_size` trims and upper-cases values, so `" xl "` and `"XL"` get
   the same code instead of becoming two categories.
3. `encode` / `encode_column` raise `UnknownCategory` for values not in the
   list (`"3XL"`, `"one size"`), and the column version reports all of them at
   once, so a new size in the catalogue is a visible pipeline failure rather
   than a silently guessed code.

With the declared order, the rule flags only L and XL, and the codes are
the same in every batch.

## Trade-offs
- Someone must keep `SIZE_ORDER` in sync with the catalogue. Failing on an
  unknown size is deliberate: it stops the batch, which is better than
  scoring with a wrong code, but it needs an owner who adds new sizes.
- Ordinal codes assume equal steps (S->M is as big as XL->XXL). Tree models
  do not care; for linear models, one-hot encoding or a learned per-size
  value avoids the assumption at the cost of more features.
- Clothing sizes are only ordered within one system. "38" (EU) and "M" can
  not be put on one line; map them to measurements first if both occur.
- In scikit-learn, pass `categories=[SIZE_ORDER]` and
  `handle_unknown="error"` to `OrdinalEncoder` for the same behaviour.

## Run
```bash
python -m unittest -v test_solution.py
```
