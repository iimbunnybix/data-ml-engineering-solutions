# 026 · A fraud model retrained on a "random sample" was great on Asia-Pacific traffic and useless everywhere else

## Problem
A payments company retrains its fraud model weekly. The training job reads a
daily transaction export (tens of millions of rows, too large to load into
memory) and needs a sample of 50,000 rows. After a pipeline rewrite, offline
metrics on the validation set stayed fine, but in production the new model's
precision dropped sharply in Europe and the Americas while Asia-Pacific
looked better than ever. Fraud losses rose for two weeks before the model was
rolled back.

## Root cause
The rewritten job sampled with the equivalent of `head(50_000)`
(`naive_head_sample`). That is only a random sample if the file is in random
order, and exports almost never are: this one is written region by region,
and by time within each region. The first 50,000 rows were all APAC traffic
from the early hours of the day. The validation set was built the same way,
so it shared the bias and hid it. In the test, a file written as 4,000 APAC,
then 4,000 EMEA, then 4,000 AMER rows gives a 1,000-row "sample" that is
100% APAC.

## Solution
**Reservoir sampling** (Algorithm R), `reservoir_sample(rows, k, rng)`:

1. Keep the first `k` rows.
2. For row number `n > k`, pick `slot = rng.randrange(n)`; if `slot < k`,
   replace that reservoir slot with the row. That happens with probability
   `k / n`.
3. By induction, after `n` rows every row is in the reservoir with
   probability exactly `k / n`, regardless of the file's order.

It is one pass, keeps only `k` rows in memory, and does not need to know the
total row count in advance. With seed 1 the test sample's region shares are
33.5% / 33.6% / 32.9%, and over 20,000 runs on a 10-row stream every position
is chosen within 5% of its expected frequency. A fixed seed makes the
training set reproducible.

## Trade-offs
- A uniform sample keeps rare classes rare. For fraud (well under 1% of
  rows) you usually want a *stratified* sample: one reservoir per class or
  per region, then reweight. Uniform is the honest baseline to start from.
- Algorithm R calls the random generator once per row. For very long
  streams, Algorithm L skips ahead geometrically and needs far fewer random
  numbers.
- In a distributed job, sample each partition with its own reservoir and
  merge by assigning each row a random key and keeping the `k` smallest keys
  overall (a weighted variant handles uneven partitions).
- SQL engines offer `TABLESAMPLE`, but `SYSTEM` sampling picks whole blocks,
  which carries the same ordering bias at block level; `BERNOULLI` is
  row-level.

## Run
```bash
python -m unittest -v test_solution.py
```
