# 032 · After a merchant-onboarding push, customers of 220 honest shops had their cards declined as fraud

## Problem
A payment processor's fraud model uses the merchant as a feature. Over one
quarter, sales onboarded thousands of small merchants, taking the platform to
about 5,000 honest merchants plus, as it turned out later, 50 merchants run
by a card-testing ring. After the next retrain, support tickets came in from
shops whose customers were being declined although nothing about them had
changed. In the replay of one month of traffic (`test_solution.py`, about
297,000 transactions), **220 honest merchants** have their transactions
declined, which is **3.1% of all honest transactions** (8,622 payments).
Among them are 6 merchants with 200 or more transactions a month. Fraud
monitoring looked healthy, because 96% of the transactions at the ring's
merchants were still being declined.

## Root cause
**The merchant feature used the hashing trick with a table sized for a much
smaller merchant base.** `merchant_id` was mapped to `hash(merchant_id) %
1024` to avoid maintaining a vocabulary. Every merchant in a bucket looks the
same to the model, so it can only learn one risk per bucket. With 5,050
merchants in 1,024 buckets, about 5,014 merchants (99%) share their bucket
with at least one other merchant (expected value under uniform hashing). Even
at 300 merchants, about 76 (25%) already shared one.

When an honest merchant shares a bucket with a ring merchant (30% fraud), it
inherits a risk far above the 5% decline threshold. For example, `m-04732`,
with 847 transactions a month, shares its bucket with ring merchant `r-00014` and is
declined. 25 buckets hold two or more merchants with 200+ transactions each.
Nothing raised an alert, because collisions are silent by design.

## Solution
**Exact indices for merchants with enough data, hashing only for the long
tail, and a collision check** (`VocabularyWithHashedTail`):

1. Every merchant with at least 200 training transactions gets its own index
   (a vocabulary built at training time and saved with the model).
2. Rare and unseen merchants are hashed into a separate range of 1,024
   buckets that starts after the vocabulary, so a tail merchant can never
   share an index with a vocabulary merchant.
3. `heavy_merchant_collisions` lists indices shared by two or more
   high-volume merchants. With the fix the list is empty; running it at each
   retrain catches a table that has become too small.
4. `stable_bucket` uses a fixed digest (BLAKE2b), not Python's `hash()`,
   which is salted per process and would differ between training and
   serving.
5. Per-index fraud rates are smoothed towards the global rate (100
   pseudo-transactions), so one fraud at a tiny merchant does not look like a
   100% fraud rate.

In the same replay, no honest merchant is declined, and all transactions at
the ring's merchants are still declined (96% with the naive hashed feature).

## Trade-offs
- The vocabulary must be versioned with the model and rebuilt at each
  retrain. A merchant that grows past 200 transactions only gets its own
  index at the next retrain.
- Tail merchants still share buckets. A new ring merchant with little
  history can still drag honest small merchants in its tail bucket up. Other
  features (card, device, velocity) have to carry those cases.
- Simply making the table much larger also works: with 2^20 buckets, about
  24 of 5,050 merchants would be expected to collide. That costs memory and
  sparse weights, and it fails silently again as the merchant count grows.
- Learned embeddings or several independent hashes per key (as in a Bloom
  filter) reduce the damage from collisions but do not remove it, and they
  make the model harder to explain to a merchant who was declined.

## Run
```bash
python -m unittest -v test_solution.py
```
