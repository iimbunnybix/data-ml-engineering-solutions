# 031 · The retention team spent a week calling happy customers while the churn model's real high-risk list went uncalled

## Problem
A subscription business scores every customer for churn risk each night and
gives its retention team the 500 highest scores as a call list. To make the
job faster, an engineer split the customer table into chunks of 1,000 and
scored them on a worker pool. The job got much faster, row counts
matched, the average score was unchanged and every dashboard looked normal.
A week later the retention team reported that the "high-risk" customers they
called were mostly active and satisfied, while customers who cancelled that
week had never appeared on the list. The offline model's quality had not
changed at all.

## Root cause
**Scores were attached to customers by row position after a step that does
not keep order.** The workers returned bare lists of scores. The pool hands
results back in the order chunks *finish* (`as_completed`,
`imap_unordered`, Spark tasks writing separate files), and the job
concatenated them and then did `dict(zip(customer_ids, scores))`. Every
score was real, but most belonged to someone else. Nothing in the output
showed a problem: the same customers and the same set of scores were there,
just paired wrongly.

In the replay (`test_solution.py`: 20,000 customers, 20 chunks, one seeded
completion order), **94.2%** of customers get another customer's score. Only
**33 of the real top 500** (6.6%) end up on the call list. The real top 500
have a mean churn score of 0.94, but the scores attached to them average
0.21, about the population mean of 0.20. During testing on one chunk of
1,000 rows the order cannot change, so the bug never showed up there.

## Solution
**Carry the entity key through the parallel step and join on it.**

1. Each worker returns `ScoredCustomer(customer_id, churn_score)` records,
   never bare scores (`score_chunk_with_keys`).
2. `join_scores_by_key` attaches scores by `customer_id`, so the completion
   order no longer matters.
3. The join is checked before anything is written: duplicate input ids, a
   score for an unknown customer, a customer scored twice (a retried chunk)
   or a missing customer (a lost chunk) all raise `JoinMismatchError`.
   A positional join would silently shift every row after such a gap.

With the fix, every customer gets their own score for any completion order,
and the call list equals the real top 500.

## Trade-offs
- Carrying the key costs a little memory and output size per row. That is
  far cheaper than a silent wrong answer.
- The one-to-one check fails the whole batch when one chunk is lost. That is
  intended: publishing yesterday's scores is better than a partial list.
  A retry of only the missing ids could follow.
- An order-preserving pool (`Executor.map`, `Pool.imap`) would also fix this
  example, but the order is still an implicit contract that the next
  refactor (Spark, a queue, a retry path) can break. Keys make the join
  explicit.
- The check proves that the pairing is complete, not that each score is right.
  A spot check that rescores a few random customers one by one and compares
  them with the batch output catches other mix-ups, for example features
  joined by position before scoring.

## Run
```bash
python -m unittest -v test_solution.py
```
