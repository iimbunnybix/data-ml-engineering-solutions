# 011 · Ticket-escalation model scored 84% in validation, then did no better than a coin flip on new customers

## Problem
A B2B support platform trained a model to flag tickets likely to escalate
to a manager, so they could be routed to senior agents immediately. The
training set had a few hundred thousand tickets from a few thousand customer
accounts; large customers filed hundreds each. Validation accuracy looked
excellent and the model shipped. Within a month, routing quality on accounts
onboarded after training was no better than random, senior agents were
swamped with tickets that never escalated, and genuinely urgent tickets from
new accounts waited in the normal queue.

## Root cause
**The train/validation split was done on rows, not on customers.** Every
ticket carries account-level fields (plan, seat count, region, integration
type) that are almost identical across one customer's tickets, and some
customers escalate far more than others. With a random row split, nearly
every validation ticket had sibling tickets from the same account in the
training set. The model learned *"this looks like account X, and account X
escalates"* — a lookup that works only for accounts it has seen.

The synthetic reproduction (`make_ticket_dataset`: 200 customers × 10
tickets, labels driven by the customer, nothing that generalises) shows the
effect with a 1-nearest-neighbour model, which can memorise entities:

| Split | Test rows whose customer is also in train | Accuracy |
|---|---|---|
| Random rows (80/20) | 100% | **84%** |
| By customer (80/20) | 0% | **46%** |
| By customer, 5-fold | 0% | 44–59% per fold, 50% mean |

The 84% is the number the team saw; ~50% is what new customers get.

## Solution
`solution.py`:

1. **`group_split`**: assign each *entity* to train or test with a stable
   hash (`sha256(salt:entity)`), so all of a customer's rows land on one side.
   Unlike Python's `hash()` it is the same in every process, and an entity
   never switches sides as more data arrives (tested), so metrics are
   comparable between retrains.
2. **`group_kfold`**: cross-validation folds with disjoint entities.
3. **`assert_no_entity_leakage`**: a guard in the training pipeline that
   fails the run if any entity appears on both sides, and reports what share
   of test rows are affected. Run it before computing any metric.

Choose the entity to match how the model will be used: if production scores
*new* customers, group by customer; if it scores new tickets from *known*
customers, a time-based split (train on the past, test on the following
weeks) is the honest one, and grouping by customer would understate accuracy.

## Trade-offs
- Group splits make metrics noisier, because the effective sample size is
  the number of entities, not rows (folds above range from 44% to 59%). Use
  more folds or report confidence intervals.
- A few huge entities can unbalance the split; check row counts per side or
  stratify buckets by entity size.
- Grouping by customer does not fix **time leakage** (features computed with
  future data); combine it with a time cut-off and point-in-time features.
- Real models do partly generalise; the gap between the row-split and
  group-split metric is itself a useful number: how much of the score is
  memorisation.
- Libraries have this built in (`GroupShuffleSplit`, `GroupKFold`,
  `StratifiedGroupKFold` in scikit-learn); the hash-based version here is
  useful in SQL/Spark pipelines where the split must be reproducible across
  jobs.

## Run
```bash
python -m unittest -v test_solution.py
```
