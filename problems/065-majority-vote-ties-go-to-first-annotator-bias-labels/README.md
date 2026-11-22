# 065 · The toxicity model got softer after a relabelling project because every annotator disagreement became the senior annotator's "ok"

## Problem
A social app trains its comment-toxicity classifier on comments labelled by
two annotators each. The labelling pipeline takes the majority label with
`Counter(labels).most_common(1)`. After a relabelling project of 2,000
comments the retrained model flagged noticeably fewer toxic comments, and
user reports of abuse went up. Nobody had changed the model code.

In the seeded simulation in the tests, 627 of the 2,000 comments (31.4%) are
really toxic, but the "majority" labels mark only 441 (22.1%) as toxic, and
186 labels are wrong.

## Root cause
**With two votes, every disagreement is a tie, and `Counter.most_common`
breaks ties by insertion order.** The first vote wins. The senior annotator
always labelled first, and she missed about a quarter of toxic comments, so
the junior annotator's "toxic" never won a disagreement. The aggregated
labels were simply the senior annotator's labels (one test checks they are
identical on all 2,000 items). The same happens with three or more classes
(`ok / toxic / spam` split three ways) or when one annotator's duplicate
submission counts twice.

Nothing failed: a tie looks exactly like a decision.

## Solution
`aggregate_labels`:

1. **One vote per annotator**, the latest one, so a resubmission cannot break
   a tie.
2. **Strict plurality only.** A label is accepted only if it has more votes
   than every other label (`plurality_label`); otherwise the item goes to an
   adjudication queue with reason `tie`.
3. **Minimum votes.** Items with fewer than `MIN_VOTES` (2) go to the queue
   with reason `too_few_votes`.
4. **Agreement rate** (share of items where all annotators agreed) is
   reported, so a drop after onboarding a new annotator is visible.

On the simulation, 1,738 items are accepted with 8 wrong labels (both
annotators wrong), and 262 go to adjudication with an 86.9% agreement rate.
202 of the 262 queued comments are really toxic, so adjudication is where
most of the missing toxic examples come back. Every item ends up either
labelled or queued, and the result no longer depends on vote order.

## Trade-offs
- Adjudication costs reviewer time (13% of items here). Cheaper options:
  a third annotator only for ties, or a documented deterministic rule such
  as "ties count as toxic" for a safety model where misses cost more.
- Training only on accepted items before adjudication is biased too: the
  hard, disputed items are exactly the ones held back (24.0% toxic among
  accepted vs 31.4% overall here). Wait for adjudication or weight them.
- Majority vote treats all annotators as equally reliable. Models such as
  Dawid-Skene estimate each annotator's error rates and weight votes; they
  need more items per annotator to be stable.
- Agreement rate is a raw percentage; Cohen's or Fleiss' kappa corrects for
  agreement by chance when classes are imbalanced.

## Run
```bash
python -m unittest -v test_solution.py
```
